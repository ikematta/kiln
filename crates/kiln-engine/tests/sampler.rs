//! Sampler behavior tests (no model required): greedy argmax, filter
//! semantics, seed determinism, and penalty math.
//!
//! Single `#[test]` because the kiln-mlx live-object counter is
//! process-global (see kiln-mlx/tests/wrappers.rs).

#![cfg(feature = "metal")]

use kiln_engine::{PenaltyOptions, Sampler, SamplingOptions, apply_penalties};
use kiln_mlx::{Array, Stream, debug, ops};

fn stream() -> Stream {
    if kiln_mlx::memory::metal_is_available() {
        Stream::gpu()
    } else {
        Stream::cpu()
    }
}

/// logprobs-shaped input: normalize a handful of raw scores.
fn logprobs(scores: &[f32], s: &Stream) -> Array {
    let raw = Array::from_f32_slice(scores, &[1, scores.len() as i32]).unwrap();
    let lse = ops::logsumexp(&raw, true, s).unwrap();
    ops::subtract(&raw, &lse, s).unwrap()
}

#[test]
fn sampler_behavior() {
    let baseline = debug::live_objects();
    {
        let s = stream();
        greedy_is_argmax(&s);
        top_k_one_is_greedy(&s);
        tight_top_p_is_greedy(&s);
        min_p_one_is_greedy(&s);
        same_seed_same_tokens(&s);
        different_seeds_eventually_differ(&s);
        penalties_match_hand_computation(&s);
    }
    assert_eq!(debug::live_objects(), baseline, "sampler leaked handles");
}

fn greedy_is_argmax(s: &Stream) {
    let lp = logprobs(&[0.1, 2.5, -1.0, 2.4], s);
    let token = Sampler::greedy()
        .sample(&lp, s)
        .unwrap()
        .item_u32()
        .unwrap();
    assert_eq!(token, 1);
}

fn top_k_one_is_greedy(s: &Stream) {
    let lp = logprobs(&[-3.0, 1.0, 4.0, 2.0, -1.0], s);
    for seed in 0..8 {
        let mut sampler = Sampler::new(SamplingOptions {
            temperature: 5.0, // deliberately hot: only the filter saves us
            top_k: 1,
            seed,
            ..SamplingOptions::default()
        });
        assert_eq!(sampler.sample(&lp, s).unwrap().item_u32().unwrap(), 2);
    }
}

fn tight_top_p_is_greedy(s: &Stream) {
    // Top token holds ~88% of the mass; top_p=0.5 keeps only it.
    let lp = logprobs(&[0.0, 3.0, 0.5, 0.2], s);
    for seed in 0..8 {
        let mut sampler = Sampler::new(SamplingOptions {
            temperature: 5.0,
            top_p: 0.5,
            seed,
            ..SamplingOptions::default()
        });
        assert_eq!(sampler.sample(&lp, s).unwrap().item_u32().unwrap(), 1);
    }
}

fn min_p_one_is_greedy(s: &Stream) {
    // min_p = 1.0 keeps only tokens at least as probable as the max.
    let lp = logprobs(&[1.0, 2.0, 0.0, 1.9], s);
    for seed in 0..8 {
        let mut sampler = Sampler::new(SamplingOptions {
            temperature: 5.0,
            min_p: 1.0,
            seed,
            ..SamplingOptions::default()
        });
        assert_eq!(sampler.sample(&lp, s).unwrap().item_u32().unwrap(), 1);
    }
}

fn same_seed_same_tokens(s: &Stream) {
    let lp = logprobs(&[1.0, 1.1, 0.9, 1.05, 0.8, 1.2], s);
    let opts = SamplingOptions {
        temperature: 1.0,
        top_p: 0.95,
        seed: 42,
        ..SamplingOptions::default()
    };
    let draw = |mut sampler: Sampler| -> Vec<u32> {
        (0..16)
            .map(|_| sampler.sample(&lp, s).unwrap().item_u32().unwrap())
            .collect()
    };
    let a = draw(Sampler::new(opts));
    let b = draw(Sampler::new(opts));
    assert_eq!(a, b, "same seed must reproduce the same token stream");
    // The key chain advances between draws: not all 16 draws identical.
    assert!(
        a.windows(2).any(|w| w[0] != w[1]),
        "key chain appears stuck: {a:?}"
    );
}

fn different_seeds_eventually_differ(s: &Stream) {
    let lp = logprobs(&[1.0, 1.1, 0.9, 1.05, 0.8, 1.2], s);
    let draw = |seed: u64| -> Vec<u32> {
        let mut sampler = Sampler::new(SamplingOptions {
            temperature: 1.5,
            seed,
            ..SamplingOptions::default()
        });
        (0..32)
            .map(|_| sampler.sample(&lp, s).unwrap().item_u32().unwrap())
            .collect()
    };
    assert_ne!(
        draw(1),
        draw(2),
        "distinct seeds produced identical streams"
    );
}

fn penalties_match_hand_computation(s: &Stream) {
    let logits = Array::from_f32_slice(&[2.0, -1.0, 0.5, 3.0], &[1, 4]).unwrap();
    // Window: token 0 twice, token 1 once.
    let recent = [0_u32, 1, 0];
    let out = apply_penalties(
        &logits,
        &recent,
        PenaltyOptions {
            repetition_penalty: 2.0,
            presence_penalty: 0.25,
            frequency_penalty: 0.1,
        },
        s,
    )
    .unwrap();
    // token 0: 2.0/2 - 0.25 - 2*0.1 = 0.55 ; token 1: -1*2 - 0.25 - 0.1 = -2.35
    let got = out.data_f32().unwrap();
    let want = [0.55, -2.35, 0.5, 3.0];
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-6, "got {got:?}, want {want:?}");
    }

    // Disabled penalties / empty window are pass-through.
    let same = apply_penalties(
        &logits,
        &[],
        PenaltyOptions {
            repetition_penalty: 2.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        },
        s,
    )
    .unwrap();
    assert_eq!(same.data_f32().unwrap(), logits.data_f32().unwrap());
}
