//! Continuous-batching correctness (SPEC §6.2 / Phase 4 part 2): the
//! batched/paged engine must produce the same greedy token streams as the
//! Phase-3 single-request contiguous path, alone and under 2-4-way
//! concurrency (SPEC §6.6/§11.3, CLAUDE.md determinism: batching must not
//! change greedy outputs).
//!
//! Two distinct guarantees back these token assertions (ADR 0002):
//! - engine at batch 1 vs the contiguous path: bit-identical by
//!   construction — both run the canonical prefill schedule (including its
//!   kernel-class padding of ragged tail pieces) and M=1 decode, so every
//!   kernel dispatch matches shape-for-shape;
//! - concurrent runs vs solo: token-id equality only. Concurrency raises
//!   the trunk's row count across MLX's qmv/qmm dispatch classes, so the
//!   logits differ from solo in ulps by library design; a pass means
//!   greedy argmax absorbed that noise for these prompts, not that the
//!   math is bit-identical.
//!
//! Skips (with a note) when `KILN_TEST_MODELS` is unset or Metal is
//! unavailable. Single `#[test]` because the kiln-mlx live-object counter
//! is process-global.

#![cfg(feature = "metal")]

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use kiln_engine::{
    Engine, EngineConfig, EngineRequest, ErrorCause, FinishKind, FinishSummary, PenaltyOptions,
    Priority, SamplingOptions, SeqEvent,
};
use kiln_mlx::{Stream, debug};
use kiln_models::{LlamaModel, generate};
use kiln_tokenize::Tokenizer;

const MODEL_NAME: &str = "llama-3.2-1b-4bit";

fn model_dir() -> Option<PathBuf> {
    let root = std::env::var_os("KILN_TEST_MODELS")?;
    let dir = PathBuf::from(root).join(MODEL_NAME);
    dir.join("config.json").is_file().then_some(dir)
}

type Collected = (Rc<RefCell<Vec<u32>>>, Rc<RefCell<Option<FinishSummary>>>);

fn request(prompt: &[u32], max_tokens: usize, stop: &[u32]) -> (EngineRequest, Collected) {
    let tokens = Rc::new(RefCell::new(Vec::new()));
    let finish = Rc::new(RefCell::new(None));
    let (t, f) = (Rc::clone(&tokens), Rc::clone(&finish));
    let request = EngineRequest {
        prompt: prompt.to_vec(),
        max_tokens,
        sampling: SamplingOptions::default(), // greedy
        penalties: PenaltyOptions {
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        },
        penalty_window: 0,
        stop_tokens: stop.iter().copied().collect(),
        priority: Priority::Interactive,
        cancel: Arc::new(AtomicBool::new(false)),
        on_event: Box::new(move |event| {
            match event {
                SeqEvent::Token(token) => t.borrow_mut().push(token),
                SeqEvent::PrefixHit { .. } => {}
                SeqEvent::Finished(summary) => *f.borrow_mut() = Some(summary),
            }
            true
        }),
    };
    (request, (tokens, finish))
}

fn drain(engine: &mut Engine<&LlamaModel>) {
    while !engine.is_idle() {
        engine.step().expect("engine step");
    }
}

/// Runs each prompt through a fresh engine, all submitted up front.
fn run_batch(
    model: &LlamaModel,
    config: EngineConfig,
    prompts: &[(&[u32], usize)],
) -> Vec<Vec<u32>> {
    let mut engine =
        Engine::new(model, model.kv_dims(), config, Stream::gpu()).expect("engine builds");
    let mut collected = Vec::new();
    for &(prompt, max_tokens) in prompts {
        let (request, results) = request_pair(prompt, max_tokens);
        engine.submit(request);
        collected.push(results);
    }
    drain(&mut engine);
    finish_tokens(collected)
}

fn request_pair(prompt: &[u32], max_tokens: usize) -> (EngineRequest, Collected) {
    request(prompt, max_tokens, &[])
}

fn finish_tokens(collected: Vec<Collected>) -> Vec<Vec<u32>> {
    collected
        .into_iter()
        .map(|(tokens, finish)| {
            let summary = finish.borrow().clone().expect("request finished");
            assert_eq!(
                summary.reason,
                FinishKind::Length,
                "expected a full-length run: {summary:?}"
            );
            tokens.borrow().clone()
        })
        .collect()
}

#[test]
fn batched_greedy_matches_single_stream() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let Some(model_dir) = model_dir() else {
        eprintln!("skipping: KILN_TEST_MODELS not set or {MODEL_NAME} missing");
        return;
    };

    let baseline = debug::live_objects();
    {
        let stream = Stream::gpu();
        let model = LlamaModel::load(&model_dir, &stream).expect("model loads");
        let tokenizer = Tokenizer::from_model_dir(&model_dir).expect("tokenizer loads");
        let encode = |text: &str| tokenizer.encode(text, true).expect("encodes");

        let short = encode("The capital of France is");
        let medium = encode(
            "Paged attention splits the key-value cache into fixed-size blocks \
             so requests can share and release memory independently. Explain why.",
        );
        let long = encode(&"The quick brown fox jumps over the lazy dog. ".repeat(16));
        assert!(long.len() > 100, "long prompt should span several chunks");
        // A prompt whose prefill limit lands a few tokens past a fine
        // boundary: its ragged tail piece is shorter than
        // PREFILL_PAD_MIN_ROWS and exercises the ADR 0002 kernel-class
        // padding on BOTH paths (engine and contiguous).
        let mut tiny_rem = encode(&"Clay hardens in the kiln's steady fire. ".repeat(18));
        let fine = kiln_engine::DEFAULT_PREFILL_FINE_CHUNK;
        tiny_rem.truncate(fine + 6); // limit = fine + 5 -> ragged piece of 5
        assert_eq!((tiny_rem.len() - 1) % fine, 5, "ragged tail size drifted");

        // Prompt sizes and budgets chosen to cross block boundaries (32)
        // during decode, to split the long prompt into several chunks
        // under the small-chunk config below, and to hit the padded
        // ragged-tail path.
        let jobs: Vec<(&[u32], usize)> = vec![
            (&short, 40),
            (&medium, 24),
            (&long, 24),
            (&short, 12),
            (&tiny_rem, 24),
        ];

        let config = EngineConfig {
            num_blocks: 64,
            ..EngineConfig::default()
        };
        let chunked = EngineConfig {
            num_blocks: 64,
            prefill_chunk: 48,
            ..EngineConfig::default()
        };

        // 1) Engine (paged, batch of 1) == Phase-3 contiguous path, with
        //    the production chunk size.
        let reference = |prompt: &[u32], max_tokens: usize| -> Vec<u32> {
            let mut sampler = kiln_engine::Sampler::greedy();
            generate(
                &model,
                prompt,
                max_tokens,
                |logprobs, s| sampler.sample(logprobs, s),
                &stream,
            )
            .expect("generates")
            .tokens
        };
        let contiguous: Vec<Vec<u32>> = jobs
            .iter()
            .map(|&(prompt, max_tokens)| reference(prompt, max_tokens))
            .collect();
        for (i, &(prompt, max_tokens)) in jobs.iter().enumerate() {
            let solo = run_batch(&model, config.clone(), &[(prompt, max_tokens)]);
            assert_eq!(
                solo[0], contiguous[i],
                "paged engine diverged from the contiguous path on job {i}"
            );
        }
        eprintln!("solo engine == contiguous path for {} jobs", jobs.len());

        // 2) 4-way concurrency (with chunked prefill interleaving decode)
        //    == solo runs under the same chunk config, twice (run-to-run
        //    determinism).
        let solo_chunked: Vec<Vec<u32>> = jobs
            .iter()
            .map(|&job| run_batch(&model, chunked.clone(), &[job]).remove(0))
            .collect();
        for round in 0..2 {
            let batched = run_batch(&model, chunked.clone(), &jobs);
            assert_eq!(
                batched, solo_chunked,
                "batched greedy outputs diverged from single-stream (round {round})"
            );
        }
        eprintln!("4-way batched == solo (2 rounds, chunk=48)");

        // 3) Late join: requests arriving mid-decode must not disturb the
        //    stream already in flight.
        {
            let mut engine = Engine::new(&model, model.kv_dims(), chunked.clone(), Stream::gpu())
                .expect("engine builds");
            let (first, first_out) = request_pair(&short, 40);
            engine.submit(first);
            for _ in 0..6 {
                engine.step().expect("engine step");
            }
            let (second, second_out) = request_pair(&long, 24);
            let (third, third_out) = request_pair(&medium, 24);
            engine.submit(second);
            engine.submit(third);
            drain(&mut engine);
            let outs = finish_tokens(vec![first_out, second_out, third_out]);
            assert_eq!(outs[0], solo_chunked[0], "late join disturbed running seq");
            assert_eq!(outs[1], solo_chunked[2], "late-joined long seq diverged");
            assert_eq!(outs[2], solo_chunked[1], "late-joined medium seq diverged");
        }
        eprintln!("late-join batching matches solo");

        // 4) Stop tokens: sampled stop finishes the request, is counted
        //    but not streamed.
        {
            let reference = &contiguous[0];
            let stop = reference[2];
            let mut engine = Engine::new(&model, model.kv_dims(), config.clone(), Stream::gpu())
                .expect("engine builds");
            let (request, (tokens, finish)) = request(&short, 40, &[stop]);
            engine.submit(request);
            drain(&mut engine);
            let summary = finish.borrow().clone().expect("finished");
            assert_eq!(summary.reason, FinishKind::Stop);
            assert_eq!(summary.matched_stop_token, Some(stop));
            assert_eq!(summary.completion_tokens, 3, "stop token is counted");
            assert_eq!(tokens.borrow().as_slice(), &reference[..2]);
        }
        eprintln!("stop-token semantics ok");

        // 5) Pool pressure: an unservable request still fails at submit
        //    (in-band, with the OOM_REJECTED-mapped capacity cause). A
        //    mid-flight exhaustion no longer fails the loser — the §6.1
        //    preemption returns the most recent request to WAITING and it
        //    resumes to a bit-exact stream once the survivor frees blocks
        //    (deeper scenarios, priorities, and the golden-fixture case
        //    live in tests/preemption.rs).
        {
            let tiny = EngineConfig {
                num_blocks: 4,
                ..EngineConfig::default()
            };
            let mut engine =
                Engine::new(&model, model.kv_dims(), tiny, Stream::gpu()).expect("engine builds");
            let (request, (_, finish)) = request_pair(&long, 24);
            engine.submit(request);
            drain(&mut engine);
            let summary = finish.borrow().clone().expect("finished");
            assert_eq!(summary.reason, FinishKind::Error);
            assert_eq!(summary.error_cause, Some(ErrorCause::Capacity));
            assert!(
                summary.error.as_deref().unwrap_or("").contains("KV pool"),
                "unexpected error: {:?}",
                summary.error
            );

            // Mid-flight pressure. Pool: 6 blocks (192 token slots). The
            // survivor (short prompt, 60 tokens) grows from 1 to 3 blocks;
            // the loser passes the submit-time capacity check
            // (150 + 24 = 174 <= 192) but needs a 6th block ~11 decode
            // steps in while the pool is full: it self-preempts (it is the
            // most recently admitted), waits for the survivor, and resumes
            // through re-prefill + replay onto the same greedy stream.
            let filler = &long[..150];
            let squeeze = EngineConfig {
                num_blocks: 6,
                ..EngineConfig::default()
            };
            let mut engine = Engine::new(&model, model.kv_dims(), squeeze, Stream::gpu())
                .expect("engine builds");
            let (first, (first_tokens, first_finish)) = request_pair(&short, 60);
            let (second, (second_tokens, second_finish)) = request_pair(filler, 24);
            engine.submit(first);
            engine.submit(second);
            // Direct successor of the pre-preemption "loser should fail
            // mid-decode, not at submit" check: the pressure event must
            // land mid-stream, after the loser already emitted tokens.
            while engine.preemptions() == 0 {
                assert!(!engine.is_idle(), "drained without any preemption");
                engine.step().expect("engine step");
            }
            assert!(
                !second_tokens.borrow().is_empty(),
                "loser should be preempted mid-decode, not before streaming"
            );
            drain(&mut engine);
            let first_summary = first_finish.borrow().clone().expect("finished");
            let second_summary = second_finish.borrow().clone().expect("finished");
            assert_eq!(
                first_summary.reason,
                FinishKind::Length,
                "survivor finishes"
            );
            assert_eq!(
                second_summary.reason,
                FinishKind::Length,
                "preempted request must finish after resuming: {second_summary:?}"
            );
            assert!(
                second_summary.preemptions >= 1,
                "the most recent request was never preempted: {second_summary:?}"
            );
            assert_eq!(
                first_summary.preemptions, 0,
                "the senior request must not be preempted"
            );
            assert_eq!(
                first_tokens.borrow().clone(),
                reference(&short, 60),
                "survivor's stream disturbed by the preempted request"
            );
            assert_eq!(
                second_tokens.borrow().clone(),
                reference(filler, 24),
                "preempted request's resumed stream diverged from its solo run"
            );
            assert!(engine.preemptions() >= 1, "engine preemption counter");
        }
        eprintln!("pool-pressure preemption ok");
    }
    assert_eq!(
        debug::live_objects(),
        baseline,
        "batching run leaked mlx handles"
    );
}
