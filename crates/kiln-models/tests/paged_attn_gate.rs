//! Phase 7 paged-attention kernel acceptance (SPEC §12): decode throughput
//! at 8k context, kernel path vs gather path, ≥ 15% or the flag stays off
//! (documented in PROGRESS.md — that outcome is a documented decision, not
//! a code defect).
//!
//! Also produces the decode-step PROFILE at 8k that the phase's
//! `mlx_compile` rider is gated on (PROGRESS 2026-07-13 scoping entry):
//! the isolated per-layer attention cost both ways, and the step-time
//! composition check (does the engine-level step delta reconcile with
//! layers x attention delta?). The residual — step time not attributable
//! to attention — bounds what op fusion could possibly recover; the
//! numbers go to PROGRESS.md as the "where profiled" evidence.
//!
//! `#[ignore]`d perf gate, same posture as the ADR 0003 throughput gate:
//! run explicitly, in release, on the serving device:
//!
//! ```text
//! KILN_TEST_MODELS=~/.kiln/test-models \
//!   cargo test -p kiln-models --release --test paged_attn_gate -- --ignored --nocapture
//! ```

#![cfg(feature = "metal")]

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use kiln_engine::{BlockManager, KvSpec, WriteRun};
use kiln_engine::{
    Engine, EngineConfig, EngineRequest, FinishKind, FinishSummary, PagedAttnInputs, PagedKv,
    PenaltyOptions, Priority, SamplingOptions, SeqEvent,
};
use kiln_mlx::fast::{self, SdpaMask};
use kiln_mlx::{Array, Dtype, Stream, ops};
use kiln_models::AnyModel;
use kiln_tokenize::Tokenizer;

const MODEL_NAME: &str = "llama-3.2-1b-4bit";
/// Prompt sized so decode runs AT 8k context: 8k prompt + decode window.
const PROMPT_TOKENS: usize = 8064;
const DECODE_TOKENS: usize = 128;
const ROUNDS: usize = 3;
/// SPEC §12 Phase 7: kernel path must gain >= 15% decode throughput at 8k.
const REQUIRED_GAIN: f64 = 1.15;

fn model_dir() -> Option<PathBuf> {
    let root = std::env::var_os("KILN_TEST_MODELS")?;
    let dir = PathBuf::from(root).join(MODEL_NAME);
    dir.join("config.json").is_file().then_some(dir)
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

type Collected = (Rc<RefCell<Vec<u32>>>, Rc<RefCell<Option<FinishSummary>>>);

fn submit(engine: &mut Engine<&AnyModel>, prompt: &[u32], max_tokens: usize) -> Collected {
    let tokens = Rc::new(RefCell::new(Vec::new()));
    let finish = Rc::new(RefCell::new(None));
    let (t, f) = (Rc::clone(&tokens), Rc::clone(&finish));
    engine.submit(EngineRequest {
        prompt: prompt.to_vec(),
        max_tokens,
        sampling: SamplingOptions::default(), // greedy: parity comparable
        penalties: PenaltyOptions {
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        },
        penalty_window: 0,
        stop_tokens: std::collections::HashSet::new(),
        grammar: None,
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
    });
    (tokens, finish)
}

/// One single-stream 8k-context run; returns (decode tok/s over the decode
/// window, generated tokens).
fn run_once(model: &AnyModel, det_width: usize, kernel: bool, prompt: &[u32]) -> (f64, Vec<u32>) {
    let mut engine = Engine::new(
        model,
        model.kv_dims(),
        EngineConfig {
            num_blocks: 320, // 10240 token slots: 8k history + headroom
            deterministic_decode_width: det_width,
            paged_attention_kernel: kernel,
            ..EngineConfig::default()
        },
        Stream::gpu(),
    )
    .expect("engine builds");
    let (tokens, finish) = submit(&mut engine, prompt, DECODE_TOKENS);
    while !engine.is_idle() {
        engine.step().expect("engine step");
    }
    assert_eq!(engine.preemptions(), 0, "8k run must not thrash");
    let summary = finish.borrow().clone().expect("request finished");
    assert_eq!(summary.reason, FinishKind::Length, "{summary:?}");
    let out = tokens.borrow().clone();
    assert_eq!(out.len(), DECODE_TOKENS);
    ((DECODE_TOKENS - 1) as f64 / summary.decode_seconds, out)
}

/// Isolated per-layer attention cost at 8k context (llama-3.2-1b geometry),
/// timed over `reps` evaluated iterations: the gather+SDPA op pair vs the
/// paged kernel, identical pool state. This is the op-level piece of the
/// decode-step profile.
fn isolated_attention_times(s: &Stream) -> (f64, f64) {
    const H: i32 = 32;
    const HK: i32 = 8;
    const D: i32 = 64;
    const BS: usize = 32;
    const N: i32 = 8192;
    const REPS: usize = 50;

    let num_blocks = (N as usize).div_ceil(BS);
    let spec = KvSpec {
        layers: 1,
        kv_heads: HK,
        head_dim: D,
        num_blocks,
        block_size: BS,
    };
    let mut mgr = BlockManager::new(num_blocks, BS).unwrap();
    let mut kv = PagedKv::new(spec);
    kv.enable_attention_kernel().unwrap();
    let blocks: Vec<_> = (0..num_blocks).map(|_| mgr.allocate().unwrap()).collect();
    for block in &blocks {
        let base = ops::zeros(&[1, HK, BS as i32, D], Dtype::Float16, s).unwrap();
        let run = [WriteRun {
            block: *block,
            row_start: 0,
            src_start: 0,
            len: BS as i32,
        }];
        kv.write(0, &run, &base, &base, s).unwrap();
    }
    kiln_mlx::eval(&kv.state()).unwrap();
    let q = ops::zeros(&[1, H, 1, D], Dtype::Float16, s).unwrap();
    let scale = (f64::from(D).powf(-0.5)) as f32;
    let scale_arr = Array::from_f32(scale);
    let inputs = PagedAttnInputs::build(blocks.iter().map(|b| b.index() as u32), N).unwrap();

    let time = |kernel: bool| -> f64 {
        // Warm (JIT/pipeline caches), then time REPS evaluated rounds.
        for phase in 0..2 {
            let start = Instant::now();
            for _ in 0..REPS {
                let out = if kernel {
                    kv.paged_sdpa(0, &q, &inputs, &scale_arr, s).unwrap()
                } else {
                    let (k, v) = kv.gather(0, &blocks, N, s).unwrap();
                    fast::scaled_dot_product_attention(&q, &k, &v, scale, SdpaMask::None, s)
                        .unwrap()
                };
                out.eval().unwrap();
            }
            if phase == 1 {
                return start.elapsed().as_secs_f64() / REPS as f64;
            }
        }
        unreachable!()
    };
    (time(false), time(true))
}

#[test]
#[ignore = "perf gate: run explicitly in release during phase acceptance"]
fn paged_attention_kernel_meets_the_8k_bar() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let Some(dir) = model_dir() else {
        eprintln!("skipping: KILN_TEST_MODELS not set or {MODEL_NAME} missing");
        return;
    };
    if cfg!(debug_assertions) {
        panic!("the 8k gate must run in release (--release)");
    }

    kiln_mlx::init();
    let stream = Stream::gpu();
    let model = AnyModel::load(&dir, &stream).expect("model loads");
    let tokenizer = Tokenizer::from_model_dir(&dir).expect("tokenizer loads");
    let det_width = model
        .calibrate_deterministic_width(&stream)
        .expect("calibrates");

    // 8k prompt: a natural seed encoding, cycled (skipping BOS) to length.
    let seed = tokenizer
        .encode(
            "Pottery is one of the oldest human inventions, and the kiln is its oldest tool. ",
            true,
        )
        .expect("encodes");
    let mut prompt = seed.clone();
    while prompt.len() < PROMPT_TOKENS {
        prompt.extend_from_slice(&seed[1..]);
    }
    prompt.truncate(PROMPT_TOKENS);
    eprintln!(
        "prompt: {} tokens, decode: {DECODE_TOKENS} tokens at ~8k context, \
         deterministic width {det_width}",
        prompt.len()
    );

    // Warm-up both paths (pipeline caches, pool allocation, kernel JIT).
    let (_, warm_gather) = run_once(&model, det_width, false, &prompt);
    let (_, warm_kernel) = run_once(&model, det_width, true, &prompt);
    // Greedy parity at the acceptance context length — the kernel is
    // bit-identical to the gather path by construction (see the
    // kiln-engine paged_attn probe), so the token streams must agree.
    assert_eq!(
        warm_gather, warm_kernel,
        "greedy divergence kernel-vs-gather at 8k context — kernel-class \
         finding (ADR 0002): characterize, do not weaken, stop"
    );

    let gather = median(
        (0..ROUNDS)
            .map(|_| run_once(&model, det_width, false, &prompt).0)
            .collect(),
    );
    let kernel = median(
        (0..ROUNDS)
            .map(|_| run_once(&model, det_width, true, &prompt).0)
            .collect(),
    );
    let gain = kernel / gather;

    // Decode-step profile at 8k (the mlx_compile rider's evidence).
    let (attn_gather_s, attn_kernel_s) = isolated_attention_times(&stream);
    let layers = 16.0; // llama-3.2-1b
    let step_gather_ms = 1000.0 / gather;
    let step_kernel_ms = 1000.0 / kernel;
    let step_delta_ms = step_gather_ms - step_kernel_ms;
    let attn_delta_ms = (attn_gather_s - attn_kernel_s) * 1000.0 * layers;
    let residual_ms = step_kernel_ms - attn_kernel_s * 1000.0 * layers;
    eprintln!(
        "decode @8k: gather {gather:.1} tok/s ({step_gather_ms:.2} ms/step), \
         kernel {kernel:.1} tok/s ({step_kernel_ms:.2} ms/step) -> {gain:.3}x \
         (bar {REQUIRED_GAIN}x)"
    );
    eprintln!(
        "profile @8k: isolated per-layer attention gather {:.3} ms vs kernel {:.3} ms; \
         step delta {step_delta_ms:.2} ms vs {layers} x attention delta {attn_delta_ms:.2} ms \
         (composition {:.0}%); non-attention residual {residual_ms:.2} ms/step \
         = trunk matmuls + sampler + dispatch (fusion headroom bound)",
        attn_gather_s * 1000.0,
        attn_kernel_s * 1000.0,
        100.0 * step_delta_ms / attn_delta_ms
    );

    assert!(
        gain >= REQUIRED_GAIN,
        "kernel path gains only {gain:.3}x at 8k context ({kernel:.1} vs {gather:.1} tok/s) — \
         below the SPEC §12 15% bar: per the spec the flag STAYS OFF and the result is \
         documented in PROGRESS.md (this failing gate is that documentation trigger, \
         not a code defect)"
    );
}
