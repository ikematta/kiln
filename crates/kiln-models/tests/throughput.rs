//! Phase 4 throughput gate (SPEC §12 / §11.3): batch-16 aggregate decode
//! throughput must be ≥ 3× single-stream on the tiny model.
//!
//! Since ADR 0002 B' the gate runs on a MIXED-traffic-representative
//! batch (8 greedy + 8 temperature-sampled): production traffic is a
//! blend, greedy rows now pay the deterministic sub-batching cost, and
//! sampled rows ride full-width — the blend is the number that matters.
//! The all-greedy (worst-case sub-batched) and mixed aggregates are both
//! measured and reported; the ≥ 3× assertion applies to both.
//!
//! `#[ignore]`d: this is a perf gate, not a correctness test — absolute
//! rates are hardware-sensitive and shared CI runners make even the ratio
//! noisy. Run it explicitly, in release, as part of a phase acceptance:
//!
//! ```text
//! cargo test -p kiln-models --release --test throughput -- --ignored --nocapture
//! ```
//!
//! Two single-stream denominators are measured and the gate holds against
//! the *stricter* (faster) one:
//! - the Phase-3 contiguous `generate` path (async_eval-pipelined — the
//!   mlx-lm-class number recorded in Phase 3), and
//! - the engine at batch 1 (the production serving path).

#![cfg(feature = "metal")]

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use kiln_engine::{
    Engine, EngineConfig, EngineRequest, FinishKind, FinishSummary, PenaltyOptions, Priority,
    SamplingOptions, SeqEvent,
};
use kiln_mlx::Stream;
use kiln_models::{LlamaModel, generate};
use kiln_tokenize::Tokenizer;

const MODEL_NAME: &str = "llama-3.2-1b-4bit";
const DECODE_TOKENS: usize = 128;
const BATCH: usize = 16;
const ROUNDS: usize = 3;

fn model_dir() -> Option<PathBuf> {
    let root = std::env::var_os("KILN_TEST_MODELS")?;
    let dir = PathBuf::from(root).join(MODEL_NAME);
    dir.join("config.json").is_file().then_some(dir)
}

fn median(mut rates: Vec<f64>) -> f64 {
    rates.sort_by(|a, b| a.total_cmp(b));
    rates[rates.len() / 2]
}

type Collected = (Rc<RefCell<Vec<u32>>>, Rc<RefCell<Option<FinishSummary>>>);

fn submit(
    engine: &mut Engine<&LlamaModel>,
    prompt: &[u32],
    max_tokens: usize,
    sampling: SamplingOptions,
) -> Collected {
    let tokens = Rc::new(RefCell::new(Vec::new()));
    let finish = Rc::new(RefCell::new(None));
    let (t, f) = (Rc::clone(&tokens), Rc::clone(&finish));
    engine.submit(EngineRequest {
        prompt: prompt.to_vec(),
        max_tokens,
        sampling,
        penalties: PenaltyOptions {
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        },
        penalty_window: 0,
        stop_tokens: std::collections::HashSet::new(),
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

/// Runs one request per sampling spec through a fresh engine; returns
/// `(aggregate tokens/sec over the whole run, per-request summaries)`.
/// The wall clock includes admission and prefill — conservative for the
/// batched numerator, symmetric for the batch-1 denominator.
fn run_engine(
    model: &LlamaModel,
    det_width: usize,
    prompt: &[u32],
    samplings: &[SamplingOptions],
) -> (f64, Vec<FinishSummary>) {
    let mut engine = Engine::new(
        model,
        model.kv_dims(),
        EngineConfig {
            deterministic_decode_width: det_width,
            ..EngineConfig::default()
        },
        Stream::gpu(),
    )
    .expect("engine builds");
    let outs: Vec<Collected> = samplings
        .iter()
        .map(|&sampling| submit(&mut engine, prompt, DECODE_TOKENS, sampling))
        .collect();
    let start = Instant::now();
    while !engine.is_idle() {
        engine.step().expect("engine step");
    }
    let elapsed = start.elapsed().as_secs_f64();
    assert_eq!(engine.preemptions(), 0, "throughput run must not thrash");
    let summaries: Vec<FinishSummary> = outs
        .iter()
        .map(|(tokens, finish)| {
            let summary = finish.borrow().clone().expect("request finished");
            assert_eq!(summary.reason, FinishKind::Length, "{summary:?}");
            assert_eq!(tokens.borrow().len(), DECODE_TOKENS);
            summary
        })
        .collect();
    (
        (samplings.len() * DECODE_TOKENS) as f64 / elapsed,
        summaries,
    )
}

#[test]
#[ignore = "perf gate: run explicitly in release during phase acceptance"]
fn batch16_aggregate_is_3x_single_stream() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let Some(dir) = model_dir() else {
        eprintln!("skipping: KILN_TEST_MODELS not set or {MODEL_NAME} missing");
        return;
    };
    if cfg!(debug_assertions) {
        panic!("throughput gate must run in release (--release)");
    }

    kiln_mlx::init();
    let stream = Stream::gpu();
    let model = LlamaModel::load(&dir, &stream).expect("model loads");
    let tokenizer = Tokenizer::from_model_dir(&dir).expect("tokenizer loads");
    let det_width = model
        .calibrate_deterministic_width(&stream)
        .expect("calibrates");
    let text = "Pottery is one of the oldest human inventions, and the kiln is its \
                oldest tool. Clay hardens into ceramic when ";
    let prompt = tokenizer.encode(text, true).expect("encodes");
    eprintln!(
        "prompt: {} tokens, decode: {DECODE_TOKENS} tokens, deterministic width {det_width}",
        prompt.len()
    );
    let greedy = SamplingOptions::default();
    // Production-representative sampled traffic: temperature + top-p, no
    // client seed -> non-deterministic class, full-width forwards.
    let sampled = SamplingOptions {
        temperature: 0.7,
        top_p: 0.9,
        ..SamplingOptions::default()
    };
    let all_greedy = vec![greedy; BATCH];
    let mixed: Vec<SamplingOptions> = (0..BATCH)
        .map(|i| if i % 2 == 0 { greedy } else { sampled })
        .collect();

    // Warm-up: shader/pipeline caches, pool allocation.
    let _ = run_engine(&model, det_width, &prompt, &[greedy]);

    // Denominator A: Phase-3 contiguous path (async_eval-pipelined).
    let gen_single = median(
        (0..ROUNDS)
            .map(|_| {
                let mut sampler = kiln_engine::Sampler::greedy();
                let out = generate(
                    &model,
                    &prompt,
                    DECODE_TOKENS,
                    |logprobs, s| sampler.sample(logprobs, s),
                    &stream,
                )
                .expect("generates");
                out.decode_tokens_per_sec()
            })
            .collect(),
    );

    // Denominator B: the engine at batch 1, decode-window rate.
    let engine_single = median(
        (0..ROUNDS)
            .map(|_| {
                let (_, summaries) = run_engine(&model, det_width, &prompt, &[greedy]);
                (DECODE_TOKENS - 1) as f64 / summaries[0].decode_seconds
            })
            .collect(),
    );

    // Numerators: batch-16 aggregate wall-clock rates. Mixed (8 greedy +
    // 8 sampled) is the production-representative gate; all-greedy is the
    // worst case for B' sub-batching (every row deterministic).
    let aggregate_mixed = median(
        (0..ROUNDS)
            .map(|_| run_engine(&model, det_width, &prompt, &mixed).0)
            .collect(),
    );
    let aggregate_greedy = median(
        (0..ROUNDS)
            .map(|_| run_engine(&model, det_width, &prompt, &all_greedy).0)
            .collect(),
    );

    let single = gen_single.max(engine_single);
    let ratio_mixed = aggregate_mixed / single;
    let ratio_greedy = aggregate_greedy / single;
    eprintln!(
        "single-stream decode: {gen_single:.1} tok/s (phase-3 pipelined path), \
         {engine_single:.1} tok/s (engine batch 1)"
    );
    eprintln!(
        "batch-{BATCH} aggregate: mixed {aggregate_mixed:.1} tok/s -> {ratio_mixed:.2}x, \
         all-greedy (B' sub-batched) {aggregate_greedy:.1} tok/s -> {ratio_greedy:.2}x \
         the stricter single-stream rate"
    );
    assert!(
        ratio_mixed >= 3.0,
        "mixed batch-{BATCH} aggregate {aggregate_mixed:.1} tok/s is only {ratio_mixed:.2}x \
         single-stream {single:.1} tok/s (SPEC §12 Phase 4 gate: >= 3x)"
    );
    assert!(
        ratio_greedy >= 3.0,
        "all-greedy batch-{BATCH} aggregate {aggregate_greedy:.1} tok/s is only \
         {ratio_greedy:.2}x single-stream {single:.1} tok/s (SPEC §12 Phase 4 gate: >= 3x)"
    );
}
