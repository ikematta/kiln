//! Batched-throughput gate under the ADR 0003 two-bar split (SPEC §11.3
//! as amended; supersedes the Phase 4 single ≥ 3× bar this test asserted
//! before the ruling — see the 2026-07-10 Phase 6 gate review entry in
//! PROGRESS.md for how the stale version was caught):
//!
//! - **Bar 1 — non-deterministic load** (every row temperature-sampled,
//!   no client seed; one full-width forward per step, B' uninvolved):
//!   SPEC §11.3's ≥ 3× aggregate target, still in force for this lane.
//!   On this tiny 1B model the absolute target is out of reach for
//!   reasons unrelated to B' (recorded pre-/post-B': 334.1/331.6 tok/s
//!   ≈ 2.7× — the small-model sampler-op artifact named in ADR 0003;
//!   the 3× bar is certified at 8B-class: 3.18×). The lane therefore
//!   passes on EITHER the absolute ≥ 3× bar OR no-regression against
//!   its recorded 1B reference.
//! - **Bar 2 — deterministic-containing loads** (greedy/seeded rows
//!   present; B' sub-batching splits every step into ≥ 2 forwards):
//!   governed by ADR 0003's measured FLOOR — the recorded price of
//!   bit-exact batched greedy — not the 3× target: 2.10× all-greedy /
//!   1.80× mixed 8+8, recorded on the dev machine at W = 9. The 50/50
//!   mixed lane is bar-2-governed per the ADR's own floor table (any
//!   deterministic admixture pays the extra full weight pass).
//!
//! Tolerance: each lane asserts ratio ≥ recorded × 0.90. The recorded
//! values are single measured samples, the ledger shows a ~9%
//! run-to-run spread on this harness (Phase 4/5 gate runs: 3.05×-3.34×),
//! and 10% is exactly SPEC §11.3's bench-regression threshold, which
//! ADR 0003 adopts for floor breaches. The 2026-07-10 gate-review
//! measurements (sampled 2.81×, greedy 2.20×, mixed 1.87×) sit above
//! every recorded value and inside that spread — treated as run-to-run
//! headroom, NOT codified as new floors: floors move only via an
//! ADR 0003 revisit and go stale at any mlx-c pin bump.
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

/// ADR 0003 recorded values (B' closeout table, 2026-07-05, dev machine,
/// W = 9): the bar-2 floors and the bar-1 small-model reference. Stale the
/// moment the mlx-c pin moves (the ADR's revisit trigger) — re-bench and
/// update via an ADR 0003 revisit, never ad hoc.
const SAMPLED_1B_RECORDED: f64 = 2.68; // 331.6 tok/s / 123.8 single-stream
const GREEDY_FLOOR: f64 = 2.10;
const MIXED_FLOOR: f64 = 1.80;
/// SPEC §11.3's bench-regression threshold (>10% fails a phase gate),
/// adopted by ADR 0003 for floor breaches; also covers the ~9%
/// run-to-run spread this harness shows across recorded gate runs.
const REGRESSION_TOL: f64 = 0.90;

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
fn batch16_aggregate_meets_adr0003_bars() {
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
    let all_sampled = vec![sampled; BATCH];
    let mixed: Vec<SamplingOptions> = (0..BATCH)
        .map(|i| if i % 2 == 0 { greedy } else { sampled })
        .collect();

    // Warm-up: shader/pipeline caches, pool allocation.
    let _ = run_engine(&model, det_width, &prompt, &[greedy]);

    // Denominator A: Phase-3 contiguous path (async_eval-pipelined).
    let gen_single = median(
        (0..ROUNDS)
            .map(|_| {
                let mut sampler = kiln_engine::Sampler::greedy().expect("sampler");
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

    // Numerators: batch-16 aggregate wall-clock rates, one per lane of
    // the ADR 0003 split. All-sampled is bar 1 (full-width forwards);
    // mixed (8 greedy + 8 sampled) and all-greedy (worst-case
    // sub-batching) are bar-2 deterministic-containing lanes.
    let aggregate_sampled = median(
        (0..ROUNDS)
            .map(|_| run_engine(&model, det_width, &prompt, &all_sampled).0)
            .collect(),
    );
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
    let ratio_sampled = aggregate_sampled / single;
    let ratio_mixed = aggregate_mixed / single;
    let ratio_greedy = aggregate_greedy / single;
    eprintln!(
        "single-stream decode: {gen_single:.1} tok/s (phase-3 pipelined path), \
         {engine_single:.1} tok/s (engine batch 1)"
    );
    eprintln!(
        "batch-{BATCH} aggregate vs the stricter single-stream rate: \
         all-sampled {aggregate_sampled:.1} tok/s -> {ratio_sampled:.2}x (bar 1), \
         mixed {aggregate_mixed:.1} tok/s -> {ratio_mixed:.2}x (bar 2, floor {MIXED_FLOOR}x), \
         all-greedy (B' sub-batched) {aggregate_greedy:.1} tok/s -> {ratio_greedy:.2}x \
         (bar 2, floor {GREEDY_FLOOR}x)"
    );
    // Bar 1: the SPEC ≥ 3x target, else no-regression against the
    // recorded small-model reference (module docs).
    assert!(
        ratio_sampled >= 3.0 || ratio_sampled >= SAMPLED_1B_RECORDED * REGRESSION_TOL,
        "non-deterministic batch-{BATCH} aggregate {aggregate_sampled:.1} tok/s is only \
         {ratio_sampled:.2}x single-stream {single:.1} tok/s — below the SPEC §11.3 3x bar \
         AND more than 10% under the recorded 1B reference {SAMPLED_1B_RECORDED}x \
         (ADR 0003 bar 1)"
    );
    // Bar 2: no-regression below the recorded ADR 0003 floors.
    assert!(
        ratio_mixed >= MIXED_FLOOR * REGRESSION_TOL,
        "mixed batch-{BATCH} aggregate {aggregate_mixed:.1} tok/s is only {ratio_mixed:.2}x \
         single-stream {single:.1} tok/s — more than 10% below the recorded ADR 0003 \
         mixed floor {MIXED_FLOOR}x (bar 2)"
    );
    assert!(
        ratio_greedy >= GREEDY_FLOOR * REGRESSION_TOL,
        "all-greedy batch-{BATCH} aggregate {aggregate_greedy:.1} tok/s is only \
         {ratio_greedy:.2}x single-stream {single:.1} tok/s — more than 10% below the \
         recorded ADR 0003 greedy floor {GREEDY_FLOOR}x (bar 2)"
    );
}
