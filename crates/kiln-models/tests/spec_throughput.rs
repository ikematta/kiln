//! The Phase 8 throughput acceptance run (SPEC §12 Phase 8): measured
//! single-stream decode speedup with speculation ON vs OFF, on the pinned
//! tiny pairs CI can hold:
//!
//! - **qwen3-0.6b-8bit target / qwen3-0.6b-4bit draft** — the certified,
//!   fully in-envelope pair (ADR 0005 bound 7, so the default gamma 4
//!   runs unclamped), measured at gamma 4 and at gamma 3 (the latter
//!   prices the ADR 0005 clamp shape on a pair where speculation has a
//!   real draft-cost gap);
//! - **qwen2.5-0.5b-4bit self-pair at gamma 3** — THE envelope-clamped
//!   model (gqa_factor 7 ⇒ ADR 0005 bound 3). A self-pair's draft costs
//!   as much as its target, so this lane measures the overhead ceiling
//!   of the clamped machinery, not a deployable win.
//!
//! This is a measurement, not a pass/fail speedup gate: the SPEC §12
//! Phase 8 bar (≥1.6× at acceptance >60%) is written for a real
//! size-gap pair ("Qwen3-0.6B drafting for a 14B target"), which no
//! pinned CI model provides — the draft:target weight-byte ratio here is
//! ~0.65 (633MB/968MB), so even perfect acceptance cannot reach 1.6×
//! (per-round cost ≈ gamma·r + 1 target-forward equivalents against
//! 1 + accepted committed tokens). The measured ratios are recorded in
//! PROGRESS.md per phase protocol; the structural gates that DO hold
//! here are the SPEC §11.3 same-family acceptance sanity (>50%) and
//! ON == OFF output equality.
//!
//! `#[ignore]`d like the batching throughput gate: absolute rates are
//! hardware-sensitive. Run explicitly, in release, as part of a phase
//! acceptance:
//!
//! ```text
//! cargo test -p kiln-models --release --test spec_throughput -- --ignored --nocapture
//! ```

#![cfg(feature = "metal")]

use std::cell::RefCell;
use std::collections::HashSet;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use kiln_engine::{
    Engine, EngineConfig, EngineRequest, FinishKind, FinishSummary, PenaltyOptions, Priority,
    SamplingOptions, SeqEvent, SpecStats,
};
use kiln_mlx::Stream;
use kiln_models::{AnyModel, DraftModel, DraftPoolSpec, check_draft_compat};
use kiln_tokenize::Tokenizer;

const DECODE_TOKENS: usize = 256;
const ROUNDS: usize = 5;

/// SPEC §11.3 / throughput.rs prose prompt: representative English text
/// a same-family draft should predict well.
const PROSE: &str = "Pottery is one of the oldest human inventions, and the kiln is the \
                     tool that turned soft clay into something permanent. Long before \
                     writing, people learned that fire changes clay forever, and ";

fn model_dir(name: &str) -> Option<PathBuf> {
    let root = std::env::var_os("KILN_TEST_MODELS")?;
    let dir = PathBuf::from(root).join(name);
    dir.join("config.json").is_file().then_some(dir)
}

fn median(mut rates: Vec<f64>) -> f64 {
    rates.sort_by(|a, b| a.total_cmp(b));
    rates[rates.len() / 2]
}

type Collected = (Rc<RefCell<Vec<u32>>>, Rc<RefCell<Option<FinishSummary>>>);

fn submit(engine: &mut Engine<&AnyModel>, prompt: &[u32]) -> Collected {
    let tokens = Rc::new(RefCell::new(Vec::new()));
    let finish = Rc::new(RefCell::new(None));
    let (t, f) = (Rc::clone(&tokens), Rc::clone(&finish));
    engine.submit(EngineRequest {
        prompt: prompt.to_vec(),
        max_tokens: DECODE_TOKENS,
        sampling: SamplingOptions::default(), // greedy: the speculating class
        penalties: PenaltyOptions {
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        },
        penalty_window: 0,
        stop_tokens: HashSet::new(),
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

/// Median single-stream decode rate over `ROUNDS` runs (plus one
/// discarded warm-up), each on a fresh engine so no state crosses runs.
/// Returns `(tok/s, tokens of the last run, spec stats of the last run)`.
fn measure(
    target: &AnyModel,
    config: &EngineConfig,
    draft_dir: Option<&PathBuf>,
    prompt: &[u32],
) -> (f64, Vec<u32>, SpecStats) {
    let stream = Stream::gpu();
    let mut rates = Vec::with_capacity(ROUNDS);
    let mut last: Option<(Vec<u32>, SpecStats)> = None;
    for round in 0..=ROUNDS {
        let mut engine = Engine::new(target, target.kv_dims(), config.clone(), Stream::gpu())
            .expect("engine builds");
        if let Some(dir) = draft_dir {
            let draft = DraftModel::load(
                dir,
                DraftPoolSpec {
                    block_size: config.block_size,
                    num_blocks: config.num_blocks,
                },
                &stream,
            )
            .expect("draft loads");
            engine.set_drafter(Box::new(draft));
        }
        let (tokens, finish) = submit(&mut engine, prompt);
        while !engine.is_idle() {
            engine.step().expect("engine step");
        }
        let summary = finish.borrow().clone().expect("request finished");
        assert_eq!(summary.reason, FinishKind::Length, "{summary:?}");
        assert_eq!(tokens.borrow().len(), DECODE_TOKENS);
        if round == 0 {
            continue; // warm-up: shader/pipeline caches, pool allocation
        }
        // Decode-window rate: first sampled token -> finish, the lane
        // speculation actually plays in (prefill is untouched by it).
        rates.push((DECODE_TOKENS - 1) as f64 / summary.decode_seconds);
        last = Some((tokens.borrow().clone(), engine.spec_stats()));
    }
    let (tokens, stats) = last.expect("at least one measured round");
    (median(rates), tokens, stats)
}

fn spec_config(target: &AnyModel, det_width: usize, gamma: usize) -> EngineConfig {
    let mut config = EngineConfig {
        num_blocks: 256,
        deterministic_decode_width: det_width,
        // No cross-run prefix reuse: every measured run pays the same
        // prefill and owns the same pool state.
        prefix_cache: false,
        ..EngineConfig::default()
    };
    // Production posture otherwise: the ADR 0005 envelope clamp exactly
    // as the worker applies it, heuristics at their defaults (single
    // stream keeps the width ramp at full gamma; healthy acceptance
    // keeps the stand-down idle — asserted below).
    config.gamma = match target.speculative_gamma_bound() {
        Some(bound) => gamma.min(bound),
        None => 0,
    };
    config
}

struct Lane {
    label: &'static str,
    gamma: usize,
    draft: Option<PathBuf>,
}

/// Runs OFF plus every ON lane for one target; asserts ON == OFF output
/// and prints the ratios. Returns `(off tok/s, per-lane (label, tok/s,
/// ratio, stats))`.
#[allow(clippy::type_complexity)]
fn run_pair(
    target_name: &str,
    target_dir: &PathBuf,
    lanes: &[Lane],
) -> (f64, Vec<(&'static str, f64, f64, SpecStats)>) {
    let stream = Stream::gpu();
    let target = AnyModel::load(target_dir, &stream).expect("target loads");
    let tokenizer = Tokenizer::from_model_dir(target_dir).expect("tokenizer loads");
    let det_width = target
        .calibrate_deterministic_width(&stream)
        .expect("calibrates");
    let prompt = tokenizer.encode(PROSE, true).expect("encodes");
    eprintln!(
        "== {target_name}: prompt {} tokens, decode {DECODE_TOKENS}, deterministic \
         width {det_width}, ADR 0005 envelope {:?}",
        prompt.len(),
        target.speculative_gamma_bound(),
    );

    let off_config = spec_config(&target, det_width, 0);
    let (off_rate, off_tokens, off_stats) = measure(&target, &off_config, None, &prompt);
    assert_eq!(
        off_stats,
        SpecStats::default(),
        "OFF lane must not speculate"
    );
    eprintln!("   speculation OFF: {off_rate:.1} tok/s");

    let mut results = Vec::new();
    for lane in lanes {
        let config = spec_config(&target, det_width, lane.gamma);
        assert_eq!(
            config.gamma, lane.gamma,
            "lane gamma {} is outside the ADR 0005 envelope for {target_name}",
            lane.gamma
        );
        if let Some(draft_dir) = &lane.draft {
            check_draft_compat(target_dir, draft_dir).expect("pair passes the compat gate");
        }
        let (rate, tokens, stats) = measure(&target, &config, lane.draft.as_ref(), &prompt);
        assert_eq!(
            tokens, off_tokens,
            "{target_name} [{}]: speculation changed greedy output",
            lane.label
        );
        assert!(
            stats.rounds_total > 0 && stats.proposed_total > 0,
            "{target_name} [{}]: speculation never engaged: {stats:?}",
            lane.label
        );
        assert_eq!(
            stats.standdowns_total, 0,
            "{target_name} [{}]: the acceptance heuristic stood a healthy pair down: \
             {stats:?}",
            lane.label
        );
        let ratio = rate / off_rate;
        let acceptance = 100.0 * stats.accepted_total as f64 / stats.proposed_total.max(1) as f64;
        eprintln!(
            "   {}: {rate:.1} tok/s -> {ratio:.2}x OFF, acceptance {}/{} ({acceptance:.1}%) \
             over {} rounds, {} rollback rounds",
            lane.label,
            stats.accepted_total,
            stats.proposed_total,
            stats.rounds_total,
            stats.rollback_rounds_total,
        );
        results.push((lane.label, rate, ratio, stats));
    }
    (off_rate, results)
}

#[test]
#[ignore = "perf measurement: run explicitly in release during phase acceptance"]
fn single_stream_speculation_speedup_measured() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    if std::env::var_os("KILN_TEST_MODELS").is_none() {
        eprintln!("skipping: KILN_TEST_MODELS not set");
        return;
    }
    if cfg!(debug_assertions) {
        panic!("the throughput measurement must run in release (--release)");
    }
    kiln_mlx::init();

    // --- The certified in-envelope pair, at the default gamma and at the
    // clamp shape.
    let (Some(qwen3_8bit), Some(qwen3_4bit)) =
        (model_dir("qwen3-0.6b-8bit"), model_dir("qwen3-0.6b-4bit"))
    else {
        panic!("qwen3-0.6b-8bit/-4bit missing — run ./scripts/fetch-test-model.sh");
    };
    let (_, qwen3) = run_pair(
        "qwen3-0.6b-8bit target / qwen3-0.6b-4bit draft",
        &qwen3_8bit,
        &[
            Lane {
                label: "ON gamma 4 (envelope: unclamped)",
                gamma: 4,
                draft: Some(qwen3_4bit.clone()),
            },
            Lane {
                label: "ON gamma 3 (the clamp shape, priced on a real pair)",
                gamma: 3,
                draft: Some(qwen3_4bit.clone()),
            },
        ],
    );
    // SPEC §11.3 sanity holds wherever this runs: same-family draft on
    // English prose accepts >50%.
    let (_, _, _, gamma4_stats) = &qwen3[0];
    assert!(
        gamma4_stats.accepted_total * 2 > gamma4_stats.proposed_total,
        "SPEC §11.3 same-family acceptance sanity failed: {gamma4_stats:?}"
    );

    // --- THE envelope-clamped model (ADR 0005 gamma 3), as a self-pair:
    // the only tokenizer-compatible draft among the pins. Measures the
    // clamped machinery's overhead ceiling (draft cost == target cost),
    // not a deployable speedup.
    let Some(qwen25) = model_dir("qwen2.5-0.5b-4bit") else {
        panic!("qwen2.5-0.5b-4bit missing — run ./scripts/fetch-test-model.sh");
    };
    {
        let stream = Stream::gpu();
        let target = AnyModel::load(&qwen25, &stream).expect("target loads");
        assert_eq!(
            target.speculative_gamma_bound(),
            Some(3),
            "qwen2.5-0.5b should be THE ADR 0005 clamped model"
        );
    }
    run_pair(
        "qwen2.5-0.5b-4bit self-pair",
        &qwen25,
        &[Lane {
            label: "ON gamma 3 (ADR 0005 clamp; self-draft)",
            gamma: 3,
            draft: Some(qwen25.clone()),
        }],
    );
}
