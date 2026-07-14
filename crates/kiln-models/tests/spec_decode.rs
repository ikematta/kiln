//! The Phase 8 part 2 correctness gate (SPEC §6.5): greedy output with
//! speculation ON must be token-identical to greedy output with
//! speculation OFF — the committed golden fixtures — for EVERY fixture,
//! for every drafter this suite can throw at the verify loop. Speculation
//! is a throughput feature that is not allowed to perturb correctness;
//! this is the full golden harness rerun under speculation, not a spot
//! check.
//!
//! Drafter matrix per fixture model:
//! - **self-draft**: the model drafts for itself (`DraftModel` over the
//!   same checkpoint — trivially tokenizer-compatible). High acceptance;
//!   exercises the accept/bonus path, draft catch-up, reconcile, and the
//!   prefix-cache composition (engine cache stays ON and is shared across
//!   fixtures, so later fixtures speculate over warm prefixes).
//! - **adversarial**: a scripted drafter proposing valid-but-wrong ids.
//!   Near-total rejection; exercises the correction + rollback path every
//!   round. Invariance under a garbage drafter is the strongest form of
//!   the claim: verification, not proposal quality, owns correctness.
//!
//! Also covered here (single #[test]: the mlx live-object counter is
//! process-global, and cases in one binary run concurrently):
//! - the drafter-attachment compatibility gate (`check_draft_compat`):
//!   the cross-tokenizer qwen3-draft/llama-target pair is rejected
//!   LOUDLY; the qwen3-0.6b-8bit/qwen3-0.6b-4bit pair passes and records
//!   the SPEC §11.3 acceptance-rate bar (>50% on English prose,
//!   same-family draft);
//! - spec_max_batch gating: with more requests admitted the drafter is
//!   never consulted (SPEC §6.5 auto-disable-by-width);
//! - the in-situ O(1) rollback measurement: rollback nanos per round,
//!   measured inside real verify rounds, must not grow with context
//!   length (the unit-level scaling companion is kiln-engine's
//!   rollback_cost suite).
//!
//! Baselines follow the ADR 0004 device split via `KILN_FIXTURE_PARITY`
//! (the preemption/prefill_pad convention): unset or `only` compares
//! speculation-ON output against the committed fixtures — the strict
//! generating-device bar (the golden suite separately proves OFF ==
//! fixture there, so this is ON == OFF by transitivity); `skip` compares
//! against a live speculation-OFF run on the same device — the
//! device-independent form of the SPEC §6.5 invariant, valid on any GPU.
//!
//! Divergences are COLLECTED across every model and drafter and reported
//! together at the end (the test still fails if any exist — the bar is
//! not weakened; the collection only completes the record for the
//! characterization workflow this build follows).
//!
//! Skips (with a note) when `KILN_TEST_MODELS` is unset or Metal is
//! unavailable; a fixture directory whose model is missing FAILS.

#![cfg(feature = "metal")]

use std::cell::RefCell;
use std::collections::HashSet;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use kiln_engine::{
    DraftError, Drafter, DrafterMemory, Engine, EngineConfig, EngineRequest, FinishKind,
    FinishSummary, PenaltyOptions, Priority, SamplingOptions, SeqEvent, SpecStats,
};
use kiln_mlx::{Stream, debug};
use kiln_models::{AnyModel, DraftModel, DraftPoolSpec, check_draft_compat};
use kiln_tokenize::{ChatMessage, ChatTemplate, Tokenizer};
use minijinja::Value;

/// Must match PINNED_DATE_STRING in scripts/gen-golden.py (and golden.rs).
const PINNED_DATE_STRING: &str = "26 Jul 2024";

#[derive(Debug, serde::Deserialize)]
struct Fixture {
    prompt: String,
    chat_template: bool,
    max_tokens: usize,
    expected_token_ids: Vec<u32>,
    #[allow(dead_code)]
    mlx_lm_version: String,
    weights_revision: String,
}

fn golden_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden")
}

fn model_root() -> Option<PathBuf> {
    std::env::var_os("KILN_TEST_MODELS").map(PathBuf::from)
}

/// A drafter that proposes syntactically valid but (for any real prompt)
/// wrong token ids — the total-rejection adversary. Every verify round
/// commits exactly the target's own token and rolls the rest back.
struct AdversarialDrafter {
    seqs: HashSet<u64>,
    /// Small ids, valid in every fixture vocab.
    script: [u32; 4],
    proposals: u64,
}

impl AdversarialDrafter {
    fn new() -> Self {
        Self {
            seqs: HashSet::new(),
            script: [11, 23, 5, 42],
            proposals: 0,
        }
    }
}

impl Drafter for AdversarialDrafter {
    fn memory(&self) -> DrafterMemory {
        DrafterMemory::default()
    }

    fn begin(&mut self, seq: u64, _prompt: &[u32], _s: &Stream) -> Result<(), DraftError> {
        self.seqs.insert(seq);
        Ok(())
    }

    fn propose(
        &mut self,
        seq: u64,
        _committed: &[u32],
        gamma: usize,
        _s: &Stream,
    ) -> Result<Vec<u32>, DraftError> {
        if !self.seqs.contains(&seq) {
            return Err(DraftError::UnknownSeq(seq));
        }
        self.proposals += 1;
        Ok(self
            .script
            .iter()
            .cycle()
            .skip((self.proposals as usize) % self.script.len())
            .take(gamma)
            .copied()
            .collect())
    }

    fn release(&mut self, seq: u64) {
        self.seqs.remove(&seq);
    }
}

/// A drafter that never proposes but counts consultations — observes the
/// SPEC §6.5 spec_max_batch gate without perturbing decoding.
struct CountingDrafter {
    seqs: HashSet<u64>,
    calls: Rc<RefCell<u64>>,
}

impl Drafter for CountingDrafter {
    fn memory(&self) -> DrafterMemory {
        DrafterMemory::default()
    }

    fn begin(&mut self, seq: u64, _prompt: &[u32], _s: &Stream) -> Result<(), DraftError> {
        self.seqs.insert(seq);
        Ok(())
    }

    fn propose(
        &mut self,
        seq: u64,
        _committed: &[u32],
        _gamma: usize,
        _s: &Stream,
    ) -> Result<Vec<u32>, DraftError> {
        if !self.seqs.contains(&seq) {
            return Err(DraftError::UnknownSeq(seq));
        }
        *self.calls.borrow_mut() += 1;
        Ok(Vec::new())
    }

    fn release(&mut self, seq: u64) {
        self.seqs.remove(&seq);
    }
}

type Collected = (Rc<RefCell<Vec<u32>>>, Rc<RefCell<Option<FinishSummary>>>);

fn submit_collected(
    engine: &mut Engine<&AnyModel>,
    prompt_ids: &[u32],
    max_tokens: usize,
) -> Collected {
    let tokens = Rc::new(RefCell::new(Vec::new()));
    let finish = Rc::new(RefCell::new(None));
    let (t, f) = (Rc::clone(&tokens), Rc::clone(&finish));
    engine.submit(EngineRequest {
        prompt: prompt_ids.to_vec(),
        max_tokens,
        sampling: SamplingOptions::default(), // greedy
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

/// One full-length greedy run; returns the tokens and the finish summary.
fn engine_generate(
    engine: &mut Engine<&AnyModel>,
    prompt_ids: &[u32],
    max_tokens: usize,
) -> (Vec<u32>, FinishSummary) {
    let (tokens, finish) = submit_collected(engine, prompt_ids, max_tokens);
    while !engine.is_idle() {
        engine.step().expect("engine step");
    }
    let summary = finish.borrow().clone().expect("request finished");
    assert_eq!(
        summary.reason,
        FinishKind::Length,
        "spec-decode runs go to max_tokens: {summary:?}"
    );
    (tokens.borrow().clone(), summary)
}

fn engine_config(model: &AnyModel, det_width: usize) -> EngineConfig {
    // Production posture, golden.rs pool sizing; prefix cache stays ON so
    // speculation and the radix cache are proven composing (SPEC §12
    // Phase 8 acceptance), incl. donation of verify-round partial tails.
    let mut config = EngineConfig {
        num_blocks: 256,
        deterministic_decode_width: det_width,
        ..EngineConfig::default()
    };
    if model.monolithic_prefill_required() {
        config.prefill_fine_chunk = config.prefill_chunk;
    }
    config
}

/// First index where two token streams disagree (or a length note).
fn divergence_note(actual: &[u32], baseline: &[u32]) -> Option<String> {
    match actual.iter().zip(baseline).position(|(a, b)| a != b) {
        Some(at) => Some(format!(
            "index {at}: {} vs baseline {}",
            actual[at], baseline[at]
        )),
        None if actual.len() != baseline.len() => Some(format!(
            "length {} vs baseline {}",
            actual.len(),
            baseline.len()
        )),
        None => None,
    }
}

/// Loads one fixture model and checks speculation-on output against the
/// baseline (committed fixture, or a live speculation-off run under
/// `KILN_FIXTURE_PARITY=skip`) for every fixture, under both drafter
/// kinds. Divergences are appended to `failures` rather than panicking,
/// so one run yields the complete matrix.
fn run_model_with_speculation(
    model_name: &str,
    model_dir: &PathBuf,
    fixture_paths: &[PathBuf],
    failures: &mut Vec<String>,
) {
    let local_revision = std::fs::read_to_string(model_dir.join(".kiln-revision"))
        .map(|text| text.trim().to_owned())
        .unwrap_or_default();
    let stream = Stream::gpu();
    let model = AnyModel::load(model_dir, &stream).expect("model loads");
    let tokenizer = Tokenizer::from_model_dir(model_dir).expect("tokenizer loads");
    let template = ChatTemplate::from_model_dir(model_dir).expect("template loads");
    let det_width = model
        .calibrate_deterministic_width(&stream)
        .expect("calibrates");
    let fixtures: Vec<(String, Fixture, Vec<u32>)> = fixture_paths
        .iter()
        .map(|path| {
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .to_owned();
            let fixture: Fixture =
                serde_json::from_str(&std::fs::read_to_string(path).expect("fixture readable"))
                    .expect("fixture parses");
            assert_eq!(
                fixture.weights_revision, local_revision,
                "fixture {model_name}/{name} was generated for a different weights revision"
            );
            let prompt_ids = if fixture.chat_template {
                let rendered = template
                    .render_with(
                        &[ChatMessage::text("user", fixture.prompt.clone())],
                        true,
                        &[("date_string", Value::from(PINNED_DATE_STRING))],
                    )
                    .expect("template renders");
                tokenizer.encode(&rendered, false).expect("encodes")
            } else {
                tokenizer.encode(&fixture.prompt, true).expect("encodes")
            };
            (name, fixture, prompt_ids)
        })
        .collect();
    let gamma_effective = EngineConfig::default()
        .gamma
        .min(det_width.saturating_sub(1));
    eprintln!(
        "== {model_name}: {} fixture(s), deterministic width {det_width}, \
         gamma effective {gamma_effective}",
        fixtures.len(),
    );

    // ADR 0004 device split: on the generating device the committed
    // fixture IS the speculation-off output (the golden suite holds that
    // bar); on foreign devices (`KILN_FIXTURE_PARITY=skip`) the baseline
    // is a live speculation-off run — the device-independent form of the
    // SPEC §6.5 on-vs-off invariant.
    let live_baseline = std::env::var("KILN_FIXTURE_PARITY").as_deref() == Ok("skip");
    let baselines: Vec<Vec<u32>> = if live_baseline {
        let mut engine = Engine::new(
            &model,
            model.kv_dims(),
            engine_config(&model, det_width),
            Stream::gpu(),
        )
        .expect("engine builds");
        fixtures
            .iter()
            .map(|(_, fixture, prompt_ids)| {
                engine_generate(&mut engine, prompt_ids, fixture.max_tokens).0
            })
            .collect()
    } else {
        fixtures
            .iter()
            .map(|(_, fixture, _)| fixture.expected_token_ids.clone())
            .collect()
    };
    let baseline_kind = if live_baseline {
        "live speculation-off run"
    } else {
        "committed fixture"
    };

    for (drafter_kind, mk_drafter) in [
        (
            "self-draft",
            Box::new(|| -> Box<dyn Drafter> {
                Box::new(
                    DraftModel::load(
                        model_dir,
                        DraftPoolSpec {
                            block_size: 32,
                            num_blocks: 256,
                        },
                        &stream,
                    )
                    .expect("self-draft loads"),
                )
            }) as Box<dyn Fn() -> Box<dyn Drafter>>,
        ),
        (
            "adversarial",
            Box::new(|| -> Box<dyn Drafter> { Box::new(AdversarialDrafter::new()) })
                as Box<dyn Fn() -> Box<dyn Drafter>>,
        ),
    ] {
        let mut engine = Engine::new(
            &model,
            model.kv_dims(),
            engine_config(&model, det_width),
            Stream::gpu(),
        )
        .expect("engine builds");
        engine.set_drafter(mk_drafter());
        for ((name, fixture, prompt_ids), baseline) in fixtures.iter().zip(&baselines) {
            let (output, _) = engine_generate(&mut engine, prompt_ids, fixture.max_tokens);
            if let Some(note) = divergence_note(&output, baseline) {
                failures.push(format!(
                    "{model_name}/{name} [{drafter_kind}, vs {baseline_kind}]: {note}"
                ));
            }
        }
        let stats = engine.spec_stats();
        if gamma_effective == 0 {
            // Dense-trunk protection (ADR 0002 B' width 1): the engine's
            // gamma clamp keeps speculation entirely off — the invariance
            // above held on the plain path, and that is the designed
            // behavior, not vacuity.
            assert_eq!(
                stats,
                SpecStats::default(),
                "width-1 model must never speculate: {stats:?}"
            );
            eprintln!(
                "{model_name} {drafter_kind}: speculation disabled by the deterministic-width \
                 clamp (width {det_width}); plain-path outputs verified"
            );
            continue;
        }
        assert!(
            stats.rounds_total > 0 && stats.proposed_total > 0,
            "{drafter_kind} speculation never engaged on {model_name} — the invariance \
             rounds above were vacuous: {stats:?}"
        );
        if drafter_kind == "self-draft" {
            assert!(
                stats.accepted_total * 2 > stats.proposed_total,
                "self-draft acceptance below 50% on {model_name}: {stats:?} — the SPEC \
                 §11.3 same-family sanity bar fails even for the identity pair"
            );
        } else {
            assert!(
                stats.rollback_rounds_total > 0,
                "an adversarial drafter must trigger rollbacks: {stats:?}"
            );
        }
        eprintln!(
            "{model_name} {drafter_kind}: {} rounds, {}/{} accepted ({:.1}%), {} rollback \
             rounds ({} tokens rolled back, mean rollback {}ns)",
            stats.rounds_total,
            stats.accepted_total,
            stats.proposed_total,
            100.0 * stats.accepted_total as f64 / stats.proposed_total.max(1) as f64,
            stats.rollback_rounds_total,
            stats.rollback_tokens_total,
            stats.rollback_nanos_total / stats.rollback_rounds_total.max(1),
        );
    }
}

/// SPEC §6.5 spec_max_batch: with more requests admitted than
/// `spec_max_batch`, the drafter is not consulted; once the batch narrows
/// it is. Observed step-by-step with a counting drafter whose proposals
/// are empty (so decoding itself stays on the plain path throughout, and
/// the outputs must equal a drafter-less run bit-for-bit).
fn check_spec_max_batch_gate(model: &AnyModel, det_width: usize) {
    let spec_max_batch = EngineConfig::default().spec_max_batch;
    let wide = spec_max_batch + 2;
    // 1-token prompts admit together (no prefill), so the batch is wide
    // from the very first step.
    let prompts: Vec<Vec<u32>> = (0..wide).map(|i| vec![100 + i as u32]).collect();
    // Three finish early, narrowing the batch below the gate.
    let max_tokens = |i: usize| if i < 3 { 8 } else { 40 };

    let baseline: Vec<Vec<u32>> = {
        let mut engine = Engine::new(
            model,
            model.kv_dims(),
            engine_config(model, det_width),
            Stream::gpu(),
        )
        .expect("engine builds");
        let handles: Vec<Collected> = prompts
            .iter()
            .enumerate()
            .map(|(i, p)| submit_collected(&mut engine, p, max_tokens(i)))
            .collect();
        while !engine.is_idle() {
            engine.step().expect("engine step");
        }
        handles.iter().map(|(t, _)| t.borrow().clone()).collect()
    };

    let calls = Rc::new(RefCell::new(0u64));
    let mut engine = Engine::new(
        model,
        model.kv_dims(),
        engine_config(model, det_width),
        Stream::gpu(),
    )
    .expect("engine builds");
    engine.set_drafter(Box::new(CountingDrafter {
        seqs: HashSet::new(),
        calls: Rc::clone(&calls),
    }));
    let handles: Vec<Collected> = prompts
        .iter()
        .enumerate()
        .map(|(i, p)| submit_collected(&mut engine, p, max_tokens(i)))
        .collect();
    let mut consulted_narrow = false;
    while !engine.is_idle() {
        let width_before = engine.num_running().max(engine.num_active());
        let calls_before = *calls.borrow();
        engine.step().expect("engine step");
        if *calls.borrow() > calls_before {
            assert!(
                width_before <= spec_max_batch,
                "drafter consulted at batch width {width_before} > spec_max_batch \
                 {spec_max_batch}"
            );
            consulted_narrow = true;
        }
    }
    assert!(
        consulted_narrow,
        "the drafter was never consulted even after the batch narrowed below \
         spec_max_batch — the gate test is vacuous"
    );
    for (i, (tokens, _)) in handles.iter().enumerate() {
        assert_eq!(
            *tokens.borrow(),
            baseline[i],
            "empty proposals changed greedy output for request {i}"
        );
    }
    eprintln!(
        "spec_max_batch gate: width {wide} never consulted the drafter; narrowed batch did \
         ({} consultations), outputs bit-identical",
        calls.borrow()
    );
}

/// In-situ O(1) rollback: mean rollback nanos per verify round, measured
/// inside real engine steps under total rejection, must not grow with the
/// context the sequence carries. Companion to kiln-engine's rollback_cost
/// unit measurement, this one exercises the real code path (settle →
/// truncate → release) with live pools.
fn check_rollback_is_flat_in_situ(model: &AnyModel, det_width: usize) {
    let mut config = engine_config(model, det_width);
    config.prefix_cache = false; // keep both runs shape-identical and lean
    let mean_rollback_nanos = |prompt_len: usize| -> u64 {
        let prompt: Vec<u32> = (0..prompt_len).map(|i| 100 + (i % 1000) as u32).collect();
        let mut engine = Engine::new(model, model.kv_dims(), config.clone(), Stream::gpu())
            .expect("engine builds");
        engine.set_drafter(Box::new(AdversarialDrafter::new()));
        let _ = engine_generate(&mut engine, &prompt, 40);
        let stats = engine.spec_stats();
        assert!(
            stats.rollback_rounds_total >= 20,
            "expected steady rejection rounds, got {stats:?}"
        );
        stats.rollback_nanos_total / stats.rollback_rounds_total
    };
    let short = mean_rollback_nanos(8);
    let long = mean_rollback_nanos(6000);
    eprintln!(
        "in-situ rollback: {short}ns/round at 8-token context, {long}ns/round at \
         6000-token context"
    );
    assert!(
        long <= short.saturating_mul(25).max(20_000),
        "in-situ rollback cost grew with context length ({short}ns -> {long}ns) — \
         the O(1) claim fails in practice"
    );
}

#[test]
fn speculation_preserves_greedy_output_and_measures_its_costs() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let Some(root) = model_root() else {
        eprintln!("skipping: KILN_TEST_MODELS not set");
        return;
    };

    kiln_mlx::init();
    let baseline_objects = debug::live_objects();
    {
        // --- Compatibility gate (SPEC §6.5, this part): the cross-family
        // pair part 1 loaded for isolation is REJECTED for drafting, and
        // loudly; the same-tokenizer cross-quant pair passes; self-pairs
        // pass trivially.
        let llama = root.join("llama-3.2-1b-4bit");
        let qwen3_4bit = root.join("qwen3-0.6b-4bit");
        let qwen3_8bit = root.join("qwen3-0.6b-8bit");
        if llama.join("config.json").is_file() && qwen3_4bit.join("config.json").is_file() {
            let err = check_draft_compat(&llama, &qwen3_4bit)
                .expect_err("a qwen3 draft under a llama target must be rejected");
            let msg = err.to_string();
            assert!(
                msg.contains("incompatible"),
                "rejection must be loud and name the cause: {msg}"
            );
            eprintln!("compat gate rejects qwen3-draft/llama-target: {msg}");
        }
        if qwen3_8bit.join("config.json").is_file() && qwen3_4bit.join("config.json").is_file() {
            check_draft_compat(&qwen3_8bit, &qwen3_4bit)
                .expect("same-tokenizer cross-quant pair must pass the compat gate");
        }

        // --- The full golden harness with speculation ON.
        let mut model_names: Vec<String> = std::fs::read_dir(golden_root())
            .expect("tests/golden exists")
            .filter_map(|entry| {
                let entry = entry.expect("readable dir entry");
                entry
                    .path()
                    .is_dir()
                    .then(|| entry.file_name().to_string_lossy().into_owned())
            })
            .collect();
        model_names.sort();
        assert!(!model_names.is_empty(), "no golden fixture directories");
        let mut failures: Vec<String> = Vec::new();
        for model_name in &model_names {
            let model_dir = root.join(model_name);
            assert!(
                model_dir.join("config.json").is_file(),
                "fixtures exist for {model_name} but the model is missing under \
                 KILN_TEST_MODELS — run ./scripts/fetch-test-model.sh"
            );
            let mut fixture_paths: Vec<PathBuf> = std::fs::read_dir(golden_root().join(model_name))
                .expect("fixture dir readable")
                .filter_map(|entry| {
                    let path = entry.expect("readable dir entry").path();
                    (path.extension().is_some_and(|e| e == "json")).then_some(path)
                })
                .collect();
            fixture_paths.sort();
            check_draft_compat(&model_dir, &model_dir)
                .expect("a model is always draft-compatible with itself");
            run_model_with_speculation(model_name, &model_dir, &fixture_paths, &mut failures);
        }

        // --- Cross-quant pair: acceptance-rate record + composition.
        if qwen3_8bit.join("config.json").is_file() && qwen3_4bit.join("config.json").is_file() {
            let stream = Stream::gpu();
            let target = AnyModel::load(&qwen3_8bit, &stream).expect("target loads");
            let tokenizer = Tokenizer::from_model_dir(&qwen3_8bit).expect("tokenizer loads");
            let det_width = target
                .calibrate_deterministic_width(&stream)
                .expect("calibrates");
            let prose = tokenizer
                .encode(
                    "Pottery is one of the oldest human inventions, and the kiln is the \
                     tool that turned soft clay into something permanent.",
                    true,
                )
                .expect("encodes");

            // Baseline (speculation off), then the same engine shape with
            // the 4-bit draft attached — outputs must match exactly.
            let mut engine = Engine::new(
                &target,
                target.kv_dims(),
                engine_config(&target, det_width),
                Stream::gpu(),
            )
            .expect("engine builds");
            let (baseline, _) = engine_generate(&mut engine, &prose, 64);
            drop(engine);

            let mut engine = Engine::new(
                &target,
                target.kv_dims(),
                engine_config(&target, det_width),
                Stream::gpu(),
            )
            .expect("engine builds");
            let draft = DraftModel::load(
                &qwen3_4bit,
                DraftPoolSpec {
                    block_size: 32,
                    num_blocks: 256,
                },
                &stream,
            )
            .expect("draft loads");
            engine.set_drafter(Box::new(draft));
            let (speculated, summary) = engine_generate(&mut engine, &prose, 64);
            assert_eq!(
                speculated, baseline,
                "qwen3-0.6b-8bit greedy output moved under a qwen3-0.6b-4bit draft"
            );
            let stats = engine.spec_stats();
            assert!(
                stats.accepted_total * 2 > stats.proposed_total,
                "SPEC §11.3 acceptance-rate sanity (>50%, same-family draft, English \
                 prose) failed: {stats:?}"
            );
            assert_eq!(
                (
                    summary.spec_tokens_proposed as u64,
                    summary.spec_tokens_accepted as u64
                ),
                (stats.proposed_total, stats.accepted_total),
                "per-request Timings counters must mirror the engine totals for a \
                 single-request engine"
            );
            // Composition (SPEC §12 Phase 8): resubmit — the radix cache
            // serves the prompt while speculation keeps decoding; both
            // must be active in one request, and the output still exact.
            let (warm, warm_summary) = engine_generate(&mut engine, &prose, 64);
            assert_eq!(warm, baseline, "warm-prefix speculation diverged");
            assert!(
                warm_summary.cached_prompt_tokens > 0,
                "the resubmit was expected to hit the prefix cache"
            );
            let warm_stats = engine.spec_stats();
            assert!(
                warm_stats.rounds_total > stats.rounds_total,
                "speculation did not run on the warm-prefix request"
            );
            eprintln!(
                "qwen3-0.6b-8bit target / qwen3-0.6b-4bit draft: {}/{} tokens accepted \
                 ({:.1}%) over {} rounds on prose; warm-prefix run reused {} prompt \
                 tokens with speculation active",
                stats.accepted_total,
                stats.proposed_total,
                100.0 * stats.accepted_total as f64 / stats.proposed_total.max(1) as f64,
                stats.rounds_total,
                warm_summary.cached_prompt_tokens,
            );
        }

        // --- Batch-width gate + in-situ rollback scaling, on the primary
        // rust-worker model.
        if llama.join("config.json").is_file() {
            let stream = Stream::gpu();
            let model = AnyModel::load(&llama, &stream).expect("model loads");
            let det_width = model
                .calibrate_deterministic_width(&stream)
                .expect("calibrates");
            check_spec_max_batch_gate(&model, det_width);
            check_rollback_is_flat_in_situ(&model, det_width);
        }

        // The invariance verdict, last so one run yields the complete
        // record (every model, both drafters, gates and measurements).
        assert!(
            failures.is_empty(),
            "greedy output moved under speculation — the SPEC §6.5 invariant is broken on \
             {} case(s):\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
    }

    kiln_mlx::memory::clear_cache().expect("cache clears");
    assert_eq!(
        debug::live_objects(),
        baseline_objects,
        "speculative decoding leaked mlx handles"
    );
}
