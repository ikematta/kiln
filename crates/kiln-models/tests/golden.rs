//! Golden-token parity harness (SPEC §11.2, the keystone test): every Rust
//! model implementation must reproduce the committed mlx-lm reference
//! fixtures token-for-token, exactly, for every model directory under
//! `tests/golden/` (architecture dispatch via `AnyModel`).
//!
//! Since Phase 4 the fixtures run through the continuous-batching engine
//! (paged KV + gather-based paged attention, SPEC §6.2/§7.4 v0) — the
//! production decode path — with the production chunk size (2048, matching
//! mlx-lm's `prefill_step_size`). Each model runs twice, against the two
//! ADR 0002 bars:
//! - request by request (single-stream): **bit-exact** — the M=1 decode and
//!   the (pad-aligned, ADR 0002) prefill build reference-class kernels, so
//!   any token difference is a model bug;
//! - with the decode batch pinned at width 16: **token-id equality** with
//!   the same reference. At width 16 the trunk matmuls cross MLX's
//!   qmv/qmm dispatch threshold, so the logits differ from single-stream
//!   in ulps by library design — a pass here means greedy argmax absorbed
//!   that noise on every fixture token (the SPEC §6.6/§11.3 invariant:
//!   batching must not change greedy *output*), not that the math is
//!   bit-identical. `tests/batching.rs` separately pins engine == Phase-3
//!   contiguous path.
//!
//! Fixture semantics (mirrored from `scripts/gen-golden.py` — keep in sync):
//! - `chat_template: true`  -> render the model's template for one user
//!   message with `add_generation_prompt` and the pinned `date_string`,
//!   encode WITHOUT special tokens (the template emits BOS itself);
//! - `chat_template: false` -> encode the raw prompt WITH special tokens;
//! - greedy, exactly `max_tokens` tokens, no EOS stopping.
//!
//! Skips (with a note) when `KILN_TEST_MODELS` is unset or Metal is
//! unavailable. A fixture directory whose model is missing under
//! `KILN_TEST_MODELS` FAILS (fetch-test-model.sh and tests/golden must not
//! drift apart), as does a local model revision differing from the one the
//! fixtures were generated against.

#![cfg(feature = "metal")]

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use kiln_engine::{
    Engine, EngineConfig, EngineRequest, FinishKind, FinishSummary, PenaltyOptions, Priority,
    SamplingOptions, SeqEvent,
};
use kiln_mlx::{Stream, debug};
use kiln_models::AnyModel;
use kiln_tokenize::{ChatMessage, ChatTemplate, Tokenizer};
use minijinja::Value;

/// Must match PINNED_DATE_STRING in scripts/gen-golden.py.
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

type Collected = (Rc<RefCell<Vec<u32>>>, Rc<RefCell<Option<FinishSummary>>>);

/// Submits one greedy request (no stop tokens) and returns its
/// token-stream/finish handles.
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

fn assert_full_length(finish: &Rc<RefCell<Option<FinishSummary>>>) -> FinishSummary {
    let summary = finish.borrow().clone().expect("request finished");
    assert_eq!(
        summary.reason,
        FinishKind::Length,
        "fixtures and fillers run to max_tokens: {summary:?}"
    );
    summary
}

/// Runs one greedy request through the batching engine and returns every
/// generated token (no stop tokens, so the token stream is complete).
fn engine_generate(
    engine: &mut Engine<&AnyModel>,
    prompt_ids: &[u32],
    max_tokens: usize,
) -> Vec<u32> {
    let (tokens, finish) = submit_collected(engine, prompt_ids, max_tokens);
    while !engine.is_idle() {
        engine.step().expect("engine step");
    }
    assert_full_length(&finish);
    tokens.borrow().clone()
}

/// Safety bound for the batch-16 rounds (~200 steps expected each).
const MAX_STEPS: usize = 4000;

/// Runs one fixture request with its decode batch pinned at width 16 (the
/// Phase-4 / mlx#3120 checkpoint: trunk-matmul kernel dispatch depends on
/// M, so bit-parity at M=1 does not extend to M=16 — the assertion this
/// feeds is the ADR 0002 token-id bar, not a bit bar).
///
/// 15 filler requests are submitted (and, being FIFO, admitted) first and
/// sized to outlive the fixture, so every sampled position of the fixture
/// comes from a decode step carrying exactly 16 sequences. Asserted, not
/// assumed: the width stays 16 from fixture admission to fixture finish,
/// all 15 fillers are still running when the fixture finishes, and no
/// preemption occurred.
fn engine_generate_at_width16(
    engine: &mut Engine<&AnyModel>,
    prompt_ids: &[u32],
    max_tokens: usize,
    filler_prompt: &[u32],
) -> Vec<u32> {
    assert!(
        engine.is_idle(),
        "width-16 rounds start from an idle engine"
    );
    let preemptions_before = engine.preemptions();
    // Admission is staggered (one prefill per step), so a filler admitted
    // k steps before the fixture needs at most max_tokens + 16 tokens to
    // outlive it; +24 leaves margin.
    let filler_max = max_tokens + 24;
    let fillers: Vec<Collected> = (0..15)
        .map(|_| submit_collected(engine, filler_prompt, filler_max))
        .collect();
    let (tokens, finish) = submit_collected(engine, prompt_ids, max_tokens);

    let mut width16_seen = false;
    let mut steps = 0;
    while finish.borrow().is_none() {
        assert!(steps < MAX_STEPS, "width-16 round livelocked");
        engine.step().expect("engine step");
        steps += 1;
        let width = engine.num_running();
        width16_seen |= width == 16;
        if width16_seen && finish.borrow().is_none() {
            assert_eq!(width, 16, "decode width fell below 16 mid-fixture");
        }
    }
    assert!(width16_seen, "the batch never reached width 16");
    for (i, (_, filler_finish)) in fillers.iter().enumerate() {
        assert!(
            filler_finish.borrow().is_none(),
            "filler {i} finished before the fixture — the fixture's tail \
             decoded below width 16 (resize filler_max)"
        );
    }
    assert_eq!(
        engine.preemptions(),
        preemptions_before,
        "width-16 round preempted; the pool is undersized for this fixture"
    );
    assert_full_length(&finish);
    let out = tokens.borrow().clone();
    while !engine.is_idle() {
        assert!(steps < MAX_STEPS, "width-16 drain livelocked");
        engine.step().expect("engine step");
        steps += 1;
    }
    for (_, filler_finish) in &fillers {
        assert_full_length(filler_finish);
    }
    out
}

/// Loads one fixture-model directory and proves parity for all its fixtures,
/// request-by-request and at decode width 16.
fn run_model(model_name: &str, model_dir: &PathBuf, fixture_paths: &[PathBuf]) {
    let local_revision = std::fs::read_to_string(model_dir.join(".kiln-revision"))
        .map(|text| text.trim().to_owned())
        .unwrap_or_default();

    let stream = Stream::gpu();
    let model = AnyModel::load(model_dir, &stream).expect("model loads");
    let tokenizer = Tokenizer::from_model_dir(model_dir).expect("tokenizer loads");
    let template = ChatTemplate::from_model_dir(model_dir).expect("template loads");
    eprintln!(
        "== {model_name}: model_type={}, {} fixture(s)",
        model.model_type(),
        fixture_paths.len()
    );
    // Production config except pool size: 256 blocks x 32 = 8192 token
    // slots, ample for every fixture. The engine is reused across
    // fixtures, exercising block recycling between requests.
    let mut engine = Engine::new(
        &model,
        model.kv_dims(),
        EngineConfig {
            num_blocks: 256,
            ..EngineConfig::default()
        },
        Stream::gpu(),
    )
    .expect("engine builds");

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
                "fixture {model_name}/{name} was generated for a different weights revision \
                 than the local test model — refetch models or regenerate fixtures (when told to)"
            );
            let prompt_ids = if fixture.chat_template {
                let rendered = template
                    .render_with(
                        &[ChatMessage {
                            role: "user".into(),
                            content: fixture.prompt.clone(),
                        }],
                        true,
                        &[("date_string", Value::from(PINNED_DATE_STRING))],
                    )
                    .expect("template renders");
                // BOS contract: the rendered template already contains BOS.
                tokenizer.encode(&rendered, false).expect("encodes")
            } else {
                tokenizer.encode(&fixture.prompt, true).expect("encodes")
            };
            (name, fixture, prompt_ids)
        })
        .collect();

    for (name, fixture, prompt_ids) in &fixtures {
        let output = engine_generate(&mut engine, prompt_ids, fixture.max_tokens);
        assert_eq!(
            output,
            fixture.expected_token_ids,
            "greedy token divergence on fixture {model_name}/{name} \
             (prompt tokens: {})",
            prompt_ids.len()
        );
        eprintln!(
            "golden {model_name}/{name}: {} prompt tokens, {} generated — exact match \
             (batched/paged engine)",
            prompt_ids.len(),
            fixture.max_tokens
        );
    }

    // Width-16 rounds: the same fixtures under the ADR 0002 batched bar —
    // token-id equality with the single-stream reference. The logits at
    // width 16 differ from M=1 in ulps by MLX kernel-dispatch design
    // (qmv/qmm classes); greedy argmax must absorb that noise on every
    // token. A failure here with the sequential rounds green is therefore
    // a *batched-enablement* gate for this architecture at this pin, not
    // evidence of a model-math bug (ADR 0002 "Consequences").
    let filler_prompt = tokenizer
        .encode("Pottery is one of the oldest human inventions", true)
        .expect("encodes");
    for (name, fixture, prompt_ids) in &fixtures {
        let output =
            engine_generate_at_width16(&mut engine, prompt_ids, fixture.max_tokens, &filler_prompt);
        assert_eq!(
            output,
            fixture.expected_token_ids,
            "greedy token-id divergence on fixture {model_name}/{name} at decode width 16 \
             (prompt tokens: {}) — batched decode fails the ADR 0002 token-equality bar \
             for this architecture at this kernel pin",
            prompt_ids.len()
        );
        eprintln!(
            "golden {model_name}/{name}: {} prompt tokens, {} generated — token-id match \
             at decode width 16 (bit-exactness not implied at M>1; ADR 0002)",
            prompt_ids.len(),
            fixture.max_tokens
        );
    }
}

#[test]
fn greedy_parity_is_exact_for_every_fixture_model() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let Some(root) = std::env::var_os("KILN_TEST_MODELS") else {
        eprintln!("skipping: KILN_TEST_MODELS not set");
        return;
    };
    let root = PathBuf::from(root);

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

    let baseline = debug::live_objects();
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
        assert!(
            !fixture_paths.is_empty(),
            "no fixtures under tests/golden/{model_name}"
        );
        run_model(model_name, &model_dir, &fixture_paths);
    }
    assert_eq!(
        debug::live_objects(),
        baseline,
        "golden run leaked mlx handles"
    );
}
