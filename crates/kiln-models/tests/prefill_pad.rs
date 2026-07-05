//! ADR 0002 kernel-class padding on real weights (Phase 6 / B1): the
//! `raw-tiny-remainder` fixture's prefill limit is 137 = 128 + 9, so the
//! canonical schedule ends in a 9-row ragged piece — below every
//! qmv/qmm dispatch threshold in the pinned MLX, hence computed in the
//! vector-kernel class unless padded. Reproducing this fixture at all is
//! proof the padding puts those rows back in the reference class.
//!
//! The engine-side row-accounting contract is pinned by
//! `kiln-engine/tests/prefill_pad.rs` with a checksum mock; this test pins
//! the value side on the real model:
//! - cold single-stream output over the padded piece is bit-exact vs the
//!   committed mlx-lm fixture (the ADR 0002 single-stream bar);
//! - a containment rerun is served the rows the padded piece wrote and
//!   stays bit-identical to the cold run (pads never leaked into cached
//!   blocks);
//! - an extension prompt resuming at a canonical boundary equals its
//!   cache-cold run (donated coverage is genuine rows only).
//!
//! The rerun and extension checks compare against the same-process cold
//! run, so they are device-independent; only the cold-vs-fixture assert is
//! cross-device sensitive (ADR 0002 — the fixture was generated on the dev
//! machine). `KILN_FIXTURE_PARITY` splits them for CI (PROGRESS
//! 2026-07-05, Option B ruling): `skip` = device-independent checks only
//! (blocking step), `only` = the fixture comparison only (advisory step),
//! unset = everything — the dev-machine bar is unchanged.
//!
//! Skips when `KILN_TEST_MODELS` is unset or Metal is unavailable.

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
use kiln_tokenize::Tokenizer;

const MODEL_NAME: &str = "llama-3.2-1b-4bit";

/// CI scenario split — see the module doc. Keep in sync with the copy in
/// `tests/preemption.rs` (integration tests cannot share modules without
/// a common-mod file; two 15-line copies beat that machinery).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    All,
    DeviceIndependent,
    FixtureOnly,
}

impl Scope {
    /// Warm-vs-cold checks: same-device comparisons, valid on any GPU.
    fn runs_device_independent(self) -> bool {
        self != Scope::FixtureOnly
    }
    /// Comparison against the committed dev-machine fixture (ADR 0002).
    fn runs_fixture(self) -> bool {
        self != Scope::DeviceIndependent
    }
}

fn scope() -> Scope {
    match std::env::var("KILN_FIXTURE_PARITY").as_deref() {
        Err(_) | Ok("") => Scope::All,
        Ok("skip") => Scope::DeviceIndependent,
        Ok("only") => Scope::FixtureOnly,
        Ok(other) => {
            panic!("KILN_FIXTURE_PARITY must be unset, empty, \"skip\", or \"only\": {other:?}")
        }
    }
}

#[derive(Debug, serde::Deserialize)]
struct Fixture {
    prompt: String,
    max_tokens: usize,
    expected_token_ids: Vec<u32>,
}

struct Outcome {
    tokens: Rc<RefCell<Vec<u32>>>,
    hits: Rc<RefCell<Vec<u32>>>,
    finish: Rc<RefCell<Option<FinishSummary>>>,
}

fn run(engine: &mut Engine<&AnyModel>, prompt: &[u32], max_tokens: usize) -> Outcome {
    let outcome = Outcome {
        tokens: Rc::new(RefCell::new(Vec::new())),
        hits: Rc::new(RefCell::new(Vec::new())),
        finish: Rc::new(RefCell::new(None)),
    };
    let (t, h, f) = (
        Rc::clone(&outcome.tokens),
        Rc::clone(&outcome.hits),
        Rc::clone(&outcome.finish),
    );
    engine.submit(EngineRequest {
        prompt: prompt.to_vec(),
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
                SeqEvent::PrefixHit { tokens, .. } => h.borrow_mut().push(tokens),
                SeqEvent::Finished(summary) => *f.borrow_mut() = Some(summary),
            }
            true
        }),
    });
    while !engine.is_idle() {
        engine.step().expect("engine step");
    }
    let summary = outcome.finish.borrow().clone().expect("request finished");
    assert_eq!(summary.reason, FinishKind::Length, "full run: {summary:?}");
    outcome
}

#[test]
fn padded_ragged_piece_is_reference_exact_and_cache_clean() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let Some(root) = std::env::var_os("KILN_TEST_MODELS") else {
        eprintln!("skipping: KILN_TEST_MODELS not set");
        return;
    };
    let model_dir = PathBuf::from(root).join(MODEL_NAME);
    if !model_dir.join("config.json").is_file() {
        panic!("{MODEL_NAME} missing under KILN_TEST_MODELS — run ./scripts/fetch-test-model.sh");
    }
    let scope = scope();
    let fixture: Fixture = serde_json::from_str(
        &std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/golden")
                .join(MODEL_NAME)
                .join("raw-tiny-remainder.json"),
        )
        .expect("fixture readable"),
    )
    .expect("fixture parses");

    let baseline = debug::live_objects();
    {
        let stream = Stream::gpu();
        let model = AnyModel::load(&model_dir, &stream).expect("model loads");
        let tokenizer = Tokenizer::from_model_dir(&model_dir).expect("tokenizer loads");
        let prompt = tokenizer.encode(&fixture.prompt, true).expect("encodes");
        assert_eq!(
            (prompt.len() - 1) % 128,
            9,
            "fixture drifted: the prompt no longer produces a 9-row ragged piece"
        );
        let config = EngineConfig {
            num_blocks: 128,
            ..EngineConfig::default()
        };
        let mut warm = Engine::new(&model, model.kv_dims(), config.clone(), Stream::gpu())
            .expect("engine builds");

        // 1) Cold: bit-exact vs the mlx-lm fixture THROUGH the padded
        // ragged piece (unpadded, those 9 rows compute in the wrong
        // kernel class on every GPU in the dispatch table). The cold run
        // always executes — it is also the local reference for checks
        // 2 and 3.
        let cold = run(&mut warm, &prompt, fixture.max_tokens);
        let cold_tokens = cold.tokens.borrow().clone();
        if scope.runs_fixture() {
            assert_eq!(
                cold_tokens.as_slice(),
                fixture.expected_token_ids.as_slice(),
                "padded single-stream run diverged from the mlx-lm reference"
            );
            eprintln!("padded piece: cold == committed fixture, exact");
        } else {
            eprintln!(
                "KILN_FIXTURE_PARITY=skip: cold-vs-fixture compare deferred to the advisory CI step"
            );
        }

        if scope.runs_device_independent() {
            // 2) Containment rerun: served from the blocks the padded piece
            // wrote; any pad-row leakage into those blocks shifts the
            // stream from the cold run's.
            let rerun = run(&mut warm, &prompt, fixture.max_tokens);
            assert_eq!(
                rerun.tokens.borrow().as_slice(),
                cold_tokens.as_slice(),
                "warm rerun over the padded piece's cached rows diverged from the cold run"
            );
            let hits = rerun.hits.borrow().clone();
            assert_eq!(hits.len(), 1, "identical prompt must hit the cache");
            assert!(
                hits[0] as usize >= 128,
                "containment should serve through the ragged region: {hits:?}"
            );

            // 3) Extension: full first stream + a divergent tail, resuming at
            // the canonical 128 boundary below the ragged region. Warm == cold
            // proves the donated coverage holds genuine rows only.
            let mut extended = prompt.clone();
            extended.extend(cold_tokens.iter().copied());
            extended.extend(
                tokenizer
                    .encode(" The cones mentioned above", false)
                    .expect("encodes"),
            );
            let mut cold_engine =
                Engine::new(&model, model.kv_dims(), config, Stream::gpu()).expect("engine builds");
            let ext_cold = run(&mut cold_engine, &extended, 24);
            let ext_warm = run(&mut warm, &extended, 24);
            assert_eq!(
                ext_warm.tokens.borrow().as_slice(),
                ext_cold.tokens.borrow().as_slice(),
                "extension resume served pad-tainted or stale rows"
            );
            let hits = ext_warm.hits.borrow().clone();
            assert_eq!(hits.len(), 1, "shared prefix must hit");
            assert!(
                hits[0].is_multiple_of(128),
                "resume lands on a canonical boundary: {hits:?}"
            );
            eprintln!(
                "padded piece: rerun hit {} tokens, extension hit {} tokens — warm == cold exact",
                rerun.hits.borrow()[0],
                hits[0]
            );
        } else {
            eprintln!(
                "KILN_FIXTURE_PARITY=only: rerun/extension checks skipped (they gate in the blocking CI step)"
            );
        }
    }
    assert_eq!(
        debug::live_objects(),
        baseline,
        "padded runs leaked mlx handles"
    );
}
