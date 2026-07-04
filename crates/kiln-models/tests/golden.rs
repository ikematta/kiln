//! Golden-token parity harness (SPEC §11.2, the keystone test): the Rust
//! Llama implementation must reproduce the committed mlx-lm reference
//! fixtures token-for-token, exactly.
//!
//! Since Phase 4 the fixtures run through the continuous-batching engine
//! (paged KV + gather-based paged attention, SPEC §6.2/§7.4 v0) — the
//! production decode path — with the production chunk size (2048, matching
//! mlx-lm's `prefill_step_size`). Batching and paging must not change
//! greedy outputs (CLAUDE.md determinism), so parity here is the Phase 4
//! acceptance gate; `tests/batching.rs` separately pins engine == Phase-3
//! contiguous path under concurrency.
//!
//! Fixture semantics (mirrored from `scripts/gen-golden.py` — keep in sync):
//! - `chat_template: true`  -> render the model's template for one user
//!   message with `add_generation_prompt` and the pinned `date_string`,
//!   encode WITHOUT special tokens (the template emits BOS itself);
//! - `chat_template: false` -> encode the raw prompt WITH special tokens;
//! - greedy, exactly `max_tokens` tokens, no EOS stopping.
//!
//! Skips (with a note) when `KILN_TEST_MODELS` is unset or Metal is
//! unavailable. Fails loudly if the local model revision differs from the
//! one the fixtures were generated against.

#![cfg(feature = "metal")]

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use kiln_engine::{
    Engine, EngineConfig, EngineRequest, FinishKind, PenaltyOptions, SamplingOptions, SeqEvent,
};
use kiln_mlx::{Stream, debug};
use kiln_models::LlamaModel;
use kiln_tokenize::{ChatMessage, ChatTemplate, Tokenizer};
use minijinja::Value;

/// Must match PINNED_DATE_STRING in scripts/gen-golden.py.
const PINNED_DATE_STRING: &str = "26 Jul 2024";

const MODEL_NAME: &str = "llama-3.2-1b-4bit";

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

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(MODEL_NAME)
}

fn model_dir() -> Option<PathBuf> {
    let root = std::env::var_os("KILN_TEST_MODELS")?;
    let dir = PathBuf::from(root).join(MODEL_NAME);
    dir.join("config.json").is_file().then_some(dir)
}

/// Runs one greedy request through the batching engine and returns every
/// generated token (no stop tokens, so the token stream is complete).
fn engine_generate(
    engine: &mut Engine<&LlamaModel>,
    prompt_ids: &[u32],
    max_tokens: usize,
) -> Vec<u32> {
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
        cancel: Arc::new(AtomicBool::new(false)),
        on_event: Box::new(move |event| {
            match event {
                SeqEvent::Token(token) => t.borrow_mut().push(token),
                SeqEvent::Finished(summary) => *f.borrow_mut() = Some(summary),
            }
            true
        }),
    });
    while !engine.is_idle() {
        engine.step().expect("engine step");
    }
    let summary = finish.borrow().clone().expect("request finished");
    assert_eq!(
        summary.reason,
        FinishKind::Length,
        "fixtures run to max_tokens: {summary:?}"
    );
    tokens.borrow().clone()
}

#[test]
fn llama_32_1b_greedy_parity_is_exact() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let Some(model_dir) = model_dir() else {
        eprintln!("skipping: KILN_TEST_MODELS not set or {MODEL_NAME} missing");
        return;
    };

    let mut fixture_paths: Vec<PathBuf> = std::fs::read_dir(golden_dir())
        .expect("tests/golden fixtures directory exists")
        .filter_map(|entry| {
            let path = entry.expect("readable dir entry").path();
            (path.extension().is_some_and(|e| e == "json")).then_some(path)
        })
        .collect();
    fixture_paths.sort();
    assert!(!fixture_paths.is_empty(), "no golden fixtures found");

    let local_revision = std::fs::read_to_string(model_dir.join(".kiln-revision"))
        .map(|text| text.trim().to_owned())
        .unwrap_or_default();

    let baseline = debug::live_objects();
    {
        let stream = Stream::gpu();
        let model = LlamaModel::load(&model_dir, &stream).expect("model loads");
        let tokenizer = Tokenizer::from_model_dir(&model_dir).expect("tokenizer loads");
        let template = ChatTemplate::from_model_dir(&model_dir).expect("template loads");
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

        for path in &fixture_paths {
            let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
            let fixture: Fixture =
                serde_json::from_str(&std::fs::read_to_string(path).expect("fixture readable"))
                    .expect("fixture parses");
            assert_eq!(
                fixture.weights_revision, local_revision,
                "fixture {name} was generated for a different weights revision than the \
                 local test model — refetch models or regenerate fixtures (when told to)"
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

            let output = engine_generate(&mut engine, &prompt_ids, fixture.max_tokens);
            assert_eq!(
                output,
                fixture.expected_token_ids,
                "greedy token divergence on fixture {name} \
                 (prompt tokens: {})",
                prompt_ids.len()
            );
            eprintln!(
                "golden {name}: {} prompt tokens, {} generated — exact match \
                 (batched/paged engine)",
                prompt_ids.len(),
                fixture.max_tokens
            );
        }
    }
    assert_eq!(
        debug::live_objects(),
        baseline,
        "golden run leaked mlx handles"
    );
}
