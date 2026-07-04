//! Golden-token parity harness (SPEC §11.2, the keystone test): the Rust
//! Llama implementation must reproduce the committed mlx-lm reference
//! fixtures token-for-token, exactly.
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

use std::path::PathBuf;

use kiln_engine::Sampler;
use kiln_mlx::{Stream, debug};
use kiln_models::{LlamaModel, generate};
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

            let mut sampler = Sampler::greedy();
            let output = generate(
                &model,
                &prompt_ids,
                fixture.max_tokens,
                |logprobs, s| sampler.sample(logprobs, s),
                &stream,
            )
            .expect("generates");

            assert_eq!(
                output.tokens,
                fixture.expected_token_ids,
                "greedy token divergence on fixture {name} \
                 (prompt tokens: {})",
                prompt_ids.len()
            );
            eprintln!(
                "golden {name}: {} prompt tokens, {} generated — exact match",
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
