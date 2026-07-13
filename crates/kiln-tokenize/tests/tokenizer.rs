//! Tokenizer loading + the BOS/special-token contract against the pinned
//! Llama-3.2 test model (skipped when `KILN_TEST_MODELS` is unset).

use std::path::PathBuf;

use kiln_tokenize::{ChatMessage, ChatTemplate, Tokenizer};
use minijinja::Value;

fn llama_dir() -> Option<PathBuf> {
    let root = std::env::var_os("KILN_TEST_MODELS")?;
    let dir = PathBuf::from(root).join("llama-3.2-1b-4bit");
    dir.join("tokenizer.json").is_file().then_some(dir)
}

const LLAMA_BOS: u32 = 128000;

#[test]
fn special_token_flag_controls_bos() {
    let Some(dir) = llama_dir() else {
        eprintln!("skipping: KILN_TEST_MODELS not set or model missing");
        return;
    };
    let tokenizer = Tokenizer::from_model_dir(&dir).expect("loads tokenizer.json");

    let with_special = tokenizer.encode("The kiln is hot", true).expect("encodes");
    let without = tokenizer.encode("The kiln is hot", false).expect("encodes");

    assert_eq!(
        with_special[0], LLAMA_BOS,
        "special encode must prepend BOS"
    );
    assert_ne!(without[0], LLAMA_BOS, "plain encode must not prepend BOS");
    assert_eq!(with_special[1..], without[..], "BOS is the only difference");
}

#[test]
fn rendered_template_already_carries_bos() {
    let Some(dir) = llama_dir() else {
        eprintln!("skipping: KILN_TEST_MODELS not set or model missing");
        return;
    };
    let tokenizer = Tokenizer::from_model_dir(&dir).expect("loads");
    let template = ChatTemplate::from_model_dir(&dir).expect("loads");
    let rendered = template
        .render(&[ChatMessage::text("user", "hi")], true)
        .expect("renders");

    // The template text itself starts with BOS…
    assert!(rendered.starts_with("<|begin_of_text|>"));
    // …so the no-special encode already yields exactly one BOS id, and a
    // special-tokens encode would double it — the failure mode the crate
    // docs forbid.
    let ids = tokenizer.encode(&rendered, false).expect("encodes");
    assert_eq!(ids[0], LLAMA_BOS);
    assert_ne!(ids[1], LLAMA_BOS);
    let doubled = tokenizer.encode(&rendered, true).expect("encodes");
    assert_eq!(doubled[0], LLAMA_BOS);
    assert_eq!(doubled[1], LLAMA_BOS, "special encode doubles BOS");
}

#[test]
fn decode_round_trips() {
    let Some(dir) = llama_dir() else {
        eprintln!("skipping: KILN_TEST_MODELS not set or model missing");
        return;
    };
    let tokenizer = Tokenizer::from_model_dir(&dir).expect("loads");
    let ids = tokenizer
        .encode("the kiln 窯 is hot \u{1f525}", false)
        .expect("encodes");
    let text = tokenizer.decode(&ids, false).expect("decodes");
    assert_eq!(text, "the kiln 窯 is hot \u{1f525}");
}

#[test]
fn render_with_pins_date_string() {
    // Uses the vendored fixture template (no model download needed): the
    // Llama 3.x template interpolates strftime_now() unless date_string is
    // supplied — the golden harness depends on this pinning.
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/llama32");
    let template = ChatTemplate::from_model_dir(fixture).expect("loads");
    let messages = [ChatMessage::text("user", "hi")];

    let pinned = template
        .render_with(
            &messages,
            true,
            &[("date_string", Value::from("26 Jul 2024"))],
        )
        .expect("renders");
    assert!(pinned.contains("26 Jul 2024"));

    let today = chrono::Local::now().format("%d %b %Y").to_string();
    let unpinned = template.render(&messages, true).expect("renders");
    assert!(unpinned.contains(&today));
}
