//! Model-directory template loading + rendering against vendored fixtures.
//!
//! `fixtures/llama32` carries the exact chat template of the pinned
//! `mlx-community/Llama-3.2-1B-Instruct-4bit` revision (trimmed
//! tokenizer_config.json); it is the template the Phase 2 gateway serves.

use std::path::PathBuf;

use kiln_tokenize::{ChatMessage, ChatTemplate, TemplateError};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn msg(role: &str, content: &str) -> ChatMessage {
    ChatMessage::text(role, content)
}

#[test]
fn llama32_renders_generation_prompt() {
    let template = ChatTemplate::from_model_dir(fixture("llama32")).expect("loads");
    assert_eq!(template.bos_token(), "<|begin_of_text|>");
    assert_eq!(template.eos_token(), "<|eot_id|>");

    let rendered = template
        .render(
            &[
                msg("system", "You are terse."),
                msg("user", "Say hi."),
                msg("assistant", "Hi."),
                msg("user", "Again."),
            ],
            true,
        )
        .expect("renders");

    // BOS comes from the template itself — the tokenize step must therefore
    // not add special tokens again (double-BOS guard).
    assert!(rendered.starts_with("<|begin_of_text|>"), "{rendered}");
    assert_eq!(rendered.matches("<|begin_of_text|>").count(), 1);
    // System block carries the strftime_now-derived date.
    let date = chrono::Local::now().format("%d %b %Y").to_string();
    assert!(
        rendered.contains(&format!("Today Date: {date}")),
        "{rendered}"
    );
    // The template inserts the knowledge-cutoff/date lines between the system
    // header and the system message, so assert both pieces independently.
    assert!(rendered.contains("<|start_header_id|>system<|end_header_id|>\n\n"));
    assert!(rendered.contains("You are terse.<|eot_id|>"));
    assert!(rendered.contains("<|start_header_id|>user<|end_header_id|>\n\nSay hi.<|eot_id|>"));
    assert!(rendered.contains("<|start_header_id|>assistant<|end_header_id|>\n\nHi.<|eot_id|>"));
    // add_generation_prompt=true leaves the assistant header open at the end.
    assert!(
        rendered.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"),
        "{rendered}"
    );
}

#[test]
fn llama32_source_hash_matches_python_worker_scheme() {
    // The Python worker hashes sha256(template_text); recompute independently
    // from the fixture to pin the cross-language comparison contract.
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture("llama32").join("tokenizer_config.json"))
            .expect("fixture readable"),
    )
    .expect("fixture parses");
    let source = config["chat_template"].as_str().expect("template string");

    use sha2::{Digest, Sha256};
    let expected = format!("{:x}", Sha256::digest(source.as_bytes()));

    let template = ChatTemplate::from_model_dir(fixture("llama32")).expect("loads");
    assert_eq!(template.source_hash(), expected);
}

#[test]
fn jinja_file_takes_precedence_and_added_token_form_parses() {
    let template = ChatTemplate::from_model_dir(fixture("chatml")).expect("loads");
    assert_eq!(template.bos_token(), "<|begin|>");
    assert_eq!(template.eos_token(), "<|im_end|>");

    let rendered = template
        .render(&[msg("user", "  hello  ")], true)
        .expect("renders");
    // .strip() only works through the pycompat unknown-method callback.
    assert_eq!(
        rendered,
        "<|begin|><|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    );
    // The bogus tokenizer_config.json chat_template must have been ignored.
    assert!(!rendered.contains("IGNORED"));
}

#[test]
fn chatml_rejects_unknown_role_via_raise_exception() {
    let template = ChatTemplate::from_model_dir(fixture("chatml")).expect("loads");
    let err = template
        .render(&[msg("tool", "x")], false)
        .expect_err("must fail");
    assert!(err.to_string().contains("unsupported role"), "{err}");
}

#[test]
fn missing_template_is_a_clean_error() {
    let dir = std::env::temp_dir().join("kiln-tokenize-empty-model-dir");
    std::fs::create_dir_all(&dir).expect("mkdir");
    let err = ChatTemplate::from_model_dir(&dir).expect_err("no template");
    assert!(matches!(err, TemplateError::Missing(_)), "{err}");
}
