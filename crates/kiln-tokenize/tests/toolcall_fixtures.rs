//! Tool-call parser correctness against REAL captured model output
//! (SPEC §8.2, Phase 7). Fixtures live in tests/fixtures/toolcall/ and are
//! regenerated only via scripts/gen-tool-fixtures.py: greedy completions
//! from the pinned test models (kind "generation", with the exact token
//! ids a worker streams) plus the family templates' own serializations
//! (kind "template", for formats no pinned model emits — Qwen3-Coder XML).
//!
//! Two layers, same invariant as the Phase 4/5 pipelined-decode tests: a
//! tool call split across many small chunks must reassemble identically to
//! one delivered in a single chunk.
//!
//! - Text level (always runs): the fixture text pushed whole vs. split at
//!   every size in a schedule must reassemble to the fixture's `expected`.
//! - Token level (env-gated on `KILN_TEST_MODELS`): the fixture's token
//!   ids replayed through the real [`StreamingDecoder`] one token at a
//!   time — the exact segments the gateway feeds the parser for a Rust
//!   worker — must reassemble to the same `expected`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use kiln_tokenize::{
    ChatMessage, ChatTemplate, MessageToolCall, MessageToolFunction, StreamingDecoder, Tokenizer,
    ToolCallFormat, ToolCallParser, ToolEvent,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    family: String,
    source: Source,
    tools: Vec<serde_json::Value>,
    #[serde(default)]
    messages: Vec<serde_json::Value>,
    /// The transformers-rendered prompt (generation fixtures only).
    prompt_text: Option<String>,
    token_ids: Option<Vec<u32>>,
    text: String,
    expected: Expected,
}

#[derive(Deserialize)]
struct Source {
    kind: String,
    model: String,
    #[serde(default)]
    template_kwargs: serde_json::Map<String, serde_json::Value>,
}

#[derive(Deserialize, Debug, PartialEq)]
struct Expected {
    content: String,
    calls: Vec<ExpectedCall>,
}

#[derive(Deserialize, Debug, PartialEq)]
struct ExpectedCall {
    name: String,
    arguments: String,
}

fn fixtures() -> Vec<(String, Fixture)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/toolcall");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("fixture dir exists") {
        let path = entry.expect("readable dir entry").path();
        if path.extension().is_some_and(|e| e == "json") {
            let text = std::fs::read_to_string(&path).expect("fixture reads");
            let fixture: Fixture = serde_json::from_str(&text).expect("fixture parses");
            out.push((
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .expect("utf8 name")
                    .to_owned(),
                fixture,
            ));
        }
    }
    assert!(
        out.len() >= 10,
        "fixture suite went missing ({} found)",
        out.len()
    );
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn format_of(fixture: &Fixture) -> ToolCallFormat {
    match fixture.family.as_str() {
        "hermes" => ToolCallFormat::Hermes,
        "llama" => ToolCallFormat::Llama,
        "qwen_xml" => ToolCallFormat::QwenXml,
        other => panic!("unknown family {other}"),
    }
}

/// Reassembles an event stream into (content, calls) for comparison.
fn reassemble(events: &[ToolEvent], name: &str) -> Expected {
    let mut content = String::new();
    let mut calls: Vec<ExpectedCall> = Vec::new();
    let mut open: Option<usize> = None;
    for event in events {
        match event {
            ToolEvent::Content(text) => content.push_str(text),
            ToolEvent::CallStart { index, name: call } => {
                assert_eq!(*index, calls.len(), "{name}: call indices sequential");
                assert_eq!(open, None, "{name}: calls must not interleave");
                open = Some(*index);
                calls.push(ExpectedCall {
                    name: call.clone(),
                    arguments: String::new(),
                });
            }
            ToolEvent::CallArgs { index, delta } => {
                assert_eq!(open, Some(*index), "{name}: args only for the open call");
                calls[*index].arguments.push_str(delta);
            }
            ToolEvent::CallEnd { index } => {
                assert_eq!(open.take(), Some(*index), "{name}: end matches start");
            }
        }
    }
    assert_eq!(open, None, "{name}: every started call must end");
    Expected { content, calls }
}

fn parse_segments<'a>(
    fixture: &Fixture,
    segments: impl Iterator<Item = &'a str>,
    name: &str,
) -> Expected {
    let mut parser = ToolCallParser::new(format_of(fixture), &fixture.tools);
    let mut events = Vec::new();
    for segment in segments {
        events.extend(parser.push(segment));
    }
    events.extend(parser.finish());
    reassemble(&events, name)
}

/// Text-level: whole-push and several chunk schedules must all reassemble
/// to the fixture's expected output. Runs without any model present.
#[test]
fn fixtures_reassemble_identically_under_any_chunking() {
    for (name, fixture) in fixtures() {
        let whole = parse_segments(&fixture, std::iter::once(fixture.text.as_str()), &name);
        assert_eq!(
            whole, fixture.expected,
            "{name}: single-chunk parse must match the fixture"
        );

        let chars: Vec<char> = fixture.text.chars().collect();
        for chunk_len in [1usize, 2, 3, 5, 11] {
            let segments: Vec<String> = chars
                .chunks(chunk_len)
                .map(|c| c.iter().collect())
                .collect();
            let chunked = parse_segments(&fixture, segments.iter().map(String::as_str), &name);
            assert_eq!(
                chunked, fixture.expected,
                "{name}: chunk_len={chunk_len} diverged from single-chunk"
            );
        }
    }
}

/// Render parity: the gateway renders tool requests through this crate's
/// minijinja environment; the model was prompted with the transformers
/// rendering. The two must agree byte-for-byte, or the e2e model sees a
/// different prompt than the one the fixtures captured.
#[test]
fn tool_prompts_render_identically_to_transformers() {
    let Some(root) = std::env::var_os("KILN_TEST_MODELS") else {
        eprintln!("skipping: KILN_TEST_MODELS not set");
        return;
    };
    let root = PathBuf::from(root);
    let mut ran = 0;
    for (name, fixture) in fixtures() {
        let Some(reference) = &fixture.prompt_text else {
            continue; // template-kind fixture: no reference prompt
        };
        let model_dir = root.join(&fixture.source.model);
        if !model_dir.join("config.json").is_file() {
            eprintln!("skipping {name}: {} not fetched", fixture.source.model);
            continue;
        }
        let template = ChatTemplate::from_model_dir(&model_dir).expect("template loads");

        let messages: Vec<ChatMessage> = fixture
            .messages
            .iter()
            .map(|m| ChatMessage {
                role: m["role"].as_str().expect("role").to_owned(),
                content: m["content"].as_str().unwrap_or_default().to_owned(),
                tool_calls: m.get("tool_calls").map(|calls| {
                    calls
                        .as_array()
                        .expect("array")
                        .iter()
                        .map(|call| MessageToolCall {
                            call_type: "function".to_owned(),
                            function: MessageToolFunction {
                                name: call["function"]["name"].as_str().expect("name").to_owned(),
                                arguments: call["function"]["arguments"].clone(),
                            },
                        })
                        .collect()
                }),
            })
            .collect();

        let mut extra: Vec<(&str, minijinja::Value)> = fixture
            .source
            .template_kwargs
            .iter()
            .map(|(key, value)| (key.as_str(), minijinja::Value::from_serialize(value)))
            .collect();
        if !fixture.tools.is_empty() {
            extra.push(("tools", minijinja::Value::from_serialize(&fixture.tools)));
        }
        let rendered = template
            .render_with(&messages, true, &extra)
            .expect("renders");
        assert_eq!(&rendered, reference, "{name}: prompt render diverged");
        ran += 1;
    }
    assert!(ran > 0, "no generation fixtures ran; models missing?");
}

/// Token-level: replay the captured token ids through the real streaming
/// decoder (token-at-a-time — the exact segment boundaries the gateway
/// produces for a Rust worker) and parse the resulting segments.
#[test]
fn fixtures_reassemble_from_streaming_decoder_segments() {
    let Some(root) = std::env::var_os("KILN_TEST_MODELS") else {
        eprintln!("skipping: KILN_TEST_MODELS not set");
        return;
    };
    let root = PathBuf::from(root);
    let mut ran = 0;
    for (name, fixture) in fixtures() {
        let Some(ids) = &fixture.token_ids else {
            continue; // template-kind fixture: no real token stream exists
        };
        assert_eq!(fixture.source.kind, "generation");
        let model_dir = root.join(&fixture.source.model);
        if !model_dir.join("tokenizer.json").is_file() {
            eprintln!("skipping {name}: {} not fetched", fixture.source.model);
            continue;
        }
        let tokenizer =
            Arc::new(Tokenizer::from_model_dir(&model_dir).expect("pinned tokenizer loads"));

        // Sanity: the fixture text is exactly the decode of the ids.
        assert_eq!(
            tokenizer.decode(ids, false).expect("decodes"),
            fixture.text,
            "{name}: fixture text out of sync with token ids"
        );

        for chunk_len in [1usize, 2, 3] {
            let mut decoder = StreamingDecoder::new(Arc::clone(&tokenizer));
            let mut segments = Vec::new();
            for chunk in ids.chunks(chunk_len) {
                segments.push(decoder.push(chunk).expect("decoder push"));
            }
            segments.push(decoder.finalize().expect("decoder finalize"));
            let got = parse_segments(&fixture, segments.iter().map(String::as_str), &name);
            assert_eq!(
                got, fixture.expected,
                "{name}: decoder-fed parse (chunk_len={chunk_len}) diverged"
            );
        }
        ran += 1;
    }
    assert!(ran > 0, "no generation fixtures ran; models missing?");
}
