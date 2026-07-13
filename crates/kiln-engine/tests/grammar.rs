//! Host-level grammar unit tests (SPEC §12 Phase 7): compile specs against
//! a real model tokenizer and drive the mask/commit machinery without
//! weights or a GPU — the schema/regex walk picks arbitrary allowed
//! tokens, which is exactly the guarantee llguidance masking provides
//! (any allowed pick yields grammar-valid output).
//!
//! Skips (with a note) when `KILN_TEST_MODELS` is unset; the model-gated
//! CI step runs it (tokenizer.json only — no Metal work in here).

#![cfg(feature = "metal")]

use std::path::PathBuf;

use kiln_engine::{GrammarEnv, GrammarError};

const MODEL_NAME: &str = "llama-3.2-1b-4bit";
/// The pinned model's config.json vocab_size (also asserted below via
/// mask width, so a pin bump that changes it fails loudly here).
const VOCAB: u32 = 128256;
/// Llama 3.2 instruct EOS ids: <|eot_id|>, <|end_of_text|>.
const EOS: [u32; 2] = [128009, 128001];

fn tokenizer_json() -> Option<PathBuf> {
    let root = std::env::var_os("KILN_TEST_MODELS")?;
    let path = PathBuf::from(root).join(MODEL_NAME).join("tokenizer.json");
    path.is_file().then_some(path)
}

fn env() -> Option<GrammarEnv> {
    let Some(path) = tokenizer_json() else {
        eprintln!("skipping: KILN_TEST_MODELS not set or {MODEL_NAME} missing");
        return None;
    };
    Some(GrammarEnv::load(&path, VOCAB, &EOS).expect("grammar env builds"))
}

/// Walks a grammar by always committing the first allowed non-EOS token,
/// asserting invariants at every step; returns the number of committed
/// tokens. Any allowed pick must keep the grammar satisfiable — that is
/// the whole constrained-decoding contract.
fn walk(grammar: &mut kiln_engine::Grammar, max_steps: usize) -> usize {
    for step in 0..max_steps {
        let mask = grammar.allowed_tokens().expect("mask computes");
        assert_eq!(mask.len(), VOCAB as usize, "mask sized to the model vocab");
        let Some(token) = mask
            .iter()
            .enumerate()
            .find(|&(id, &allowed)| allowed == 1 && !EOS.contains(&(id as u32)))
            .map(|(id, _)| id as u32)
        else {
            // EOS-only mask: the grammar is complete.
            assert!(
                EOS.iter().any(|&id| mask[id as usize] == 1),
                "empty mask must never surface (allowed_tokens errors instead)"
            );
            return step;
        };
        if grammar.commit(token).expect("allowed token commits") {
            return step + 1;
        }
    }
    panic!("grammar did not terminate within {max_steps} steps");
}

#[test]
fn regex_grammar_masks_and_terminates() {
    let Some(env) = env() else { return };
    let mut grammar = env.compile_regex("[0-9]{3}-[0-9]{4}").expect("compiles");

    // First step: only digit-prefixed tokens are allowed, never EOS.
    let mask = grammar.allowed_tokens().expect("mask computes");
    let allowed: Vec<u32> = (0..VOCAB).filter(|&id| mask[id as usize] == 1).collect();
    assert!(!allowed.is_empty());
    for &id in &EOS {
        assert_eq!(mask[id as usize], 0, "EOS allowed before the match started");
    }

    let steps = walk(&mut grammar, 16);
    assert!(steps >= 1, "regex must need at least one token");
    // Once complete, commit is terminal and idempotent.
    assert!(grammar.commit(EOS[0]).expect("EOS commit after stop"));
}

#[test]
fn json_schema_grammar_walk_terminates() {
    let Some(env) = env() else { return };
    let mut grammar = env
        .compile_json_schema(
            r#"{
                "x-guidance": {"whitespace_flexible": false},
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["cat", "dog"]},
                    "count": {"type": "integer", "minimum": 0, "maximum": 9}
                },
                "required": ["kind", "count"],
                "additionalProperties": false
            }"#,
        )
        .expect("compiles");
    walk(&mut grammar, 64);
}

#[test]
fn compile_errors_are_reported() {
    let Some(env) = env() else { return };
    for (name, result) in [
        (
            "bad schema",
            env.compile_json_schema(r#"{"type": "nonsense"}"#),
        ),
        ("bad json", env.compile_json_schema("{not json")),
        ("bad regex", env.compile_regex("(unclosed")),
    ] {
        match result {
            Err(GrammarError::Compile(detail)) => {
                assert!(!detail.is_empty(), "{name}: detail must not be empty");
            }
            Err(other) => panic!("{name}: expected Compile, got {other}"),
            Ok(_) => panic!("{name}: must not compile"),
        }
    }
}

#[test]
fn load_rejects_bad_inputs() {
    let Some(path) = tokenizer_json() else {
        eprintln!("skipping: KILN_TEST_MODELS not set or {MODEL_NAME} missing");
        return;
    };
    // Unknown vocab size.
    assert!(matches!(
        GrammarEnv::load(&path, 0, &EOS),
        Err(GrammarError::Load(_))
    ));
    // Out-of-range EOS must error, not panic (llguidance asserts on it).
    assert!(matches!(
        GrammarEnv::load(&path, VOCAB, &[VOCAB + 5]),
        Err(GrammarError::Load(_))
    ));
    // Missing tokenizer.json.
    assert!(matches!(
        GrammarEnv::load(&path.with_file_name("nope.json"), VOCAB, &EOS),
        Err(GrammarError::Load(_))
    ));
}
