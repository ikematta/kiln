//! Grammar-constrained decoding via llguidance (SPEC §3 "Structured
//! output", §12 Phase 7): `json_schema` and `regex` grammar specs compile
//! to a per-request [`Grammar`], whose allowed-token mask is applied to the
//! logits inside the decode step — the same per-request logit-processor
//! slot as the penalties (SPEC §6.2 step 3).
//!
//! Ownership and threading: [`GrammarEnv`] is built once per worker from
//! the model's `tokenizer.json` (token byte representations + trie) and is
//! `Send + Sync` — grammar compilation runs on the gRPC handler tasks.
//! A compiled [`Grammar`] is `Send` (it crosses the submission channel
//! once) but stateful: after that, only the engine thread touches it. The
//! mask for step N depends on every token committed through step N-1,
//! which is why grammar-constrained sequences never ride the async_eval
//! decode pipeline (the engine gates it in `pipeline_ok`).
//!
//! Contract with the engine:
//! - masking is a strict no-op for requests without a grammar — the decode
//!   step adds zero ops to their graph (golden parity depends on this);
//! - the mask is sized to the worker's configured vocab (`n_vocab`, the
//!   model's `config.json` `vocab_size`, which the trie is padded to at
//!   load), so it matches the logits' vocab dimension; padding token ids
//!   past the tokenizer's real vocab are never allowed;
//! - when the grammar reaches an accepting state the mask allows the
//!   model's EOS token(s) (wired into the trie at load); once the grammar
//!   is complete ([`Grammar::commit`] returns `true`) the engine finishes
//!   the request with `Stop`.

use std::path::Path;

use kiln_mlx::{Array, Dtype, MlxError};
use llguidance::api::TopLevelGrammar;
use llguidance::{Matcher, ParserFactory};
use thiserror::Error;
use toktrie_hf_tokenizers::ByteTokenizer;

#[derive(Debug, Error)]
pub enum GrammarError {
    /// The tokenizer-derived environment could not be built; the worker
    /// serves without `CAPABILITY_GRAMMAR`.
    #[error("grammar environment unavailable: {0}")]
    Load(String),
    /// The submitted grammar spec does not compile (proto
    /// `WORKER_ERROR_GRAMMAR_COMPILE`).
    #[error("grammar failed to compile: {0}")]
    Compile(String),
    /// Mask computation or token commit failed mid-decode (an in-band
    /// per-request `Finished{error}`, `WORKER_ERROR_INTERNAL`).
    #[error("grammar runtime fault: {0}")]
    Runtime(String),
    #[error(transparent)]
    Mlx(#[from] MlxError),
}

/// Compile-time guarantees for the ownership story in the module docs.
fn _assert_thread_bounds() {
    fn send_sync<T: Send + Sync>() {}
    fn send<T: Send>() {}
    send_sync::<GrammarEnv>();
    send::<Grammar>();
}

/// Per-worker grammar compiler: the token trie + parser factory llguidance
/// needs, built once from the model directory's `tokenizer.json`.
pub struct GrammarEnv {
    factory: ParserFactory,
    n_vocab: usize,
}

impl GrammarEnv {
    /// Builds the environment from `tokenizer.json`.
    ///
    /// `n_vocab` is the model's logits vocab (`config.json` `vocab_size`);
    /// the trie is padded up to it so masks always match the logits shape.
    /// `eos_ids` are the model's EOS token ids (kiln-models
    /// `eos_token_ids`): the grammar allows exactly these once it reaches
    /// an accepting state, keeping grammar termination aligned with the
    /// engine's stop-token handling regardless of what the tokenizer file
    /// claims EOS is.
    pub fn load(
        tokenizer_json: &Path,
        n_vocab: u32,
        eos_ids: &[u32],
    ) -> Result<Self, GrammarError> {
        let load = |detail: String| GrammarError::Load(detail);
        if n_vocab == 0 {
            return Err(load("model vocab_size is unknown (0)".to_owned()));
        }
        let mut tokenizer = ByteTokenizer::from_file(tokenizer_json)
            .map_err(|err| load(format!("{}: {err:#}", tokenizer_json.display())))?;
        // set_eos_tokens asserts ids are in range; pre-validate so a
        // malformed config degrades to "no grammar support", never a panic.
        let in_range: Vec<u32> = eos_ids
            .iter()
            .copied()
            .filter(|&id| id < tokenizer.tokrx_info().vocab_size)
            .collect();
        if in_range.len() != eos_ids.len() {
            return Err(load(format!(
                "eos ids {eos_ids:?} exceed the tokenizer vocab ({})",
                tokenizer.tokrx_info().vocab_size
            )));
        }
        if !in_range.is_empty() {
            tokenizer.set_eos_tokens(&in_range);
        }
        let tok_env = tokenizer
            .into_tok_env(Some(n_vocab as usize))
            .map_err(|err| load(format!("token trie construction failed: {err:#}")))?;
        let factory = ParserFactory::new_simple(&tok_env)
            .map_err(|err| load(format!("parser factory construction failed: {err:#}")))?;
        Ok(Self {
            factory,
            n_vocab: n_vocab as usize,
        })
    }

    /// Compiles a `GrammarSpec.json_schema` payload (a JSON Schema
    /// document as a string).
    pub fn compile_json_schema(&self, schema: &str) -> Result<Grammar, GrammarError> {
        let schema: serde_json::Value = serde_json::from_str(schema).map_err(|err| {
            GrammarError::Compile(format!("json_schema is not valid JSON: {err}"))
        })?;
        self.compile(TopLevelGrammar::from_json_schema(schema))
    }

    /// Compiles a `GrammarSpec.regex` payload.
    pub fn compile_regex(&self, regex: &str) -> Result<Grammar, GrammarError> {
        self.compile(TopLevelGrammar::from_regex(regex))
    }

    fn compile(&self, grammar: TopLevelGrammar) -> Result<Grammar, GrammarError> {
        let parser = self
            .factory
            .create_parser(grammar)
            .map_err(|err| GrammarError::Compile(format!("{err:#}")))?;
        Ok(Grammar {
            matcher: Matcher::new(Ok(parser)),
            n_vocab: self.n_vocab,
        })
    }
}

/// One request's compiled grammar constraint: the llguidance matcher plus
/// the mask geometry. Advances strictly by sampled tokens ([`Self::commit`]
/// once per sample), so its state tracks the generated text exactly —
/// preemption replay never re-commits because replay never samples.
pub struct Grammar {
    matcher: Matcher,
    n_vocab: usize,
}

impl Grammar {
    /// Allowed-token bytes (1 = allowed) for the next sample, one byte per
    /// vocab entry. Once the grammar is complete this is the EOS-only mask.
    pub fn allowed_tokens(&mut self) -> Result<Vec<u8>, GrammarError> {
        let mask = self
            .matcher
            .compute_mask_or_eos()
            .map_err(|err| GrammarError::Runtime(format!("mask computation failed: {err:#}")))?;
        let mut bytes = vec![0u8; self.n_vocab];
        let mut allowed = 0usize;
        mask.iter_set_entries(|token| {
            if let Some(slot) = bytes.get_mut(token) {
                *slot = 1;
                allowed += 1;
            }
        });
        if allowed == 0 {
            // A well-formed grammar always allows something (EOS once
            // accepting); an empty mask means the parser is wedged.
            return Err(GrammarError::Runtime(
                self.matcher
                    .get_error()
                    .unwrap_or_else(|| "grammar produced an empty token mask".to_owned()),
            ));
        }
        Ok(bytes)
    }

    /// [`Self::allowed_tokens`] as a `[1, n_vocab]` bool MLX array, ready
    /// for `where_cond` against a logits row of the same width.
    pub(crate) fn mask_array(&mut self) -> Result<Array, GrammarError> {
        let bytes = self.allowed_tokens()?;
        Ok(Array::from_raw_bytes(
            &bytes,
            &[1, self.n_vocab as i32],
            Dtype::Bool,
        )?)
    }

    /// Advances the grammar by one sampled token. Returns `true` when the
    /// grammar is complete (the engine finishes the request with `Stop`).
    /// The token must have been sampled under this grammar's mask; a
    /// rejection is therefore an internal fault, not a client error.
    pub fn commit(&mut self, token: u32) -> Result<bool, GrammarError> {
        if self.matcher.is_stopped() {
            return Ok(true);
        }
        self.matcher
            .consume_token(token)
            .map_err(|err| GrammarError::Runtime(format!("token {token} rejected: {err:#}")))?;
        Ok(self.matcher.is_stopped())
    }
}
