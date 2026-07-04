//! Tokenizer loading + encode/decode over the HF `tokenizers` crate
//! (SPEC §3: gateway-side tokenization for Rust workers).
//!
//! Loads `tokenizer.json` from a model directory — the exact same file the
//! Python worker's tokenizer reads, so ids agree across workers.
//!
//! See the crate docs for the BOS/special-token contract governing
//! `add_special_tokens`: rendered chat templates are ALWAYS encoded with
//! `add_special_tokens = false` (the template text already contains BOS);
//! `true` is only for raw, non-templated prompts.

use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("failed to load tokenizer from {path}: {message}")]
    Load { path: String, message: String },
    #[error("encode failed: {0}")]
    Encode(String),
    #[error("decode failed: {0}")]
    Decode(String),
}

/// A loaded `tokenizer.json` vocabulary + pipeline.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab_size())
            .finish()
    }
}

impl Tokenizer {
    /// Loads `<dir>/tokenizer.json`.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let path = dir.as_ref().join("tokenizer.json");
        let inner =
            tokenizers::Tokenizer::from_file(&path).map_err(|source| TokenizerError::Load {
                path: path.display().to_string(),
                message: source.to_string(),
            })?;
        Ok(Self { inner })
    }

    /// Encodes `text` to token ids.
    ///
    /// `add_special_tokens` runs the tokenizer's post-processor (BOS for
    /// Llama-family models). Pass `false` for rendered chat templates,
    /// `true` for raw prompts — see the crate-level BOS contract.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decodes token ids to text (full decode; the incremental streaming
    /// decoder is a later-phase addition).
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| TokenizerError::Decode(e.to_string()))
    }

    /// Vocabulary size including added special tokens.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
