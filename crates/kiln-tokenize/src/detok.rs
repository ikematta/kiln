//! Incremental, UTF-8-safe detokenization for streamed token ids.
//!
//! Used by the GATEWAY for Rust workers (Phase 3): the Rust worker streams
//! bare token ids (`TokenChunk.text` empty) and the gateway turns them into
//! SSE text deltas. The Python worker detokenizes itself; the gateway passes
//! its text through.
//!
//! The sliding two-offset scheme (as used by HF text-generation-inference)
//! handles both hard cases from CLAUDE.md:
//! - multi-token UTF-8: a codepoint split across tokens decodes to U+FFFD
//!   until its continuation bytes arrive — text is held, never emitted as a
//!   partial codepoint;
//! - context-dependent decoders (SentencePiece leading-space rules,
//!   byte-fallback): new text is always computed as `decode(window+new) -
//!   decode(window)`, never by decoding tokens in isolation.

use std::sync::Arc;

use crate::tokenizer::{Tokenizer, TokenizerError};

/// Streaming decoder for one generation stream.
pub struct StreamingDecoder {
    tokenizer: Arc<Tokenizer>,
    ids: Vec<u32>,
    /// Start of the decode window: tokens before this are already emitted
    /// and no longer influence new text.
    prefix_offset: usize,
    /// End of the already-emitted region within the window.
    read_offset: usize,
}

impl StreamingDecoder {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        Self {
            tokenizer,
            ids: Vec::new(),
            prefix_offset: 0,
            read_offset: 0,
        }
    }

    /// Feeds new token ids; returns the text that is now safe to emit
    /// (possibly empty while a codepoint is still incomplete).
    ///
    /// Decodes with special tokens included — the caller decides which ids
    /// reach the decoder (workers do not chunk their stop token).
    pub fn push(&mut self, new_ids: &[u32]) -> Result<String, TokenizerError> {
        self.ids.extend_from_slice(new_ids);
        let prefix = self
            .tokenizer
            .decode(&self.ids[self.prefix_offset..self.read_offset], false)?;
        let full = self
            .tokenizer
            .decode(&self.ids[self.prefix_offset..], false)?;

        // Hold while: nothing new, trailing replacement char (incomplete
        // UTF-8 sequence), or the window rewrote earlier text (defensive —
        // the boundary check keeps slicing panic-free even then).
        if full.len() > prefix.len()
            && !full.ends_with('\u{FFFD}')
            && full.is_char_boundary(prefix.len())
        {
            let emitted = full[prefix.len()..].to_string();
            self.prefix_offset = self.read_offset;
            self.read_offset = self.ids.len();
            Ok(emitted)
        } else {
            Ok(String::new())
        }
    }

    /// Releases whatever is still held (call once the stream is over).
    /// Genuinely incomplete trailing bytes surface as U+FFFD — a complete
    /// codepoint, so still safe for SSE.
    pub fn finalize(&mut self) -> Result<String, TokenizerError> {
        let prefix = self
            .tokenizer
            .decode(&self.ids[self.prefix_offset..self.read_offset], false)?;
        let full = self
            .tokenizer
            .decode(&self.ids[self.prefix_offset..], false)?;
        self.prefix_offset = self.ids.len();
        self.read_offset = self.ids.len();
        if full.len() > prefix.len() && full.is_char_boundary(prefix.len()) {
            Ok(full[prefix.len()..].to_string())
        } else {
            Ok(String::new())
        }
    }
}
