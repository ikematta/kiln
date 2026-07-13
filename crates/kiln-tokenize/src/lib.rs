#![deny(unsafe_code)]
//! `kiln-tokenize`: tokenizer loading, chat templates (minijinja), streaming
//! UTF-8-safe incremental detokenization, and tool-call parsers (SPEC §8.2).
//!
//! # BOS / special-token contract
//!
//! Chat-template rendering ([`ChatTemplate::render`]) produces the *complete*
//! prompt text: the template source itself emits BOS (and every other special
//! token) exactly where the model expects it — e.g. Llama 3.x templates start
//! with `<|begin_of_text|>`. Therefore rendered text must be encoded with
//! `add_special_tokens = false` everywhere; encoding it with special tokens
//! enabled would run the tokenizer's post-processor and prepend a second BOS,
//! which silently degrades model output.
//!
//! The invariant is identical on both worker paths — only *who* runs the
//! encode differs:
//!
//! - **Rust worker** (`SubmitRequest.token_ids`): the gateway tokenizes
//!   locally with this crate and MUST pass `add_special_tokens = false` on
//!   rendered templates. The worker never re-tokenizes and never re-applies
//!   special tokens to the ids it receives.
//! - **Python worker** (`CAPABILITY_TOKENIZER_OWNED`): the worker owns the
//!   tokenizer, so the gateway sends the rendered text through the `Tokenize`
//!   RPC with `add_special_tokens = false` — same contract, delegated over
//!   RPC instead of executed in-process.
//!
//! The `Tokenize` RPC's `add_special_tokens` flag (and the `true` path of
//! this crate's tokenizer `encode`) exists for *non-templated* input — raw
//! `/v1/completions`-style prompts, where no template ran and BOS must come
//! from the tokenizer itself. Do not copy that flag value onto the
//! chat-template path.
//!
//! Phase 2 shipped chat templating; tokenizer loading landed with the Rust
//! worker (Phase 3), incremental detokenization with Phase 4, and the
//! streaming tool-call parsers ([`toolcall`]) with Phase 7.

pub mod chat_template;
pub mod detok;
pub mod stops;
pub mod tokenizer;
pub mod toolcall;

pub use chat_template::{
    ChatMessage, ChatTemplate, MessageToolCall, MessageToolFunction, TemplateError,
};
pub use detok::StreamingDecoder;
pub use stops::StopStringMatcher;
pub use tokenizer::{Tokenizer, TokenizerError};
pub use toolcall::{ToolCallFormat, ToolCallParser, ToolEvent};
