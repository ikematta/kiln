#![deny(unsafe_code)]
//! `kiln-tokenize`: tokenizer loading, chat templates (minijinja), streaming
//! UTF-8-safe incremental detokenization, and tool-call parsers (SPEC §8.2).
//!
//! Phase 2 ships chat templating; tokenizer loading and incremental
//! detokenization land with the Rust worker (Phase 3), tool-call parsing in
//! Phase 7.

pub mod chat_template;

pub use chat_template::{ChatMessage, ChatTemplate, TemplateError};
