#![deny(unsafe_code)]
//! `kiln-engine`: continuous batching loop, paged KV cache, radix prefix
//! cache, sampler, speculative decoding, and SSD cold tier (SPEC §6).
//!
//! Phase 3 ships the sampler (§6.6); the batching engine and its data
//! structures arrive with Phase 4.

#[cfg(feature = "metal")]
pub mod sampler;

#[cfg(feature = "metal")]
pub use sampler::{PenaltyOptions, Sampler, SamplingOptions, apply_penalties};
