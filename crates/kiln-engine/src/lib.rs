#![deny(unsafe_code)]
//! `kiln-engine`: continuous batching loop, paged KV cache, radix prefix
//! cache, sampler, speculative decoding, and SSD cold tier (SPEC §6).
//!
//! Phase 3 shipped the sampler (§6.6); Phase 4 builds the batching engine,
//! starting with the paged-KV block manager (§6.3). The block manager is
//! pure bookkeeping and builds without the `metal` feature.

pub mod block;
#[cfg(feature = "metal")]
pub mod sampler;

pub use block::{AppendPlan, BlockError, BlockId, BlockManager, BlockTable, CowCopy, CowOutcome};
#[cfg(feature = "metal")]
pub use sampler::{PenaltyOptions, Sampler, SamplingOptions, apply_penalties};
