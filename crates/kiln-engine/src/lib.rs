#![deny(unsafe_code)]
//! `kiln-engine`: continuous batching loop, paged KV cache, radix prefix
//! cache, sampler, speculative decoding, and SSD cold tier (SPEC §6).
//!
//! Phase 3 shipped the sampler (§6.6); Phase 4 adds the paged-KV block
//! manager (§6.3), the physical pools + gather-based paged attention
//! (§7.4 v0), and the continuous batching loop (§6.2). The block manager
//! is pure bookkeeping and builds without the `metal` feature; everything
//! touching MLX arrays is `metal`-gated.

pub mod block;
#[cfg(feature = "metal")]
pub mod engine;
#[cfg(feature = "metal")]
pub mod paged;
pub mod radix;
#[cfg(feature = "metal")]
pub mod sampler;
pub mod ssd;
#[cfg(feature = "metal")]
pub mod step;

pub use block::{AppendPlan, BlockError, BlockId, BlockManager, BlockTable, CowCopy, CowOutcome};
#[cfg(feature = "metal")]
pub use engine::{
    DEFAULT_BLOCK_SIZE, DEFAULT_MAX_BATCH_TOKENS, DEFAULT_NUM_BLOCKS, DEFAULT_PREFILL_CHUNK,
    Engine, EngineConfig, EngineError, EngineRequest, ErrorCause, EventSink, FinishKind,
    FinishSummary, KvDims, Priority, SeqEvent,
};
#[cfg(feature = "metal")]
pub use paged::{KvSpec, PagedKv, WriteRun};
pub use radix::{ChainHash, RadixCache};
#[cfg(feature = "metal")]
pub use sampler::{PenaltyOptions, Sampler, SamplingOptions, apply_penalties};
pub use ssd::{SlabGeometry, SsdStore};
#[cfg(feature = "metal")]
pub use step::{SeqStep, StepBatch, StepInput, StepModel};
