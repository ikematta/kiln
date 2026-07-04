//! The step contract between the batching loop and a model implementation
//! (SPEC §6.2 step 2 / §7.2): the engine describes one forward pass —
//! which sequences participate, where their K/V rows land, which blocks to
//! gather — and the model executes it against the paged pools.

use kiln_mlx::{Array, MlxError, Stream};

use crate::block::BlockId;
use crate::paged::{PagedKv, WriteRun};

/// One sequence's contribution to a step.
#[derive(Debug)]
pub struct SeqStep {
    /// Tokens this sequence feeds this step (1 for decode, the chunk
    /// length during prefill).
    pub len: i32,
    /// Tokens already in this sequence's KV before the step — its RoPE
    /// offset.
    pub offset: i32,
    /// Whether the segment's last position is sampled (its logits are
    /// returned).
    pub sample: bool,
    /// All blocks backing the sequence through this step, in token order.
    pub blocks: Vec<BlockId>,
    /// Where this step's K/V rows land in the pools.
    pub writes: Vec<WriteRun>,
}

/// The step's input tokens: host-known ids on the synchronous path, or a
/// lazily computed `[1, n]` u32 array on the pipelined decode path — the
/// engine feeds the previous step's sampled tokens back into the next
/// forward *before* reading their values to the host (the async_eval
/// pipeline, mirroring `generate.rs`), so no id vector exists at
/// graph-build time.
#[derive(Debug)]
pub enum StepInput {
    Ids(Vec<u32>),
    Lazy(Array),
}

impl StepInput {
    /// Total positions in the step (shape-only for `Lazy`; MLX shapes are
    /// known without evaluation).
    pub fn num_tokens(&self) -> usize {
        match self {
            StepInput::Ids(ids) => ids.len(),
            StepInput::Lazy(tokens) => tokens.dim(1) as usize,
        }
    }
}

/// One forward pass: `input` is the concatenation of every sequence's
/// segment, in `seqs` order.
#[derive(Debug)]
pub struct StepBatch {
    pub input: StepInput,
    pub seqs: Vec<SeqStep>,
}

impl StepBatch {
    /// Total positions in the step.
    pub fn num_tokens(&self) -> usize {
        self.input.num_tokens()
    }
}

/// A model the batching loop can drive (SPEC §7.2, paged form).
pub trait StepModel {
    /// Runs one forward pass over `batch`, writing this step's K/V into
    /// `kv`, and returns logits `[1, n_sampled, vocab]` for the sampled
    /// positions in `seqs` order — `None` when no sequence samples (a pure
    /// prefill chunk; implementations skip the lm_head for those).
    fn forward_step(
        &self,
        batch: &StepBatch,
        kv: &mut PagedKv,
        s: &Stream,
    ) -> Result<Option<Array>, MlxError>;
}

impl<M: StepModel + ?Sized> StepModel for &M {
    fn forward_step(
        &self,
        batch: &StepBatch,
        kv: &mut PagedKv,
        s: &Stream,
    ) -> Result<Option<Array>, MlxError> {
        (**self).forward_step(batch, kv, s)
    }
}
