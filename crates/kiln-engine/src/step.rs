//! The step contract between the batching loop and a model implementation
//! (SPEC §6.2 step 2 / §7.2): the engine describes one forward pass —
//! which sequences participate, where their K/V rows land, which blocks to
//! gather — and the model executes it against the paged pools.

use kiln_mlx::{Array, MlxError, Stream};

use crate::block::BlockId;
use crate::paged::{PagedKv, WriteRun};
use crate::paged_attn::PagedAttnInputs;

/// One sequence's contribution to a step.
#[derive(Debug)]
pub struct SeqStep {
    /// Tokens this sequence feeds this step (1 for decode, the chunk
    /// length during prefill).
    pub len: i32,
    /// Tokens already in this sequence's KV before the step — its RoPE
    /// offset.
    pub offset: i32,
    /// How many of the segment's trailing positions are sampled (their
    /// logits are returned): 0 for prefill chunks and post-preemption
    /// replay, 1 for plain decode, `len` for a speculative verify segment
    /// (SPEC §6.5 — the target scores the fed token AND every draft token
    /// in one forward). Never negative, never more than `len`.
    pub sample_rows: i32,
    /// All blocks backing the sequence through this step, in token order.
    pub blocks: Vec<BlockId>,
    /// Where this step's K/V rows land in the pools.
    pub writes: Vec<WriteRun>,
    /// Prepared inputs for the block-table-aware attention kernel (SPEC
    /// §7.4 Phase 7). `Some` only when the engine's kernel flag is on AND
    /// the segment is decode-shaped (`len == 1`); models take the
    /// [`PagedKv::paged_sdpa`] route iff this is set and the step carries
    /// no pad rows and the architecture uses fused SDPA (no softcap).
    pub paged_attn: Option<PagedAttnInputs>,
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
    /// Kernel-class pad rows the model must append to the trunk (and
    /// prepend as SDPA query rows) for this step — ADR 0002 / the
    /// [`crate::PREFILL_PAD_MIN_ROWS`] rule. Set only on single-sequence
    /// ragged prefill pieces; `input` and `num_tokens()` cover REAL tokens
    /// only. Pad rows are pure compute filler: implementations must never
    /// write their K/V, sample them, or emit them.
    pub pad_rows: i32,
}

impl StepBatch {
    /// Total REAL positions in the step (pad rows excluded).
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
