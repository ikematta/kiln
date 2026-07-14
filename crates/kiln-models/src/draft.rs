//! The `DraftModel` speculation mode (SPEC §6.5): a second, smaller model
//! loaded in the same worker process as the target — its own weight set
//! and its own paged KV pool, sharing nothing but the Metal device/stream
//! (every op still issues from the single engine thread).
//!
//! Isolation is structural: the draft's weights are a separate
//! [`AnyModel`] over separate arrays, and its KV lives in a separate
//! [`PagedKv`] addressed by a separate [`BlockManager`] — no pool, block
//! id, or array is shared with the target, so draft activity cannot
//! perturb target numerics (asserted bit-level by `tests/draft.rs`).
//! Accounting is shared: the draft's weights and pool bytes report
//! through [`Drafter::memory`] into the same worker heartbeat totals the
//! target uses (SPEC §2.3), so gateway budget math sees one combined
//! footprint per worker.
//!
//! Phase 8 part 1 is loading + accounting + the [`Drafter`] state
//! lifecycle only: [`Drafter::propose`] returns the empty proposal ("no
//! speculation this round") until the draft decode loop lands in part 2.

use std::collections::HashSet;
use std::path::Path;

use kiln_engine::{
    BlockManager, DEFAULT_BLOCK_SIZE, DEFAULT_NUM_BLOCKS, DraftError, Drafter, DrafterMemory,
    KvDims, KvSpec, PagedKv,
};
use kiln_mlx::Stream;

use crate::model::AnyModel;
use crate::nn::ModelError;

/// Geometry of the draft's own KV pool. Defaults mirror the engine's
/// target-pool defaults so the draft can shadow the same token capacity
/// the target pool admits — a sequence the target accepted never needs a
/// draft-side capacity decision (auto-disable heuristics are a later
/// Phase 8 part, not a loading concern).
#[derive(Debug, Clone, Copy)]
pub struct DraftPoolSpec {
    /// Tokens per block (SPEC §6.3: power of two).
    pub block_size: usize,
    /// Blocks in the draft pool.
    pub num_blocks: usize,
}

impl Default for DraftPoolSpec {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            num_blocks: DEFAULT_NUM_BLOCKS,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DraftLoadError {
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error("invalid draft KV pool spec: {0}")]
    Pool(String),
    #[error("failed to scan draft weights at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

/// A loaded draft model plus its own KV pool bookkeeping.
#[derive(Debug)]
pub struct DraftModel {
    model: AnyModel,
    mgr: BlockManager,
    kv: PagedKv,
    /// Sequences with live draft-side state (arrival-keyed, like the
    /// engine). Tables/blocks per sequence arrive with the decode loop.
    seqs: HashSet<u64>,
    /// Sum of the checkpoint's `.safetensors` file sizes — the same
    /// convention the worker's `StaticInfo.weights_bytes` uses for the
    /// target, so the two add up coherently in `MemoryReport`.
    weights_bytes: u64,
}

impl DraftModel {
    /// Loads the draft checkpoint at `dir` (any supported architecture)
    /// and sizes — but does not materialize — its private KV pool. Pool
    /// bytes stay 0 until first written, exactly like the target's.
    pub fn load(
        dir: impl AsRef<Path>,
        pool: DraftPoolSpec,
        s: &Stream,
    ) -> Result<Self, DraftLoadError> {
        let dir = dir.as_ref();
        pool.num_blocks
            .checked_mul(pool.block_size)
            .filter(|&tokens| tokens <= i32::MAX as usize)
            .ok_or_else(|| {
                DraftLoadError::Pool(format!(
                    "pool of {} blocks x {} tokens overflows i32 addressing",
                    pool.num_blocks, pool.block_size
                ))
            })?;
        let model = AnyModel::load(dir, s)?;
        let dims = model.kv_dims();
        let mgr = BlockManager::new(pool.num_blocks, pool.block_size)
            .map_err(|err| DraftLoadError::Pool(err.to_string()))?;
        let kv = PagedKv::new(KvSpec {
            layers: dims.layers,
            kv_heads: dims.kv_heads,
            head_dim: dims.head_dim,
            num_blocks: pool.num_blocks,
            block_size: pool.block_size,
        });
        let weights_bytes = weights_bytes(dir)?;
        Ok(Self {
            model,
            mgr,
            kv,
            seqs: HashSet::new(),
            weights_bytes,
        })
    }

    pub fn model(&self) -> &AnyModel {
        &self.model
    }

    pub fn kv_dims(&self) -> KvDims {
        self.model.kv_dims()
    }

    /// The draft's own KV pool (engine/test-facing; the decode loop
    /// writes through it in Phase 8 part 2).
    pub fn kv(&self) -> &PagedKv {
        &self.kv
    }

    pub fn kv_mut(&mut self) -> &mut PagedKv {
        &mut self.kv
    }
}

/// Weight footprint by the `StaticInfo.weights_bytes` convention: the
/// checkpoint's `.safetensors` file sizes summed.
fn weights_bytes(dir: &Path) -> Result<u64, DraftLoadError> {
    let entries = std::fs::read_dir(dir).map_err(|source| DraftLoadError::Io {
        path: dir.display().to_string(),
        source,
    })?;
    let mut bytes = 0;
    for entry in entries {
        let entry = entry.map_err(|source| DraftLoadError::Io {
            path: dir.display().to_string(),
            source,
        })?;
        let name = entry.file_name();
        if name.to_string_lossy().ends_with(".safetensors") {
            let meta = entry.metadata().map_err(|source| DraftLoadError::Io {
                path: entry.path().display().to_string(),
                source,
            })?;
            bytes += meta.len();
        }
    }
    Ok(bytes)
}

impl Drafter for DraftModel {
    fn memory(&self) -> DrafterMemory {
        let live = (self.mgr.capacity() - self.mgr.num_free()) as u64;
        DrafterMemory {
            weights_bytes: self.weights_bytes,
            kv_allocated_bytes: self.kv.allocated_bytes(),
            kv_used_bytes: live * self.kv.bytes_per_block(),
        }
    }

    fn begin(&mut self, seq: u64, _prompt: &[u32], _s: &Stream) -> Result<(), DraftError> {
        // Re-begin of a known sequence is the preemption-resume reset
        // (trait contract); with no per-sequence KV state yet, both cases
        // reduce to registration.
        self.seqs.insert(seq);
        Ok(())
    }

    fn propose(
        &mut self,
        seq: u64,
        _committed: &[u32],
        _gamma: usize,
        _s: &Stream,
    ) -> Result<Vec<u32>, DraftError> {
        if !self.seqs.contains(&seq) {
            return Err(DraftError::UnknownSeq(seq));
        }
        // Phase 8 part 1: the draft decode loop is not built, so every
        // round is "no speculation" — a legal proposal under the trait
        // contract, and the answer that leaves target decoding untouched.
        Ok(Vec::new())
    }

    fn release(&mut self, seq: u64) {
        self.seqs.remove(&seq);
    }
}
