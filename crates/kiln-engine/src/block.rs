//! Paged-KV block manager (SPEC §6.3): a fixed pool of KV-cache blocks
//! with refcounting and copy-on-write, plus the per-request [`BlockTable`]s
//! that map token positions to pool blocks.
//!
//! Pure bookkeeping, deliberately MLX-free: the manager deals in block
//! *indices*. The physical per-layer K/V pool arrays — and the data copy a
//! [`CowOutcome::Copied`] instructs — belong to the paged-attention layer
//! that lands later in Phase 4, so this module builds and tests without
//! the `metal` feature.
//!
//! Sharing model: a block's refcount is its number of owners — a
//! [`BlockTable`] entry today, a radix-tree node once Phase 5's prefix
//! cache lands. The manager never hands out a shared block for writing:
//! a writer of a block with refcount > 1 gets a fresh block to copy into
//! ([`BlockManager::copy_on_write`]), so shared contents are immutable by
//! construction.
//!
//! Determinism: the free list is LIFO and all operations are sequential
//! on the engine thread, so block placement is reproducible run-to-run —
//! part of the greedy bit-reproducibility contract (CLAUDE.md).

use thiserror::Error;

/// Index of one fixed-size block in the (per-layer) KV pools.
///
/// Ids are minted only by a [`BlockManager`]; presenting an id to a
/// manager other than the one that issued it is a logic error (detected
/// only when the id is out of range for that pool).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(u32);

impl BlockId {
    /// Position of this block in the pool arrays.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum BlockError {
    /// The pool has fewer free blocks than the operation required.
    #[error("block pool exhausted: {needed} free block(s) needed, pool capacity {capacity}")]
    OutOfBlocks { capacity: usize, needed: usize },
    /// The id does not name a live block of this pool (already free, or
    /// minted by a different manager).
    #[error("block {0:?} is not allocated")]
    NotAllocated(BlockId),
    /// More than `u32::MAX` owners requested for one block.
    #[error("refcount overflow on block {0:?}")]
    RefcountOverflow(BlockId),
    /// Block size must be a nonzero power of two (SPEC §6.3).
    #[error("invalid block size {0}: must be a nonzero power of two")]
    InvalidBlockSize(usize),
    /// The pool needs at least one block and block ids are `u32`.
    #[error("invalid pool capacity {0}: must be in 1..=u32::MAX")]
    InvalidCapacity(usize),
    /// A table's token count overflowed `usize`.
    #[error("token count overflow")]
    TokenCountOverflow,
}

/// Physical copy the engine must perform: block `src`'s rows into `dst`,
/// before any new token is written to `dst`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CowCopy {
    pub src: BlockId,
    pub dst: BlockId,
}

/// Resolution of a write intent against a block (see
/// [`BlockManager::copy_on_write`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CowOutcome {
    /// The caller is the sole owner; write in place.
    Writable(BlockId),
    /// The block was shared; the caller's ownership moved to the fresh
    /// `dst` block, which must be populated per the enclosed [`CowCopy`].
    Copied(CowCopy),
}

/// Fixed-capacity pool of KV blocks: allocation, refcounting, and
/// copy-on-write resolution. Holds no KV data — only ownership state.
#[derive(Debug)]
pub struct BlockManager {
    block_size: usize,
    /// Per-block owner count; 0 means the block is on the free list.
    refcounts: Vec<u32>,
    /// LIFO free list.
    free: Vec<BlockId>,
}

impl BlockManager {
    /// A pool of `capacity` blocks of `block_size` tokens each.
    pub fn new(capacity: usize, block_size: usize) -> Result<Self, BlockError> {
        if block_size == 0 || !block_size.is_power_of_two() {
            return Err(BlockError::InvalidBlockSize(block_size));
        }
        if capacity == 0 {
            return Err(BlockError::InvalidCapacity(capacity));
        }
        let capacity_u32 =
            u32::try_from(capacity).map_err(|_| BlockError::InvalidCapacity(capacity))?;
        Ok(Self {
            block_size,
            refcounts: vec![0; capacity],
            // Reversed so blocks are first issued in 0, 1, 2, ... order.
            free: (0..capacity_u32).rev().map(BlockId).collect(),
        })
    }

    /// Tokens per block.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Total blocks in the pool.
    pub fn capacity(&self) -> usize {
        self.refcounts.len()
    }

    /// Blocks currently on the free list.
    pub fn num_free(&self) -> usize {
        self.free.len()
    }

    /// Owner count for `id`: 0 means free, `None` means the id is not
    /// from this pool.
    pub fn refcount(&self, id: BlockId) -> Option<u32> {
        self.refcounts.get(id.index()).copied()
    }

    /// Takes a block off the free list, returning it with refcount 1.
    pub fn allocate(&mut self) -> Result<BlockId, BlockError> {
        let capacity = self.capacity();
        let id = self.free.pop().ok_or(BlockError::OutOfBlocks {
            capacity,
            needed: 1,
        })?;
        *self.slot_mut(id)? = 1;
        Ok(id)
    }

    /// Adds an owner to an allocated block (prefix reuse, table fork).
    pub fn retain(&mut self, id: BlockId) -> Result<(), BlockError> {
        let slot = self.slot_mut(id)?;
        if *slot == 0 {
            return Err(BlockError::NotAllocated(id));
        }
        *slot = slot
            .checked_add(1)
            .ok_or(BlockError::RefcountOverflow(id))?;
        Ok(())
    }

    /// Drops one owner; the block returns to the free list when its last
    /// owner releases it.
    pub fn release(&mut self, id: BlockId) -> Result<(), BlockError> {
        let slot = self.slot_mut(id)?;
        if *slot == 0 {
            return Err(BlockError::NotAllocated(id));
        }
        *slot -= 1;
        if *slot == 0 {
            self.free.push(id);
        }
        Ok(())
    }

    /// Resolves a write intent against `id` (SPEC §6.3 copy-on-write).
    ///
    /// Sole owner → [`CowOutcome::Writable`]: write in place. Shared →
    /// the caller's ownership of `id` moves to a freshly allocated block
    /// and [`CowOutcome::Copied`] instructs the engine to copy the
    /// physical rows before writing; the shared source itself is never
    /// handed out as writable. On [`BlockError::OutOfBlocks`] nothing
    /// changes.
    pub fn copy_on_write(&mut self, id: BlockId) -> Result<CowOutcome, BlockError> {
        let refcount = self.refcount(id).ok_or(BlockError::NotAllocated(id))?;
        match refcount {
            0 => Err(BlockError::NotAllocated(id)),
            1 => Ok(CowOutcome::Writable(id)),
            _ => {
                // Allocate the copy target first so exhaustion leaves the
                // pool untouched; the release then drops the caller's
                // ownership of the shared source (count stays >= 1, so
                // the source cannot be freed under the pending copy).
                let dst = self.allocate()?;
                self.release(id)?;
                Ok(CowOutcome::Copied(CowCopy { src: id, dst }))
            }
        }
    }

    fn slot_mut(&mut self, id: BlockId) -> Result<&mut u32, BlockError> {
        self.refcounts
            .get_mut(id.index())
            .ok_or(BlockError::NotAllocated(id))
    }
}

/// What the engine must do to the physical pools after a table append.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct AppendPlan {
    /// A shared tail block was detached: copy `src`'s rows into `dst`
    /// before writing the new tokens (the table now lists `dst`).
    pub cow: Option<CowCopy>,
    /// Fresh blocks appended to the table, in order.
    pub appended: Vec<BlockId>,
}

/// Per-request map from token positions to pool blocks (SPEC §6.3):
/// token position `p` lives in `blocks()[p / block_size]` at row
/// `p % block_size`.
///
/// The table owns one refcount on every block it lists. It holds no
/// manager handle, so dropping it does *not* release its blocks — request
/// teardown must call [`release`](Self::release) (or move ownership, e.g.
/// into the Phase 5 prefix cache) explicitly.
#[derive(Debug, Default)]
#[must_use]
pub struct BlockTable {
    blocks: Vec<BlockId>,
    num_tokens: usize,
}

impl BlockTable {
    /// An empty table owning no blocks.
    pub fn new() -> Self {
        Self::default()
    }

    /// Blocks backing this request, in token order.
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    /// Logical tokens stored (the last block may be partially filled).
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Grows the table by `n` token slots, allocating fresh blocks and
    /// resolving a copy-on-write when the first new token would land in a
    /// shared, partially filled tail block.
    ///
    /// Atomic on failure: if the pool cannot cover the whole append
    /// (including the COW target), neither the table nor the manager is
    /// modified.
    pub fn append_tokens(
        &mut self,
        mgr: &mut BlockManager,
        n: usize,
    ) -> Result<AppendPlan, BlockError> {
        if n == 0 {
            return Ok(AppendPlan::default());
        }
        let block_size = mgr.block_size();
        let new_total = self
            .num_tokens
            .checked_add(n)
            .ok_or(BlockError::TokenCountOverflow)?;

        // The tail block is written by this append only if it is
        // partially filled; a shared one must be detached first.
        let cow_src = if self.num_tokens.is_multiple_of(block_size) {
            None
        } else {
            self.blocks
                .last()
                .copied()
                .filter(|&tail| matches!(mgr.refcount(tail), Some(rc) if rc > 1))
        };

        let fresh = new_total
            .div_ceil(block_size)
            .saturating_sub(self.blocks.len());
        let needed = fresh + usize::from(cow_src.is_some());
        if mgr.num_free() < needed {
            return Err(BlockError::OutOfBlocks {
                capacity: mgr.capacity(),
                needed,
            });
        }

        // Nothing below can fail: the free list covers `needed`.
        let cow = if let Some(src) = cow_src {
            match mgr.copy_on_write(src)? {
                CowOutcome::Copied(copy) => {
                    if let Some(tail) = self.blocks.last_mut() {
                        *tail = copy.dst;
                    }
                    Some(copy)
                }
                // Unreachable: cow_src is only set for refcount > 1.
                CowOutcome::Writable(_) => None,
            }
        } else {
            None
        };
        let mut appended = Vec::with_capacity(fresh);
        for _ in 0..fresh {
            let id = mgr.allocate()?;
            self.blocks.push(id);
            appended.push(id);
        }
        self.num_tokens = new_total;
        Ok(AppendPlan { cow, appended })
    }

    /// Clones the table, adding an owner to every block (prefix sharing).
    /// On failure the retains taken so far are rolled back, leaving the
    /// manager unchanged.
    pub fn fork(&self, mgr: &mut BlockManager) -> Result<Self, BlockError> {
        for (retained, &id) in self.blocks.iter().enumerate() {
            if let Err(err) = mgr.retain(id) {
                // Releasing a block we hold a fresh reference on cannot
                // fail; discard the Results.
                for &undo in &self.blocks[..retained] {
                    let _ = mgr.release(undo);
                }
                return Err(err);
            }
        }
        Ok(Self {
            blocks: self.blocks.clone(),
            num_tokens: self.num_tokens,
        })
    }

    /// Releases this table's ownership of every block, consuming the
    /// table. Blocks whose last owner this was return to the free list.
    pub fn release(self, mgr: &mut BlockManager) -> Result<(), BlockError> {
        for &id in &self.blocks {
            mgr.release(id)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_block_size() {
        assert_eq!(
            BlockManager::new(4, 0).err(),
            Some(BlockError::InvalidBlockSize(0))
        );
        assert_eq!(
            BlockManager::new(4, 24).err(),
            Some(BlockError::InvalidBlockSize(24))
        );
    }

    #[test]
    fn rejects_zero_capacity() {
        assert_eq!(
            BlockManager::new(0, 32).err(),
            Some(BlockError::InvalidCapacity(0))
        );
    }

    #[test]
    fn allocates_in_order_until_exhausted() {
        let mut mgr = BlockManager::new(3, 32).unwrap();
        let ids: Vec<usize> = (0..3).map(|_| mgr.allocate().unwrap().index()).collect();
        assert_eq!(ids, vec![0, 1, 2]);
        assert_eq!(
            mgr.allocate().err(),
            Some(BlockError::OutOfBlocks {
                capacity: 3,
                needed: 1
            })
        );
        assert_eq!(mgr.num_free(), 0);
    }

    #[test]
    fn refcount_lifecycle_and_reuse() {
        let mut mgr = BlockManager::new(2, 32).unwrap();
        let id = mgr.allocate().unwrap();
        assert_eq!(mgr.refcount(id), Some(1));
        mgr.retain(id).unwrap();
        assert_eq!(mgr.refcount(id), Some(2));
        mgr.release(id).unwrap();
        assert_eq!(mgr.refcount(id), Some(1));
        assert_eq!(mgr.num_free(), 1);
        mgr.release(id).unwrap();
        assert_eq!(mgr.refcount(id), Some(0));
        assert_eq!(mgr.num_free(), 2);
        // Free block: further owner operations are errors.
        assert_eq!(mgr.release(id).err(), Some(BlockError::NotAllocated(id)));
        assert_eq!(mgr.retain(id).err(), Some(BlockError::NotAllocated(id)));
        assert_eq!(
            mgr.copy_on_write(id).err(),
            Some(BlockError::NotAllocated(id))
        );
    }

    #[test]
    fn foreign_ids_are_rejected() {
        let mut mgr = BlockManager::new(4, 32).unwrap();
        let foreign = BlockId(7);
        assert_eq!(mgr.refcount(foreign), None);
        assert_eq!(
            mgr.retain(foreign).err(),
            Some(BlockError::NotAllocated(foreign))
        );
        assert_eq!(
            mgr.release(foreign).err(),
            Some(BlockError::NotAllocated(foreign))
        );
    }

    #[test]
    fn cow_sole_owner_writes_in_place() {
        let mut mgr = BlockManager::new(2, 32).unwrap();
        let id = mgr.allocate().unwrap();
        assert_eq!(mgr.copy_on_write(id).unwrap(), CowOutcome::Writable(id));
        assert_eq!(mgr.refcount(id), Some(1));
        assert_eq!(mgr.num_free(), 1);
    }

    #[test]
    fn cow_shared_allocates_copy_and_moves_ownership() {
        let mut mgr = BlockManager::new(2, 32).unwrap();
        let id = mgr.allocate().unwrap();
        mgr.retain(id).unwrap();
        let CowOutcome::Copied(copy) = mgr.copy_on_write(id).unwrap() else {
            panic!("expected a copy for a shared block");
        };
        assert_eq!(copy.src, id);
        assert_ne!(copy.dst, id);
        // One owner moved off the shared block onto the fresh copy.
        assert_eq!(mgr.refcount(id), Some(1));
        assert_eq!(mgr.refcount(copy.dst), Some(1));
        assert_eq!(mgr.num_free(), 0);
    }

    #[test]
    fn cow_exhausted_pool_leaves_state_unchanged() {
        let mut mgr = BlockManager::new(1, 32).unwrap();
        let id = mgr.allocate().unwrap();
        mgr.retain(id).unwrap();
        assert_eq!(
            mgr.copy_on_write(id).err(),
            Some(BlockError::OutOfBlocks {
                capacity: 1,
                needed: 1
            })
        );
        assert_eq!(mgr.refcount(id), Some(2));
        assert_eq!(mgr.num_free(), 0);
    }

    #[test]
    fn table_append_allocates_blocks_and_tracks_tokens() {
        let mut mgr = BlockManager::new(4, 4).unwrap();
        let mut table = BlockTable::new();
        assert_eq!(
            table.append_tokens(&mut mgr, 0).unwrap(),
            AppendPlan::default()
        );

        let plan = table.append_tokens(&mut mgr, 6).unwrap();
        assert_eq!(plan.cow, None);
        assert_eq!(plan.appended.len(), 2);
        assert_eq!(table.blocks(), plan.appended.as_slice());
        assert_eq!(table.num_tokens(), 6);

        // Fits in the partial (unshared) tail: no allocation, no COW.
        let plan = table.append_tokens(&mut mgr, 2).unwrap();
        assert_eq!(plan, AppendPlan::default());
        assert_eq!(table.num_tokens(), 8);
        table.release(&mut mgr).unwrap();
        assert_eq!(mgr.num_free(), 4);
    }

    #[test]
    fn table_append_cows_shared_partial_tail() {
        let mut mgr = BlockManager::new(4, 4).unwrap();
        let mut table = BlockTable::new();
        table.append_tokens(&mut mgr, 6).unwrap();
        let forked = table.fork(&mut mgr).unwrap();
        let shared_tail = *table.blocks().last().unwrap();
        assert_eq!(mgr.refcount(shared_tail), Some(2));

        let plan = table.append_tokens(&mut mgr, 1).unwrap();
        let copy = plan.cow.expect("shared partial tail must be copied");
        assert_eq!(copy.src, shared_tail);
        assert_eq!(table.blocks().last(), Some(&copy.dst));
        // The fork still owns the original tail, untouched.
        assert_eq!(forked.blocks().last(), Some(&shared_tail));
        assert_eq!(mgr.refcount(shared_tail), Some(1));

        // A shared but *full* tail is never written, so never copied.
        let mut aligned = BlockTable::new();
        aligned.append_tokens(&mut mgr, 4).unwrap();
        let aligned_fork = aligned.fork(&mut mgr).unwrap();
        // Pool is now exhausted (3 + 1 blocks live); free one first.
        table.release(&mut mgr).unwrap();
        let plan = aligned.append_tokens(&mut mgr, 1).unwrap();
        assert_eq!(plan.cow, None);
        assert_eq!(plan.appended.len(), 1);

        forked.release(&mut mgr).unwrap();
        aligned.release(&mut mgr).unwrap();
        aligned_fork.release(&mut mgr).unwrap();
        assert_eq!(mgr.num_free(), 4);
    }

    #[test]
    fn table_append_is_atomic_on_exhaustion() {
        let mut mgr = BlockManager::new(3, 4).unwrap();
        let mut table = BlockTable::new();
        table.append_tokens(&mut mgr, 6).unwrap();
        let forked = table.fork(&mut mgr).unwrap();
        let blocks_before = table.blocks().to_vec();

        // Needs a COW target + 1 fresh block, but only 1 block is free.
        assert_eq!(
            table.append_tokens(&mut mgr, 3).err(),
            Some(BlockError::OutOfBlocks {
                capacity: 3,
                needed: 2
            })
        );
        assert_eq!(table.blocks(), blocks_before.as_slice());
        assert_eq!(table.num_tokens(), 6);
        assert_eq!(mgr.num_free(), 1);

        // The COW alone still fits.
        let plan = table.append_tokens(&mut mgr, 2).unwrap();
        assert!(plan.cow.is_some());
        assert!(plan.appended.is_empty());
        assert_eq!(table.num_tokens(), 8);

        table.release(&mut mgr).unwrap();
        forked.release(&mut mgr).unwrap();
        assert_eq!(mgr.num_free(), 3);
    }
}
