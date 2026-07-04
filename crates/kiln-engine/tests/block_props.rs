//! Model-based property tests for the paged-KV block manager (SPEC §6.3;
//! Phase 4 task "block manager (+proptests)").
//!
//! Each case drives `BlockManager`/`BlockTable` with a random op sequence
//! against a naive shadow model and asserts after every op that:
//! - a block is never issued twice while an owner still holds it,
//! - refcounts equal the shadow owner counts, and a block returns to the
//!   free pool exactly when its last owner releases it,
//! - copy-on-write never exposes a shared block for writing: every owner's
//!   view of its block's contents survives other owners' writes.
//!
//! The `*_extended` variants rerun the same properties at a higher case
//! count; they are `#[ignore]`d from the default suite and run via
//! `cargo test -p kiln-engine -- --ignored` (CLAUDE.md convention).

use std::collections::HashMap;

use kiln_engine::block::{BlockError, BlockId, BlockManager, BlockTable, CowCopy, CowOutcome};
use proptest::prelude::*;
use proptest::sample::Index;
use proptest::test_runner::TestCaseError;

// ---------------------------------------------------------------------------
// Manager-level model: raw allocate/retain/release/copy_on_write.

#[derive(Debug, Clone)]
enum MgrOp {
    Alloc,
    /// Add an owner to the block held by a random live owner.
    Retain(Index),
    /// Drop a random live owner.
    Release(Index),
    /// A random owner writes its block: COW resolution + simulated write.
    Write(Index),
}

fn mgr_op() -> impl Strategy<Value = MgrOp> {
    prop_oneof![
        3 => Just(MgrOp::Alloc),
        2 => any::<Index>().prop_map(MgrOp::Retain),
        3 => any::<Index>().prop_map(MgrOp::Release),
        3 => any::<Index>().prop_map(MgrOp::Write),
    ]
}

fn mgr_params() -> impl Strategy<Value = (usize, usize, Vec<MgrOp>)> {
    (1usize..48, 0u32..8, prop::collection::vec(mgr_op(), 1..200))
        .prop_map(|(capacity, bs_exp, ops)| (capacity, 1usize << bs_exp, ops))
}

fn run_manager_model(
    capacity: usize,
    block_size: usize,
    ops: Vec<MgrOp>,
) -> Result<(), TestCaseError> {
    let mut mgr = BlockManager::new(capacity, block_size)
        .map_err(|e| TestCaseError::fail(format!("constructor: {e}")))?;
    // One entry per live ownership reference: (block, content this owner
    // last observed or wrote).
    let mut owners: Vec<(BlockId, u64)> = Vec::new();
    // Shadow of the physical block contents, one value per block.
    let mut contents: HashMap<BlockId, u64> = HashMap::new();
    let mut next_val: u64 = 0;

    for op in ops {
        match op {
            MgrOp::Alloc => {
                let free_before = mgr.num_free();
                match mgr.allocate() {
                    Ok(id) => {
                        prop_assert!(free_before > 0, "allocated from an empty free list");
                        prop_assert!(id.index() < capacity);
                        prop_assert!(
                            owners.iter().all(|&(b, _)| b != id),
                            "double-issued {:?}",
                            id
                        );
                        next_val += 1;
                        contents.insert(id, next_val);
                        owners.push((id, next_val));
                    }
                    Err(BlockError::OutOfBlocks { .. }) => prop_assert_eq!(free_before, 0),
                    Err(e) => return Err(TestCaseError::fail(format!("allocate: {e}"))),
                }
            }
            MgrOp::Retain(idx) => {
                if owners.is_empty() {
                    continue;
                }
                let (id, seen) = owners[idx.index(owners.len())];
                prop_assert!(mgr.retain(id).is_ok());
                owners.push((id, seen));
            }
            MgrOp::Release(idx) => {
                if owners.is_empty() {
                    continue;
                }
                let (id, _) = owners.swap_remove(idx.index(owners.len()));
                prop_assert!(mgr.release(id).is_ok());
                let remaining = owners.iter().filter(|&&(b, _)| b == id).count();
                prop_assert_eq!(
                    mgr.refcount(id),
                    Some(remaining as u32),
                    "refcount must hit 0 exactly when the last owner releases"
                );
                if remaining == 0 {
                    contents.remove(&id);
                }
            }
            MgrOp::Write(idx) => {
                if owners.is_empty() {
                    continue;
                }
                let i = idx.index(owners.len());
                let (id, _) = owners[i];
                let shared = owners.iter().filter(|&&(b, _)| b == id).count() > 1;
                match mgr.copy_on_write(id) {
                    Ok(CowOutcome::Writable(w)) => {
                        prop_assert_eq!(w, id);
                        prop_assert!(!shared, "shared block {:?} handed out as writable", id);
                        next_val += 1;
                        contents.insert(id, next_val);
                        owners[i] = (id, next_val);
                    }
                    Ok(CowOutcome::Copied(CowCopy { src, dst })) => {
                        prop_assert_eq!(src, id);
                        prop_assert!(shared, "sole-owned block {:?} was copied", id);
                        prop_assert_ne!(dst, src);
                        prop_assert!(
                            owners.iter().all(|&(b, _)| b != dst),
                            "COW double-issued {:?}",
                            dst
                        );
                        // The engine would copy src -> dst, then write dst.
                        next_val += 1;
                        contents.insert(dst, next_val);
                        owners[i] = (dst, next_val);
                    }
                    Err(BlockError::OutOfBlocks { .. }) => {
                        prop_assert!(shared, "COW of a sole-owned block needs no allocation");
                        prop_assert_eq!(mgr.num_free(), 0);
                    }
                    Err(e) => return Err(TestCaseError::fail(format!("copy_on_write: {e}"))),
                }
            }
        }

        // Invariants after every op.
        let mut refs: HashMap<BlockId, u32> = HashMap::new();
        for &(b, _) in &owners {
            *refs.entry(b).or_insert(0) += 1;
        }
        for (&b, &rc) in &refs {
            prop_assert_eq!(mgr.refcount(b), Some(rc), "refcount mismatch on {:?}", b);
        }
        prop_assert_eq!(mgr.num_free(), capacity - refs.len());
        // "COW never mutates a shared block": every owner still observes
        // the content it last saw, no matter what other owners did.
        for &(b, seen) in &owners {
            prop_assert_eq!(
                contents.get(&b).copied(),
                Some(seen),
                "owner view of {:?} changed underneath it",
                b
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Table-level model: append/fork/release with per-token contents, checking
// the engine-facing AppendPlan contract (COW exactly when a shared partial
// tail would be written; fresh blocks really fresh; failed appends atomic).

#[derive(Debug, Clone)]
enum TblOp {
    New,
    Append { table: Index, n: usize },
    Fork(Index),
    Release(Index),
}

fn tbl_op() -> impl Strategy<Value = TblOp> {
    prop_oneof![
        1 => Just(TblOp::New),
        4 => (any::<Index>(), 1usize..40).prop_map(|(table, n)| TblOp::Append { table, n }),
        2 => any::<Index>().prop_map(TblOp::Fork),
        1 => any::<Index>().prop_map(TblOp::Release),
    ]
}

fn tbl_params() -> impl Strategy<Value = (usize, usize, Vec<TblOp>)> {
    (1usize..32, 0u32..5, prop::collection::vec(tbl_op(), 1..150))
        .prop_map(|(capacity, bs_exp, ops)| (capacity, 1usize << bs_exp, ops))
}

fn run_table_model(
    capacity: usize,
    block_size: usize,
    ops: Vec<TblOp>,
) -> Result<(), TestCaseError> {
    let mut mgr = BlockManager::new(capacity, block_size)
        .map_err(|e| TestCaseError::fail(format!("constructor: {e}")))?;
    // Live tables with the logical token stream each expects to read back.
    let mut tables: Vec<(BlockTable, Vec<u64>)> = Vec::new();
    // Shadow of the physical pools: rows written so far, per block.
    let mut pool: HashMap<BlockId, Vec<u64>> = HashMap::new();
    let mut next_val: u64 = 0;

    for op in ops {
        match op {
            TblOp::New => tables.push((BlockTable::new(), Vec::new())),
            TblOp::Append { table, n } => {
                if tables.is_empty() {
                    continue;
                }
                let t = table.index(tables.len());
                // Pre-op facts the plan assertions need.
                let partial = !tables[t].0.num_tokens().is_multiple_of(block_size);
                let old_tail = tables[t].0.blocks().last().copied();
                let tail_shared = old_tail.is_some_and(|tail| {
                    tables
                        .iter()
                        .enumerate()
                        .any(|(j, (tb, _))| j != t && tb.blocks().contains(&tail))
                });
                let blocks_before = tables[t].0.blocks().to_vec();
                let tokens_before = tables[t].0.num_tokens();
                let free_before = mgr.num_free();

                let (tbl, logical) = &mut tables[t];
                match tbl.append_tokens(&mut mgr, n) {
                    Ok(plan) => {
                        if partial && tail_shared {
                            let copy = plan.cow.ok_or_else(|| {
                                TestCaseError::fail("shared partial tail appended without COW")
                            })?;
                            prop_assert_eq!(Some(copy.src), old_tail);
                            prop_assert!(
                                !pool.contains_key(&copy.dst),
                                "COW dst {:?} already live",
                                copy.dst
                            );
                            let src_rows = pool[&copy.src].clone();
                            pool.insert(copy.dst, src_rows);
                        } else {
                            prop_assert!(plan.cow.is_none(), "unexpected COW: {:?}", plan.cow);
                        }
                        for &fresh in &plan.appended {
                            prop_assert!(!pool.contains_key(&fresh), "double-issued {:?}", fresh);
                            pool.insert(fresh, Vec::new());
                        }
                        // Simulate the engine's writes for the n new tokens.
                        for k in 0..n {
                            let pos = tokens_before + k;
                            let bid = tbl.blocks()[pos / block_size];
                            let rows = pool.get_mut(&bid).expect("write target not in pool");
                            prop_assert_eq!(rows.len(), pos % block_size, "misaligned write");
                            rows.push(next_val);
                            logical.push(next_val);
                            next_val += 1;
                        }
                        prop_assert_eq!(tbl.num_tokens(), tokens_before + n);
                    }
                    Err(BlockError::OutOfBlocks { .. }) => {
                        // Failed appends must not disturb table or pool state.
                        prop_assert_eq!(tbl.blocks(), blocks_before.as_slice());
                        prop_assert_eq!(tbl.num_tokens(), tokens_before);
                        prop_assert_eq!(mgr.num_free(), free_before);
                    }
                    Err(e) => return Err(TestCaseError::fail(format!("append: {e}"))),
                }
            }
            TblOp::Fork(idx) => {
                if tables.is_empty() {
                    continue;
                }
                let t = idx.index(tables.len());
                let forked = tables[t]
                    .0
                    .fork(&mut mgr)
                    .map_err(|e| TestCaseError::fail(format!("fork: {e}")))?;
                let logical = tables[t].1.clone();
                tables.push((forked, logical));
            }
            TblOp::Release(idx) => {
                if tables.is_empty() {
                    continue;
                }
                let (tbl, _) = tables.swap_remove(idx.index(tables.len()));
                let ids = tbl.blocks().to_vec();
                tbl.release(&mut mgr)
                    .map_err(|e| TestCaseError::fail(format!("release: {e}")))?;
                for id in ids {
                    if !tables.iter().any(|(tb, _)| tb.blocks().contains(&id)) {
                        pool.remove(&id);
                    }
                }
            }
        }

        // Invariants after every op.
        let mut refs: HashMap<BlockId, u32> = HashMap::new();
        for (tb, _) in &tables {
            for &b in tb.blocks() {
                *refs.entry(b).or_insert(0) += 1;
            }
        }
        for (&b, &rc) in &refs {
            prop_assert_eq!(mgr.refcount(b), Some(rc), "refcount mismatch on {:?}", b);
        }
        prop_assert_eq!(mgr.num_free(), capacity - refs.len());
        // Every table's logical stream must read back exactly through its
        // block table — a mutated shared block breaks a sibling here.
        for (tb, logical) in &tables {
            prop_assert_eq!(tb.num_tokens(), logical.len());
            prop_assert_eq!(tb.blocks().len(), logical.len().div_ceil(block_size));
            for (pos, &expected) in logical.iter().enumerate() {
                let bid = tb.blocks()[pos / block_size];
                prop_assert_eq!(
                    pool[&bid][pos % block_size],
                    expected,
                    "table view mutated at position {}",
                    pos
                );
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Default runs (proptest default case count, honors PROPTEST_CASES).

proptest! {
    #[test]
    fn manager_model((capacity, block_size, ops) in mgr_params()) {
        run_manager_model(capacity, block_size, ops)?;
    }

    #[test]
    fn table_model((capacity, block_size, ops) in tbl_params()) {
        run_table_model(capacity, block_size, ops)?;
    }
}

// Extended runs: same properties at 16x the default case count.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    #[test]
    #[ignore = "extended case count; run with: cargo test -p kiln-engine -- --ignored"]
    fn manager_model_extended((capacity, block_size, ops) in mgr_params()) {
        run_manager_model(capacity, block_size, ops)?;
    }

    #[test]
    #[ignore = "extended case count; run with: cargo test -p kiln-engine -- --ignored"]
    fn table_model_extended((capacity, block_size, ops) in tbl_params()) {
        run_table_model(capacity, block_size, ops)?;
    }
}
