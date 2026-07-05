//! Radix-tree property tests (SPEC §11.3): random interleavings of
//! donation, matching, request-style sharing, and eviction against a
//! naive reference model, checking
//! - longest-prefix matches equal the reference (a flat set of
//!   block-aligned prefixes),
//! - block-manager conservation (free + cache-held + request-held ==
//!   capacity) with no double ownership,
//! - eviction only ever frees sole-owned leaves, and (without an SSD
//!   tier) removes exactly that one prefix from matchability.
//!
//! MLX-free (like the tree itself): runs in every CI shape.

use std::collections::HashMap;

use kiln_engine::radix::{ROOT, RadixCache};
use kiln_engine::{BlockId, BlockManager};
use proptest::prelude::*;

const BLOCK: usize = 4;
const CAPACITY: usize = 64;

#[derive(Debug, Clone)]
enum Op {
    /// Donate a token stream of `chunks` blocks drawn from a tiny
    /// alphabet (dense prefix sharing).
    Donate { seed: u64, chunks: usize },
    /// Longest-prefix match of a stream, holding the matched blocks like
    /// an admitted request would (refcount++).
    MatchAndHold { seed: u64, chunks: usize },
    /// Release the oldest held match (request finished; no re-donation —
    /// the engine's dedup makes that a no-op for shared prefixes).
    ReleaseHeld,
    /// Evict one LRU sole-owned leaf, as pool pressure would.
    EvictOne,
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        (any::<u64>(), 1_usize..6).prop_map(|(seed, chunks)| Op::Donate { seed, chunks }),
        (any::<u64>(), 1_usize..6).prop_map(|(seed, chunks)| Op::MatchAndHold { seed, chunks }),
        Just(Op::ReleaseHeld),
        Just(Op::EvictOne),
    ]
}

/// Deterministic token stream with heavy shared prefixes: chunk `i` of a
/// stream is one of two variants, picked by bit `i` of the seed.
fn stream(seed: u64, chunks: usize) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(chunks * BLOCK);
    for i in 0..chunks {
        let variant = (seed >> i) & 1;
        for j in 0..BLOCK {
            tokens.push((i as u32) * 16 + (variant as u32) * 8 + j as u32);
        }
    }
    tokens
}

/// The reference: the set of matchable block-aligned prefixes, plus the
/// per-prefix physical block for ownership cross-checks.
#[derive(Default)]
struct Reference {
    present: HashMap<Vec<u32>, BlockId>,
}

impl Reference {
    fn longest_match(&self, tokens: &[u32]) -> usize {
        let mut chunks = 0;
        while (chunks + 1) * BLOCK <= tokens.len()
            && self.present.contains_key(&tokens[..(chunks + 1) * BLOCK])
        {
            chunks += 1;
        }
        chunks
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        .. ProptestConfig::default()
    })]

    #[test]
    fn radix_matches_reference_and_conserves_blocks(ops in proptest::collection::vec(op_strategy(), 1..80)) {
        let mut mgr = BlockManager::new(CAPACITY, BLOCK).expect("manager");
        let mut tree = RadixCache::new(BLOCK, [0x5a; 32]);
        let mut reference = Reference::default();
        let mut cache_live = 0_usize;
        // Matches held by "running requests": vectors of retained blocks.
        let mut held: Vec<Vec<BlockId>> = Vec::new();

        for op in ops {
            match op {
                Op::Donate { seed, chunks } => {
                    let tokens = stream(seed, chunks);
                    let mut cur = ROOT;
                    for (i, chunk) in tokens.chunks_exact(BLOCK).enumerate() {
                        let Ok(block) = mgr.allocate() else { break };
                        cur = match tree.child(cur, chunk) {
                            Some(node) => {
                                prop_assert!(tree.node_block(node).is_some(),
                                    "non-resident node in an SSD-less tree");
                                mgr.release(block).expect("release duplicate");
                                tree.touch(node);
                                node
                            }
                            None => {
                                let node = tree.insert_child(cur, chunk);
                                tree.set_resident(node, block);
                                cache_live += 1;
                                reference.present.insert(tokens[..(i + 1) * BLOCK].to_vec(), block);
                                node
                            }
                        };
                    }
                }
                Op::MatchAndHold { seed, chunks } => {
                    let tokens = stream(seed, chunks);
                    // Engine walk: descend while resident, retain each block.
                    let mut cur = ROOT;
                    let mut got = Vec::new();
                    for chunk in tokens.chunks_exact(BLOCK) {
                        let Some(node) = tree.child(cur, chunk) else { break };
                        let Some(block) = tree.node_block(node) else { break };
                        mgr.retain(block).expect("retain matched block");
                        tree.touch(node);
                        got.push(block);
                        cur = node;
                    }
                    prop_assert_eq!(got.len(), reference.longest_match(&tokens),
                        "match length diverged from the reference");
                    // While held, every matched block is shared.
                    for block in &got {
                        prop_assert!(mgr.refcount(*block).unwrap_or(0) >= 2);
                    }
                    if !got.is_empty() {
                        held.push(got);
                    }
                }
                Op::ReleaseHeld => {
                    if !held.is_empty() {
                        for block in held.remove(0) {
                            mgr.release(block).expect("release held");
                        }
                    }
                }
                Op::EvictOne => {
                    let candidate = tree.evict_candidate(&mgr);
                    if let Some(node) = candidate {
                        let block = tree.node_block(node).expect("candidate resident");
                        // Only sole-owned (cache-only) blocks are eligible.
                        prop_assert_eq!(mgr.refcount(block), Some(1));
                        let freed = tree.evict(&mut mgr, node);
                        prop_assert_eq!(freed, Some(block));
                        prop_assert_eq!(mgr.refcount(block), Some(0));
                        cache_live -= 1;
                        // SSD-less eviction prunes exactly that prefix.
                        let prefix = reference.present.iter()
                            .find(|(_, b)| **b == block)
                            .map(|(prefix, _)| prefix.clone())
                            .expect("evicted block tracked");
                        reference.present.remove(&prefix);
                        // Leaf-first: nothing extending it may remain.
                        for other in reference.present.keys() {
                            prop_assert!(!other.starts_with(&prefix[..]) || other.len() <= prefix.len(),
                                "evicted an interior node");
                        }
                    }
                }
            }

            // Conservation: every block is exactly one of free /
            // cache-held / additionally request-shared.
            let held_extra: usize = held.iter().map(Vec::len).sum();
            let live = CAPACITY - mgr.num_free();
            prop_assert_eq!(live, cache_live,
                "cache residency out of sync with the pool (held extras: {})", held_extra);
            prop_assert_eq!(tree.resident_blocks(), cache_live);
            prop_assert_eq!(reference.present.len(), cache_live);
        }

        // Teardown: release holds, evict everything; the pool must refill.
        for blocks in held.drain(..) {
            for block in blocks {
                mgr.release(block).expect("release held");
            }
        }
        while let Some(node) = tree.evict_candidate(&mgr) {
            tree.evict(&mut mgr, node);
        }
        prop_assert_eq!(mgr.num_free(), CAPACITY, "eviction must drain the tree");
        prop_assert_eq!(tree.resident_blocks(), 0);
    }
}
