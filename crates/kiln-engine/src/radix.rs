//! Radix prefix cache (SPEC §6.3): a tree keyed on token-id sequences in
//! block-aligned chunks — one node per full KV block — mapping prefixes of
//! past requests to pool blocks that can be reused instead of re-prefilled.
//!
//! Pure bookkeeping, deliberately MLX-free like the block manager: the
//! tree deals in [`BlockId`]s and token ids. The engine drives it — the
//! walk during admission, donation at finish, eviction under pool
//! pressure, and the SSD load/flush hand-off all live in `engine.rs`.
//!
//! Ownership: the cache holds exactly one [`BlockManager`] owner reference
//! for every *resident* node (a node whose block is in the pool). Requests
//! that reuse a node take their own reference (`retain`), so a shared
//! block has refcount >= 2 and the manager's copy-on-write keeps its
//! contents immutable. Eviction only ever targets nodes the cache solely
//! owns (refcount 1) with no resident children, LRU-first — residency is
//! therefore prefix-closed: a resident node's ancestors are resident,
//! which is what makes a match walk meaningful.
//!
//! ## The settled-rows donation invariant (Phase 4 carry-forward)
//!
//! The async_eval decode pipeline can discard a speculative row: a
//! sequence that stops at a pipelined apply has one already-scheduled,
//! never-read row in flight, whose KV write lands in a block the sequence
//! owned at build time. Under Phase 4's free-list-only reuse that was safe
//! because every future owner *rewrites* each row below its own length
//! (stream-ordered after the stale write) — but a radix-shared block is
//! read by requests that never wrote it, so that argument dies here.
//!
//! Resolution (Phase 5 decision): **only settled rows are donatable.** At
//! finish, exactly `processed + fed` rows are settled — each was an
//! ancestor of a token readback or a step-boundary eval, so its contents
//! are materialized and final. A discarded speculative row sits at slot
//! `processed + fed`, strictly beyond that range; the donation in
//! `engine.rs` covers `floor((processed + fed) / block_size)` full blocks
//! and releases everything past them to the free list, where the Phase-4
//! rewrite argument still holds (free-list blocks are only reacquired by
//! writers). The alternative — forcing the discarded row's readback to
//! keep its block cache-eligible — was rejected: it would stall the
//! pipeline at every stop and rest correctness on the value of a row no
//! reader ever checked, to save at most one block per pipelined finish.
//! `tests/pipeline_discard.rs` pins this: a pipelined stop immediately
//! followed by a prefix match over that block's region must neither serve
//! the in-flight block nor any stale data.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use crate::block::{BlockId, BlockManager};

/// Chained prefix digest: `hash(node) = sha256(hash(parent) || tokens)`.
/// Collision-resistant so it can key the persisted SSD index across
/// restarts (a 64-bit hash would make cross-request KV poisoning a
/// birthday attack; see SPEC §6.4 fingerprint discussion).
pub type ChainHash = [u8; 32];

/// Root of the tree (always present, never evicted).
pub const ROOT: usize = 0;

/// One tree node. `tokens` is the edge label from the parent: exactly
/// `block_size` ids for a full node, fewer for a partial leaf (a donated
/// settled tail — see the module docs; partial nodes never have children
/// and never reach the SSD tier).
struct Node {
    /// Generation guard: arena slots are recycled, so queued references
    /// (`(index, generation)`) can detect staleness.
    generation: u64,
    parent: usize,
    tokens: Vec<u32>,
    /// This node keys fewer than `block_size` tokens (leaf-only).
    partial: bool,
    children: HashMap<Vec<u32>, usize>,
    /// Partial leaves under this node (variable-length labels cannot live
    /// in the exact-chunk `children` map).
    partials: Vec<usize>,
    /// Pool block holding this node's KV rows, when resident. The cache
    /// owns one manager reference for it.
    block: Option<BlockId>,
    /// The chained prefix digest identifying this node in the SSD index.
    chain_hash: ChainHash,
    /// A confirmed copy exists on the SSD tier (write acked).
    ssd: bool,
    /// A flush is enqueued or in flight; suppresses re-enqueue.
    flush_pending: bool,
    /// LRU clock value of the last touch.
    last_hit: u64,
    /// Children with `block.is_some()` — eviction is leaf-first.
    resident_children: u32,
}

/// The tree. See the module docs for the ownership contract.
pub struct RadixCache {
    block_size: usize,
    nodes: Vec<Node>,
    free_slots: Vec<usize>,
    clock: u64,
    generations: u64,
    /// Resident nodes (gauge; also bounds eviction scans).
    resident: usize,
}

impl RadixCache {
    /// `seed` is folded into every chain hash — derive it from the model
    /// fingerprint so slabs from a different model/dtype can never collide
    /// in the SSD index even if a cache directory is shared.
    pub fn new(block_size: usize, seed: ChainHash) -> Self {
        Self {
            block_size,
            nodes: vec![Node {
                generation: 0,
                parent: ROOT,
                tokens: Vec::new(),
                partial: false,
                children: HashMap::new(),
                partials: Vec::new(),
                block: None,
                chain_hash: seed,
                ssd: false,
                flush_pending: false,
                last_hit: 0,
                resident_children: 0,
            }],
            free_slots: Vec::new(),
            clock: 0,
            generations: 0,
            resident: 0,
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Resident nodes (blocks the cache holds in the pool).
    pub fn resident_blocks(&self) -> usize {
        self.resident
    }

    /// The child of `node` labeled with exactly `chunk`, if present.
    pub fn child(&self, node: usize, chunk: &[u32]) -> Option<usize> {
        self.nodes[node].children.get(chunk).copied()
    }

    /// The digest a child of `node` labeled `chunk` has (or would have).
    pub fn chain_hash_of(&self, node: usize, chunk: &[u32]) -> ChainHash {
        let mut hasher = Sha256::new();
        hasher.update(self.nodes[node].chain_hash);
        for token in chunk {
            hasher.update(token.to_le_bytes());
        }
        hasher.finalize().into()
    }

    pub fn node_generation(&self, node: usize) -> u64 {
        self.nodes[node].generation
    }

    pub fn node_hash(&self, node: usize) -> ChainHash {
        self.nodes[node].chain_hash
    }

    pub fn node_block(&self, node: usize) -> Option<BlockId> {
        self.nodes[node].block
    }

    pub fn node_tokens(&self, node: usize) -> &[u32] {
        &self.nodes[node].tokens
    }

    /// Keyed token rows of this node (`block_size` unless partial).
    pub fn node_rows(&self, node: usize) -> usize {
        self.nodes[node].tokens.len()
    }

    pub fn node_is_partial(&self, node: usize) -> bool {
        self.nodes[node].partial
    }

    /// Full children of `node`, for the containment probe (their labels
    /// are exact `block_size` chunks; the probe compares label prefixes).
    pub fn full_children(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.nodes[node].children.values().copied()
    }

    /// Partial leaves under `node`.
    pub fn partial_children(&self, node: usize) -> &[usize] {
        &self.nodes[node].partials
    }

    /// Creates a partial leaf under `parent` keying `tokens`
    /// (`1..block_size` ids). Pool-only: partial leaves are never flushed
    /// to the SSD tier and never have children.
    pub fn insert_partial_child(&mut self, parent: usize, tokens: &[u32]) -> usize {
        debug_assert!(!tokens.is_empty() && tokens.len() < self.block_size);
        debug_assert!(!self.nodes[parent].partial, "partials are leaves");
        let chain_hash = self.chain_hash_of(parent, tokens);
        self.generations += 1;
        self.clock += 1;
        let node = Node {
            generation: self.generations,
            parent,
            tokens: tokens.to_vec(),
            partial: true,
            children: HashMap::new(),
            partials: Vec::new(),
            block: None,
            chain_hash,
            ssd: false,
            flush_pending: false,
            last_hit: self.clock,
            resident_children: 0,
        };
        let index = match self.free_slots.pop() {
            Some(slot) => {
                self.nodes[slot] = node;
                slot
            }
            None => {
                self.nodes.push(node);
                self.nodes.len() - 1
            }
        };
        self.nodes[parent].partials.push(index);
        index
    }

    /// Re-keys a partial leaf with a longer settled tail, moving its
    /// residency to `block` (the caller transfers one owner reference in
    /// and releases the returned old block itself).
    pub fn upgrade_partial(
        &mut self,
        node: usize,
        tokens: &[u32],
        block: BlockId,
    ) -> Option<BlockId> {
        debug_assert!(self.nodes[node].partial);
        debug_assert!(!tokens.is_empty() && tokens.len() < self.block_size);
        let parent = self.nodes[node].parent;
        let chain_hash = self.chain_hash_of(parent, tokens);
        let old = self.nodes[node].block.replace(block);
        if old.is_none() {
            self.nodes[parent].resident_children += 1;
            self.resident += 1;
        }
        self.nodes[node].tokens = tokens.to_vec();
        self.nodes[node].chain_hash = chain_hash;
        self.touch(node);
        old
    }

    pub fn node_on_ssd(&self, node: usize) -> bool {
        self.nodes[node].ssd
    }

    /// Marks a confirmed SSD copy (flush acked / index hit verified).
    pub fn set_on_ssd(&mut self, node: usize) {
        self.nodes[node].ssd = true;
        self.nodes[node].flush_pending = false;
    }

    pub fn flush_pending(&self, node: usize) -> bool {
        self.nodes[node].flush_pending
    }

    pub fn set_flush_pending(&mut self, node: usize) {
        self.nodes[node].flush_pending = true;
    }

    /// A flush failed; allow a later donation to re-enqueue it.
    pub fn clear_flush_pending(&mut self, node: usize) {
        self.nodes[node].flush_pending = false;
    }

    /// A queued reference is stale once the slot was pruned and recycled.
    pub fn is_live(&self, node: usize, generation: u64) -> bool {
        node < self.nodes.len()
            && self.nodes[node].generation == generation
            && (node == ROOT || !self.nodes[node].tokens.is_empty())
    }

    /// Bumps the node's LRU clock.
    pub fn touch(&mut self, node: usize) {
        self.clock += 1;
        self.nodes[node].last_hit = self.clock;
    }

    /// Creates a (non-resident) child of `parent` labeled `chunk`.
    pub fn insert_child(&mut self, parent: usize, chunk: &[u32]) -> usize {
        debug_assert_eq!(chunk.len(), self.block_size, "chunk must be one block");
        debug_assert!(self.child(parent, chunk).is_none(), "child exists");
        let chain_hash = self.chain_hash_of(parent, chunk);
        self.generations += 1;
        self.clock += 1;
        let node = Node {
            generation: self.generations,
            parent,
            tokens: chunk.to_vec(),
            partial: false,
            children: HashMap::new(),
            partials: Vec::new(),
            block: None,
            chain_hash,
            ssd: false,
            flush_pending: false,
            last_hit: self.clock,
            resident_children: 0,
        };
        let index = match self.free_slots.pop() {
            Some(slot) => {
                self.nodes[slot] = node;
                slot
            }
            None => {
                self.nodes.push(node);
                self.nodes.len() - 1
            }
        };
        self.nodes[parent].children.insert(chunk.to_vec(), index);
        index
    }

    /// Makes `node` resident, adopting one owner reference on `block` that
    /// the caller already holds (donation or SSD load).
    pub fn set_resident(&mut self, node: usize, block: BlockId) {
        debug_assert!(self.nodes[node].block.is_none(), "already resident");
        self.nodes[node].block = Some(block);
        let parent = self.nodes[node].parent;
        self.nodes[parent].resident_children += 1;
        self.resident += 1;
    }

    /// The best eviction victim: LRU among resident nodes with no resident
    /// children whose block the cache solely owns (refcount 1 — evicting a
    /// block a running request shares would free nothing).
    pub fn evict_candidate(&self, mgr: &BlockManager) -> Option<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|&(index, node)| {
                index != ROOT
                    && !node.tokens.is_empty()
                    && node.resident_children == 0
                    && node
                        .block
                        .is_some_and(|block| mgr.refcount(block) == Some(1))
            })
            .min_by_key(|(_, node)| node.last_hit)
            .map(|(index, _)| index)
    }

    /// Drops `node`'s residency, releasing the cache's owner reference.
    /// A node with no SSD copy becomes unreachable garbage and is pruned
    /// together with its (necessarily non-resident) subtree. Returns the
    /// freed block.
    pub fn evict(&mut self, mgr: &mut BlockManager, node: usize) -> Option<BlockId> {
        let block = self.nodes[node].block.take()?;
        let released = mgr.release(block);
        debug_assert!(released.is_ok(), "cache released a foreign block");
        let parent = self.nodes[node].parent;
        self.nodes[parent].resident_children -= 1;
        self.resident -= 1;
        if !self.nodes[node].ssd {
            self.prune(node);
        }
        Some(block)
    }

    /// Removes `node` and its whole subtree from the tree. Any resident
    /// descendants' references are released to `mgr` — used both for
    /// SSD-less eviction and for dropping a subtree whose backing slab
    /// failed verification. Pruning through `Self::evict` has already
    /// detached the block; direct calls release here.
    pub fn prune_subtree(&mut self, mgr: &mut BlockManager, node: usize) {
        if node == ROOT {
            return;
        }
        // Detach from the parent first so the DFS below owns the subtree;
        // the parent's resident count only ever tracked `node` itself.
        let parent = self.nodes[node].parent;
        if self.nodes[node].block.is_some() {
            self.nodes[parent].resident_children -= 1;
        }
        self.detach(parent, node);
        let mut stack = vec![node];
        while let Some(index) = stack.pop() {
            if let Some(block) = self.nodes[index].block.take() {
                let released = mgr.release(block);
                debug_assert!(released.is_ok(), "cache released a foreign block");
                self.resident -= 1;
            }
            stack.extend(std::mem::take(&mut self.nodes[index].children).into_values());
            stack.extend(std::mem::take(&mut self.nodes[index].partials));
            self.recycle(index);
        }
    }

    fn prune(&mut self, node: usize) {
        let parent = self.nodes[node].parent;
        self.detach(parent, node);
        // Descendants of an evicted node are non-resident (leaf-first
        // eviction) — anything left is SSD-only and now unreachable, or
        // pending-flush and now lost; recycle it all.
        let mut stack: Vec<usize> = std::mem::take(&mut self.nodes[node].children)
            .into_values()
            .chain(std::mem::take(&mut self.nodes[node].partials))
            .collect();
        self.recycle(node);
        while let Some(index) = stack.pop() {
            debug_assert!(self.nodes[index].block.is_none(), "resident under evictee");
            stack.extend(std::mem::take(&mut self.nodes[index].children).into_values());
            stack.extend(std::mem::take(&mut self.nodes[index].partials));
            self.recycle(index);
        }
    }

    /// Removes `node` from its parent's child index (map or partial list).
    fn detach(&mut self, parent: usize, node: usize) {
        if self.nodes[node].partial {
            self.nodes[parent].partials.retain(|&child| child != node);
        } else {
            let label = std::mem::take(&mut self.nodes[node].tokens);
            self.nodes[parent].children.remove(&label);
        }
    }

    fn recycle(&mut self, node: usize) {
        self.generations += 1;
        self.nodes[node] = Node {
            generation: self.generations,
            parent: ROOT,
            tokens: Vec::new(),
            partial: false,
            children: HashMap::new(),
            partials: Vec::new(),
            block: None,
            chain_hash: [0; 32],
            ssd: false,
            flush_pending: false,
            last_hit: 0,
            resident_children: 0,
        };
        self.free_slots.push(node);
    }

    /// Drops every node (engine fault recovery: the block manager is being
    /// rebuilt, so no references are released here). SSD copies survive in
    /// the store's index and are re-discovered hash-first on later walks.
    pub fn reset(&mut self) {
        let seed = self.nodes[ROOT].chain_hash;
        let block_size = self.block_size;
        let clock = self.clock;
        let generations = self.generations + 1;
        *self = Self::new(block_size, seed);
        self.clock = clock;
        self.generations = generations;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockManager;

    fn chunk(base: u32) -> Vec<u32> {
        (base..base + 4).collect()
    }

    #[test]
    fn insert_descend_and_hash_chaining() {
        let mut tree = RadixCache::new(4, [7; 32]);
        let a = tree.insert_child(ROOT, &chunk(0));
        let b = tree.insert_child(a, &chunk(4));
        assert_eq!(tree.child(ROOT, &chunk(0)), Some(a));
        assert_eq!(tree.child(a, &chunk(4)), Some(b));
        assert_eq!(tree.child(a, &chunk(8)), None);
        // Chain hashes are deterministic and parent-dependent.
        assert_eq!(tree.chain_hash_of(ROOT, &chunk(0)), tree.node_hash(a));
        assert_eq!(tree.chain_hash_of(a, &chunk(4)), tree.node_hash(b));
        assert_ne!(tree.node_hash(a), tree.node_hash(b));
        let other = RadixCache::new(4, [8; 32]);
        assert_ne!(
            other.chain_hash_of(ROOT, &chunk(0)),
            tree.node_hash(a),
            "seed must isolate models"
        );
    }

    #[test]
    fn eviction_is_lru_leaf_first_and_skips_shared() {
        let mut mgr = BlockManager::new(4, 4).unwrap();
        let mut tree = RadixCache::new(4, [0; 32]);
        let a = tree.insert_child(ROOT, &chunk(0));
        let b = tree.insert_child(a, &chunk(4));
        let c = tree.insert_child(ROOT, &chunk(8));
        for node in [a, b, c] {
            let block = mgr.allocate().unwrap();
            tree.set_resident(node, block);
        }
        assert_eq!(tree.resident_blocks(), 3);
        // `a` has a resident child, so candidates are `b` and `c`; `b` is
        // older (insert order sets last_hit).
        assert_eq!(tree.evict_candidate(&mgr), Some(b));
        tree.touch(b);
        assert_eq!(tree.evict_candidate(&mgr), Some(c));
        // A share on c's block (a running request) exempts it.
        mgr.retain(tree.node_block(c).unwrap()).unwrap();
        assert_eq!(tree.evict_candidate(&mgr), Some(b));
        mgr.release(tree.node_block(c).unwrap()).unwrap();

        // Evicting `b` (no SSD copy) prunes it; `a` becomes a leaf.
        let freed = tree.evict(&mut mgr, b).unwrap();
        assert_eq!(mgr.refcount(freed), Some(0));
        assert_eq!(tree.child(a, &chunk(4)), None);
        assert_eq!(tree.resident_blocks(), 2);
        assert!([Some(a), Some(c)].contains(&tree.evict_candidate(&mgr)));
    }

    #[test]
    fn ssd_backed_eviction_keeps_the_node() {
        let mut mgr = BlockManager::new(2, 4).unwrap();
        let mut tree = RadixCache::new(4, [0; 32]);
        let a = tree.insert_child(ROOT, &chunk(0));
        let block = mgr.allocate().unwrap();
        tree.set_resident(a, block);
        tree.set_on_ssd(a);
        tree.evict(&mut mgr, a);
        assert_eq!(mgr.num_free(), 2);
        // Still discoverable for an SSD reload.
        assert_eq!(tree.child(ROOT, &chunk(0)), Some(a));
        assert!(tree.node_on_ssd(a));
        assert_eq!(tree.node_block(a), None);
    }

    #[test]
    fn generations_guard_recycled_slots() {
        let mut mgr = BlockManager::new(2, 4).unwrap();
        let mut tree = RadixCache::new(4, [0; 32]);
        let a = tree.insert_child(ROOT, &chunk(0));
        let generation = tree.node_generation(a);
        let block = mgr.allocate().unwrap();
        tree.set_resident(a, block);
        assert!(tree.is_live(a, generation));
        tree.evict(&mut mgr, a); // no SSD copy -> pruned + recycled
        assert!(!tree.is_live(a, generation));
        let b = tree.insert_child(ROOT, &chunk(4));
        assert_eq!(a, b, "slot is recycled");
        assert!(!tree.is_live(a, generation), "old generation stays dead");
        assert!(tree.is_live(b, tree.node_generation(b)));
    }

    #[test]
    fn prune_subtree_releases_resident_descendants() {
        let mut mgr = BlockManager::new(4, 4).unwrap();
        let mut tree = RadixCache::new(4, [0; 32]);
        let a = tree.insert_child(ROOT, &chunk(0));
        let b = tree.insert_child(a, &chunk(4));
        for node in [a, b] {
            let block = mgr.allocate().unwrap();
            tree.set_resident(node, block);
        }
        tree.prune_subtree(&mut mgr, a);
        assert_eq!(mgr.num_free(), 4);
        assert_eq!(tree.resident_blocks(), 0);
        assert_eq!(tree.child(ROOT, &chunk(0)), None);
    }

    #[test]
    fn reset_drops_everything_but_keeps_the_seed() {
        let mut tree = RadixCache::new(4, [3; 32]);
        let a = tree.insert_child(ROOT, &chunk(0));
        let hash = tree.node_hash(a);
        tree.reset();
        assert_eq!(tree.child(ROOT, &chunk(0)), None);
        assert_eq!(tree.resident_blocks(), 0);
        // Hash-first rediscovery must produce the same digests.
        assert_eq!(tree.chain_hash_of(ROOT, &chunk(0)), hash);
    }
}
