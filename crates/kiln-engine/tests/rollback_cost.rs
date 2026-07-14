//! Measures the SPEC §6.5 rollback primitive — `BlockTable::truncate` —
//! rather than trusting its O(1) label: a speculative verify round that
//! rejects draft tokens must cost block-release bookkeeping only, with no
//! dependence on how long the sequence already is. A design that rolled
//! back by copying or recomputing KV would scale with context length;
//! this suite pins the measured shape (flat across a 512x span of table
//! sizes) and prints the absolute numbers for the acceptance record. The
//! in-situ engine-loop counterpart (rollback nanos measured inside real
//! verify rounds, short vs long context) lives in
//! kiln-models/tests/spec_decode.rs.
//!
//! Block-manager only — no MLX, no Metal: rollback moves no data by
//! construction, which is the point.

use std::time::Instant;

use kiln_engine::{BlockManager, BlockTable};

const BLOCK_SIZE: usize = 32;

/// Mean nanos for one speculative append + reject-all rollback cycle
/// (the worst-case verify round: gamma+1 slots appended, gamma+1 - 1
/// committed... none committed here — every slot rolls back) against a
/// table already holding `tokens` tokens.
fn rollback_cycle_nanos(tokens: usize, stale: usize, cycles: usize) -> f64 {
    let mut mgr = BlockManager::new(
        tokens.div_ceil(BLOCK_SIZE) + stale / BLOCK_SIZE + 2,
        BLOCK_SIZE,
    )
    .expect("manager");
    let mut table = BlockTable::new();
    table.append_tokens(&mut mgr, tokens).expect("base append");
    // Warm-up: fix the table's Vec capacity so measured cycles do not
    // include a one-time reallocation.
    for _ in 0..64 {
        table.append_tokens(&mut mgr, stale).expect("warm append");
        table.truncate(&mut mgr, tokens).expect("warm rollback");
    }
    let started = Instant::now();
    for _ in 0..cycles {
        table.append_tokens(&mut mgr, stale).expect("append");
        table.truncate(&mut mgr, tokens).expect("rollback");
    }
    let nanos = started.elapsed().as_nanos() as f64 / cycles as f64;
    table.release(&mut mgr).expect("release");
    nanos
}

/// The O(1) claim, measured: the cost of rolling back a verify round's
/// rejected slots does not grow with the sequence the round extends.
/// A rollback that copied or re-derived per-token state would scale with
/// the 512x table-size span below; bookkeeping-only rollback stays flat.
/// The assertion allows 25x of noise headroom — far under the 512x a
/// linear design would show, far over scheduler jitter.
#[test]
fn rollback_cost_is_flat_across_sequence_lengths() {
    // gamma = 4 rejected wholesale, the worst standard round (SPEC §6.5
    // default gamma; the verify segment is gamma+1 slots, at least one of
    // which always commits — 5 stale slots bounds it).
    const STALE: usize = 5;
    const CYCLES: usize = 20_000;
    let sizes = [128usize, 1024, 8192, 65_536];
    let mut means = Vec::with_capacity(sizes.len());
    for &tokens in &sizes {
        let nanos = rollback_cycle_nanos(tokens, STALE, CYCLES);
        eprintln!(
            "rollback of {STALE} slots against a {tokens}-token table: {nanos:.0} ns/cycle \
             (append+truncate, mean of {CYCLES})"
        );
        means.push(nanos);
    }
    let (min, max) = (
        means.iter().copied().fold(f64::INFINITY, f64::min),
        means.iter().copied().fold(0.0, f64::max),
    );
    assert!(
        max <= min * 25.0,
        "rollback cost grew with sequence length: {means:?} ns across tables of {sizes:?} \
         tokens — the O(1) block-release claim does not hold"
    );
}

/// Companion scale check: rollback cost tracks the number of BLOCKS
/// released, not the table size. Rolling back 16 blocks costs more than
/// rolling back one — that is the expected per-block bookkeeping — but
/// both stay in the sub-microsecond regime and neither depends on the
/// 64k tokens already in the table. Informational bounds only (printed
/// numbers are the record); the hard assertion is the flatness above.
#[test]
fn rollback_cost_tracks_released_blocks() {
    const TOKENS: usize = 65_536;
    const CYCLES: usize = 20_000;
    let one_block = rollback_cycle_nanos(TOKENS, BLOCK_SIZE, CYCLES);
    let many_blocks = rollback_cycle_nanos(TOKENS, 16 * BLOCK_SIZE, CYCLES);
    eprintln!(
        "rollback of 1 block: {one_block:.0} ns/cycle; of 16 blocks: {many_blocks:.0} ns/cycle \
         (64k-token table)"
    );
    // A per-token (copying) design would separate these by the token
    // ratio and push the absolute numbers orders of magnitude up; block
    // bookkeeping keeps even the 16-block case well under 100us.
    assert!(
        many_blocks < 100_000.0,
        "16-block rollback took {many_blocks:.0} ns — not bookkeeping-scale"
    );
}
