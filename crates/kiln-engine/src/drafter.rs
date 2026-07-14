//! The `Drafter` abstraction (SPEC §6.5): one interface over the two
//! speculative-decoding modes — `DraftModel` (a second, smaller model
//! loaded in the same worker process with its own weights and its own KV
//! pool, sharing only the Metal device/stream) and, reserved for a later
//! phase, `SelfDraft`/MTP-style heads. Both propose tokens; the engine
//! verifies them with a single target-model forward inside the normal
//! batch step and commits the longest agreeing prefix.
//!
//! The engine drives this interface from inside `run_iteration` (Phase 8
//! part 2): eligible greedy requests get a proposal per step, the target
//! scores the proposed tokens in one gamma+1-row verify forward, the
//! longest agreeing prefix plus the target's own next token commits, and
//! rejected positions roll back by block release (O(1) — no data moves).
//!
//! Threading: a drafter lives on the engine thread with everything else
//! that touches MLX (its methods take the engine's `Stream`, and
//! implementations holding arrays are `!Send` like the engine itself).
//! Memory: a drafter's weights and KV pool are part of the worker's
//! budget envelope (SPEC §2.3) — [`Drafter::memory`] feeds the heartbeat
//! `MemoryReport`, so the gateway's machine-level accounting sees the
//! draft as part of this worker's footprint, never as invisible overhead.

use kiln_mlx::{MlxError, Stream};
use thiserror::Error;

/// SPEC §6.5: tokens proposed per speculation round (config `gamma`).
pub const DEFAULT_GAMMA: usize = 4;

/// SPEC §6.5: speculation auto-disables per-request when the decode batch
/// is wider than this (config `spec_max_batch`) — batching already
/// saturates the GPU there.
pub const DEFAULT_SPEC_MAX_BATCH: usize = 4;

/// Default `EngineConfig::spec_min_acceptance` (Phase 8 auto-disable
/// heuristics): the verified acceptance rate (accepted / proposed) below
/// which a warmed-up request stands down from speculation. 1-in-8 is the
/// break-even acceptance for a draft costing ~1/8th of the target per
/// forward — speculation pays iff the acceptance fraction exceeds the
/// draft/target cost ratio: a round spends `gamma·c_draft + c_verify` to
/// commit `1 + accepted` tokens where a plain step spends `c_target` for
/// one. Permissive enough that a properly-sized deployment pair (SPEC
/// §12 Phase 8 names 0.6B drafting for 14B, cost ratio ≈ 0.05) never
/// trips it on honest content, while a mis-paired draft proposing noise
/// stops burning draft forwards after the warmup window.
pub const DEFAULT_SPEC_MIN_ACCEPTANCE: f64 = 0.125;

/// Proposed tokens a request must accumulate before
/// `spec_min_acceptance` may judge it — acceptance over fewer tokens is
/// mostly noise (a single rejected round would condemn a healthy pair).
pub const SPEC_ACCEPTANCE_WARMUP_PROPOSED: u32 = 16;

/// The SPEC §6.5 batch-width stand-down, as a per-round gamma bound:
/// full `gamma` single-stream, ramping down linearly as the admitted
/// batch approaches `spec_max_batch` (batching alone is saturating the
/// GPU there, so speculation's extra verify rows pay less and less), at
/// least 1 while `width <= spec_max_batch` (SPEC §6.5 disables strictly
/// ABOVE the threshold), and 0 — speculation off — beyond it.
///
/// This is a heuristic bound, applied UNDER the hard ADR 0005 cutoffs
/// (per-model envelope gamma, deterministic width, verify key length):
/// it only ever shrinks the round, never widens the envelope.
pub fn spec_gamma_at_width(gamma: usize, spec_max_batch: usize, width: usize) -> usize {
    let width = width.max(1);
    if gamma == 0 || width > spec_max_batch {
        return 0;
    }
    // Ceil division keeps the ramp ≥ 1 across the whole admitted range
    // and exactly `gamma` at width 1.
    (gamma * (spec_max_batch + 1 - width)).div_ceil(spec_max_batch)
}

/// Draft-side failure. Never fatal to the request: the engine treats any
/// drafter error as "no proposal this round" at worst, or surfaces it as
/// an in-band request error — a drafter must not be able to take down a
/// generation that plain decoding would have served.
#[derive(Debug, Error)]
pub enum DraftError {
    #[error(transparent)]
    Mlx(#[from] MlxError),
    /// The drafter holds no state for the sequence (an engine/drafter
    /// bookkeeping desync — an internal bug, reported in-band).
    #[error("drafter has no state for sequence {0}")]
    UnknownSeq(u64),
    /// The draft-side KV pool cannot cover the sequence.
    #[error("draft KV capacity exhausted: {0}")]
    Capacity(String),
}

/// Memory footprint of a drafter, for the worker heartbeat (SPEC §2.3):
/// summed into the proto `MemoryReport`'s `weights_bytes` /
/// `kv_pool_allocated_bytes` / `kv_pool_used_bytes` totals.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DrafterMemory {
    /// Draft weight bytes (0 for self-drafting modes, which reuse the
    /// target's weights).
    pub weights_bytes: u64,
    /// Bytes backing the draft's own KV pools (0 until first use — the
    /// pools materialize lazily, exactly like the target's).
    pub kv_allocated_bytes: u64,
    /// Bytes of draft pool blocks currently owned by live sequences.
    pub kv_used_bytes: u64,
}

/// A token proposer the engine can drive per speculating sequence
/// (SPEC §6.5). Sequences are keyed by the engine's arrival number, which
/// is unique per admission and stable across preemption.
///
/// Contract (drives the Phase 8 part 2 verify loop; only the state
/// lifecycle is exercised today):
///
/// 1. [`begin`](Self::begin) — called at admission with the full prompt.
///    Beginning an already-known sequence discards its draft state and
///    starts over; that is also the preemption-resume path (the target
///    re-prefills, so the drafter restarts from the committed context).
/// 2. [`propose`](Self::propose) — called between steps with the tokens
///    committed target-side since the previous call (accepted proposals
///    plus the bonus token). A drafter that speculatively advanced its
///    own KV past the committed context must reconcile — truncate to the
///    committed length — before proposing again; `committed` is exactly
///    the information needed to do so. Returns at most `gamma` proposed
///    token ids; fewer, including none, is a legal answer meaning "no
///    speculation this round" (the step then decodes normally).
/// 3. [`release`](Self::release) — the sequence finished or was
///    cancelled; drop its draft state. Idempotent.
pub trait Drafter {
    /// Current footprint for the worker's memory report.
    fn memory(&self) -> DrafterMemory;

    /// Starts (or restarts) draft-side tracking of a sequence from its
    /// prompt.
    fn begin(&mut self, seq: u64, prompt: &[u32], s: &Stream) -> Result<(), DraftError>;

    /// Proposes up to `gamma` tokens continuing `seq`'s committed
    /// context; see the trait docs for the `committed` reconciliation
    /// contract.
    fn propose(
        &mut self,
        seq: u64,
        committed: &[u32],
        gamma: usize,
        s: &Stream,
    ) -> Result<Vec<u32>, DraftError>;

    /// Drops all draft-side state for `seq`. Unknown sequences are a
    /// no-op (finish paths may race a never-begun sequence).
    fn release(&mut self, seq: u64);
}

#[cfg(test)]
mod tests {
    use super::spec_gamma_at_width;

    #[test]
    fn width_ramp_defaults() {
        // gamma 4, spec_max_batch 4 (the SPEC §6.5 defaults).
        assert_eq!(spec_gamma_at_width(4, 4, 1), 4, "full gamma single-stream");
        assert_eq!(spec_gamma_at_width(4, 4, 2), 3);
        assert_eq!(spec_gamma_at_width(4, 4, 3), 2);
        assert_eq!(spec_gamma_at_width(4, 4, 4), 1, "stood down, still on");
        assert_eq!(spec_gamma_at_width(4, 4, 5), 0, "off strictly above");
        assert_eq!(spec_gamma_at_width(4, 4, 100), 0);
    }

    #[test]
    fn width_ramp_stays_on_through_spec_max_batch() {
        // SPEC §6.5 disables strictly ABOVE spec_max_batch: for any
        // nonzero gamma the ramp keeps at least 1 through the admitted
        // range, and is monotone in width.
        for gamma in 1..=8 {
            for smb in 1..=8 {
                let mut prev = usize::MAX;
                for width in 1..=smb {
                    let g = spec_gamma_at_width(gamma, smb, width);
                    assert!(
                        (1..=gamma).contains(&g),
                        "gamma {gamma} smb {smb} width {width} -> {g}"
                    );
                    assert!(g <= prev, "ramp must not grow with width");
                    prev = g;
                }
                assert_eq!(spec_gamma_at_width(gamma, smb, smb + 1), 0);
            }
        }
    }

    #[test]
    fn width_ramp_edges() {
        assert_eq!(spec_gamma_at_width(0, 4, 1), 0, "gamma 0 is off");
        assert_eq!(spec_gamma_at_width(4, 0, 1), 0, "spec_max_batch 0 is off");
        // Width 0 (no admitted requests — not reachable from the per-seq
        // path) behaves as single-stream rather than dividing by zero.
        assert_eq!(spec_gamma_at_width(4, 4, 0), 4);
        // The clamped-envelope shape (qwen2.5-0.5b, ADR 0005 gamma 3).
        assert_eq!(spec_gamma_at_width(3, 4, 1), 3);
        assert_eq!(spec_gamma_at_width(3, 4, 4), 1);
        assert_eq!(spec_gamma_at_width(3, 4, 5), 0);
    }
}
