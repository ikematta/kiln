//! Drafter trait shape (SPEC §6.5, Phase 8 part 1): a scripted stub
//! exercises the state-lifecycle contract — begin/propose/release keyed
//! by the engine's arrival numbers, the `committed` reconciliation
//! feed-through, short and empty proposals as legal answers, and use as
//! a boxed trait object (how the engine owns it). No model, no
//! generation; the real `DraftModel` mode is covered by
//! kiln-models/tests/draft.rs.

#![cfg(feature = "metal")]

use std::collections::HashMap;

use kiln_engine::{DEFAULT_GAMMA, DEFAULT_SPEC_MAX_BATCH, DraftError, Drafter, DrafterMemory};
use kiln_mlx::Stream;

fn stream() -> Stream {
    if kiln_mlx::memory::metal_is_available() {
        Stream::gpu()
    } else {
        Stream::cpu()
    }
}

/// Per-sequence stub state: the committed context as the drafter has
/// reconciled it, plus how far it speculated past that context.
struct SeqState {
    context: Vec<u32>,
    speculated: usize,
}

/// Proposes a fixed script of tokens, `gamma` at a time, and tracks the
/// trait's reconciliation contract: `committed` tokens extend the
/// context, and any speculation beyond them is rolled back.
struct ScriptedDrafter {
    script: Vec<u32>,
    seqs: HashMap<u64, SeqState>,
}

impl ScriptedDrafter {
    fn new(script: Vec<u32>) -> Self {
        Self {
            script,
            seqs: HashMap::new(),
        }
    }
}

impl Drafter for ScriptedDrafter {
    fn memory(&self) -> DrafterMemory {
        DrafterMemory::default() // stub holds no weights and no pool
    }

    fn begin(&mut self, seq: u64, prompt: &[u32], _s: &Stream) -> Result<(), DraftError> {
        // Re-begin discards prior state (the preemption-resume reset).
        self.seqs.insert(
            seq,
            SeqState {
                context: prompt.to_vec(),
                speculated: 0,
            },
        );
        Ok(())
    }

    fn propose(
        &mut self,
        seq: u64,
        committed: &[u32],
        gamma: usize,
        _s: &Stream,
    ) -> Result<Vec<u32>, DraftError> {
        let state = self.seqs.get_mut(&seq).ok_or(DraftError::UnknownSeq(seq))?;
        // Reconcile: committed tokens are now real context; whatever was
        // speculated beyond them is discarded (the O(1)-rollback slot).
        state.context.extend_from_slice(committed);
        state.speculated = 0;
        let at = state.context.len() % self.script.len();
        let proposal: Vec<u32> = self
            .script
            .iter()
            .cycle()
            .skip(at)
            .take(gamma.min(self.script.len().saturating_sub(at)))
            .copied()
            .collect();
        state.speculated = proposal.len();
        Ok(proposal)
    }

    fn release(&mut self, seq: u64) {
        self.seqs.remove(&seq);
    }
}

#[test]
fn drafter_contract_via_trait_object() {
    let s = stream();
    // Owned exactly as the engine owns it: behind `Box<dyn Drafter>`.
    let mut drafter: Box<dyn Drafter> = Box::new(ScriptedDrafter::new(vec![10, 11, 12, 13, 14]));

    assert_eq!(drafter.memory(), DrafterMemory::default());

    // Unknown sequences are rejected, and release is a safe no-op.
    let err = drafter
        .propose(7, &[], DEFAULT_GAMMA, &s)
        .expect_err("propose before begin");
    assert!(matches!(err, DraftError::UnknownSeq(7)), "{err}");
    drafter.release(7);

    // begin -> propose: at most gamma tokens.
    drafter.begin(7, &[1, 2, 3], &s).expect("begin");
    let proposal = drafter.propose(7, &[], DEFAULT_GAMMA, &s).expect("propose");
    assert_eq!(proposal, vec![13, 14], "script offset 3, capped at end");
    assert!(proposal.len() <= DEFAULT_GAMMA);

    // Feed-through: commit one accepted token + bonus, propose again.
    let proposal = drafter.propose(7, &[13, 99], 2, &s).expect("propose");
    assert_eq!(proposal, vec![10, 11], "context grew by 2, gamma capped 2");

    // The empty proposal is legal ("no speculation this round").
    drafter.begin(8, &[], &s).expect("begin");
    let proposal = drafter.propose(8, &[], 0, &s).expect("propose");
    assert!(proposal.is_empty());

    // Sequences are independent; release drops exactly one.
    drafter.release(7);
    let err = drafter
        .propose(7, &[], DEFAULT_GAMMA, &s)
        .expect_err("released sequence");
    assert!(matches!(err, DraftError::UnknownSeq(7)), "{err}");
    drafter
        .propose(8, &[], DEFAULT_GAMMA, &s)
        .expect("other sequence unaffected");

    // Re-begin resets (the preemption-resume path).
    drafter.begin(8, &[1], &s).expect("re-begin");
    let proposal = drafter.propose(8, &[], DEFAULT_GAMMA, &s).expect("propose");
    assert_eq!(
        proposal,
        vec![11, 12, 13, 14],
        "state restarted from prompt"
    );

    // The SPEC §6.5 defaults are what the config layer wires through.
    assert_eq!(DEFAULT_GAMMA, 4);
    assert_eq!(DEFAULT_SPEC_MAX_BATCH, 4);
}
