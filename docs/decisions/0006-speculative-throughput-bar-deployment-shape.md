# ADR 0006: The Phase 8 speculative-throughput bar is a deployment-shape claim

- Status: accepted (PM-directed, 2026-07-14)
- Date: 2026-07-14 (Phase 8 part 3 closeout)
- Relationship: continues ADR 0005 (which fixed WHERE speculation may run
  correctly at this pin) by settling what its throughput promise means
  and where it can be measured. Like ADR 0003, it amends a *performance
  acceptance target*; the correctness bars of ADRs 0002/0004/0005 are
  untouched and remain blocking.

## Context

SPEC §12 Phase 8's acceptance reads: "with Qwen3-0.6B drafting for a 14B
target (or tiny-pair in CI), single-stream decode ≥1.6× baseline at
acceptance >60%". The parenthetical implies the tiny-pair CI fleet can
stand in for the deployment pair. It cannot: the bar as originally
written implicitly assumed a deployment shape — a small draft, a large
target, and a meaningful cost asymmetry between them — and the pinned
test fleet, chosen for CI cost and speed (every model sub-1B; ADR 0001,
`scripts/fetch-test-model.sh`), cannot produce that shape.

Measured (PROGRESS 2026-07-14; `spec_throughput.rs` release perf lane,
dev machine, deterministic width 9, 256-token decode):

| pair (target / draft)                          | γ | OFF tok/s | ON tok/s | ratio | acceptance |
|------------------------------------------------|---|-----------|----------|-------|------------|
| qwen3-0.6b-8bit / qwen3-0.6b-4bit (unclamped)   | 4 | 104.9     | 74.1     | 0.71× | 83.8%      |
| qwen3-0.6b-8bit / qwen3-0.6b-4bit (clamp shape) | 3 | 104.9     | 71.0     | 0.68× | 89.9%      |
| qwen2.5-0.5b-4bit self-pair (ADR 0005 clamp)    | 3 | 224.6     | 140.5    | 0.63× | 100.0%     |

The loss is structural at these pins, not tunable:

- The draft:target weight-byte ratio is ~0.65 (633,442,994 /
  968,893,578 bytes). A speculation round spends γ draft forwards plus
  one (γ+1)-row verify to replace the plain steps it covers; at cost
  ratio ~0.65 the round costs more than what it replaces even at 100%
  acceptance — the ceiling sits below ~1.4× before overheads, and the
  measured reality is 0.63–0.71×. A speculating request additionally
  forfeits async_eval pipelining.
- Acceptance is NOT the problem: every measured pair sits far above the
  >60% qualifier (83.8–100%) with exact greedy outputs — which is also
  why the Phase 8 part 3 acceptance auto-disable cannot catch this loss.
  It is cost-ratio-driven and invisible to any acceptance signal.
- The ADR 0005 γ clamp is a minor erosion, not the cause (0.71× → 0.68×
  on the one pair it clamps): tiny pairs are unprofitable with or
  without the envelope.

Speculation's value case rests entirely on the size-gap pair SPEC §12
names (0.6B drafting 8–14B; weight-cost ratio ≈ 0.05–0.1), which no
pinned CI model provides. The 2026-07-14 DECISION NEEDED offered:
(A) amend the bar by ADR to name the deployment shape, keeping CI lanes
correctness-only; or (B) pin a 7–8B model solely for a perf-lane
measurement. The PM ruling is option A; option B is declined for pinned-
fleet cost (a ~4.5 GB pin whose only consumer would be one perf number).

## Decision

1. **Correctness remains the permanent, blocking CI gate for
   CAPABILITY_SPECULATIVE — nothing about correctness changes.** Golden
   parity under speculation (greedy outputs identical, speculation on vs
   off), the ADR 0005 verify-kernel envelope, and the auto-disable
   heuristics (width ramp, acceptance stand-down) stay blocking in the
   model-gated CI lane exactly as today.
2. **The ≥1.6× throughput claim is decoupled from CI.** It is restated
   as a documented deployment-shape precondition, not an acceptance
   test: expected to hold for draft/target pairs with substantial size
   asymmetry (roughly draft ≤1B, target ≥7–8B), and UNVERIFIED in CI
   until such a pair enters the pinned fleet. The `spec_throughput.rs`
   perf lane keeps recording what the pinned fleet actually measures
   (currently: a loss); it records, it does not gate.
3. **An operator enabling speculation on a small/small pair should
   expect a throughput loss, not a gain** — 0.63–0.71× measured at
   sub-1B scale — regardless of the acceptance rate they observe. The
   operator documentation for `[model.speculative]` (config wiring
   pending) must state this precondition.
4. **Revisit triggers:** a size-asymmetric pair (target ≥7–8B) entering
   the pinned fleet re-arms the measurement — the perf lane then
   measures the ≥1.6× claim on it and the "unverified" status above is
   lifted. Independently, the cost figures in this ADR are re-benched at
   every mlx-c/core-MLX pin bump under ADR 0001's standing process, like
   ADR 0003's, and are stale the moment the pin moves.

## Consequences

- SPEC §12 Phase 8's acceptance line references this ADR (doc-only
  amendment; the task list is unchanged).
- **BACKLOG (tracked in SPEC §6.5 alongside the feature):**
  attachment-time weights-byte-ratio guard in the worker. The worker
  knows both byte counts at drafter attachment (it logs them today) and
  should warn — or reject, behind config — when the ratio predicts a
  loss. The acceptance heuristic structurally cannot do this (see
  Context); only the attach site can.
- CAPABILITY_SPECULATIVE advertisement (the remaining Phase 8 part) is
  gated on the correctness suite only; advertising the capability does
  not imply a speedup on the operator's chosen pair.
