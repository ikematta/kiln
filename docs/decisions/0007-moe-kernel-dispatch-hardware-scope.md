# ADR 0007: MoE kernel-dispatch verification is scoped to M4-class hardware; M3 Ultra deployment is a tracked, unverified gap

- Status: recorded (directed by the 2026-07-25 MoE session-1 task; PM
  ratification pending like any ADR)
- Date: 2026-07-25 (MoE arc, session 1)
- Relationship: the fifth instance of the structural pattern ADRs
  0002/0003/0004/0005 record — the pinned MLX selects between Metal
  kernels by operand shape and device class, and bit-level behavior is
  only known where it has been measured. 0004 established that golden
  bit-exactness is a same-device bar; this ADR applies the same honesty
  to a hardware class Kiln's MoE code has NEVER run on.

## Context

Session 1 of the MoE arc implemented expert routing/gating and the MoE
forward pass (`kiln-models` `moe.rs` + `olmoe.rs`) and proved it against
the golden harness using the pinned proxy model
`mlx-community/OLMoE-1B-7B-0125-Instruct-4bit` (6.9B total / 1.3B active
parameters, ~3.9 GB at 4-bit) — the smallest well-supported mlx-community
MoE checkpoint, chosen because it fits the 16 GB M4 dev machine. The
eventual deployment target for MoE support is different in every axis
that has historically mattered for kernel dispatch:

- **Hardware class:** M3 Ultra, 128 GB unified memory. The dev machine is
  a 16 GB M4. MLX's dispatch tables are explicitly GPU-class-dependent —
  ADR 0002 measured qmv/qmm thresholds of 6–32 varying by GPU class (the
  table's maximum occurs on the `d`-suffix class), and ADR 0005's 2-pass
  SDPA boundary differs on `d`/`s` classes. M3 Ultra is exactly such a
  different dispatch table, plus a paired-die GPU topology no pinned
  test has ever exercised.
- **Model scale:** large MoE architectures (tens of billions of total
  parameters, larger expert counts/widths). Every gather_qmm shape —
  expert count, per-expert width, group sizes crossed with row counts —
  lands in regions of the dispatch table the proxy never touches.
- **Op family:** the expert path (`gather_qmm`/`gather_mm`, the SwitchGLU
  sort threshold) is a kernel family none of ADRs 0002–0005 measured at
  all; session 1's measurements of it exist ONLY on the M4 dev machine.

The mitigations session 1 built are calibration-based, not table-based —
`calibrate_deterministic_width` probes the whole MoE block (router,
top-k, expert dispatch, combine) on the running device, so the B'
deterministic width adapts wherever the code runs. That protects the
batched-greedy invariant on any device it runs on, but it is a per-device
measurement, not a portability proof, and nothing has measured it on the
target class.

## Decision

1. **What is verified:** MoE routing/gating/expert-dispatch correctness
   — strict single-stream golden parity and the width-16 batched gate —
   on the fixture-generating M4-class dev machine, via the pinned OLMoE
   proxy. This carries exactly the ADR 0004 same-device scope and no
   more.
2. **What is explicitly NOT verified: kernel-dispatch-class safety of
   the MoE path on M3 Ultra / 128 GB hardware at deployment scale.** No
   claim is made — and none should be inferred from green CI or green
   dev-machine gates — that the expert-dispatch kernel classes, the
   SwitchGLU sort-threshold behavior, the calibrated deterministic
   width, or greedy bit-stability transfer to that hardware or to
   large-MoE shapes. The trust level ADR 0005 established for the
   speculative envelope (source-verified dispatch predicates + measured
   probes on the running class) does NOT extend to MoE on M3 Ultra
   until an equivalent verification pass runs on real target hardware.
3. **The gap is a tracked backlog item, not a silent assumption.**
   Closing it requires, on an actual M3 Ultra (or the deployment-class
   machine of record), with a large MoE checkpoint of the deployment
   family: the full golden-style parity run (single-stream strict,
   width-16 batched), the calibration probe's measured width recorded in
   PROGRESS.md, and a re-read of the gather_qmm dispatch path at the
   then-current pin. Since no such machine is available to this project
   today, this is flagged for community verification once Kiln is
   open-sourced (or for a PM-provisioned hardware pass) — whichever
   comes first. Until then, `docs/SPEC.md` §7.2's MoE backlog note is
   the user-facing statement of scope.
4. **Speculation on MoE targets stays off** (`speculative_gamma_bound`
   returns `None` for MoE architectures) independent of this gap: the
   expert gather path has no kernel-class certificate, so ADR 0005
   decision (2)'s new-architecture precondition (documented geometry
   review + green spec_decode gate on the generating device) has not
   been met. Meeting it is a separate work item from the hardware gap,
   and both must close before speculative decoding runs on a deployed
   MoE model.

## Consequences

- SPEC §7.2 gains a MoE backlog note referencing this ADR (doc
  amendment landed with this ADR).
- Any future session that adds a second MoE architecture, a quantization
  variant, or `worker = "auto"` routing for MoE inherits this scope
  statement: new work extends the M4-verified surface, never the
  M3-Ultra-verified surface, until decision (3)'s pass exists.
- CI scope is narrower still: hosted macos-14 runners (7 GB) can at most
  exercise the 3.9 GB proxy — and per ADR 0004 only as an advisory
  cross-device signal — and can NEVER exercise deployment scale;
  memory alone excludes large MoE checkpoints there. CI green on MoE
  means "proxy-scale, foreign-device, advisory" and nothing more.
- Revisit triggers: (a) target-class hardware becomes available —
  run decision (3)'s pass and record it; (b) any mlx-c/core-MLX bump
  (ADR 0001's standing quarterly process) — the M4-side measurements
  this ADR scopes go stale with the pin like every other ADR's.
