# ADR 0004: Golden parity is a same-device bar; dtype-aware relaxed-bar delta clause

- Status: accepted (PM-directed, 2026-07-05)
- Date: 2026-07-05 (Phase 6)

## Context

SPEC §11.2 defines the keystone bar: the Rust worker reproduces the
committed mlx-lm golden fixtures token-for-token, exactly, with an escape
hatch ("relaxed bar") for legitimate op-ordering divergences: first
divergence beyond token 48 AND logprob delta < 1e-3 at the divergence,
invocable only via an ADR. The fixtures are generated once, by
`scripts/gen-golden.py`, on the Phase 6 dev machine.

ADR 0002 established that the pinned MLX selects different Metal kernels
by row count M, and that different kernel classes reduce in different
orders, producing bit-different, ulp-level-close results — on a single
device. The 2026-07-05 CI work (Option B, PROGRESS) ran the golden
harness on a foreign GPU (the macos-14 paravirtual runner) for the first
time and produced the first cross-device datapoint, CI run 28753659372:

- `gemma-2-2b`: all 12 rounds (6 fixtures, single-stream + width-16)
  token-exact on the foreign device.
- `gemma-3-1b/chat-basic`: token-exact through generated token 49, then
  the runner sampled 188797 where the fixture holds 195597 at token 50
  of 64. Tokens 1–49 matched, so the input state at the divergence
  position was identical on both devices; only accumulated activation
  rounding differed.

Measured on the generating device, replaying the exact fixture path
(mlx-lm `generate_step`, greedy, M=1 decode): `logprob[195597] =
-2.40625` (top-1), `logprob[188797] = -2.46875` (top-2) — delta
**6.25e-2**, failing the 1e-3 clause 62x while satisfying the position
clause (50 > 48).

Two follow-up measurements complete the picture:

1. **The runtime logit dtype is float16, and the delta is 4 ULPs.** The
   checkpoint config declares `torch_dtype: bfloat16`, but the loaded
   pipeline computes fp16 logits (measured: `generate_step` logprobs and
   raw logits are `mlx.core.float16`; a 2026-07-05 PROGRESS note calling
   these bf16 on a 1/128 grid was imprecise — corrected here). The
   candidates' raw logits sit at ~16.72, in the fp16 binade [16, 32)
   whose ULP is 2^-6 = 1.5625e-2. The measured 6.25e-2 delta is exactly
   4 fp16 ULPs. The minimum NONZERO logit delta at this magnitude is one
   ULP = 1.5625e-2 — 15.6x above the 1e-3 threshold (bf16 would be
   2^-3 = 0.125, 125x above). The 1e-3 constant equals hundreds of f32
   ULPs at these logit magnitudes: a meaningful "the race was close"
   test for f32, but mathematically unsatisfiable for fp16 or bf16
   logits short of an exact tie. The clause was implicitly calibrated
   for f32 and is dead for every half-precision model.
2. **The "cross-device" flip reproduces on the generating device by
   kernel class alone.** Recomputing the same 70-token state in a single
   prefill pass (M=70, tiled kernel class) on the dev machine yields an
   exact fp16 tie — both candidates' logits are 16.71875 — and argmax
   breaks to 188797, the foreign-device outcome. A device change is a
   kernel-class change (different reduction orders), not a new
   phenomenon beyond ADR 0002.

## Decision

1. **Golden-token bit-exactness is a same-device guarantee.** The SPEC
   §11.2 bar binds on the fixture-generating device class and there it
   is unchanged and strict: a golden failure on the dev machine is a
   correctness bug, full stop. Token-id equality on any other device is
   explicitly NOT a Kiln correctness bar — greedy argmax surviving a
   kernel-class change is margin, not a guarantee (ADR 0002), and the
   gemma-3 case is the proof: a 4-ULP fp16 race that ties exactly under
   the neighboring kernel class. Consequently the CI golden step
   (`.github/workflows/ci.yml`, "Golden parity vs committed fixtures")
   is **permanently advisory** — `continue-on-error`, never promoted to
   blocking, no matter how long it happens to stay green. Its value is
   drift detection: a CHANGE in its failure pattern (new fixture, new
   model, new divergence position) accompanying a code change is a
   signal worth a PROGRESS note and investigation, never a merge
   blocker. The same scope applies to the other committed-fixture
   comparisons (preemption scenario 4, prefill_pad's cold-vs-fixture
   assert): blocking on the generating device, advisory on CI. Foreign-
   device correctness is gated by the device-independent tier (the
   Option B blocking step), which asserts same-device invariants that
   must hold on ANY device.
2. **The relaxed-bar delta clause becomes dtype-aware.** The fixed
   `< 1e-3` is replaced by: *logprob delta at the divergence position of
   at most **4 ULPs of the logit compute dtype at the divergence
   candidates' raw-logit magnitude***, measured on the reference
   implementation (mlx-lm `generate_step` logprobs over the fixture
   path; logprob deltas equal raw-logit deltas up to logsumexp rounding,
   so either is measurable). Rationale: "the candidates were tied up to
   kernel-order noise" is a statement about resolution, and resolution
   is dtype x magnitude — a fixed constant cannot express it. Concretely
   at logit magnitude [16, 32): f32 → 7.6e-6, fp16 → 6.25e-2, bf16 →
   0.5. For f32 this TIGHTENS the old bound (1e-3 was ~500 f32 ULPs
   there); the escape hatch has never been invoked, so nothing breaks.
   The 4-ULP width covers the observed legitimate flip (exactly 4 fp16
   ULPs on the decode path, an exact tie one kernel class over) without
   admitting real distribution differences, which sit orders of
   magnitude higher. The position clause (first divergence beyond token
   48) is unchanged. Invoking the relaxed bar still requires a further
   ADR naming the specific model and reason — this ADR calibrates the
   clause; it pre-approves nothing.

## Consequences

- SPEC §11.2 is amended to state the same-device scope and the
  dtype-aware clause, referencing this ADR (the SPEC edit lands with
  this ADR; CLAUDE.md's description of the relaxed bar remains accurate
  as written).
- The advisory CI step's log is the only place its result is visible
  (the job stays green by design). Reading it is part of assessing
  golden status on CI; its result is recorded in PROGRESS when it
  changes.
- Fixtures continue to be generated only on the pinned dev-machine
  setup and only when explicitly instructed. If the fixture-generating
  device class itself changes (new dev machine), the full golden suite
  must be re-run there before any fixture is trusted, and divergences
  handled as re-generation events, not relaxed-bar events.
- Re-verify at every mlx-c / core-MLX bump as part of ADR 0001's
  standing quarterly process: re-run the goldens on the generating
  device (strict) and read the advisory lane for pattern changes. If a
  bump makes kernel dispatch M-invariant and cross-device deterministic,
  the advisory lane's promotion can be reconsidered — via a new ADR.
- The gemma-3 chat-basic case needs no action under this ADR: on the
  generating device it is token-exact (the bar it is bound by), and its
  foreign-device flip is a 4-ULP kernel-class coin toss, not a defect.
