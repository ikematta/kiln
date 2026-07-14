# ADR 0005: The speculative-verify kernel envelope (dispatch-derived gamma clamp)

- Status: accepted (PM-directed, 2026-07-13)
- Date: 2026-07-13 (Phase 8 part 2)
- Relationship: the fourth instance of the structural pattern ADRs
  0002/0003/0004 record — the pinned MLX selects between Metal kernels by
  operand shape, kernels reduce in different orders, and greedy argmax
  races (measured down to 1 ULP) are decided differently across the
  class boundary. 0002 met it on trunk-matmul row count and prefill SDPA
  query length; 0003 priced its batching consequence; 0004 met it across
  devices; this ADR meets it on the speculative verify forward's query
  count and defines where speculation may run at all.

## Context

SPEC §6.5's verify step scores a drafter's `gamma` proposed tokens plus
the fed token in ONE target forward — `gamma+1` query rows where plain
decode runs one. Phase 8 part 2's greedy-invariance gate (every golden
fixture rerun with speculation on) caught a divergence:
qwen2.5-0.5b-4bit/chat-basic flips at generated index 33 under a
gamma=4 verify, on a position whose top-2 raw-logit gap is exactly
1 fp16 ULP (16.765625 vs 16.75 at binade [16,32), ULP 2^-6). A 5-row
verify-shaped forward from the IDENTICAL KV state reproduces the flip;
2-, 3-, and 4-row shapes are bit-identical to plain decode. The flip is
deterministic per process layout but varies across allocation histories
— no fixture rerun can certify the offending shape.

Root cause, verified from the pinned source (MLX v0.31.1 via mlx-c
v0.6.0, `mlx/backend/metal/scaled_dot_product_attention.cpp`,
`ScaledDotProductAttention::use_fallback`), not inferred from behavior:

```cpp
const bool sdpa_vector_supported_head_dim =
    query_head_dim == value_head_dim &&
    (query_head_dim == 64 || query_head_dim == 96 ||
     query_head_dim == 128 || query_head_dim == 256);
const bool supports_sdpa_full = query_sequence_length > 8 && ...;
const bool supports_sdpa_vector = (query_sequence_length <= 8) &&
    (query_sequence_length <= key_sequence_length) &&
    sdpa_vector_supported_head_dim &&
    (query_sequence_length * gqa_factor) <= 32;
```

qwen2.5-0.5b has gqa_factor 7 (14 Q heads / 2 KV heads): a gamma=4
verify (qL 5, 5x7 = 35 > 32) satisfies NEITHER predicate and silently
takes the UNFUSED composed-op attention — a different kernel class from
the qL=1 vector kernel plain decode uses. Every other pinned model has
gqa_factor <= 4 and stays inside the predicate at qL 5, which is exactly
the observed pass/fail matrix.

The predicate is uniform at the pin: `use_fallback` contains no device
or dtype branching. dtype selects a kernel template but never the
dispatch decision, so it does NOT enter the clamp formula. `head_dim`
DOES — as a set-membership precondition, not a gamma term.

Two further source facts complete the envelope:

1. **Within the 1-pass vector kernel, per-row bits are invariant to
   query count and key length by construction** (`kernels/sdpa_vector.h`):
   one threadgroup per query row (`grid_dims(B*H, qL, 1)`, fixed
   `group_dims(1024,1,1)`); keys are assigned to simdgroups by fixed
   stride 32 independent of N; each thread's online softmax accumulates
   in ascending key order; the cross-simdgroup reduction tree is a fixed
   32x32 shape; and the bottom-right causal predicate
   `i <= N - qL + q_seq_idx` makes verify row j use exactly the key set
   plain decode uses at that position — masked keys never touch the
   accumulators. This is the source-level proof behind the measured
   2/3/4-row bit-identity, and the reason a clamped verify commits
   exactly the plain path's tokens by induction.
2. **The 2-pass vector variant is outside any such certificate**: the
   host dispatch (`sdpa_vector_2pass`) selects it at key length >= 1024
   on 'd'/'s'-class GPUs, or >= 4096 with GQA on any device, and its
   partition count (`blocks`) depends on `n_simds = gqa_factor * qL` and
   on N — the query count itself changes the reduction partitioning, so
   verify and plain provably differ there.

The trunk is not at issue: verify matmuls at M = gamma+1 <= W ride the
same calibrated row-bit-stability the B' width-16 goldens already prove
(ADR 0002). Two attention paths have no fused-class certificate at all
and are excluded outright: gemma2's manual softcapped attention (its
score/probs matmuls change gemv/gemm class with the query row count) and
dense (unquantized) trunks (no fine-shape bit guarantee — the ADR 0002
addendum precedent).

## Decision

1. **Speculation runs only inside the certified envelope**, enforced at
   drafter attachment (worker: `AnyModel::speculative_gamma_bound`,
   config-derived, never hardcoded per model) and per round (engine):
   - the target's architecture module must take the fused-SDPA decode
     path (no manual softcapped attention), with a quantized trunk, and
     `head_dim` in {64, 96, 128, 256} (equal Q/V head dims);
   - `gamma + 1 <= min(W, 8, floor(32 / gqa_factor))` — W is the
     ADR 0002 calibrated deterministic width; 8 and 32 are the pinned
     `supports_sdpa_vector` constants;
   - the verify's key length (`offset + gamma + 1`) stays within the
     1-pass region on every device class:
     `kiln_engine::VERIFY_MAX_KEY_LEN = 1023`. Requests past it stop
     speculating (plain decode continues). Refinement to the per-device
     boundary (4096 off 'd'/'s' classes, via `mlx_device_info`'s
     architecture string, which the pinned mlx-c exposes) is recorded as
     an optimization, not implemented.
   A target outside the envelope with a draft configured is a LOAD
   FAILURE (worker UNHEALTHY, "outside the ADR 0005 speculation
   envelope") — the same loudness as an incompatible tokenizer; a worker
   never serves with requested speculation silently inert. At this pin:
   llama/qwen3/gemma3 targets keep gamma 4, qwen2.5-0.5b is clamped to
   gamma 3, gemma2 and dense checkpoints cannot speculate.
2. **Precondition for new architectures** (this is the documented gate,
   not an implicit consequence of the formula): enabling speculation for
   any new model family — or any checkpoint with unusual attention
   geometry (higher gqa_factor, new head_dim, non-fused attention,
   different masking) — requires (a) reviewing its geometry against the
   envelope above, including re-reading the dispatch predicate if the
   pin has moved, and (b) a green run of the full spec_decode gate
   (every fixture x self-draft + adversarial drafters) on the
   fixture-generating device. `speculative_gamma_bound` returning `Some`
   for an unreviewed geometry is not by itself permission.
3. **Enforcement is the spec_decode gate**, now CI-blocking: on the
   generating device it compares speculation-on output against the
   committed fixtures; on foreign devices (`KILN_FIXTURE_PARITY=skip`)
   against a live speculation-off run — the device-independent form of
   the SPEC §6.5 invariant, valid anywhere because the envelope keeps
   every verify inside the source-certified 1-pass class on any device.
4. **Revisit at every mlx-c/core-MLX bump** under ADR 0001's standing
   quarterly process: re-read `use_fallback` and the 1-pass/2-pass
   dispatch, re-run tests/spec_probe.rs (which deliberately constructs
   the out-of-envelope gamma=4 shape on qwen2.5 and must keep printing
   its divergence for as long as the pin dispatches this way), and
   re-run the full spec_decode gate. If a bump makes SDPA dispatch
   query-length-invariant, the clamp constants can be relaxed — via a
   new ADR.

## Consequences

- SPEC §6.5's "proposes gamma tokens (default 4)" is qualified by this
  envelope; the per-model effective gamma is an attachment-time fact
  logged by the worker and observable in acceptance metrics.
- Speculation is unavailable beyond ~1k tokens of context at this pin
  (the conservative universal 1-pass bound). The device-aware refinement
  (4096 on non-'d'/'s' classes) is the recorded path if long-context
  speculation matters before the pin moves.
- gemma2 and dense checkpoints serve without speculation; configuring a
  draft for them is a hard error by design.
- The characterization instruments stay in-tree
  (kiln-models/tests/spec_probe.rs) as reproducible evidence bound to
  this pin.
