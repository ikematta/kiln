# ADR 0002: Parity bars under MLX's M-dependent kernel dispatch

- Status: accepted (PM-directed, 2026-07-04)
- Date: 2026-07-04 (Phase 6)

## Context

MLX v0.31.1 (transitively pinned by mlx-c v0.6.0, ADR 0001) selects between
distinct Metal kernels for the two op families on Kiln's hot path based on
M, the row count of the operation:

- **Quantized matmul** (`mlx/backend/metal/quantized.cpp`,
  `get_qmv_batch_limit`): a vector kernel (`qmv`/`qmv_fast`) below a
  threshold, a tiled matrix kernel (`qmm`) at or above it. The threshold
  depends on the weight shape and the GPU architecture: 6–32 across the
  dispatch table (measured on the Phase 6 dev machine: 18 for
  K,N ≤ 2048 projections, 10–12 for the large MLP/lm_head shapes; the
  table's maximum, 32, occurs on the `d`-suffix GPU class).
- **`mx.fast` SDPA**: a vector path for short query lengths and a tiled
  two-pass path otherwise.

The two kernels in each family reduce in different orders and therefore
produce **bit-different but numerically close (ulp-level) results** for the
same rows. Measured at the pin, on real checkpoint weights, for every
architecture Kiln implements (llama, qwen2, qwen3): rows are bit-identical
*within* a kernel class regardless of M, and differ *across* the class
boundary, for every projection including the tied lm_head. SDPA behaves the
same way (query length 32 vs 260: bit-identical; 4 vs 260: not).

This is a characteristic of the pinned library and hardware, **not a Kiln
defect**: no schedule or model code choice can make one forward pass at
M = 16 reproduce the bits of sixteen forward passes at M = 1 without
forfeiting batching itself (per-sequence trunk matmuls would re-read the
weights per request, destroying the SPEC §11.3 batch-throughput target).

Consequently, bit-exact logit equality between batched (M > 1) and
single-stream (M = 1) decoding was never guaranteed. The Phase 4/5 Llama
width-16 golden results were **token-id** equality; the logits already
differed in ulps at width 16 (the lm_head crosses its dispatch threshold
near width 10–12), and greedy argmax happened to be robust to that noise on
those fixtures.

## Decision

1. **Single-stream (M = 1) parity vs the mlx-lm reference remains strictly
   bit-exact — token-for-token over every committed fixture, no
   exceptions.** This is the SPEC §11.2 keystone bar, unchanged.
2. **Batched decode (M > 1) parity is defined as token-id equality with the
   single-stream reference** — matching the invariant SPEC §6.6/§11.3 and
   CLAUDE.md actually state (greedy *output* must not change with
   batching), NOT bit-exact logit equality.
3. **Prefill pieces are kept in the reference kernel class** by padding:
   any ragged tail piece shorter than `PREFILL_PAD_MIN_ROWS` (32) rows that
   does not start on a `prefill_chunk` boundary is computed with pad rows
   appended to the trunk and pad query rows prepended for SDPA, and the pad
   rows are discarded — never written to KV, never sampled, never emitted.
   32 covers the maximum `qmv` threshold in the dispatch table across GPU
   classes and the SDPA vector-kernel bound. A piece that *does* start on a
   `prefill_chunk` boundary and is short is exactly the piece the mlx-lm
   reference also computes at that size, and is deliberately not padded.
   This keeps bar (1) satisfiable at every prompt length under the Phase 5
   fine-grid schedule (which is otherwise unchanged: boundaries, resume
   semantics, and cache eligibility are as before).

## Consequences

- Tests asserting batched-vs-single-stream agreement (the golden width-16
  rounds, `tests/batching.rs` concurrency sections) verify **token-id
  equality**; their documentation must not claim bit-exactness. Where the
  greedy stream survives, that is argmax margin absorbing ulp noise, and it
  is fixture-, model-, and GPU-dependent.
- Measured status at this pin (2026-07-04, M4-class dev machine): llama
  holds token-id equality at decode width 16 across all committed fixtures;
  qwen2.5/qwen3 chat fixtures flip a token near position 28–33 at width 16.
  Batched-decode enablement per architecture is gated on the width-16
  golden rounds under bar (2).
- Revisit at any future mlx-c / core-MLX bump, as part of the same standing
  quarterly process as ADR 0001's C1 plan: re-run the kernel-class probes
  and the full golden suite; if a bump makes dispatch M-invariant (or adds
  a switch), bar (2) can be tightened.

## Addendum (2026-07-10): bar (3) scope limit — pad does not cover dense trunks

Appended at explicit PM instruction (docs/decisions/ is otherwise agent
read-only). Records a measured scope limit; bars (1) and (2) are
unchanged.

**Bar (3)'s pad-to-32 construction does NOT guarantee reference-
kernel-class placement for dense (unquantized) trunks.** Falsified
during Phase 6 Task 3 (PROGRESS 2026-07-10) on
smollm2-135m-bf16/raw-tiny-remainder (133-token prompt → fine-grid
prefill 128 + 4-row ragged tail padded to 32): greedy divergence at
generated token index 45 (engine 260 vs fixture 284). A pure-mlx-lm
replica of the padded fine-grid schedule reproduces the same flip at
the same index, while the reference-shaped (single 132-row piece)
replica reproduces the fixture — the implementation is exonerated; the
padded schedule itself is not bit-reproducible against the reference
for this dense bf16 trunk at the pin.

The real-tensor bisect is the substance of the falsification:
synthetic probes at the exact shapes and geometry — Gaussian and
outlier-heavy — false-negatived every op family; only real activations
exposed the divergence. On real tensors, the unpadded tail's layer-1
SDPA (query length 5, vector class) diverges from the reference class
(measured real-data query-class boundary: 9), and under the pad
(query length 32, reference class at layer 1, all layer-1 sub-ops
bit-equal) the full padded schedule STILL flips — an op in a deeper
layer crosses kernel class on its own data, independent of the padded
piece's row count. Kernel-class placement is therefore not certifiable
per-op for dense trunks, and the pad guarantee has no measurable
foundation there.

**Resolution: dense trunks never fragment prefill.** Unquantized
checkpoints take the monolithic prefill override
(`AnyModel::monolithic_prefill_required` = gemma2-softcap OR
`quantization.is_none()`, honored by engine builders as
`prefill_fine_chunk = prefill_chunk`): every prefill piece is
reference-shaped by construction — identical to mlx-lm's own prefill
loop — and bar (3)'s pad rule never triggers (every piece starts on a
`prefill_chunk` boundary). This sidesteps the pad guarantee entirely
rather than repairing it.

**Bar (3) remains valid and in effect for quantized trunks only**,
where it continues to hold empirically: all quantized fixture models
(4-bit and 8-bit, fp16 activations) are token-exact at every committed
prompt length under the fine-grid schedule, re-verified in the same
Task 3 run. Re-examine, with the rest of this ADR, at every
mlx-c/core-MLX bump under ADR 0001's standing quarterly process.
