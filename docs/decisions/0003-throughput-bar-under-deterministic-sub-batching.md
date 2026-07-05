# ADR 0003: Throughput acceptance under deterministic sub-batched decode (B')

- Status: accepted (PM-directed, 2026-07-05)
- Date: 2026-07-05 (Phase 6)
- Relationship: directly continues ADR 0002's determinism-bar discussion.
  Recorded as a new ADR rather than an appended section because
  `docs/decisions/` files are agent-read-only once landed (CLAUDE.md; the
  ADR 0001 addenda were explicit one-off PM instructions), and because
  ADR 0002 defines *correctness* bars while this ADR amends a
  *performance acceptance target* — a separable decision with its own
  revisit trigger.

## Context

ADR 0002 bar (2) — batched greedy output must equal single-stream — is,
since the B' implementation (PROGRESS 2026-07-05), guaranteed by
construction: rows of deterministic requests (temperature = 0 or explicit
client seed) decode in sub-batches of at most the startup-calibrated
width W (the device's row-bit-stability boundary; 9 on the Phase 6 dev
machine), keeping every trunk matmul in the M=1 kernel class.
Non-deterministic rows ride one unrestricted full-width forward. All 16
golden fixtures are bit-exact single-stream AND at decode width 16, all
three architectures.

**The measured price at the pinned mlx-c v0.6.0 / MLX v0.31.1
(ADR 0001):** deterministic rows pay ceil(rows/W) weight
streams per step instead of one. Engine benchmarks at decode width 16:

| load                | llama-3.2-1b-4bit    | qwen3-8b-4bit        |
|---------------------|----------------------|----------------------|
| greedy16, unpartitioned (old) | 410.4 tok/s, 3.31x | 67.5 tok/s, 3.46x |
| greedy16, B'        | 259.9 tok/s, 2.10x   | 44.8 tok/s, 2.30x    |
| mixed16, B'         | 222.3 tok/s, 1.80x   | 45.7 tok/s, 2.34x    |
| sampled16 (non-det) | 331.6 tok/s, 2.68x*  | 62.1 tok/s, 3.18x    |

(x = multiple of single-stream throughput; * the 1B sampled-load miss is
a small-model sampler-op artifact, pre-existing and unrelated to B'.)

The 2026-07-05 op-level-split investigation (PROGRESS REPORT entry)
established that this cost is **a property of the pinned kernel dispatch
table, not an implementation defect, and is not recoverable by
re-partitioning**:

- The penalty is qmv-vs-qmm weight re-streaming at the kernel-class
  boundary. MLP and lm_head shapes — 80%+ of streamed bytes — sit at
  dispatch threshold 10 at both 1B and 8B scale and must chunk to ≤ 9
  rows under any bit-exact scheme.
- A per-shape composition of directly measured kernel times reproduces
  93% of the engine-measured B' step-time delta at 1B (20.9 vs
  22.6 ms/step) — the cost is kernel weight-streaming time, not
  scheduling or dispatch overhead.
- Op-level splitting (chunking only threshold-crossing matmuls inside
  one full-width forward, sharing attention/norms) recovers 0% at 1B
  (attention splitting is cost-neutral there) and ~1% at 8B (attention
  shapes' own threshold, 12, is below width 16 and forces chunking
  anyway); ≤ ~3% crediting every avoidable overhead.

No partition arrangement at this pin closes the gap. The only structural
lever is upstream: a batched qmv kernel that streams weights once across
row groups while preserving row bits.

## Decision

1. **The SPEC §11.3 batch-16 ≥ 3x single-stream acceptance target applies
   to non-deterministic and mixed-majority load.** It holds under B'
   (3.18x at 8B; the 1B sampled number is a known small-model sampler
   artifact, tracked separately).
2. **Deterministic-containing batch loads (greedy or explicit-seed rows
   present) are governed by a separate, explicitly lower bar.** The
   current measured floor is recorded as the B' price of bit-exact
   batched greedy: **~2.1–2.3x aggregate at decode width 16** (2.10x
   greedy / 1.80x mixed at 1B; 2.30x greedy / 2.34x mixed at 8B; dev
   machine W = 9). This is a **measured floor, not a target**: phase
   benches record it, and a regression below the recorded floor fails a
   phase gate exactly like any other >10% bench regression (SPEC §11.3).
   Single-stream latency is untouched by B' and keeps its existing bar.
3. **Revisit trigger:** re-measure this penalty at every future
   mlx-c/core-MLX version bump, as part of ADR 0001's standing quarterly
   C1 process. The startup calibration adapts W automatically if
   dispatch changes, but the cost figures in this ADR must be re-benched
   and are to be treated as **stale the moment the pin moves**. An
   upstream kernel change that streams weights once across row groups
   would close the gap and allow re-unifying the two bars.

## Consequences

- SPEC §11.3's perf bullet references this ADR for the batched-throughput
  acceptance split (doc-only amendment; the section is otherwise
  unchanged).
- `scripts/bench.sh` phase runs should report deterministic-load and
  non-deterministic-load batch-16 numbers separately so the two bars are
  individually gateable.
- The 14B-class deployment bench (deferred from the B' closeout: dev
  machine is 16GB) records its deterministic-load floor against bar (2)
  when first run on M4 Pro/Max-class hardware.
- Option B from the 2026-07-05 DECISION NEEDED (per-op chunking) is
  struck permanently on the investigation's measurements.
