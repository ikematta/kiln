# ADR 0001: mlx-c pinned at v0.6.0

- Status: accepted
- Date: 2026-07-02 (Phase 0)

## Context

`kiln-mlx` hand-rolls FFI over mlx-c (SPEC §3, §7.1). mlx-c is vendored as a
git submodule at `crates/kiln-mlx/vendor/mlx-c` and built by `build.rs` via
cmake, statically linked. The FFI declarations in `kiln-mlx/src/sys.rs` are
written against the exact headers at the pin, so any bump can silently change
signatures or semantics under the bindings.

## Decision

Pin `ml-explore/mlx-c` at tag **v0.6.0**, commit
`0726ca922fc902c4c61ef9c27d94132be418e945`.

This transitively pins MLX itself: mlx-c's CMakeLists FetchContent-pins
`ml-explore/mlx` at **v0.31.1**. The MLX version therefore moves only when
this submodule pin moves.

## Consequences

- The agent must not bump the submodule (CLAUDE.md). Bumps happen via a
  scheduled quarterly task: bump, rebuild, full golden-parity re-run
  (SPEC §11.2, §14), human approval.
- An mlx-c API needed but missing at this pin is a `DECISION NEEDED:` entry
  in PROGRESS.md, not a bump and not a workaround.
- Stale-build recovery after checkout changes:
  `rm -rf target/mlx-c-build && cargo build -p kiln-mlx`.

## Addendum (2026-07-03): Python worker MLX version alignment

Appended at explicit PM instruction (docs/decisions/ is otherwise agent
read-only). Recording only; no pins changed.

`kiln_worker_py` pins `mlx-lm==0.31.3` (pyproject.toml), whose `mlx` wheel
dependency resolves to **mlx.core 0.31.2** in the worker venv (verified:
`uv run --project python/kiln_worker_py python -c "import mlx.core as mx;
print(mx.__version__)"` → `0.31.2`; uv.lock agrees). This **drifts by one
patch version** from the MLX the Rust side builds: the vendored mlx-c v0.6.0
FetchContent-pins MLX **v0.31.1** (see Decision above).

Consequence to keep in mind: the two workers run different MLX patch
releases, so bit-exact cross-worker output identity is not guaranteed even
for greedy decoding. Golden-parity fixtures (SPEC §11.2) are generated from
mlx-lm as the reference — if patch-level drift ever surfaces as a fixture
mismatch, reconcile the pins (quarterly mlx-c bump task) rather than relaxing
the parity bar.
