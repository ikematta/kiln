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
