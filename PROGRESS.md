## [2026-07-02] Phase 0 / Scaffold & Contract (workspace, CI, proto codegen, mlx-c submodule + smoke) — DONE
- What:
  - Cargo workspace (resolver 3, edition 2024) with the 8 crates from SPEC §4: kiln-proto, kiln-mlx, kiln-models, kiln-engine, kiln-worker, kiln-gateway, kiln-tokenize, kiln-jobs. Libraries/binaries other than kiln-mlx are stubs with `#![deny(unsafe_code)]`.
  - kiln-proto: Rust codegen from `proto/kiln/v1/worker.proto` via tonic-prost-build (build.rs), exposed as `kiln_proto::v1`. Proto file used exactly as seeded — no field changes.
  - python/kiln_worker_py: uv-managed package (requires-python >=3.12); generated `worker_pb2`/`worker_pb2_grpc` committed under `src/kiln_worker_py/gen/`, excluded from ruff.
  - kiln-mlx: mlx-c vendored as git submodule at `crates/kiln-mlx/vendor/mlx-c`, pinned at v0.6.0 (`0726ca922fc902c4c61ef9c27d94132be418e945`, ADR docs/decisions/0001-mlx-c-pin.md). build.rs drives cmake into `target/mlx-c-build`, statically links libmlxc/libmlx + frameworks. Hand-written minimal `sys.rs` (5 symbols, matched against pinned headers) + `smoke` safe wrapper + `examples/smoke.rs`.
  - CI (.github/workflows/ci.yml): ubuntu lint (fmt, clippy --no-default-features -D warnings, ruff check+format), ubuntu compile-check (--no-default-features), macos-14 full build + clippy + tests + smoke + python codegen drift check.
- Decisions:
  - tonic 0.14 series splits prost integration: deps are tonic + tonic-prost (runtime), tonic-prost-build (build). Uses system protoc (35.1 locally; CI installs via apt/brew).
  - mlx-c pinned at v0.6.0 (latest release tag); transitively pins MLX at v0.31.1 via mlx-c's FetchContent. Recorded in ADR 0001.
  - kiln-mlx default feature `metal` gates the entire mlx-c build/link; `--no-default-features` is the Linux compile-check path per CLAUDE.md.
  - build.rs passes `CMAKE_POLICY_VERSION_MINIMUM=3.5` (local cmake is 4.3.4, which refuses fetched subprojects declaring < 3.5).
  - Smoke test computes on the MLX default CPU stream (deterministic on any macOS box; GPU paths start Phase 3).
  - Generated Python stubs are committed; CI regenerates and fails on drift (`git diff --exit-code`).
  - MSRV left unset — SPEC §14 lists it as an open pre-Phase-0 PM decision. Non-blocking; flagging for review.
- Deviations:
  - This machine has Command Line Tools only (no full Xcode → no `metal` compiler), so MLX builds with `MLX_BUILD_METAL=OFF` here. build.rs autodetects the Metal toolchain (`xcrun -sdk macosx metal`) and enables Metal where available (e.g. macos-14 CI runners); `KILN_MLX_METAL=0|1` overrides. CLAUDE.md's "full build incl. Metal" is not achievable on this host until Xcode is installed.
  - `sys.rs` is hand-written for the smoke surface only; the bindgen-generated full binding surface is part of the Phase 3 FFI task (noted in the file header).
  - "CI green" cannot be verified: the repo has no git remote. All CI job commands were run locally instead (outputs below).
  - Installed rustup + stable toolchain (1.96.1) on this machine — no Rust toolchain was present.
- Acceptance:
  ```
  $ cargo build --workspace
  warning: kiln-mlx@0.0.1: kiln-mlx: building vendored mlx-c (MLX_BUILD_METAL=OFF)
      Finished `dev` profile [unoptimized + debuginfo] target(s) in 2m 27s

  $ cargo run -p kiln-mlx --example smoke
       Running `target/debug/examples/smoke`
  3.0

  $ cargo test --workspace
  test smoke::tests::one_plus_two_is_three ... ok
  test result: ok. 1 passed; 0 failed; ... (all other crates: 0 tests, ok)

  $ cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
      Finished `dev` profile [unoptimized + debuginfo] target(s) in 3.21s   (clean)
  $ cargo build --workspace --no-default-features && cargo clippy --workspace --all-targets --no-default-features -- -D warnings
      Finished `dev` profile [unoptimized + debuginfo] target(s)            (clean)

  $ uv run --project python/kiln_worker_py python -m grpc_tools.protoc -Iproto \
      --python_out=... --grpc_python_out=... proto/kiln/v1/worker.proto     (ok)
  $ python -c "from kiln.v1 import worker_pb2, worker_pb2_grpc; ..."
  python proto OK: r1 [1, 2, 3] 8 type
  $ uv run --project python/kiln_worker_py ruff check python/ && ... ruff format --check python/
  All checks passed! / 1 file already formatted
  ```
- Next: remaining Phase 0 item — `kiln.toml` parsing with figment + `kiln.toml.example` (SPEC §10, §12 Phase 0). Then Phase 1 (Python worker end-to-end).

## [2026-07-03] Phase 0 / Correction — Metal toolchain now available on the dev machine
- What:
  - Corrects the 2026-07-02 entry's deviation note "full build incl. Metal is not achievable on this host": the Metal Toolchain has since been installed (`xcodebuild -downloadComponent MetalToolchain`), and `xcrun -sdk macosx metal --version` now succeeds (Apple metal 32023.883). Re-verified Metal-ON build + smoke locally (output below). build.rs autodetection now selects `MLX_BUILD_METAL=ON` on this machine without the `KILN_MLX_METAL=1` override.
  - Remote housekeeping: `origin` verified as `https://github.com/ikematta/kiln.git`, `main` pushed with upstream tracking. GitHub Actions ran on the pushed head and is green — closes the "CI green" Phase 0 acceptance item that was previously unverifiable (run 28638701212, `completed success`, 5m57s).
- Decisions: none (no code changed; the original entry stands as written per append-only rule)
- Deviations: none
- Acceptance:
  ```
  $ git remote -v
  origin  https://github.com/ikematta/kiln.git (fetch/push)
  $ git push -u origin main
  branch 'main' set up to track 'origin/main'.        (Everything up-to-date)
  $ gh run list --repo ikematta/kiln --limit 1
  completed  success  P0: record Phase 0 scaffold completion...  CI  main  push  28638701212  5m57s

  $ KILN_MLX_METAL=1 cargo build -p kiln-mlx
  warning: kiln-mlx@0.0.1: kiln-mlx: building vendored mlx-c (MLX_BUILD_METAL=ON)
      Finished `dev` profile [unoptimized + debuginfo] target(s)
  $ grep MLX_BUILD_METAL target/mlx-c-build/build/CMakeCache.txt
  MLX_BUILD_METAL:BOOL=ON
  $ cargo run -p kiln-mlx --example smoke
  3.0
  $ cargo test -p kiln-mlx
  test smoke::tests::one_plus_two_is_three ... ok    (1 passed)
  ```
- Next: kiln.toml parsing with figment (this session, entry below).
