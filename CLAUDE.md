# CLAUDE.md — Kiln Agent Operating Manual

You are building **Kiln**, a Rust LLM inference server for Apple Silicon over MLX.
The authoritative specification is `docs/SPEC.md`. Read it before any work. When this
file and SPEC.md conflict, SPEC.md wins; flag the conflict in PROGRESS.md.

## Session protocol (non-negotiable)

1. Read `PROGRESS.md` (tail 100 lines) to find current phase/task state.
2. Do exactly the task you were given in the prompt. Do not start the next task.
3. Before claiming done: run the acceptance commands for the task, paste real output.
4. Append a PROGRESS.md entry (format below). Never edit or delete prior entries.
5. If blocked on a design choice not covered by SPEC.md: write the options in
   PROGRESS.md under `DECISION NEEDED:`, pick nothing, and stop.

## Hard rules

- **No monkey-patching, ever.** Not in Rust, not in the Python worker. Model-specific
  behavior lives in Kiln-owned modules keyed by `model_type`. If a dependency is
  broken, vendor the file or add a build-time patch under `patches/` with an ADR.
- **The proto is frozen after Phase 2.** `proto/kiln/v1/worker.proto` wire semantics
  may not change without a new ADR in `docs/decisions/` and explicit human approval.
  Additive changes (new fields, new capability flags) are allowed; renumbering,
  retyping, or repurposing fields is forbidden. Never reuse a removed field number —
  add it to `reserved`.
- **`docs/decisions/` is read-only for you** once a file exists there. Propose changes
  via PROGRESS.md.
- **No new dependencies without justification.** Any `Cargo.toml` or `pyproject.toml`
  dependency addition must be explained in the commit message: what it does, why std
  or an existing dep can't, license. No git dependencies except the vendored mlx-c
  submodule. No `openssl` (use rustls).
- **FFI discipline (kiln-mlx):**
  - All `unsafe` lives in `crates/kiln-mlx/src/sys.rs` and the safe wrapper modules
    of that crate. Zero `unsafe` anywhere else in the workspace. `#![deny(unsafe_code)]`
    in every other crate.
  - Every mlx-c `*_new` call has exactly one matching `*_free`, enforced via `Drop`.
    The debug-build allocation counter (`kiln_mlx::debug::live_objects()`) must return
    to baseline in every test that constructs arrays.
  - Install the custom error handler (`mlx_set_error_handler`) at worker startup.
    The default handler calls `exit()` — a worker that dies on a bad shape is a bug.
  - All MLX operations are issued from the single engine thread. `Stream` is `!Send`.
    Do not "fix" that by wrapping it in a Mutex.
- **Error handling:** no `unwrap()`/`expect()` in library code (tests and examples
  fine). Fallible paths return `Result` with `thiserror` types. Worker must never
  panic on malformed client input — it returns a `Finished{error}` event.
- **Never weaken a test to make it pass.** If a test seems wrong, say so in
  PROGRESS.md and stop. Golden fixtures in `tests/golden/` are regenerated only via
  `scripts/gen-golden.py` and only when explicitly instructed.
- **Determinism:** greedy decoding must be bit-reproducible run-to-run on the same
  build. Batching, prefix caching, and speculative decoding must not change greedy
  outputs — several tests assert exactly this; treat a violation as a correctness
  bug in your change, not in the test.
- Formatting/lint gates: `cargo fmt --check`, `cargo clippy --workspace -- -D warnings`,
  `ruff check python/ tests/e2e scripts`, `ruff format --check python/ tests/e2e scripts`.
  Run before every commit.
- Commits: small, one logical change, imperative subject, body explains why.
  Reference the phase/task, e.g. `P4: block manager COW path (task 4.2)`.

## Build & test commands

```bash
# One-time setup
git submodule update --init --recursive          # pulls vendored mlx-c (pinned)
./scripts/fetch-test-model.sh                    # downloads pinned tiny models to ~/.kiln/test-models
uv sync --project python/kiln_worker_py          # python worker venv

# Build
cargo build --workspace                          # macOS: full build incl. kiln-mlx/Metal
cargo build --workspace --no-default-features    # linux CI compile-check (no Metal)

# Proto codegen (runs automatically via build.rs; force with:)
cargo build -p kiln-proto
python -m grpc_tools.protoc -Iproto --python_out=python/kiln_worker_py/src/kiln_worker_py/gen \
  --grpc_python_out=python/kiln_worker_py/src/kiln_worker_py/gen proto/kiln/v1/worker.proto

# Tests
cargo test --workspace                           # unit + integration (Metal tests auto-skip if no GPU)
cargo test -p kiln-engine -- --ignored           # property tests (slow)
cargo test -p kiln-models --test golden          # golden-token parity harness
pytest python/kiln_worker_py/tests               # python worker unit tests
uv run --project tests/e2e pytest tests/e2e     # black-box HTTP tests (full stack; `uv sync --project tests/e2e` once)

# Benchmarks & soak
cargo bench -p kiln-engine                       # criterion: step overhead, block mgr
./scripts/bench.sh --model qwen3-0.6b-4bit       # throughput/TTFT; writes results to bench/results/
./scripts/soak.sh --minutes 30                   # leak gate: RSS slope + mlx live-object counter

# Run locally
cargo run -p kiln-gateway -- --config kiln.toml
cargo run -p kiln-worker -- --model ~/.kiln/test-models/llama-3.2-1b-4bit --socket /tmp/kiln-test.sock
```

## mlx-c pin

Vendored at `crates/kiln-mlx/vendor/mlx-c` (git submodule). The pinned commit is
recorded in `docs/decisions/0001-mlx-c-pin.md`. `build.rs` builds it with cmake in
Release and links statically. **Do not bump the submodule.** If an mlx-c API you
need is missing at the pin, note it under `DECISION NEEDED:` and stop. After any
clean checkout, a stale build error usually means: `rm -rf target/mlx-c-build &&
cargo build -p kiln-mlx`.

## Golden-token parity harness

Purpose: prove Rust model implementations reproduce mlx-lm exactly.

- Fixtures: `tests/golden/<model>/<case>.json` → `{prompt, chat_template: bool,
  max_tokens, expected_token_ids, mlx_lm_version, weights_revision}`.
- Generate (only when told): `python scripts/gen-golden.py --model <id> --out tests/golden/<model>/`.
- Run: `cargo test -p kiln-models --test golden`. Pass = exact token-id match for
  every fixture. The relaxed bar in SPEC §11.2 applies only after a human-approved
  ADR names the specific model and reason.
- When adding a new architecture: fixtures first (from mlx-lm reference), then the
  implementation, then iterate until green. Never the other way around.

## Test models (pinned)

Fetched by `scripts/fetch-test-model.sh` into `~/.kiln/test-models/` at exact HF
revisions (revisions live in the script — treat as frozen):

- `mlx-community/Llama-3.2-1B-Instruct-4bit`
- `mlx-community/Qwen3-0.6B-4bit`            (also the draft model in spec-decode tests)
- `mlx-community/gemma-3-1b-it-4bit`
- one BF16 tiny model (see script) for the unquantized path

Tests reference them via the `KILN_TEST_MODELS` env var; never hardcode home paths.

## Repository map (orientation)

```
proto/kiln/v1/worker.proto   the contract (frozen semantics after Phase 2)
crates/kiln-mlx              unsafe FFI + safe wrappers (Array, Stream, errors)  [only unsafe crate]
crates/kiln-models           model impls + config.json parsing (llama, qwen*, gemma*)
crates/kiln-engine           batching loop, paged KV, radix prefix cache, sampler, spec decode, SSD tier
crates/kiln-worker           binary: gRPC server = kiln-engine + kiln-models
crates/kiln-gateway          binary: axum, API adapters, router, supervisor, metrics
crates/kiln-tokenize         tokenizer, chat templates (minijinja), tool-call parsers
crates/kiln-proto            prost codegen + shared enums
crates/kiln-jobs             download/quantize job runner (wraps `mlx_lm convert`)
crates/kiln-cli              binary `kiln`: serve/models/bench wrappers over the above
python/kiln_worker_py        mlx-lm fallback worker (same proto; no monkey-patching)
Formula/kiln.rb              Homebrew formula       packaging/  launchd plist template
tests/golden                 parity fixtures        tests/e2e   black-box HTTP suite
docs/SPEC.md                 the spec               docs/decisions/  ADRs (read-only)
```

## PROGRESS.md entry format (append-only)

```markdown
## [2026-07-02] Phase 4 / Task 4.2 — DONE | PARTIAL | BLOCKED
- What: <2–5 bullets, what changed and where>
- Decisions: <choices made within spec latitude, with one-line rationale>
- Deviations: <any departure from SPEC.md, or "none">
- Acceptance:
  ```
  <trimmed real output of the acceptance commands>
  ```
- Next: <the single next task per SPEC §12>
- DECISION NEEDED: <only if blocked; state options A/B with tradeoffs>
```

## Known sharp edges

- MLX is lazy: nothing computes until `eval`. Call `mlx_eval`/`async_eval` only at
  step boundaries on sampled-token arrays; reading data from an unevaluated array
  returns NULL from the `data_*` accessors.
- Incremental detokenization must handle byte-fallback and multi-token UTF-8; never
  emit partial code points over SSE. Use the streaming decoder in `kiln-tokenize`,
  fuzz-tested against full decode.
- `tokio` tasks must never block on Metal work; the engine thread is a dedicated
  OS thread communicating with the gRPC layer via channels.
- SSD slab files include a model fingerprint header; a fingerprint mismatch is a
  silent skip + counter increment, never an error surfaced to a request.
- macOS file descriptors: raise `RLIMIT_NOFILE` at worker start (mmap'd slabs + sockets).
