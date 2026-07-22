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

## [2026-07-03] Phase 0 / kiln.toml parsing with figment + kiln.toml.example — DONE
- What:
  - `kiln-gateway` gains a lib target with `config` module: full SPEC §10 schema (`[server]`, `[memory]`, `[defaults]`, `[[model]]` incl. `[model.speculative]`, `[auth]` + `[[auth.api_keys]]`), all defaults per spec.
  - Loading via figment: `Toml::file_exact` merged with `Env::prefixed("KILN_").split("__")` (e.g. `KILN_SERVER__PORT=9090`), then validation: non-zero port, `budget_fraction` in (0,1], power-of-two `block_size`, non-zero `prefill_chunk`, `max_batch_tokens >= block_size`, unique non-empty model ids/paths, `gamma >= 1`.
  - `kiln.toml.example` at repo root (valid-TOML rendering of the §10 sketch, with env-override docs).
  - 8 unit tests: example-file parse, empty-file defaults, env override, unrelated-`KILN_*`-var isolation, missing-file error, three validation rejections.
- Decisions:
  - Config module lives in kiln-gateway (its consumer; SPEC §4 defines no config crate). Can be split out later if workers/jobs need it.
  - Env overrides restricted to SERVER/MEMORY/DEFAULTS/AUTH prefixes so unrelated vars like `KILN_TEST_MODELS` (CLAUDE.md) cannot corrupt config keys; `[[model]]` entries are file-only.
  - Missing config file is a hard error (`file_exact`) — silently serving defaults on a typo'd `--config` path is a footgun.
  - `worker` defaults to `"auto"`; `~` in paths kept verbatim, expansion deferred to use sites (Phase 2).
  - New workspace deps, all MIT/Apache-2.0: figment (toml+env features; the SPEC §3-mandated config loader), serde (derive), thiserror (config error type). figment "test" feature (Jail) as dev-dependency only.
  - `allow(clippy::result_large_err)` scoped to the test module — figment's `Jail::expect_with` fixes the closure's error type; library code is unaffected.
- Deviations: none
- Acceptance:
  ```
  $ cargo test -p kiln-gateway
  test config::tests::duplicate_model_ids_are_rejected ... ok
  test config::tests::empty_file_yields_spec_defaults ... ok
  test config::tests::env_overrides_file_values ... ok
  test config::tests::missing_file_is_an_error ... ok
  test config::tests::non_power_of_two_block_size_is_rejected ... ok
  test config::tests::out_of_range_budget_fraction_is_rejected ... ok
  test config::tests::parses_the_committed_example_file ... ok
  test config::tests::unrelated_kiln_env_vars_are_ignored ... ok
  test result: ok. 8 passed; 0 failed

  $ cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings \
      && cargo clippy --workspace --all-targets --no-default-features -- -D warnings
      (clean)
  $ cargo build --workspace && cargo build --workspace --no-default-features
      Finished `dev` profile [unoptimized + debuginfo] target(s)   (both clean)
  $ cargo test --workspace     (9 passed total, 0 failed)
  $ ruff check python/ && ruff format --check python/
  All checks passed! / 1 file already formatted
  ```
- Next: Phase 0 complete. Phase 1 — Python worker end-to-end (SPEC §12), pending PM phase gate.

## [2026-07-03] Phase 1 / Python worker end-to-end — DONE
- What:
  - `kiln_worker_py` now implements the Phase 1 protocol surface over gRPC/UDS: GetInfo,
    Health, Submit (server-streamed), Cancel, plus Tokenize (required by
    CAPABILITY_TOKENIZER_OWNED). Drain/Stats remain UNIMPLEMENTED until their phases.
  - Sequential generation via mlx-lm `generate_step` on a single dedicated engine thread
    (all MLX ops confined there). raw_text and token_ids inputs; temp/top_p/top_k/min_p/seed
    plus repetition/frequency/presence penalties; stop strings via an incremental matcher
    that never streams matched text; stop token ids; ignore_eos; usage counts + timings +
    seed echo in `Finished`. Malformed input returns `Finished{ERROR}` — worker never dies.
  - `scripts/fetch-test-model.sh` created (SPEC §4 lists it; Phase 0 didn't build it):
    stdlib-only downloader, four models pinned at exact HF revisions, sha256-verified.
    All four fetched locally.
  - Test suite: 9 unit tests (stop matcher) + 19 integration tests driving the real pinned
    llama-3.2-1b-4bit over UDS (`KILN_TEST_MODELS`, skip-with-hint when absent).
  - CLI: `python -m kiln_worker_py --model <dir> --socket <path>`; RLIMIT_NOFILE raised at
    start; SIGTERM/SIGINT graceful shutdown; socket binds immediately, LOADING → READY.
- Decisions:
  - Test-model pins = repo HEADs as of 2026-07-03 (llama 0823137, qwen3 73e3e38,
    gemma3 2d44e83). BF16 tiny model slot: `mlx-community/SmolLM2-135M-Instruct`
    (422de22) — smallest unquantized candidate (~270 MB) and llama-arch, so it doubles as
    the unquantized path for Phase 3 golden fixtures.
  - `mlx-lm==0.31.3` pinned exactly, marker `sys_platform == 'darwin'` (mlx has no wheels
    for the linux CI lint job, which runs `uv sync`). `psutil` added for current-process
    RSS (`ru_maxrss` is peak, would mislabel the proto field). Both justified in commits.
  - Cancel guarantee: `generate_step` yields one evaluated token per step; the cancel flag
    is checked between yields, so overshoot ≤ the one pipelined step. Test asserts
    `steps_done - cancel_step <= 2` white-box on the live request object.
  - Every sampled token (incl. the finishing eos/stop token) is delivered in a TokenChunk,
    so `sum(chunk.token_ids) == completion_tokens`; tests assert the invariant. Text held
    back by the stop matcher rides on later/final chunks (proto allows lagging text).
  - `weights_fingerprint` = sha256(config.json bytes + weight file names/sizes): cheap
    identity, not a weights content hash (~1 GB hash per start not worth it for the
    fallback worker; the SSD tier scheme is the Rust worker's, Phase 5).
  - `echo_prompt` → INVALID_REQUEST for now (nothing sends it before Phase 2; honest
    rejection beats silent misbehavior). `logprobs_top_n` ignored (LOGPROBS not
    advertised). Grammar → GRAMMAR_UNSUPPORTED (capability not advertised).
  - Submit during LOADING aborts UNAVAILABLE (proto comment) rather than Finished{error}.
  - Generated-stub import shim (`_gen.py`) inserts `gen/` on sys.path once — the stubs'
    absolute `kiln.v1` imports require it; committed stub layout is CI-checked, unchanged.
- Deviations: none
- Acceptance:
  ```
  $ export KILN_TEST_MODELS=~/.kiln/test-models
  $ uv run --project python/kiln_worker_py pytest python/kiln_worker_py/tests -v
  tests/test_worker.py::test_submit_streams_tokens PASSED
  tests/test_worker.py::test_health_reports_memory_numbers PASSED
  tests/test_worker.py::test_cancel_mid_stream_stops_within_two_steps PASSED
  tests/test_worker.py::test_greedy_is_deterministic PASSED
  tests/test_worker.py::test_seeded_sampling_is_reproducible PASSED
  tests/test_worker.py::test_stop_string_ends_generation_and_is_excluded PASSED
  ... (28 passed in 6.70s)

  $ python -m kiln_worker_py --model ~/.kiln/test-models/llama-3.2-1b-4bit \
      --socket /tmp/kiln-test.sock &   # then, from a grpc client on that socket:
  state: WORKER_STATE_READY | rss MB: 194 | mlx_active MB: 695
  model: llama-3.2-1b-4bit llama q4_g64
  text: 'Paris.\nThe capital of France is Paris.\nThe capital of'
  finish: FINISH_REASON_LENGTH | prompt: 6 | completion: 12

  $ cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
      (clean; Rust untouched this task)
  $ ruff check python/ && ruff format --check python/
  All checks passed! / 11 files already formatted
  ```
- Next: Phase 2 — Gateway v0 (axum app, supervisor spawning this worker, OpenAI chat
  completions, /v1/models, auth, metrics), pending PM phase gate. Note for PM: CI does not
  yet run the python worker pytest suite (needs test-model caching on the macos runner) —
  worth adding when the cache strategy is decided.

## [2026-07-03] Phase 1 / Follow-up — python worker suite in CI + ADR 0001 addendum — DONE
- What:
  - `test-macos` CI job now runs `pytest python/kiln_worker_py/tests` against the pinned
    test models, with `actions/cache` on `~/.kiln/test-models` keyed by
    `hashFiles('scripts/fetch-test-model.sh')` — the pins (revisions + sha256s) live in
    that script, so a pin bump changes the key and re-fetches; otherwise the ~2 GB model
    download is skipped (`Fetch pinned test models` step is `if: cache-hit != 'true'`).
  - Appended an addendum to `docs/decisions/0001-mlx-c-pin.md` at explicit PM instruction
    (that directory is otherwise agent read-only): `mlx-lm==0.31.3` resolves to
    **mlx.core 0.31.2** in the worker venv, one patch version ahead of the **MLX v0.31.1**
    that vendored mlx-c v0.6.0 builds for kiln-mlx. Verified both sides
    (`import mlx.core` in the venv; `GIT_TAG v0.31.1` in mlx-c's CMakeLists). Recorded
    only — no pins changed.
- Decisions:
  - Cache key hashes the whole fetch script, not just the pin lines: any script change
    invalidates the cache (conservative; worst case one re-download).
  - This session's push also published the five Phase 1 commits (they were local-only).
- Deviations: none
- Acceptance:
  ```
  push 41563f9 → run 28671645169: all jobs green, test-macos 4m14s (cache MISS)
    Cache not found for input keys: test-models-e57549323c60...
    Fetch pinned test models: ==> llama-3.2-1b-4bit ... ==> smollm2-135m-bf16 (all 4 fetched)
    Python worker tests: ============ 28 passed in 12.14s ============
    Post Cache: Cache saved with key: test-models-e57549323c60...

  push cd3aa22 → run 28671908976: all jobs green, test-macos 2m40s (cache HIT)
    Cache restored from key: test-models-e57549323c60...
    - Fetch pinned test models        (skipped)
    Python worker tests: ============ 28 passed in 15.93s ============

  $ uv run --project python/kiln_worker_py python -c "import mlx.core as mx; print(mx.__version__)"
  0.31.2
  ```
- Next: Phase 2 — Gateway v0 (SPEC §12), pending PM phase gate.

## [2026-07-03] Phase 1 / Investigation — mlx-c tag for core MLX 0.31.2 — DONE (decision left to PM)
- What: investigated (read-only; no pins changed, submodule untouched) whether an mlx-c
  tagged release builds core MLX v0.31.2, to close the one-patch drift recorded in the
  ADR 0001 addendum (python worker on 0.31.2 vs kiln-mlx on 0.31.1).
- Findings (all via GitHub/PyPI API, 2026-07-03):
  - **No mlx-c tag pins core 0.31.2.** Newest tag is v0.6.0 (2026-03-20, our pin);
    its CMakeLists pins `GIT_TAG v0.31.1` (verified at the tag). No CHANGELOG file,
    no GitHub releases — tags only. Hypothesized option A is unavailable today.
  - **mlx-c lags core by design but the work exists upstream:** main is 4 commits ahead
    of v0.6.0 — `#110` distributed_group_free, `#111` gguf, `#112` graph export, and
    `#114` "regenerate bindings for MLX 0.31.2" (2026-04-24, two days after core
    v0.31.2 shipped on 2026-04-22). Untagged for >2 months since.
  - **mlx-lm==0.31.3 hard-requires `mlx>=0.31.2`** on Darwin — it cannot run on core
    0.31.1. But **mlx-lm==0.31.2 requires only `mlx>=0.30.4`**, so pinning
    `mlx-lm==0.31.2` plus an explicit `mlx==0.31.1` aligns the python worker with
    kiln-mlx's core exactly. What 0.31.3 adds over 0.31.2: batch-KV-cache fixes,
    server/tool-call fixes, thread-local generation stream — none load-bearing for our
    sequential Phase-1 worker.
  - **The 0.31.1→0.31.2 core delta is NOT parity-benign:** it includes Metal split-K
    quantized matmul for small M (mlx#3120) — a reduction-order change in exactly the
    4-bit matmul path our pinned test models hit — plus fp16/bf16 sort-NaN fixes
    (mlx#3269). Cross-version bitwise logit divergence on 4-bit models is plausible to
    likely, so "generate goldens on 0.31.2, verify Rust on 0.31.1" risks failing the
    SPEC §11.2 exact bar for reasons that are neither side's bug.
- Deviations: none
- DECISION NEEDED: how to reconcile MLX core versions before Phase 3 golden fixtures
  are generated (fixtures record mlx_lm_version + expected ids; generating them before
  deciding bakes the drift in):
  - **Option A (unavailable):** bump submodule to an mlx-c tag pinning core 0.31.2 —
    no such tag exists as of today.
  - **Option B1 (agent's recommendation):** pin python worker down to core 0.31.1:
    `mlx-lm==0.31.2` + explicit `mlx==0.31.1` in pyproject.toml. Pros: both workers on
    identical core; goldens generated from the mlx-lm reference then target exactly the
    bytes kiln-mlx builds; preserves the exact-parity bar; smallest change (two dep
    lines + lockfile + ADR addendum update + suite re-run). Cons: forgoes mlx-lm
    0.31.3's fixes (assessed irrelevant to this worker today); a later un-pin needs a
    golden re-run anyway.
  - **Option B2 (not recommended):** keep 0.31.3/0.31.2 and accept drift via a dated
    ADR invoking SPEC §11.2's relaxed bar. Cons: mlx#3120 makes divergence likely on
    precisely the 4-bit fixture models, so the relaxed bar could become the de-facto
    norm across all models — permanently weakening the keystone test.
  - **Option C1 (durable fix, timing not ours):** when mlx-c tags its next release
    (main already regenerates for 0.31.2), bump the submodule via the ADR 0001
    quarterly process (bump → rebuild → full golden re-run → human approval) and keep
    mlx-lm current. Combine: B1 now, C1 when the tag lands.
  - **Option C2/C3 (rejected):** pin submodule to untagged main (4 unreviewed commits;
    ADR pins tags deliberately), or force core v0.31.2 under v0.6.0 bindings via a
    FetchContent override (#114 exists precisely because bindings needed regenerating —
    that's the "silent semantic change under the bindings" ADR 0001 guards against).
- Next: PM decision on the above; then Phase 2 — Gateway v0 (SPEC §12), pending phase gate.

## [2026-07-03] Phase 1 / Follow-up — core MLX alignment via option B1 — DONE
- What:
  - Implements the PM-approved option B1 from this date's DECISION NEEDED entry (commit
    061e399): `kiln_worker_py` now pins `mlx-lm==0.31.2` + explicit `mlx==0.31.1`, so
    both workers share core MLX v0.31.1 — the version vendored mlx-c v0.6.0 builds.
    **The DECISION NEEDED flag from commit 061e399 is CLOSED** (this session).
  - Lockfile delta is only mlx/mlx-metal/mlx-lm (re-locked conservatively;
    transformers 5.12.1 etc. unchanged).
  - ADR 0001 gains a follow-up note (appended after the drift addendum, at explicit PM
    instruction) recording the resolution and the standing C1 plan: when mlx-c tags its
    next release, bump submodule + return mlx-lm/mlx to current as one change with a
    full golden re-run; revisit pins before any golden-fixture regeneration.
- Decisions:
  - Verified before downgrading that no worker code depends on 0.31.3-only mlx-lm APIs:
    make_sampler / make_logits_processors (incl. presence/frequency penalties) /
    generate_step (evaluated-token yield) / fresh detokenizer / eos_token_ids are
    identical at 0.31.2. Nothing to report as blocked.
  - `mlx==0.31.1` added as an explicit direct dependency (already transitive): pins the
    exact core so the resolver can't drift it independently of the submodule pin.
- Deviations: none
- Acceptance:
  ```
  $ uv sync --project python/kiln_worker_py && uv run --project python/kiln_worker_py \
      python -c "import mlx.core as mx; print(mx.__version__)"
  0.31.1

  $ KILN_TEST_MODELS=~/.kiln/test-models uv run --project python/kiln_worker_py \
      pytest python/kiln_worker_py/tests -v
  ============ 28 passed in 8.34s ============   (all Submit/Cancel/Health/Tokenize green)

  push 568369b → run 28691082336: all jobs green, test-macos 1m35s
    uv sync:  + mlx==0.31.1 / + mlx-lm==0.31.2
    Cache restored from key: test-models-e57549323c60...   (- Fetch pinned: skipped)
    Python worker tests: ============ 28 passed in 11.03s ============
  ```
- Next: Phase 2 — Gateway v0 (SPEC §12), pending PM phase gate.

## [2026-07-04] Phase 2 / Gateway v0 (tracer bullet complete) — DONE
- What:
  - `kiln-tokenize`: chat templating via minijinja — model-dir loading
    (chat_template.jinja else tokenizer_config.json, incl. named-list form),
    bos/eos extraction (string + AddedToken forms), pycompat callback,
    `raise_exception`/`strftime_now`, lenient undefined; sha256 source hash
    mirroring the worker's `chat_template_hash` scheme. Tests include the
    vendored template of the pinned Llama-3.2-1B revision.
  - `kiln-gateway` v0 (SPEC §8): model registry from kiln.toml (path
    resolution, `$runtime_dir/worker-<hash12>.sock` per §3 with a 104-byte
    macOS sun_path guard); lazy tonic-over-UDS channels that survive worker
    restarts; supervisor per §2.2 — spawns the python worker, forwards its
    output into gateway logs, 1s Health poll on the frozen proto (2s RPC
    deadline / 3s missed-deadline budget), child-exit watch, exponential
    backoff restart (500ms doubling, cap 10s, max 3; 60s-stable Ready resets
    the counter, then Failed until manual reset), GetInfo cache + template
    hash verification.
  - OpenAI surface: POST /v1/chat/completions (stream SSE + non-stream) per
    §8.2 — validate → render template → worker Tokenize → Submit(token_ids) →
    TokenEvent translation with usage/finish_reason; GET /v1/models;
    /healthz; /readyz (per-model worker states); Prometheus GET /metrics
    (http/chat/token/restart counters, worker_up gauge, latency histogram);
    argon2 API-key auth with sha256-keyed verify cache + `hash-key`
    subcommand; structured JSON logs with per-request UUIDv7 ids (echoed as
    x-request-id, reused as worker request_id).
  - `tests/e2e`: own uv project; full-stack fixture (gateway builds itself,
    throwaway config, /readyz gate); 8 tests driving the real `openai` SDK,
    incl. kill -9 crash/recovery and metrics. CI macos job runs it; ruff
    scope widened to tests/e2e; CLAUDE.md commands updated.
- Decisions:
  - New config knob `server.python_worker_argv` (§10 doesn't say how the
    gateway finds the worker); default is the CLAUDE.md uv-run invocation,
    checkout-relative, overridable for packaged installs.
  - Prompt path tokenizes via the worker's Tokenize RPC with
    add_special_tokens=false rather than submitting raw_text: the rendered
    template already contains BOS, and worker-side encode would add a second
    one (verified against the Llama-3.2 tokenizer/template).
  - No usable API-key hashes ⇒ auth disabled with a loud warning (localhost
    bind default per §8.1); empty/malformed hash entries are skipped.
  - Default max_tokens when the client omits it: remaining context
    (max_context_len − prompt), 1024 fallback if the worker reported no ctx.
  - Unsupported OpenAI features (n>1, logprobs, tools, response_format other
    than text, tool/function roles) are explicit 400s, not silent drops.
  - Shutdown kills workers with SIGKILL; graceful Drain+SIGTERM arrives with
    eviction (Phase 9). worker="rust" is a startup error until Phase 3;
    "auto" resolves to python.
- Deviations:
  - §8.3 per-key rate limits (rpm/tpm) and TTFT/total timeouts are NOT
    implemented — absent from the Phase 2 task list; config keys parse but
    are unenforced. Flagging here so a later phase picks them up explicitly.
  - Admin endpoints, /v1/completions, /v1/embeddings, /v1/messages: later
    phases per §12 (not attempted).
- Acceptance:
  ```
  $ uv run --project tests/e2e pytest tests/e2e -v        (real model, full stack)
  test_chat.py::test_models_list PASSED
  test_chat.py::test_auth_rejects_bad_and_missing_keys PASSED
  test_chat.py::test_chat_completion_non_streaming PASSED
  test_chat.py::test_greedy_is_reproducible PASSED
  test_chat.py::test_chat_completion_streaming PASSED
  test_chat.py::test_validation_errors_are_openai_shaped PASSED
  test_crash_recovery.py::test_kill9_mid_request_yields_502_then_recovers PASSED
  test_metrics.py::test_request_counters_increment PASSED
  ============ 8 passed in 14.52s ============

  $ (acceptance demo: openai SDK vs live stack)
  == non-streaming chat (temperature=0):
     content       = 'A kiln is a large, heated oven or furnace used for ...'
     finish_reason = stop, usage = 46+28=74
  == streaming chat:
     streamed 9 deltas -> '1\n2\n3\n4\n5'   (stream usage = 46+10)
  == kill -9 mid-request:
     killed worker pid(s) [4126, 4127] mid-generation
     client got HTTP 502: {'error': {'code': 'worker_crashed', 'type':
       'server_error', 'message': 'worker RPC failed: Unknown error', ...}}
  == auto-restart:
     next request succeeded 2.2s after crash: "I'm ready."
  == /metrics (selected):
     kiln_chat_completions_total{...,outcome="ok"} 3
     kiln_http_requests_total{...,path="/v1/chat/completions",status="200"} 3
     kiln_http_requests_total{...,path="/v1/chat/completions",status="502"} 2
     kiln_worker_restarts_total{model="llama-3.2-1b-4bit"} 1
     kiln_worker_up{model="llama-3.2-1b-4bit"} 1

  $ cargo test --workspace     → all green (22 gateway, 12 tokenize, 1 mlx)
  $ pytest python/kiln_worker_py/tests   → 28 passed
  $ cargo fmt --check / clippy -D warnings / ruff check+format → clean
  ```
- Next: Phase 3 — Rust worker v0: single-request Llama (SPEC §12), pending
  PM phase gate.

## [2026-07-04] Phase 2 / Follow-up — worker process-group kill — DONE
- What: post-acceptance sweep found the supervisor orphaning python workers:
  it killed its direct child (the `uv run` wrapper) but not the model-loaded
  python process underneath, leaking ~1 GB RSS per shutdown/recycle. Workers
  now run in their own process group (pgid == spawned pid) and every exit
  path SIGKILLs the group via /bin/kill (libc::kill would need unsafe, which
  is confined to kiln-mlx). e2e teardown now fails the suite on any surviving
  worker process — regression guard.
- Decisions: /bin/kill subprocess over adding a libc/nix dependency or
  relaxing the unsafe-code rule; group-kill also runs in the
  child-exited-on-its-own arms, where only the wrapper may have died.
- Deviations: none
- Acceptance:
  ```
  $ uv run --project tests/e2e pytest tests/e2e -v
  ============ 8 passed in 9.97s ============
  $ pgrep -fl kiln_worker_py   (after suite)
  (none)
  $ cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings
  clean
  ```
- Next: Phase 3 — Rust worker v0: single-request Llama (SPEC §12), pending
  PM phase gate.

## [2026-07-04] Phase 3 / Rust worker v0: single-request Llama — DONE
- What:
  - kiln-tokenize: crate-level BOS/special-token contract doc (written
    first, per task instruction); tokenizer.json loading + encode/decode
    over the HF `tokenizers` crate; `ChatTemplate::render_with` for pinned
    template vars (date_string).
  - scripts/gen-golden.py + tests/golden/llama-3.2-1b-4bit/ (5 fixtures):
    generated AFTER verifying the reference stack is mlx.core 0.31.1 /
    mlx-lm 0.31.2 (ADR 0001 follow-up B1 state — the same core MLX the
    vendored mlx-c v0.6.0 builds). Script hard-refuses any other mlx.core.
    Fixtures pin date_string="26 Jul 2024" (Llama templates interpolate
    strftime_now() otherwise) and compare exactly max_tokens greedy tokens,
    no EOS stop.
  - kiln-mlx: bindgen-generated sys bindings from the pinned headers (SPEC
    §7.1); safe Array/Stream RAII wrappers (!Send/!Sync, Clone via
    mlx_array_set); recording error handler replacing mlx-c's exit()-ing
    default; debug live-object leak counter; ops/fast/random/memory wrapper
    surface incl. quantized_matmul/dequantize, fused rms_norm/rope/SDPA;
    mmap io module (memmap2 confined here — unsafe stays in kiln-mlx).
  - kiln-models: Llama config.json parsing (rope_scaling default/linear/
    llama3), mmap'd safetensors loader (sharded or single-file), Llama
    forward ported op-for-op from mlx_lm.models.llama (quantized linear/
    embedding, tied lm_head, Llama3RoPE freqs in the same f32 MLX graph),
    contiguous per-layer KV cache with mlx-lm's 256-step growth,
    generate loop with chunked prefill + async_eval pipelining.
  - kiln-engine: sampler (§6.6) — greedy argmax; temp/top-p/top-k/min-p +
    seeded per-request categorical (key chain from seed, no global RNG);
    repetition/presence/frequency penalties as a logits-processor fn.
  - Tests: golden harness (exact token-id equality, revision-checked),
    1k-iteration leak gate, sampler behavior suite, tokenizer BOS
    single/double proof, wrapper FFI discipline suite; release bench
    example for the tok/s comparison.
- Decisions:
  - bindgen as a kiln-mlx build-dependency (SPEC §7.1 names bindgen for
    sys.rs); generation is metal-gated so the Linux compile-check never
    needs libclang.
  - mmap lives in kiln-mlx::io because Mmap::map is unsafe and unsafe is
    confined to kiln-mlx; kiln-models consumes a safe &[u8].
  - Fixtures store generated-only token ids and both sides re-derive
    prompt ids (template render + encode) — tokenization parity is tested
    together with model parity.
  - A module is quantized iff its .scales tensor exists (mlx-lm
    class_predicate semantics); dense f16/bf16 Linear/Embedding also
    implemented (exercised fully in Phase 6's dtype matrix).
  - Penalties are a separate apply_penalties on raw logits (mlx-lm
    logits-processor semantics), wired ahead of the sampler when the
    Phase 4 per-request pipeline exists; generate() takes a sampler
    callback on logprobs.
- Deviations:
  - Contiguous KvCache lives in kiln-models, not kiln-engine — it is the
    explicitly temporary v0 cache the Phase 4 paged block manager
    replaces; the sampler is in kiln-engine per the repo map.
  - Per-module quantization overrides in config.json (mixed-precision
    dicts) are not parsed; uniform group_size/bits only. Such checkpoints
    fail loudly at load. Revisit with the Phase 6 quantization matrix.
  - SPEC §12 Phase 3 also lists gateway-side pieces (token_ids submit
    path via the worker protocol, incremental detok in the gateway) and
    the kiln-worker gRPC binary — not in this task's scope per the prompt;
    they are the Phase 3 remainder (see Next).
- Acceptance:
  ```
  $ KILN_TEST_MODELS=~/.kiln/test-models cargo test -p kiln-models --test golden -- --nocapture
  golden chat-basic:        48 prompt tokens,  64 generated — exact match
  golden chat-code:         47 prompt tokens, 128 generated — exact match
  golden chat-multibyte:    51 prompt tokens,  64 generated — exact match
  golden raw-continuation:   6 prompt tokens,  64 generated — exact match
  golden raw-long-prefill: 249 prompt tokens,  64 generated — exact match
  test result: ok. 1 passed (all 5 fixtures exact)

  $ single-stream decode, same model/prompt/256 tokens (3 runs each):
  mlx-lm generate: 98.2 / 100.0 / 100.2 tok/s   (median 100.0)
  kiln  (release): 98.3 /  96.6 /  99.4 tok/s   (median 98.3)
  -> -1.7% median (-0.8% best-vs-best), within the -10% bar.
     Peak memory: mlx-lm 0.742 GB, kiln 0.752 GB.

  $ KILN_TEST_MODELS=~/.kiln/test-models cargo test -p kiln-models --test leak -- --nocapture
  leak gate: 1000 decode iterations at 94.4 tok/s, live objects during run: 389
  leak gate: mlx active memory 0B -> 0B, live objects 0 -> 0
  test result: ok. 1 passed

  $ cargo test --workspace          -> all green (incl. 22 gateway, 16 tokenize,
                                       2 mlx, 1 engine, 5 models targets)
  $ cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings -> clean
  $ cargo build --workspace --no-default-features -> clean (linux compile-check path)
  $ ruff check/format python/ tests/e2e scripts/gen-golden.py -> clean
  $ pytest python/kiln_worker_py/tests -> 9 passed, 19 skipped (no-GPU skips unchanged)
  ```
- Next: Phase 3 remainder — kiln-worker gRPC binary (Submit/token_ids path
  over UDS, GetInfo/Health from the loaded model) + gateway routing to the
  Rust worker with gateway-side tokenization and incremental detok; then
  Phase 4 per SPEC §12.

## [2026-07-04] Phase 3 / Follow-up — per-module quantization overrides — DONE
- What: closeout review found the "Deviations" claim in the previous entry
  ("mixed-precision checkpoints fail loudly at load") was WRONG and untested:
  Quantization's derived Deserialize silently ignored unknown keys, so a
  config.json with per-module override entries loaded as plain uniform
  quantization and would have failed at the first forward pass with an
  opaque MLX shape error — or quietly misread a module. Verified by probe
  (loader returned Ok on an override-carrying config), then fixed:
  LlamaConfig::from_json_str now validates the raw quantization object —
  per-module keys are rejected at load with UnsupportedQuantization naming
  the offending modules and the python-worker route; non-affine modes
  rejected by name; "mode": "affine" accepted. Unit tests cover both
  override forms (dict and false), the error variant, message contents,
  and that the uniform golden-model block still loads.
- Decisions: manual raw-JSON validation over serde deny_unknown_fields —
  the latter would return a generic Parse error, reject the legitimate
  "mode": "affine" key, and not name the route-to-python remedy.
- Deviations: none (this corrects the record of the previous entry).
- Acceptance:
  ```
  $ cargo test -p kiln-models   (KILN_TEST_MODELS set)
  test config::tests::per_module_quantization_overrides_are_a_named_load_error ... ok
  test config::tests::quantization_mode_affine_accepted_others_named ... ok
  test config::tests::uniform_quantization_still_loads ... ok
  (+ 3 prior config tests, golden parity, 1k-iteration leak gate: all ok)
  $ cargo fmt --check && cargo clippy -p kiln-models --all-targets -- -D warnings
  clean
  ```
- Next: unchanged — Phase 3 remainder (kiln-worker gRPC binary + gateway
  token_ids path + incremental detok), pending PM phase gate.

## [2026-07-04] Phase 3 / Closeout — Rust worker wired end-to-end — DONE
- What:
  - kiln-tokenize: StreamingDecoder (TGI-style two-offset incremental
    detokenization; U+FFFD holdback so partial code points never reach SSE;
    fuzzed against full decode — 300 random id sequences + ZWJ-emoji/CJK
    corpora at multiple chunk schedules) and StopStringMatcher (port of the
    python worker's stops.py; gateway-side for rust workers).
  - kiln-models: generate_with (logits-processor slot pre-normalization with
    mlx-lm history semantics + per-token callback whose false return stops
    generation — the ≤2-step cancel bound); eos_token_ids() from config.json
    (int|list — same set mlx-lm stops on). Golden parity + leak gate re-run
    green after the refactor.
  - kiln-mlx: os module (RLIMIT_NOFILE raise per the CLAUDE.md sharp edge;
    process RSS via proc_pidinfo for MemoryReport). New dep libc, confined
    to the unsafe crate.
  - kiln-worker: the Rust worker binary — tonic gRPC over UDS behind the
    frozen proto, same argv contract as the python worker; one engine
    thread owns model+Stream (!Send), single request at a time;
    GetInfo/Health (fingerprint + template-hash schemes byte-identical to
    modelinfo.py), Submit (python-parity validation as in-band
    Finished{ERROR}; bare-id TokenChunks — text empty by design; stop token
    counted but never chunked), Cancel (flag between steps), Drain/Stats/
    Tokenize UNIMPLEMENTED per phase/design. UNHEALTHY-with-detail on load
    failure so the supervisor recycles.
  - kiln-gateway: worker="rust" spawns kiln-worker (rust_worker_argv,
    default = sibling binary of the gateway); startup validation of rust-
    servable models via kiln-models config parsing (no-MLX dep); gateway
    tokenizes locally (BOS contract) and detokenizes incrementally;
    stop strings matched gateway-side with Cancel-on-match and honest
    usage (drain to Finished); auto still resolves to python until the
    Phase 6 routing matrix.
  - tests/e2e: stack fixture parametrized over both workers (same Phase 2
    suite gates the rust worker forever); new cross-worker parity tests.
- Decisions:
  - The rust worker REJECTS stop_strings/raw_text (INVALID_REQUEST) instead
    of silently ignoring them — the gateway owns text; a misconfigured
    caller stays loud.
  - The stop-triggering token is counted in usage but not chunked: the
    gateway decodes chunks verbatim and stop text is excluded by contract.
  - Penalty-enabled requests forfeit one step of async_eval pipelining
    (host-side penalty window needs the token value); the default path is
    unchanged.
  - Gateway-side stop-string match ends the request as finish_reason
    "stop" regardless of the worker's terminal event (usually our own
    CANCELLED).
- Deviations: none against the task; worker="auto"→python note stands
  (Phase 6).
- Acceptance:
  ```
  $ uv run --project tests/e2e pytest tests/e2e -v     (full stack, real model)
  test_chat.py (6 tests)                  [python] PASSED   [rust] PASSED
  test_crash_recovery.py::kill9...        [python] PASSED   [rust] PASSED
  test_metrics.py::counters               [python] PASSED   [rust] PASSED
  test_cross_worker_parity.py::greedy_outputs_identical_across_workers PASSED
  test_cross_worker_parity.py::streaming_text_identical_across_workers PASSED
  test_cross_worker_parity.py::stop_strings_work_on_both_workers       PASSED
  ============ 19 passed in 36.69s ============

  $ side-by-side, same prompt, temperature=0 (one gateway, both workers):
  llama-py [stop] usage=48+39: 'A kiln is a type of furnace or oven used for
    various industrial and commercial applications, such as firing ceramics,
    baking materials, and testing materials, to achieve specific physical or
    chemical properties.'
  llama-rs [stop] usage=48+39: <byte-identical text, identical usage>

  $ standalone worker smoke (python grpc client over UDS): READY health with
    memory report (rss 0.72GB), GetInfo (q4_g64, template hash matches),
    48-token greedy chat streamed at 101.5 tok/s decode, STOP on <|eot_id|>,
    in-band INVALID_REQUEST for malformed input, Tokenize UNIMPLEMENTED.

  $ cargo test --workspace            -> 22 test targets, all ok
    (incl. golden parity exact + 1k-iteration leak gate re-run post-refactor)
  $ cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings -> clean
  $ cargo build --workspace --no-default-features -> clean
  $ ruff check/format python/ tests/e2e scripts/gen-golden.py -> clean
  $ pytest python/kiln_worker_py/tests -> 28 passed
  ```
- Next: Phase 4 — paged KV + continuous batching (kiln-engine core), per
  SPEC §12, pending PM phase gate.

## [2026-07-04] Phase 3 / Follow-up — stop-string usage parity — DONE
- What: closeout verification (PM request) of two properties of the
  gateway-side stop-string path.
  1. completion_tokens parity on stop-string matches: NOT identical before
     this change — the rust path passed the worker's Finished count through,
     which includes cancel-overshoot tokens (reproduced: python 7 vs rust 8
     for identical text '1\n2\n3\n', stop=["4"]). The gateway now freezes
     the token count at the chunk whose text completed the stop string and
     overrides completion_tokens on matched requests — the client-visible
     completion length, equal to the tokenizer-owning worker's count. The
     cross-worker parity test now asserts full (content, finish_reason,
     prompt_tokens, completion_tokens) equality for greedy AND stop-string
     cases.
  2. finish_reason precedence is now documented in kiln-gateway chat.rs
     module docs (not just ledger prose): a gateway-side match always
     reports "stop" regardless of the worker's terminal event
     (CANCELLED/STOP/LENGTH/ERROR). The LENGTH-as-"stop" case is accepted
     and correct, not a race bug: the client's completion was truncated at
     the match; the worker's reason describes an uncancelled continuation
     the client never saw; the python worker reports STOP at the same point
     for the identical request. Usage guarantee stated alongside.
- Decisions: usage overridden gateway-side rather than trying to stop the
  worker synchronously — the ≤2-step cancel bound makes worker-side counts
  inherently overshooting; the pipeline is the authority on what the client
  saw.
- Deviations: none.
- Acceptance:
  ```
  $ pre-fix (chat.rs stashed), strengthened test:
  AssertionError: stop-string usage/text divergence:
      python: ('1\n2\n3\n', 'stop', 48, 7)
      rust:   ('1\n2\n3\n', 'stop', 48, 8)
  $ post-fix:
  test_greedy_outputs_identical_across_workers PASSED
  test_streaming_text_identical_across_workers PASSED
  test_stop_strings_work_on_both_workers PASSED
  ============ 3 passed in 9.53s ============
  $ cargo fmt --check / clippy -D warnings / ruff -> clean
  ```
- Next: unchanged — Phase 4 per SPEC §12, pending PM phase gate.

## [2026-07-04] Phase 3 / Follow-up — Linux lint CI fix — DONE
- What: PR #1's lint job (clippy --all-targets --no-default-features,
  Linux, no submodules) failed: kiln-models' dev-dependency forced
  kiln-engine/metal unconditionally, feature-unifying MLX into a graph
  that cannot build there. The metal enable now forwards through
  kiln-models' own metal feature; dev-dep default-features off.
- Acceptance: exact CI lint command locally → zero mlx-c builds, clippy
  clean; full-featured clippy + golden parity + leak gate still green.
- Next: unchanged — Phase 4 pending PM phase gate.

## [2026-07-04] Phase 3 / Follow-up — unsafe-surface review findings — DONE
- What: pre-merge manual review of the kiln-mlx unsafe surface (SPEC §14 /
  CLAUDE.md: reviewed regardless of green tests) found the safe host-read
  APIs unsound: MLX's item<T>/data<T> are raw reinterpreting accessors
  (verified in the vendored sources), so wrong-dtype or non-contiguous
  reads through the safe wrappers were UB (incl. out-of-bounds reads for
  wider-T dtype mismatches). Latent only — every call site reads fresh
  typed contiguous op outputs. Fixed with dtype guards on item_*/data_*,
  a strides-based row-contiguity guard on data_*, and an ops::contiguous
  escape hatch; wrapper tests pin all failure modes. Also re-audited the
  metal feature graph after the CI fix: cargo tree shows [] for every kiln
  crate under --no-default-features (dev-deps included) and no dependency
  edge anywhere enables metal unconditionally; the Linux lint job is the
  standing tripwire.
- Deviations: none.
- Acceptance:
  ```
  $ cargo test -p kiln-mlx --test wrappers -> ok (incl. new guard cases)
  $ cargo clippy --workspace --all-targets [-D warnings] and
    --no-default-features variant -> both clean
  $ cargo test --workspace -> 22 targets ok (golden parity + leak gate green)
  ```
- Next: unchanged — Phase 4 pending PM phase gate; PR #1 merge is the PM's
  call after this review.

## [2026-07-04] Phase 4 / part 1/4 — block manager (§6.3) — DONE
- What:
  - crates/kiln-engine/src/block.rs: paged-KV block manager as a standalone
    unit — fixed block pool (power-of-two block size) with LIFO free list,
    per-block refcounts, allocate/retain/release, and copy_on_write: sole
    owner → write in place; shared → caller's ownership moves to a fresh
    block plus a (src, dst) copy instruction; a shared block is never
    handed out writable. BlockTable maps request token positions → block
    indices: append_tokens (allocates fresh blocks, COWs a shared partial
    tail, atomic on OutOfBlocks), fork (retain-all with rollback on
    failure), explicit release. No MLX types, no model/worker wiring; the
    module is feature-ungated so the Linux --no-default-features check
    compiles and lints it.
  - tests: inline unit tests (constructor validation, allocation order and
    exhaustion, refcount lifecycle, foreign-id rejection, COW edge cases,
    table append/fork/release incl. atomicity); model-based proptest suite
    in tests/block_props.rs — a manager model and a table model run random
    op sequences against shadow state, asserting after every op: a block is
    never double-issued while owned, refcounts equal owner counts and hit 0
    exactly at the last release, and COW never mutates a shared block
    (every owner's content view survives sibling writes; read-back through
    each table stays exact). 4096-case extended variants are #[ignore]d per
    the CLAUDE.md `-- --ignored` convention for slow property tests.
  - deps: proptest 1 added as a workspace-managed dev-dependency of
    kiln-engine (rationale in commit message).
- Decisions:
  - The manager holds no KV data: COW returns a copy instruction for the
    engine's physical per-layer pools (arriving later in Phase 4), keeping
    this unit MLX-free and exhaustively testable.
  - BlockTable does not auto-release on Drop (it holds no manager handle);
    request teardown owns the release. #[must_use] flags accidental drops.
  - Refcount 0 puts a block straight on the free list; §6.2's "return to
    prefix cache, not freed" is Phase 5's radix tree holding its own
    refcount — no cached/evictable state belongs in the manager.
  - LIFO free list, deterministic issue order — block placement is
    reproducible run-to-run, in line with the greedy determinism gate.
- Deviations: none.
- Acceptance:
  ```
  $ cargo test -p kiln-engine                       (default features)
  block::tests ......... 11 passed (unit suite)
  block_props .......... manager_model ok, table_model ok; 2 passed,
                         2 ignored (extended) in 1.19s
  sampler_behavior ..... ok (unchanged)
  $ cargo test -p kiln-engine --test block_props -- --ignored
  manager_model_extended ok, table_model_extended ok   (4096 cases each,
  2 passed in 17.99s)
  $ cargo test -p kiln-engine --no-default-features -> same block suite green
  $ cargo fmt --check -> clean
  $ cargo clippy --workspace --all-targets -- -D warnings -> clean
  $ cargo clippy --workspace --all-targets --no-default-features -- -D warnings -> clean
  ```
- Next: Phase 4 part 2/4 (await prompt) — per SPEC §12 presumably the
  gather-based paged attention v0 (§7.4) wiring block tables to per-layer
  K/V pools.

## [2026-07-04] Phase 4 / part 2/4 — batching loop + paged attention v0 (§6.2, §7.4) — DONE
- What:
  - crates/kiln-engine/src/paged.rs: PagedKv — per-layer K/V pools as
    preallocated MLX arrays `[num_blocks, kv_heads, block_size, head_dim]`
    (lazy dtype on first write), functional slice_update writes per
    WriteRun, gather-based paged attention v0 (take → reshape → slice into
    a contiguous per-request `[1, H, T, D]` view whose stride pattern
    matches what the Phase-3 contiguous cache fed fused SDPA), COW block
    copies, state()/byte accounting, fault reset.
  - crates/kiln-engine/src/step.rs: the engine↔model step contract —
    StepBatch/SeqStep (len, offset, sample flag, gather blocks, write
    runs) and the StepModel trait (blanket impl for &M).
  - crates/kiln-engine/src/engine.rs: the §6.2 loop — per iteration:
    cancel sweep, admit while budget allows (one prefill in flight at a
    time), build (block-table appends + write-run derivation, OutOfBlocks
    → in-band request error), forward, per-request penalties → logprobs →
    sample, single eval at the step boundary, emit/stop-check/release,
    clear_cache maintenance (after chunks + every 256 iterations).
    Chunked prefill mirrors mlx-lm/Phase-3 exactly: chunks of
    prefill_chunk (default 2048) over prompt[..n-1], last prompt token fed
    by the first sampled step. Step-level MLX faults fail all in-flight
    requests in-band and rebuild manager+pools; the engine stays serving.
  - crates/kiln-models: llama.rs implements StepModel (forward_step per
    Attention/Block; lm_head only over sampled positions — prefill chunks
    skip it, as the Phase-3 dead-graph did numerically) + kv_dims();
    kiln-engine promoted from dev-dependency to dependency (intra-
    workspace; metal forwarding unchanged).
  - crates/kiln-worker: engine.rs rewired from the single-request loop to
    Engine<LlamaModel> — drains the submission channel between steps so
    requests join the running batch; event sink translates SeqEvent →
    TokenChunk/Finished with the same reason/counter/timing semantics;
    RequestHandle.cancelled is now Arc<AtomicBool> handed to the engine;
    heartbeat now reports kv_pool_allocated/used bytes; GetInfo reports
    kv_block_size=32; prefill estimate reads kiln_engine::DEFAULT_PREFILL_CHUNK.
  - tests: kiln-engine/tests/paged.rs (write/gather round-trips across
    block boundaries, COW isolation, accounting, error paths);
    kiln-models/tests/batching.rs (paged engine ≡ Phase-3 contiguous path
    solo; 4-way concurrent greedy ≡ solo bit-exact across 2 rounds with
    chunked prefill interleaving decode at chunk=48; late join; stop-token
    semantics; submit-time and mid-flight pool exhaustion with survivor
    stream untouched); golden harness rewired to drive fixtures through
    the batching engine (production chunk size, engine reused across
    fixtures, leak-checked).
- Decisions:
  - Each iteration issues TWO forward calls sharing the KV pools: the
    prefill chunk alone (op shapes/chunk boundaries identical to the
    solo path — prefill numerics can never depend on batch composition),
    plus the concatenated decode tokens (§6.2 step 2). Batched-decode
    bit-parity at 2-4 concurrency is asserted empirically in
    tests/batching.rs; part 4 must re-verify at batch 16 (row-count-
    dependent matmul kernel dispatch is the risk).
  - Pool layout `[blocks, H, bs, D]` (head-major in-block): writes need no
    transpose; gathers land in the same per-head row-contiguous stride
    pattern SDPA saw from the Phase-3 cache, avoiding kernel-path drift.
  - forward_step returns logits only for sampled positions (Option<Array>)
    rather than SPEC §7.2's sketched `[n_positions, vocab]` — evaluating a
    step would otherwise force the lm_head over whole prefill chunks that
    mlx-lm (and Phase 3) never computed. Trait takes &self (model weights
    are immutable during forward).
  - No async_eval decode pipelining this session (Phase-3's one-step
    pipeline): correctness scope only. Expect single-stream throughput
    below the Phase-3 recorded number until part 4, which owns the
    benches (<200µs step overhead, batch-16 ≥3×) and re-pipelining; not
    measured this session.
  - Pool exhaustion mid-request → in-band Finished{error} (preemption is
    part 3); submit rejects requests that can never fit. Worker pool
    fixed at EngineConfig defaults (512×32 tokens) until §6.4 admission.
  - Phase-3 generate/kv_cache stay (leak gate + example + the reference
    the batching tests pin against); retire when a later phase supersedes.
- Deviations: forward signature vs SPEC §7.2 sketch as above (returns
  sampled-position logits, &self, split StepBatch/PagedKv types); §6.2
  step 4 "return to prefix cache" remains free-list release until Phase 5
  (per part 1's decision). Otherwise none.
- Acceptance:
  ```
  $ cargo test -p kiln-models --test golden -- --nocapture   (CRITICAL GATE)
  golden chat-basic:        48 prompt tokens,  64 generated — exact match (batched/paged engine)
  golden chat-code:         47 prompt tokens, 128 generated — exact match (batched/paged engine)
  golden chat-multibyte:    51 prompt tokens,  64 generated — exact match (batched/paged engine)
  golden raw-continuation:   6 prompt tokens,  64 generated — exact match (batched/paged engine)
  golden raw-long-prefill: 249 prompt tokens,  64 generated — exact match (batched/paged engine)
  test result: ok. 1 passed (no leaked mlx handles)
  $ cargo test -p kiln-models --test batching
  solo engine == contiguous path for 4 jobs
  4-way batched == solo (2 rounds, chunk=48)
  late-join batching matches solo / stop-token semantics ok / pool-exhaustion ok
  test result: ok. 1 passed in 7.45s
  $ cargo test --workspace -> all targets ok (incl. paged.rs unit suite,
    block_props, sampler, leak gate 1k-iteration loop)
  $ uv run --project tests/e2e pytest tests/e2e -q -> 19 passed in 30.42s
    (full stack; incl. greedy/streaming parity across rust+python workers)
  $ cargo fmt --all --check -> clean
  $ cargo clippy --workspace --all-targets -- -D warnings -> clean
  $ cargo clippy --workspace --all-targets --no-default-features -- -D warnings -> clean  (exact CI lint shape)
  $ cargo build --workspace --no-default-features -> clean                (exact CI compile-linux shape)
  $ ruff check / ruff format --check python/ tests/e2e -> clean
  ```
- Next: Phase 4 part 3/4 (await prompt) — preemption under memory
  pressure (§6.1), proper Cancel/Drain semantics, per-worker admission;
  then part 4/4 owns the batch-16 ≥3× throughput gate, golden re-run,
  criterion step-overhead bench (<200µs), and the leak/soak acceptance.
- Addendum (2026-07-04, PM instruction, cross-reference only): the batch-16
  decode-parity risk flagged under Decisions above is the same mechanism the
  2026-07-03 drift investigation (feeding ADR 0001's B1 follow-up) pinned on
  mlx#3120 — Metal split-K quantized matmul for small M, a reduction-order
  change in exactly our 4-bit matmul path. Same dependency, M-dependent
  kernel/reduction dispatch; note #3120 landed in core 0.31.2, so it is not
  in our pinned 0.31.1 build today but arrives with any quarterly mlx-c bump.
  If part 4's batch-16 golden re-verification diverges, check first which
  quantized-matmul kernel variant dispatches at the batched M vs M=1.

## [2026-07-04] Phase 4 / part 3/4 — preemption (§6.1), Cancel, Drain — DONE
- What:
  - Step-zero check (per prompt): pre-preemption pool exhaustion was already
    a clean in-band Finished{error} on both paths — submit() rejects
    requests that can never fit, and mid-flight OutOfBlocks failed the one
    request while survivors stayed bit-exact (asserted in batching.rs).
    The one gap was the code: the worker mapped every engine error to
    WORKER_ERROR_INTERNAL; capacity refusals now carry
    FinishSummary.error_cause=Capacity → WORKER_ERROR_OOM_REJECTED.
  - crates/kiln-engine/src/engine.rs: preemption in the step planner. On
    OutOfBlocks the planner frees the least-deserving block-holder —
    lowest Priority (new: Batch < Interactive, from proto) first, then
    most-recently-admitted (stable `arrival` number assigned at submit,
    kept across preemption so a resumed request keeps seniority) — and
    retries; if the requester itself is least deserving it self-preempts.
    Preemption releases the whole block table and rewinds
    `processed`/`fed` to 0; the request returns to WAITING (queue kept
    arrival-sorted) with `history` (generated tokens) intact.
  - Resume = re-prefill prompt[..n-1] with the original chunk boundaries,
    then replay generated tokens as single-token non-sampled steps
    (skipping lm_head via the existing sample flag), then continue
    sampling. Replay emits nothing and never re-checks stop/max caps.
    Seq state reworked to `history` + `fed` cursor (replaces `next_input`;
    computation-neutral for never-preempted requests — golden re-verified).
  - Admission projection (§6.4-lite): the waiting-queue head is admitted
    only once free blocks cover its path to the *next sampled token*
    (full prefill + replay), so every admission generates ≥1 token —
    kills preemption thrash while keeping FIFO order.
  - crates/kiln-worker: Drain implemented per proto. GRACEFUL sets an
    escalate-only AtomicU8 posture (survives Loading→Ready), rejects new
    Submits in-band with WORKER_ERROR_DRAINING (double-check after
    registry insert closes the Submit/Drain race), lets in-flight finish;
    deadline_ms > 0 arms a task that cancels stragglers at the deadline;
    IMMEDIATE flags every registry handle cancelled — the same flags
    Cancel uses, so the engine loop needs no drain-specific path. Health
    derives DRAINING over Ready; requests_waiting now includes the
    engine's queue; requests_preempted mirrored into Shared for the
    part-4 Stats RPC. Submission carries proto Priority into the engine.
  - tests: kiln-models/tests/preemption.rs (same-priority victim resumes
    bit-exact vs solo; BATCH self-preempts under an younger INTERACTIVE
    and both match solo; cancel honored ≤2 steps mid-stream and while
    preempted-in-WAITING; golden chat-code fixture forced through
    preemption reproduces expected_token_ids exactly; leak-checked).
    kiln-worker/tests/rpc.rs (black-box: spawns the real binary on a UDS,
    tonic client — Cancel found/CANCELLED/not-found, GRACEFUL drain with
    deadline escalation, DRAINING health + in-band rejects, IMMEDIATE
    drain cancels in-flight). batching.rs test 5 updated: mid-flight
    exhaustion now asserts preempt-and-resume bit-exactness instead of
    the old in-band failure (that failure mode is superseded, not
    weakened — the old assertion text even pointed here).
- Decisions:
  - Replay-as-single-token-steps rather than chunked re-prefill of
    generated tokens: a chunk would change trunk-matmul M and give
    attention a multi-token query — exactly the M-dependent kernel
    dispatch class flagged in the mlx#3120 addendum — while len=1 decode
    steps are the shapes already empirically pinned bit-exact. Perf of
    resume is part 4's problem if it matters (preemption should be rare
    at production pool sizes).
  - "Most-recently-admitted" implemented as submit-order `arrival`,
    stable across preemption; re-admission would otherwise mark the
    victim newest and starve it. Waiting queue stays FIFO-by-arrival;
    priority affects only victim choice (gateway-level priority
    scheduling is Phase 9 per SPEC §12).
  - Drain deadline semantics (proto leaves them open): GRACEFUL +
    deadline_ms>0 escalates stragglers to cancellation at the deadline;
    deadline_ms=0 waits indefinitely. DrainAck.requests_remaining =
    live registry count at ack. Python worker Drain stays UNIMPLEMENTED
    (its batching upgrade is Phase 9).
  - Dev-deps added to kiln-worker for tests/rpc.rs: tower + hyper-util
    (UDS connector, same pattern/deps as kiln-gateway/src/uds.rs; both
    already in the workspace tree).
- Deviations: none beyond part 2's recorded forward-signature deltas.
- Acceptance:
  ```
  $ cargo test -p kiln-models --test preemption -- --nocapture
  same-priority preemption: victim resumed bit-exact
  priority preemption: BATCH yielded to INTERACTIVE, both bit-exact
  cancel honored within 1 step(s) of the flag
  cancel-while-preempted ok, survivor bit-exact
  golden chat-code under preemption: exact match after resume
  test result: ok. 1 passed (no leaked mlx handles)
  $ cargo test -p kiln-models --test golden -- --nocapture   (CRITICAL GATE, re-run)
  golden chat-basic/chat-code/chat-multibyte/raw-continuation/raw-long-prefill
    — exact match (batched/paged engine); test result: ok. 1 passed
  $ cargo test -p kiln-models --test batching -> ok (incl. rewritten
    pool-pressure case: loser preempted, resumed, both streams == solo)
  $ cargo test -p kiln-worker --test rpc -- --nocapture
  worker 1: cancel + graceful drain (deadline escalation) ok
  worker 2: immediate drain ok
  test result: ok. 1 passed in 3.89s
  $ cargo test --workspace -> 27/27 test targets ok (leak gate incl.)
  $ uv run --project tests/e2e pytest tests/e2e -q -> 19 passed in 27.37s
  $ uv run --project python/kiln_worker_py pytest python/kiln_worker_py/tests -q -> 28 passed
  $ cargo fmt --all --check -> clean
  $ cargo clippy --workspace --all-targets -- -D warnings -> clean
  $ cargo clippy --workspace --all-targets --no-default-features -- -D warnings -> clean  (exact CI lint shape)
  $ cargo build --workspace --no-default-features -> clean                (exact CI compile-linux shape)
  $ ruff check / ruff format --check python/ tests/e2e -> clean
  ```
- Next: Phase 4 part 4/4 (await prompt) — batch-16 ≥3× throughput gate
  (re-verify decode parity at batch 16 against the mlx#3120 addendum),
  golden re-run, criterion step-overhead bench (<200µs), async_eval decode
  re-pipelining, Stats RPC, leak/soak acceptance.

## [2026-07-04] Phase 4 / Verification — adversarial review of parts 1-3 (§6.1-6.4) — DONE
- What (findings first, fixes after):
  - (1) batching.rs test 5, old vs new, verdict: NOT a literal strict
    superset as shipped. Mapping: submit-time in-band rejection kept and
    strengthened (+error_cause=Capacity); survivor Length + bit-exact
    stream kept verbatim; loser "Error/exhausted" replaced by the
    strictly stronger Length + preemptions>=1 + bit-exact-vs-solo (the
    old failure mode intentionally no longer exists, so a superset of
    the *assertion* is impossible — the invariants are what carried
    over). One old invariant had NO successor: `completion_tokens > 0`
    ("failed mid-decode, not at submit") — mid-stream-ness was only
    guaranteed by sizing, unasserted. Fixed: test 5 now steps until the
    first preemption and asserts the loser had already streamed tokens
    at that instant (the direct successor), then drains.
  - (2) tests/preemption.rs section 5: BATCH request under sustained
    pressure from sequential younger INTERACTIVE arrivals — preempted on
    both cycles (preemptions >= 2), never starved (MAX_STEPS liveness
    bound), resumes onto its solo stream bit-exact; the interactive
    juniors finish untouched. Section 6 (seniority discriminator): after
    J is preempted and resumes, a younger equal-priority K forces the
    next collision — victim must be K; asserts J.preemptions == 1
    exactly, which fails if arrival were (buggily) renewed on
    re-admission and J ranked newest.
  - (3) Coverage gap found in the pass: no test preempted a request
    mid-PREFILL (blocks held, zero tokens generated). Section 7 covers
    it (chunk=8, pool 9, 120-tok senior needing its 5th block 10 decode
    steps in): junior self-preempts at 128/149 prompt tokens with an
    asserted-empty stream, later re-prefills from scratch bit-exact.
    First sizing attempt admitted-then-decoded before pressure landed —
    the admission projection makes mid-prefill preemption reachable only
    when a senior grows *during* the junior's prefill window; the final
    sizing pins that.
  - (4) Honesty correction to part 3's ledger claim that the admission
    projection "kills preemption thrash": it is a snapshot heuristic,
    not a reservation — section 7's scenario shows bounded churn
    (readmit → partial re-prefill → re-preempt while a senior grows),
    and a churn cycle can produce zero tokens for the churned request.
    Global progress still holds (the most-deserving runner always
    advances; all scenarios drain well inside the 2000-step bound). A
    real reservation is Phase 9 admission-control territory.
  - (5) Replay-cost forward reference recorded: new row in SPEC §14's
    risk table (single-token replay = O(generated) forwards per resume;
    revisit before Phase 9 makes preemption routine; chunked replay
    gated on part 4's batched-M parity evidence / mlx#3120). SPEC edit
    made under the review prompt's explicit sanction ("PROGRESS.md or
    SPEC's risk table"); no normative section touched.
  - (6) Remaining sweeps, clean: no other test file weakened across
    adc7b03..4ba9239 (golden.rs semantics unchanged — engine rewiring +
    priority field only; all other suites purely additive); zero
    unwrap()/expect() outside in-file test modules in all touched crates
    (verified per-file against `mod tests` line boundaries); proto/ and
    docs/decisions/ untouched; dependency deltas are proptest (dev-only,
    named by SPEC §12 Phase 4), kiln-engine dev→regular promotion
    (intra-workspace, recorded in part 2), tower+hyper-util (dev-only,
    justified in part 3's commit); the one added #[allow(dead_code)]
    mirrors golden.rs's fixture-schema precedent; num_active stays live
    via is_idle().
- Decisions: kept the bounded-churn behavior rather than adding
  admission reservations now (correctness unaffected, Phase 9 owns
  policy); asserted J.preemptions == 1 exactly (not >=) in section 6
  because the equality is the discriminator.
- Deviations: none. SPEC §14 row is additive and PM-sanctioned.
- Acceptance:
  ```
  $ cargo test -p kiln-models --test preemption -- --nocapture
  same-priority preemption: victim resumed bit-exact
  priority preemption: BATCH yielded to INTERACTIVE, both bit-exact
  cancel honored within 1 step(s) of the flag
  cancel-while-preempted ok, survivor bit-exact
  golden chat-code under preemption: exact match after resume
  double preemption under sustained pressure: resumed bit-exact both times
  arrival seniority stable across resume: younger K preempted, J untouched
  mid-prefill preemption: junior re-prefilled from scratch, bit-exact
  test result: ok. 1 passed in 13.46s (no leaked mlx handles)
  $ cargo test -p kiln-models --test batching -> ok (test 5 now also
    asserts the loser streamed before its first preemption)
  $ cargo test --workspace -> 27/27 test targets ok
  $ uv run --project tests/e2e pytest tests/e2e -q -> 19 passed in 25.54s
  $ cargo fmt --all --check -> clean
  $ cargo clippy --workspace --all-targets -- -D warnings -> clean
  $ cargo clippy --workspace --all-targets --no-default-features -- -D warnings -> clean  (exact CI lint shape)
  $ cargo build --workspace --no-default-features -> clean                (exact CI compile-linux shape)
  $ ruff check / ruff format --check python/ tests/e2e -> clean
  ```
- Next: Phase 4 part 4/4 (await prompt) — batch-16 ≥3× throughput gate
  (re-verify decode parity at batch 16 against the mlx#3120 addendum),
  golden re-run, criterion step-overhead bench (<200µs), async_eval decode
  re-pipelining, Stats RPC, leak/soak acceptance.

## [2026-07-04] Phase 4 / part 4/4 — acceptance run (SPEC §12 Phase 4 gates) — DONE
- What:
  - crates/kiln-models/tests/throughput.rs (new; #[ignore]d perf gate, run
    explicitly in release): batch-16 aggregate vs single-stream decode on
    llama-3.2-1b-4bit (27-token prompt, 128 decode tokens per request).
    Measured on the dev machine: single-stream 124.3 tok/s (Phase-3
    pipelined path; the engine at batch 1 does 116.5 tok/s), batch-16
    aggregate 378.7 tok/s = **3.05x** the stricter denominator. Gate
    >= 3x: PASS — thin margin, stable across three runs (3.05/3.06/3.05).
  - tests/golden.rs: every fixture re-verified **bit-exact with the decode
    batch pinned at width 16** — the mlx#3120 checkpoint flagged since
    part 2. Mechanism: 15 longer-lived fillers admitted (FIFO) ahead of
    the fixture; asserted, not assumed: width stays 16 from fixture
    admission to fixture finish, all fillers outlive it, zero preemptions.
    All 5 fixtures exact -> no M-dependent quantized-matmul divergence at
    the core 0.31.1 pin; the exact-parity bar stands untouched, no kernel
    investigation triggered, no ADR needed. These width-16 rounds are now
    a permanent part of the keystone test (+~14s locally).
  - tests/preemption.rs section 8: preemption under the batch-16 load
    shape (not a 2-3 request micro-pool): a 16-deep interactive burst
    (8x 64-token + 8x 96-token prompts, interleaved) against a 54-block
    pool whose peak demand is 64 blocks. Width demonstrably reached 16,
    the youngest request was preempted mid-decode and resumed, all 16
    streams bit-exact vs their solo runs, both most-senior requests
    untouched.
  - tests/leak_batched.rs (new; own file because the live-object counter
    is process-global): the 1k-iteration leak gate re-run through the full
    batched/paged/preemption path — 1020 engine iterations over 15 waves
    of 16 requests (chunk=48 so long prompts prefill in 4 chunks; pool 42
    vs ~54-block peak), 45 preemptions, 15 mid-stream cancellations, every
    request Finished{Length|Cancelled} as expected; live objects 0 -> 0.
    tests/leak.rs (Phase-3 contiguous path) still passes unchanged.
  - crates/kiln-engine/benches/step_overhead.rs (new criterion bench):
    an Engine::step() around a null StepModel at steady-state batch 16 is
    272.6µs wall. Two companion benches attribute it: the standalone
    eval+readback round-trip of the 16 sampling graphs is 255.3µs (a
    null-forward artifact — production pays one step-boundary eval that
    the real forward dominates) and host-side sampling-graph construction
    is 22.2µs. Non-GPU engine overhead per step ≈ 272.6 − 255.3 ≈ 17µs
    (~40µs counting graph build) — SPEC §6.2 target < 200µs: PASS.
- Decisions:
  - Throughput gate is an #[ignore]d release-only cargo test rather than a
    CI test: perf ratios on shared CI runners are noise, and SPEC §13.4
    has the PM re-run phase gates anyway. It asserts against the stricter
    of the two single-stream denominators. scripts/bench.sh (§11.3 load
    harness, referenced by CLAUDE.md) still does not exist — the §12
    Phase 4 gate needs only the recorded ratio; full harness flagged for
    the packaging/tooling pass.
  - Bench methodology: with a null forward, the step-boundary eval stands
    alone and carries a fixed Metal round-trip production absorbs inside
    the forward's eval, so the bench reports the attribution triplet
    instead of pretending the 272µs headline is scheduler cost. All three
    numbers recorded; the headline stays the conservative bound.
  - New dev-dependency criterion 0.8 (workspace-wide, dev-only, minimal
    features — no plotters/rayon): named by SPEC §3's testing stack and
    CLAUDE.md's `cargo bench -p kiln-engine`; cargo-test alone cannot do
    steady-state statistical sampling. Apache-2.0 OR MIT.
- Deviations: none.
- Acceptance:
  ```
  $ cargo test -p kiln-models --release --test throughput -- --ignored --nocapture
  prompt: 27 tokens, decode: 128 tokens
  single-stream decode: 124.3 tok/s (phase-3 pipelined path), 116.5 tok/s (engine batch 1)
  batch-16 aggregate: 378.7 tok/s -> 3.05x the stricter single-stream rate
  test result: ok. 1 passed (re-runs: 3.06x @ 379.1, 3.05x @ 383.0 tok/s)
  $ cargo test -p kiln-models --test golden -- --nocapture   (CRITICAL GATE)
  golden chat-basic/chat-code/chat-multibyte/raw-continuation/raw-long-prefill
    — exact match (batched/paged engine)
    — exact match at decode width 16          <- the mlx#3120 checkpoint
  test result: ok. 1 passed in 27.57s (no leaked mlx handles)
  $ cargo test -p kiln-models --test preemption -- --nocapture
  (sections 1-7 unchanged, all ok) ... batch-16 pressure: width 16,
  1 preemption(s) across 1 request(s), all 16 streams bit-exact
  test result: ok. 1 passed in 17.08s (no leaked mlx handles)
  $ cargo test -p kiln-models --test leak_batched -- --nocapture
  leak gate (batched): 1020 engine iterations over 15 waves, 45 preemption(s),
  15 cancellation(s); mlx active memory 0B -> 0B, live objects 0 -> 0
  test result: ok. 1 passed in 38.59s
  $ cargo bench -p kiln-engine --bench step_overhead
  engine/step_overhead_batch16        time: [269.69 µs 272.60 µs 276.96 µs]
  engine/sampling_graph_build_batch16 time: [21.968 µs 22.156 µs 22.320 µs]
  engine/sampling_eval_floor_batch16  time: [254.23 µs 255.32 µs 257.38 µs]
  => non-GPU overhead ≈ 272.6 − 255.3 ≈ 17µs per step at batch 16 (< 200µs)
  $ cargo test --workspace -> 28/28 test targets ok (both leak gates incl.;
    throughput #[ignore]d by default, exactly as CI runs it)
  $ cargo run -p kiln-mlx --example smoke -> 3.0        (exact CI smoke step)
  $ python proto codegen + git diff --exit-code -> clean (exact CI codegen step)
  $ uv run --project python/kiln_worker_py pytest -q -> 28 passed
  $ uv run --project tests/e2e pytest tests/e2e -q -> 19 passed in 25.79s
  $ cargo fmt --all --check -> clean
  $ cargo clippy --workspace --all-targets -- -D warnings -> clean
  $ cargo clippy --workspace --all-targets --no-default-features -- -D warnings -> clean  (exact CI lint shape)
  $ cargo build --workspace --no-default-features -> clean                (exact CI compile-linux shape)
  $ ruff check / ruff format --check python/ tests/e2e -> clean
  ```
- Next: PM phase gate on Phase 4 (SPEC §13.4: re-run bench + e2e on PM
  hardware), then Phase 5 — radix prefix cache + SSD tier (SPEC §12).
  Notes for the gate: (a) the 3.05x margin is thin; the async_eval decode
  re-pipelining forecast in the part-2/3 ledgers was NOT in this prompt's
  scope and remains unimplemented — it and Phase 7's paged-attention
  kernel are the obvious levers if the PM wants headroom; (b) the Stats
  RPC likewise was not in this prompt's scope and has no §12 Phase 4
  acceptance criterion — say when it should land; (c) the width-16
  parity evidence pins the trunk matmul at M=16 bit-exact, which is the
  batched-M half of what SPEC §14's replay-cost row asks for — chunked
  replay additionally needs the multi-token attention-query shape
  validated before it is safe.

## [2026-07-04] Phase 4 / Follow-up — async_eval decode pipelining (batched engine) — DONE
- What:
  - Implements the lever flagged in the closeout entry for the thin 3.05x
    margin: `generate.rs`'s async_eval decode pipeline, lifted to the
    batched engine. A steady-state pure-decode step now defers its token
    readback (`async_eval` on the sampled `[1]` arrays); at the next
    `step()` call the following step's forward is built *feeding those
    still-lazy arrays* (reshape + concatenate to `[1, n]`), scheduled,
    and only then is the previous step read back and settled — host-side
    graph construction (large at batch 16: per-seq per-layer gather/write
    chains) overlaps GPU execution.
  - Strict gating, so scheduling semantics are untouched: the pipeline
    engages only when nothing is waiting, every running sequence is
    sampling (no prefill, no replay), penalties are off (their windows
    need the previous token host-side, same forfeit as mlx-lm), no
    cancel flag is up, and the next single-token appends *provably* fit
    free blocks (exact per-row count, checked before any table slot is
    appended). Prefill, replay, admission, preemption, and capacity
    decisions always run on the synchronous path against fully-applied
    state — victim choice, seniority, and the admission projection are
    bit-for-bit the pre-change logic, which is why the exact-count
    preemption assertions (suite section 6) still pass unmodified.
  - Contract change (kiln-engine internal API, not the frozen proto):
    `StepBatch.tokens: Vec<u32>` became `StepBatch.input: StepInput`
    (`Ids(Vec<u32>)` | `Lazy(Array)`); `llama.rs::forward_step` accepts
    either — identical u32 values reach the embedding lookup both ways.
  - Speculative-row semantics: a sequence that stops/cancels at an apply
    has one already-scheduled row in flight; it is discarded unread. Its
    KV write lands in blocks the sequence owned at build time; releasing
    them is safe because a future owner rewrites every row below its own
    length and `PagedKv::gather` trims to that length. **Phase 5 NOTE
    (recorded in the engine module docs): radix sharing must re-review
    this invariant before blocks with a stale speculative tail row can
    enter the prefix cache.**
- Before/after (same machine, same session, release, 27-token prompt,
  128 decode tokens, median of 3):
  ```
  BEFORE (commit ff91746, re-measured this session):
    single-stream 125.6 tok/s (generate) / 116.8 tok/s (engine batch 1)
    batch-16 aggregate 385.8 tok/s -> 3.07x   (closeout entry: 378.7 -> 3.05x)
  AFTER (pipelined):
    single-stream 123.9-126.1 tok/s (generate) / 122.7-126.6 tok/s (engine batch 1)
    batch-16 aggregate 396.2 / 397.9 / 417.8 tok/s -> 3.13x / 3.16x / 3.30x
  ```
  Honest read: the margin measurably improved (median ratio 3.07x ->
  3.16x; aggregate +3-8%) but the win is modest — at batch 16 the step is
  GPU-dominated by the gather-based paged attention, so hiding host time
  buys less than at batch 1. The engine's *single-stream* deficit closed
  entirely (116.8 -> ~126 tok/s, at parity with the pipelined Phase-3
  path), which raises the gate's stricter denominator and makes the ratio
  gain conservative. The next real throughput lever is Phase 7's
  block-table-aware attention kernel (eliminates the gather), not more
  pipelining.
  - Step-overhead bench moved: 272.6µs -> 170.0µs per step()
    (null-forward headline now itself < 200µs — the standalone-eval
    round-trip previously counted there now hides behind the next step's
    build; attribution benches unchanged: build 22.4µs, eval floor
    255.3µs).
- Decisions:
  - Depth-2 pipeline only (one step in flight), mirroring generate.rs —
    matches MLX stream discipline and keeps the cancel bound at <= 2
    steps (proto promise; suite re-verified <= 1 observed).
  - Sync-fallback-on-anything-unusual over pipelining through pressure:
    correctness of preemption order and the §6.4 admission projection is
    worth more than the rare-path speedup; pressure scenarios in the
    tests therefore exercise the identical pre-change planner.
  - No new dependencies.
- Deviations: none.
- Acceptance:
  ```
  $ cargo test -p kiln-models --test golden -- --nocapture   (CRITICAL GATE)
  all 5 fixtures — exact match (batched/paged engine)
  all 5 fixtures — exact match at decode width 16
  test result: ok. 1 passed in 24.73s (no leaked mlx handles)
  $ cargo test -p kiln-models --test batching -- --nocapture -> ok (solo ==
    contiguous; 4-way == solo x2; late-join; stop-token; pool-pressure)
  $ cargo test -p kiln-models --test preemption -- --nocapture -> ok, all 9
    sections incl. exact-count seniority (J.preemptions == 1) and batch-16
    pressure, all streams bit-exact; cancel honored within 1 step
  $ cargo test -p kiln-models --test leak_batched -- --nocapture
  1035 engine iterations / 15 waves, 45 preemptions, 15 cancellations,
  live objects 0 -> 0   ($ --test leak -> also 0 -> 0)
  $ cargo test -p kiln-models --release --test throughput -- --ignored --nocapture
  single-stream decode: 125.8 tok/s (phase-3 pipelined path), 122.7 tok/s (engine batch 1)
  batch-16 aggregate: 397.9 tok/s -> 3.16x   (runs: 3.30x @ 417.8, 3.13x @ 396.2)
  $ cargo bench -p kiln-engine --bench step_overhead
  engine/step_overhead_batch16 time: [168.58 µs 169.98 µs 172.52 µs]
  $ cargo test --workspace -> all test targets ok
  $ uv run --project tests/e2e pytest tests/e2e -q -> 19 passed in 29.04s
  $ uv run --project python/kiln_worker_py pytest -q -> 28 passed
  $ cargo run -p kiln-mlx --example smoke -> 3.0 ; codegen diff -> clean
  $ cargo fmt --all --check -> clean
  $ cargo clippy --workspace --all-targets -- -D warnings -> clean
  $ cargo clippy --workspace --all-targets --no-default-features -- -D warnings -> clean  (exact CI lint shape)
  $ cargo build --workspace --no-default-features -> clean                (exact CI compile-linux shape)
  $ ruff check / ruff format --check python/ tests/e2e -> clean
  ```
- Next: PM phase gate on Phase 4 (SPEC §13.4), then Phase 5 — radix
  prefix cache + SSD tier (SPEC §12). Carry-forward for Phase 5: the
  stale-speculative-tail-row invariant above must be part of the radix
  integration review.

## [2026-07-04] Phase 5 — Radix prefix cache + SSD tier — DONE
- What:
  - **Step zero (the Phase-4 carry-forward) resolved first**: under radix
    sharing, "future owners rewrite every row below their length" no
    longer holds, so the pipeline's discarded speculative row needed a
    decision. Chosen: **(b), narrowed to the exact hazardous slot** —
    only settled rows (`processed + fed`, each forced by a token
    readback or step-boundary eval) are ever keyed by the cache; the
    discarded in-flight row sits exactly at that boundary and is never
    served. Its block either returns to the free list (only writers can
    reacquire it; stale write is stream-ordered before any rewrite) or
    carries the row beyond the keyed range where request-length trims
    and COW keep it unreachable. Option (a) (force the readback) was
    rejected: a pipeline stall at every stop to save at most one
    cacheable block per pipelined finish, resting on a four-layer
    laziness argument no reader ever checks. Pinned by
    `kiln-engine/tests/pipeline_discard.rs`: a mock model whose logits
    are checksum-tied to gathered KV runs a pipelined stop (in-flight
    row = last slot of a full block) immediately followed by a prefix
    match over that region — asserts the exact settled match bound,
    bit-equality with a cache-cold run, and a cancel variant; the test
    fails under mutation of the donation bound (verified).
  - `kiln-engine/src/radix.rs`: block-aligned radix tree (SPEC §6.3) —
    refcount integration with the Phase-4 block manager, leaf-first LRU
    eviction of sole-owned blocks ahead of any preemption, sha256 chain
    hashes, partial leaves for donated settled tails. COW integration:
    reusing a partial tail leaves the next append to the existing
    `append_tokens` copy-on-write path (previously unreachable, now the
    hot containment path). `PrefixCacheHit` events + capability flag +
    `Finished.cached_prompt_tokens` wired through the worker (existing
    proto fields only; wire semantics untouched).
  - `kiln-engine/src/ssd.rs` (SPEC §6.4): fixed-layout slabs, 64 slots
    per file, header = magic/version/geometry/dtype/model fingerprint;
    slot = chain hash + token ids (verified on read) + payload sha256
    (torn slots detectable) + raw K/V bytes. Async flush on a dedicated
    writer thread with acks (a block is readable only after its write
    acked); write-behind capture bounded to 2 blocks per synchronous
    step, never during a pipelined turn; idle worker drains the queue.
    Startup index scan from headers only; strict fingerprint check =
    silent skip + counter; LRU byte cap unlinks whole slabs. Restart
    warm-load is lazy and hash-first during prefix walks.
  - **Determinism hazard found and fixed mid-phase**: the e2e
    reproducibility gate caught warm reruns diverging bitwise — a
    sub-block remainder re-prefilled as one odd-length chunk hits
    different kernel dispatches than the cold run's chunking, and KV
    bits are chunk-shape dependent. Rule now enforced: a hit is either
    **full containment** (every prefill position served, incl. partial
    tail; nothing recomputed) or **trimmed to canonical prefill_chunk
    boundaries** (every recomputed chunk has the cold run's exact
    shape; causality keeps per-row bits independent of later tokens in
    a chunk). Consequence: sub-2048-token *divergent* overlaps are not
    served at all — resubmits/reruns (the acceptance case) get 100%
    containment.
  - Side task (metrics audit): gateway `/metrics` was gateway-local
    counters only and the worker's `Stats` RPC was UNIMPLEMENTED.
    Implemented both ends per SPEC §5/§2.3: worker `Stats` fills every
    proto field except spec-decode + step-overhead percentiles (zeros;
    Phase 8 / needs an in-engine reservoir); supervisor polls Stats on
    the 1s Health cadence and re-exports per-model `kiln_worker_*`
    gauges; python worker's UNIMPLEMENTED is skipped for that worker
    lifetime. e2e asserts the labeled gauges appear.
  - Config flags: engine `EngineConfig{prefix_cache, ssd}`; worker
    `--no-prefix-cache/--ssd-dir/--ssd-max-gb`; gateway passes
    `[defaults] ssd_tier/ssd_cache_max_gb` + `server.cache_dir` to rust
    workers; slabs at `<cache_dir>/<weights_fingerprint>/blocks/`.
- Decisions:
  - Preemption suite runs `prefix_cache: false`: it pins exact
    preemption counts and pool arithmetic that donated-but-evictable
    blocks would shift; cache-on preemption interplay (resume
    re-matching) is covered by the cache tests. Not a weakening — the
    suites' assertions are unchanged.
  - Partial tails are pool-only (not persisted): a restart therefore
    serves containment only when the donor's settled stream covers the
    request's last needed block — the tail chunk is hash-discoverable
    exactly when `(p-1) % block_size == block_size - 1` or generation
    crossed the boundary. Persisting partial slots would add slot
    upgrade churn for a 1-in-32 restart gap; revisit if warm-restart
    hit rates matter in practice.
  - Leak gate (batched) now runs cache-on: preemption churn there fell
    (3 vs 45 events — eviction absorbs pressure), which is the feature
    working; the preemption suite retains dense preemption coverage.
- Deviations:
  - SPEC §6.4 says flushes ride "a dedicated tokio blocking pool";
    kiln-engine has no tokio (the engine loop is a plain OS thread per
    SPEC §6.2), so a dedicated writer thread + ack channel implements
    the same contract.
  - SPEC §6.4 says reads are "mmap + copy"; used `pread`
    (`FileExt::read_exact_at`) — mmap requires `unsafe` outside
    kiln-mlx, forbidden by CLAUDE.md. Same copy semantics, one syscall.
- Acceptance:
  ```
  $ cargo test -p kiln-models --test prefix_cache -- --nocapture
  2k resubmit: reused 2047/2048 (100.0% skip), TTFT 1279.2ms -> 20.4ms (62.7x)
  pipelined stop on real weights: settled 87, containment rerun reused 81,
    divergent extension unserved, both == cold
  SSD restart: reused 511 tokens from disk, reads 16, output bit-exact
  corrupt slab header: ignored (rejects 1), cold run bit-exact
  test result: ok. 1 passed (live objects back to baseline)
  $ cargo test -p kiln-engine --test pipeline_discard -- --nocapture
  pipelined stop: match trimmed to settled blocks; warm == cold
  pipelined cancel: 3 generated, match 4 <= aligned settled 8
  test result: ok. 1 passed   (fails under donation-bound mutation: verified)
  $ cargo test -p kiln-engine   (radix_props + ssd + radix + block units)
  lib 20 passed; block_props 2 passed; paged/pipeline_discard/radix_props/
  sampler each ok
  $ cargo test -p kiln-worker --test rpc -- prefix_cache_stats_and_ssd_restart
  stats over RPC: requests_total 2, prefix_tokens_reused_total 63,
  kv_blocks 3+509=512, ssd_writes 2; restart hit: tokens_reused 63, from_ssd
  test result: ok
  $ cargo test -p kiln-models --test golden -- --nocapture   (CRITICAL GATE)
  all 5 fixtures — exact match (batched/paged engine, prefix cache on)
  all 5 fixtures — exact match at decode width 16
  $ cargo test -p kiln-models --test batching / preemption / leak_batched
  all sections ok; preemption exact counts unchanged (cache off there);
  leak gate: live objects 0 -> 0 (cache on)
  $ cargo test --workspace -> all test targets ok (exactly as CI runs it)
  $ uv run --project tests/e2e pytest tests/e2e -q -> 21 passed
    (incl. test_greedy_is_reproducible[rust] — the gate that caught the
     chunk-shape divergence — and the new worker-stats re-export test)
  $ uv run --project python/kiln_worker_py pytest -q -> 28 passed
  $ cargo test -p kiln-models --release --test throughput -- --ignored
  single-stream 124.4 (generate) / 124.6 (engine batch 1) tok/s
  batch-16 aggregate: 416.0 tok/s -> 3.34x   (Phase-4 gate intact)
  $ cargo bench -p kiln-engine --bench step_overhead
  engine/step_overhead_batch16 time: [166.70 µs 167.30 µs 167.93 µs]
  $ cargo run -p kiln-mlx --example smoke -> 3.0 ; proto codegen diff clean
  $ cargo fmt --all --check -> clean
  $ cargo clippy --workspace --all-targets -- -D warnings -> clean
  $ cargo clippy --workspace --all-targets --no-default-features -- -D warnings -> clean
  $ cargo build --workspace --no-default-features -> clean
  $ ruff check / ruff format --check python/ tests/e2e -> clean
  ```
- Next: PM phase gate on Phase 5 (SPEC §13.4: bench + e2e on PM
  hardware), then Phase 6 — Qwen + Gemma models, quantization matrix,
  `worker="auto"` routing (SPEC §12). Notes for the gate: (a) the
  determinism rule trades sub-chunk reuse on *divergent* prefixes for
  bit-exactness — multi-turn chat reuses nothing until the shared
  prefix crosses `prefill_chunk` (2048); if that matters before Phase 7,
  the options are canonical-shape re-prefill of the shared region
  (costs compute, keeps bits) or an ADR relaxing determinism for
  partial hits; (b) WorkerStats step-overhead percentiles are zero
  until an in-engine reservoir exists (criterion covers the gate);
  (c) partial tails are not persisted (see Decisions) — restart
  containment holds when generation crossed a block boundary.

## [2026-07-04] Phase 5 / Gate follow-up — multi-turn reuse quantification — DONE
- What:
  - PM asked for the real-world impact of the trim-to-canonical-chunk
    rule before merging. Added
    `crates/kiln-models/tests/prefix_multiturn.rs`: a realistic 8-turn
    conversation (600-token opening prompt; each turn appends the
    64-token reply plus a 90-200-token user increment) run sequentially
    against the warm cache, with a cache-cold reference per turn. The
    test asserts only correctness (warm == cold bit-exact every turn —
    holds — and that any served hit obeys the canonical-boundary rule);
    the skip percentages are measurements, not gates.
  - Measured (debug build, M-series, llama-3.2-1b-4bit):
    ```
    turn | prompt | reused | skip%  | F=32 would | warm TTFT | cold TTFT
       0 |    600 |      0 |   0.0% |          0 |   384.7ms |   369.9ms
       1 |    844 |      0 |   0.0% |        640 |   518.7ms |   520.2ms
       2 |   1028 |      0 |   0.0% |        896 |   624.7ms |   634.5ms
       3 |   1252 |      0 |   0.0% |       1088 |   769.8ms |   771.8ms
       4 |   1406 |      0 |   0.0% |       1312 |   838.3ms |   840.5ms
       5 |   1670 |      0 |   0.0% |       1440 |  1014.1ms |  1019.6ms
       6 |   1884 |      0 |   0.0% |       1728 |  1135.7ms |  1152.4ms
       7 |   2088 |      0 |   0.0% |       1920 |  1362.7ms |  1288.7ms
    ```
    Every turn: 0% skip; warm TTFT == cold TTFT, growing linearly with
    conversation length. Turn 7's donor overlap was 1947 tokens and
    still trimmed to zero (floor(1947/2048)·2048). A block-granular
    (F=32) trim would have served 76-93% per turn; F=256 within ~7% of
    that. The cache currently pays only for exact resubmits/retries
    (containment: 100% skip, 62x TTFT) and shared prefixes >= 2048.
- Direct answer: **2048-token granularity is not acceptable if
  multi-turn conversations are part of the caching value proposition.**
  For the entire realistic conversation window (< ~2k accumulated
  tokens) the prefix cache does nothing, and per-turn TTFT grows
  linearly exactly where prefix caching is supposed to flatten it.
- DECISION NEEDED: how to narrow reuse granularity without reopening
  the chunk-shape bit-divergence bug that e2e caught (KV bits depend on
  prefill chunk shapes; a warm remainder recomputed in a shape the cold
  run never uses can flip greedy outputs).
  - **Option A — keep 2048 (status quo).** Zero risk, zero work. The
    cache remains a resubmit/retry and long-RAG-prefix feature until
    revisited (e.g., alongside Phase 7 kernel work). Multi-turn gets
    nothing.
  - **Option B — hybrid canonical schedule with a fine absolute grid in
    the tail (recommended, pending approval).** Keep 2048-token bulk
    chunks; process the *final partial 2048 super-chunk* of every
    prefill on an absolute F-token grid (chunks split at absolute
    multiples of F, plus the final sub-F remainder). Warm resumes may
    then start at any F boundary: every recomputed segment is a
    (offset, length) pair the cold path uses *for that same prompt* —
    the fixed-set-of-shapes requirement — so bit-exactness holds by
    construction rather than by kernel-dispatch luck. Expected
    multi-turn skip from the measured overlaps: 76-93% per turn at
    F=32; F=64 within one block of that while halving the added
    dispatches. Costs/risks, and why this needs a PM decision:
    (1) it changes the canonical schedule for every prompt whose tail
    is < 2048 — cold KV bits shift for all sub-2k prompts — so the
    golden suite must be re-validated against the mlx-lm reference,
    whose own schedule stays one 2048 chunk; if any fixture's token ids
    diverge, proceeding requires the SPEC §11.2 relaxed bar + ADR
    (human approval), else fall back to A. This is the gating risk.
    (2) Cold prefill gains up to 2048/F extra small forwards for the
    tail region (~32 at F=64); needs a before/after TTFT bench as part
    of acceptance. (3) Containment/partial-tail behavior (exact
    resubmits at 100%) is unaffected.
  - **Option C — ADR relaxing bit-determinism for divergent-prefix hits
    only (vLLM-style).** Serve block-granular reuse and accept low-bit
    KV drift on extended prompts (exact resubmits stay bit-exact via
    containment). Maximum reuse, no schedule change, but it repeals the
    CLAUDE.md/SPEC determinism clause for a whole request class and
    requires carving exceptions into the tests that assert it — not
    recommended while B is untried.
  - Recommendation if forced to pick: **B at F=64**, gated on the
    golden suite staying exact after the schedule change and a
    cold-TTFT bench delta within noise; escalate to A-vs-C only if a
    golden fixture diverges. Not implemented — no caching-logic changes
    in this session per instruction.
- Deviations: none (investigation + one test only).
- Acceptance:
  ```
  $ cargo test -p kiln-models --test prefix_multiturn -- --nocapture
  (table above) ... warm == cold bit-exact on all 8 turns; served hits
  obey the canonical-boundary rule; live objects back to baseline
  test result: ok. 1 passed in 25.58s
  $ cargo fmt --all --check -> clean
  $ cargo clippy -p kiln-models --all-targets -- -D warnings -> clean
  ```
- Next: PM decision on the granularity options above; Phase 5 merge
  gate otherwise unchanged (previous entry).

## [2026-07-04] Phase 5 / Option B step 2 — golden gate on the fine-tail schedule — DONE (GATE PASSED)
- What: ran the full golden-token parity harness against the F=64
  canonical schedule (previous commit). Under the new schedule every
  fixture's prefill is re-chunked — e.g. raw-long-prefill's 248 prefill
  positions run as 64+64+64+56 instead of one 248-token chunk — while
  the committed fixture ids remain mlx-lm's single-2048-chunk reference.
- Acceptance:
  ```
  $ cargo test -p kiln-models --test golden -- --nocapture   (THE GATE)
  golden chat-basic:        48 prompt, 64 gen  — exact match (batched/paged engine)
  golden chat-code:         47 prompt, 128 gen — exact match
  golden chat-multibyte:    51 prompt, 64 gen  — exact match
  golden raw-continuation:   6 prompt, 64 gen  — exact match
  golden raw-long-prefill: 249 prompt, 64 gen  — exact match
  (all five again) — exact match at decode width 16   <- mlx#3120 rounds
  test result: ok. 1 passed in 23.14s
  $ cargo test -p kiln-models --test batching -- --nocapture
  solo engine == contiguous path; 4-way == solo x2; late-join;
  stop-token; pool-pressure — all ok (generate path and engine share
  the new schedule; batch-1 parity holds)
  ```
- Read: token-id parity survives the 64-grid re-chunking of every
  sub-2k prompt on this model/pin/hardware. This does NOT prove KV
  bit-equality across schedules (the e2e divergence proved bits can
  move); it proves the schedule change stays inside golden's token-id
  bar. The cache-hit path (next commit) does not rely on either fact:
  warm resumes recompute only in shapes the same prompt's cold schedule
  produces, per the resume-invariance unit test.
- Next: step 3 — serve cache hits from F-aligned boundaries.

## [2026-07-04] Phase 5 / Option B (F=64) — fine-grained prefix reuse — DONE
- What (sequenced per the PM instruction; each step its own commit):
  1. **Canonical schedule** (`canonical_prefill_len`, shared by the
     batched engine and the Phase-3 generate path): bulk chunks split at
     absolute `prefill_chunk` multiples; the final partial super-chunk
     split at absolute `prefill_fine_chunk` (default 64) multiples plus
     the sub-fine remainder. A unit test exhaustively proves
     resume-invariance: walking from any boundary reproduces exactly the
     cold schedule's suffix, for several (chunk, fine) configs — the
     property the bit-exactness argument stands on. `fine >=
     prefill_chunk` restores the old schedule (bench knob); test configs
     with `prefill_chunk <= 64` degenerate to their old schedules.
  2. **Golden gate: PASSED** (own commit, before any cache-path change).
     All 5 fixtures exact, including the width-16 rounds, with every
     sub-2k prefill re-chunked onto the 64 grid (e.g. raw-long-prefill:
     one 248-token chunk -> 64+64+64+56). Batch-1 == contiguous parity
     also re-verified.
  3. **Cache hits now resume at schedule boundaries** (fine multiples in
     the final super-chunk, chunk multiples in bulk). Re-measured the
     8-turn conversation (90-200-token increments, debug build):
     ```
     turn | prompt | reused | skip%  | warm TTFT | cold TTFT   (before: all 0%)
        1 |    844 |    640 |  75.8% |   164.8ms |    573.2ms
        2 |   1028 |    896 |  87.2% |   114.0ms |    681.4ms
        3 |   1252 |   1088 |  86.9% |   147.6ms |    842.7ms
        4 |   1406 |   1280 |  91.0% |   107.6ms |    933.9ms
        5 |   1670 |   1408 |  84.3% |   216.5ms |   1133.4ms
        6 |   1884 |   1728 |  91.7% |   142.3ms |   1291.6ms
        7 |   2088 |      0 |   0.0% |  1285.6ms |   1286.3ms
     ```
     The 76-93% estimate is confirmed for steady turns (75.8-91.7%,
     warm TTFT cut 3.5-9.1x), with one correction: **the single turn
     whose prompt crosses a 2048 super-chunk boundary serves 0** — its
     donor overlap (1920 here) lies inside the new prompt's bulk chunk,
     which is not a resumable boundary; resuming there would compute a
     shape the cold run never uses, so it is correctly refused. Reuse
     returns the following turn. Cost: one cold prefill per 2048 tokens
     of conversation growth (~1 turn in 10-15). Warm == cold bit-exact
     asserted on every turn; the divergent-extension scenario in
     prefix_cache.rs now exercises a real fine-aligned resume on real
     weights (was: expects-no-hit).
  4. **Miss-path cost** (release medians, fresh engine, cache off,
     `tests/prefill_schedule_bench.rs`):
     ```
     prompt | tail fwds | fine=64 TTFT | old sched | delta
        257 |         4 |     179.0ms |   168.2ms |  +10.7ms (+6.4%)
        512 |         8 |     352.3ms |   314.9ms |  +37.4ms (+11.9%)
       1024 |        16 |     714.6ms |   615.1ms |  +99.5ms (+16.2%)
       2048 |        32 |    1470.8ms |  1288.4ms | +182.4ms (+14.2%)
     tuning curve @2048: F=64 +13.9% | F=128 +2.9% | F=256 +0.2%
     ```
     ~6.5ms per added forward — dominated by per-step fixed cost (eval
     sync + per-chunk cache maintenance), not matmul time; hence the
     sharply non-linear curve.
  5. Full re-run: workspace 34/34 test targets ok (batching, preemption
     exact counts, both leak gates 0 -> 0, golden, all prefix suites,
     worker rpc); batch-16 throughput 398.5 tok/s -> 3.22x (gate >= 3x);
     e2e 21 passed (both worker kinds); python worker 28 passed; both CI
     shapes + fmt + ruff clean; smoke 3.0.
- Decisions:
  - F=64 shipped as the default per the instruction. **Flagging for the
    PM**: the measured curve says F=128 (+2.9% miss cost, warm recompute
    grows by at most 64 extra tokens/turn) or F=256 (+0.2%) may be the
    better default trade on this hardware; it is a one-line default
    change (`DEFAULT_PREFILL_FINE_CHUNK`), golden re-gated the same way
    (a coarser grid is strictly closer to the old schedule). Say the
    word and it lands with a fresh golden run.
  - The super-chunk-crossing seam (step 3) is inherent to keeping bulk
    chunks at 2048: finer bulk boundaries would tax every long cold
    prompt. Left as-is; the multiturn test documents it.
  - Per-step overhead, not matmuls, dominates the fine-grid cost —
    batching several consecutive fine chunks into one engine iteration
    (same forward shapes, fewer step boundaries) is the obvious lever if
    the miss cost ever matters; noted, not built.
- Deviations: none.
- Acceptance: outputs quoted above per step; gate commands identical to
  the Phase 5 closeout list.
- Next: PM phase gate on Phase 5 (SPEC §13.4) including the F-default
  choice above, then Phase 6 — Qwen + Gemma, quantization matrix,
  worker="auto" routing (SPEC §12).

## [2026-07-04] Phase 5 / Option B follow-up — default fine grid 64 -> 128 — DONE
- What: per the step-4 tuning curve and PM approval,
  `DEFAULT_PREFILL_FINE_CHUNK` changed 64 -> 128 (one-line default; the
  schedule machinery is untouched). Test adaptations only: the
  prefix_cache divergent-extension scenario's seed prompt lengthened so
  its donor overlap still crosses a fine boundary (it keys off the
  constant), and the schedule bench's first table now measures the
  shipped default rather than a hardcoded 64.
- Acceptance (same gates, same order):
  ```
  $ cargo test -p kiln-models --test golden -- --nocapture   (THE GATE)
  all 5 fixtures — exact match (batched/paged engine)
  all 5 fixtures — exact match at decode width 16            (mlx#3120 rounds)
  test result: ok. 1 passed in 22.50s
  $ cargo test -p kiln-models --test prefix_multiturn -- --nocapture
  turn | prompt | reused | skip%  | warm TTFT | cold TTFT
     1 |    844 |    640 |  75.8% |   162.5ms |    560.7ms
     2 |   1028 |    896 |  87.2% |   110.3ms |    668.3ms
     3 |   1252 |   1024 |  81.8% |   184.0ms |    828.6ms
     4 |   1406 |   1280 |  91.0% |   105.2ms |    916.8ms
     5 |   1670 |   1408 |  84.3% |   206.6ms |   1055.7ms
     6 |   1884 |   1664 |  88.3% |   170.2ms |   1190.4ms
     7 |   2088 |      0 |   0.0% |  (super-chunk crossing, unchanged seam)
  vs F=64: turns 3 and 6 reuse exactly 64 fewer tokens; the rest are
  identical. Short-increment check: turn 4 (90-token increment) reused
  1280 (91.0%) — every turn advances donor coverage by increment + 64
  generated >= 154 > 128, so the resumable boundary advances every
  turn; nothing stalls just short of 128. (Stalling would need
  increment + generation < 128 in a single turn — possible with very
  short replies, costs that turn one boundary, self-heals next turn.)
  $ cargo test -p kiln-models --release --test prefill_schedule_bench -- --ignored
  prompt | tail fwds | default F=128 TTFT | old sched | delta
     257 |         2 |        164.2ms |   161.7ms |  +2.5ms (+1.5%)
     512 |         4 |        320.2ms |   318.5ms |  +1.7ms (+0.5%)
    1024 |         8 |        672.1ms |   639.9ms | +32.2ms (+5.0%)
    2048 |        16 |       1401.2ms |  1371.5ms | +29.7ms (+2.2%)
  curve re-sample @2048: F=64 +16.9% | F=128 +3.9% | F=256 +1.3%
  -> the exploratory +2.9% figure holds as the real number within the
  run-to-run noise band (+2.2%/+3.9% across two samplings); single-digit
  at every prompt size, vs +14-17% at F=64.
  $ cargo test --workspace -> 34/34 test targets ok, zero failures
    (batching, preemption exact counts, both leak gates 0 -> 0, all
     prefix suites, worker rpc)
  $ cargo fmt --all --check / clippy (both shapes) -> clean
  $ cargo build --workspace --no-default-features -> clean
  $ uv run --project tests/e2e pytest tests/e2e -q -> 21 passed
    (incl. the greedy-reproducibility canary, both worker kinds)
  ```
- Deviations: none.
- Next: PM phase gate on Phase 5 (SPEC §13.4), then Phase 6 — Qwen +
  Gemma models, quantization matrix, worker="auto" routing (SPEC §12).

## [2026-07-04] Phase 6 / Task 1 — Qwen2.5 + Qwen3 (fixtures first) — BLOCKED
- What:
  1. ADR 0001 B1 alignment re-verified before generating fixtures: worker
     venv reports mlx.core 0.31.1 / mlx-lm 0.31.2 (gen-golden.py's hard
     refusal also passed). Reference stack unchanged.
  2. New pinned test model (no qwen2-arch model existed in the pinned set;
     the acceptance matrix requires one): `qwen2.5-0.5b-4bit` =
     mlx-community/Qwen2.5-0.5B-Instruct-4bit @ a5339a4131f1, appended to
     fetch-test-model.sh. Existing pins untouched.
  3. Fixtures first: tests/golden/qwen3-0.6b-4bit/ and
     tests/golden/qwen2.5-0.5b-4bit/ (5 cases each, standard case list).
  4. Implementation (all in working tree, uncommitted — see DECISION):
     - `qwen2.rs`/`qwen3.rs`: op-for-op ports of mlx_lm.models.qwen2/qwen3.
     - `config.rs`: Qwen2Config/Qwen3Config (field-for-field mlx-lm
       defaults; qwen2 has no head_dim override, rope_theta 1e6 default),
       `RopeScaling::Yarn`, `ArchConfig` model_type dispatch (this is the
       `worker="auto"` predicate for task 4), SUPPORTED_ARCHITECTURES.
     - Shared trunk: llama's Attention/Block/forward paths consolidated
       into crate-private `nn::CausalLm`, parameterized by GQA geometry +
       optional qwen3 qk-norm (RMSNorm slotted between the [B,L,H,D]
       reshape and the head transpose, exactly the reference op order).
       llama.rs/qwen2.rs/qwen3.rs are now thin config+loader modules; the
       (green) llama golden suite pins the refactor bit-for-bit.
     - Yarn RoPE: host f64 correction-range/mscale + f32 MLX graph
       mirroring YarnRoPE; unit test asserts freqs + mscale bit-identical
       to reference-generated constants (Qwen3 long-context recipe;
       `nn::tests::yarn_freqs_match_reference_bit_for_bit`). `ops::clip`
       added to kiln-mlx.
     - `AnyModel` enum + worker loads it; golden harness generalized to
       every tests/golden/<model>/ dir (missing local model now fails
       loudly instead of skipping).
- Result: single-stream parity is EXACT for all three architectures, but
  the golden gate is red, and the cause is fully diagnosed as MLX kernel
  dispatch at the pin — not model math:
  - Gate state @ shipped engine defaults: llama 5/5 exact (both rounds);
    qwen2.5 4/5 — raw-long-prefill (261 tok) diverges at generated token
    10; qwen3 unreached (same 261-token fixture shape ⇒ same exposure).
  - Prefill bisect (engine config only, same weights/prompt):
    fine=off (single 260-tok tail = mlx-lm's shape) → exact;
    fine=65 / fine=130 (uniform pieces) → exact;
    fine=64 / 128 / 256 (ragged 4-token final piece) → diverges @10.
    Prefix cache on/off: irrelevant.
  - Root cause (vendored MLX v0.31.1,
    mlx/backend/metal/quantized.cpp `get_qmv_batch_limit`): quantized
    matmul dispatches a vector kernel below an M threshold that varies by
    shape and GPU generation (6..32 per the table; measured here: ~18 for
    K,N ≤ 2048, ~10-12 for the mlp/lm_head shapes). Row-bit probe over
    real weights (all three models, every projection + tied lm_head):
    rows are bit-identical WITHIN a kernel class (M=2..16 == M=1 small
    shapes; M=24..260 mutually identical) and differ ACROSS the boundary
    — every arch, every projection. mx.fast SDPA has the same two-class
    structure: q-len 4 vs 260 differs; q-len 32 vs 260 bit-identical.
  - Consequence 1 (fine-tail schedule, latent since Phase 5, arch-
    independent): a ragged final piece of size limit%128 below the
    threshold computes those positions' KV in the vector-kernel class
    while the mlx-lm reference (monolithic ≤2048 tail) is in the matrix
    class. Llama's fixtures pass only by remainder luck (5/46/120 — 5 is
    vector-class on BOTH sides). A synthetic llama rem=3 prompt shows
    bit-different KV (probe) though its 64 greedy tokens happened to
    survive (margins).
  - Consequence 2 (batched decode): trunk M = step token count, so widths
    ≥ ~12 put mlp/lm_head matmuls in the matrix class vs the M=1
    reference — logit ulp noise for EVERY arch. Llama's width-16 golden
    rounds pass on argmax margins (its lm_head rows provably differ at
    M=16); qwen fixtures flip: chat-basic diverges at token 28 (qwen3) /
    33 (qwen2.5) at width 16. Bit-level "batching must not change greedy
    outputs" is not achievable under this kernel pin for any arch; token
    level it is fixture/margin/hardware dependent (CI's M1 has different
    thresholds than this machine).
- Decisions (within latitude): new pin appended, none bumped; trunk
  consolidation into nn::CausalLm (llama golden green pins it); attention
  scale computed in f64 then narrowed (matches Python; value identical
  for every pinned model).
- Deviations:
  - fetch-test-model.sh gained a NEW pinned model (revisions themselves
    frozen; adding was unavoidable for qwen2 coverage) — flagging.
  - Golden harness: fixture-dir-without-local-model is now a failure, not
    a skip (fetch script and fixtures must not drift).
- Acceptance (real outputs, trimmed):
  ```
  $ uv run --project python/kiln_worker_py python -c "import mlx.core..."
  mlx.core 0.31.1 / mlx_lm 0.31.2                       (B1 holds)
  $ cargo test -p kiln-models --lib
  11 passed (incl. yarn_freqs_match_reference_bit_for_bit)
  $ cargo test -p kiln-models --test golden -- --nocapture   (THE GATE)
  llama-3.2-1b-4bit: 5/5 exact + 5/5 exact at width 16
  qwen2.5-0.5b-4bit: chat-basic/chat-code/chat-multibyte/raw-continuation
    exact; raw-long-prefill FAILED (first divergence token 10)
  -> test result: FAILED
  single-stream cross-check, fine=off engine: 15/15 fixtures exact
    (llama, qwen2.5, qwen3 — all five each)
  width-16 probe (fine=off): qwen3 chat-basic diverges @28,
    qwen2.5 chat-basic @33; llama all exact
  $ cargo test --workspace --no-fail-fast -> 33/34 targets ok; only the
    golden gate fails (batching, preemption, both leak gates, prefix
    suites, worker rpc all green under the AnyModel/nn refactor)
  $ uv run --project tests/e2e pytest tests/e2e -q -> 21 passed
  $ uv run --project python/kiln_worker_py pytest ... -> 28 passed
  $ cargo fmt --check / clippy --all-targets (both CI shapes) -> clean
  $ cargo build --workspace --no-default-features -> clean
  ```
- DECISION NEEDED: two coupled calls; neither is covered by SPEC §11.2 as
  written (its relaxed bar — first divergence past token 48, logprob
  delta < 1e-3, per-model ADR — does not admit these: width-16 divergence
  starts at token 28). Per CLAUDE.md I am not weakening the test, not
  changing the PM-approved F=128 default, and not committing a red gate.
  A) Batched-decode parity bar:
     A1. Redefine the width-16 golden rounds as the token-level canary
         they empirically are; bit-exact bar applies to single-stream
         only. Requires ADR + SPEC amendment naming qwen2/qwen3 (and
         acknowledging llama passes on margins). Define a batched
         tolerance/canary policy for qwen serving.
     A2. Keep the bit-exact bar; rust worker serves qwen single-stream
         parity-proven but batched-unproven — i.e. defer qwen enablement
         until the quarterly mlx-c bump (note: dispatch-by-M is a perf
         feature of MLX; a bump may not change this).
     A3. Force one kernel class in the trunk (pad every decode step to
         ≥ 32 rows / per-row lm_head): rejected on analysis — it makes
         single-stream diverge from the M=1 reference instead, and/or
         burns large bandwidth.
     My read: A1 is the only workable shape; the tolerance definition is
     yours.
  B) Fine-tail ragged piece (single-stream, all archs):
     B1. Pad sub-32-row ragged tail pieces to 32 rows (trunk matmuls) and
         32 q-rows (SDPA) when the piece does not start on a 2048
         boundary. Probes show both kernel families are row-stable and
         reference-identical at ≥ 32 rows. Keeps F=128, all Option B
         boundaries, warm==cold, and donation semantics intact (ragged
         pieces are never donated). Est. ~100-150 lines (engine schedule,
         StepBatch, CausalLm); needs a golden re-gate + a deliberate
         tiny-remainder fixture.
     B2. Stopgap: revert default to the single-tail-chunk schedule
         (fine=off) — restores universal single-stream parity today,
         forfeits Phase 5 multi-turn reuse below 2048.
     My read: B1.
  Nothing committed; the full change set sits in the working tree
  (7 modified, 4 new source files, 2 new fixture dirs) for review.
- Next: on the A/B rulings — finish Task 1 acceptance, then Gemma2/3,
  the 8-bit/BF16 matrix, worker="auto" routing, /v1/completions
  (Phase 6 SPEC §12 order).

## [2026-07-04] Phase 6 / Task 1 (continued) — ADR 0002 + B1 padding — PARTIAL
- What (implements the PM's A1/B1 rulings on the previous entry):
  1. **ADR 0002** (docs/decisions/0002-parity-bars-under-mlx-kernel-dispatch.md):
     MLX v0.31.1 dispatches distinct quantized-matmul kernels (qmv/qmm)
     and a two-class SDPA path on M, bit-different but numerically close
     across the threshold — a library/hardware characteristic, not a Kiln
     defect. Bars: single-stream vs mlx-lm strictly bit-exact, no
     exceptions; batched (M>1) parity = token-id equality with the
     single-stream reference; revisit at every mlx-c/core-MLX bump per the
     ADR 0001 quarterly process.
  2. **Width-16 framing corrected** (golden.rs, batching.rs docs +
     assertion messages). Correction to a previously overstated guarantee,
     recorded per instruction: the Phase 4/5 "bit-exact at width 16 /
     bit-identical under concurrency" wording was wrong as stated — those
     tests always asserted token-id equality, and the logits at width 16
     already differed from M=1 in ulps (lm_head/mlp cross their dispatch
     threshold near M~10-12); the passes were argmax margin. **No
     assertion was weakened**: every existing width-16/concurrency
     assertion already compared token ids; only comments/docs claimed
     more than was verified.
  3. **B1** — ragged prefill tail pieces shorter than
     `PREFILL_PAD_MIN_ROWS` (32 = max qmv threshold across the GPU
     dispatch table, above the SDPA vector bound) and off the super-chunk
     grid now run with kernel-class pad rows:
     - kiln-engine: `StepBatch::pad_rows` + the rule at prefill
       scheduling (pad capped at the piece offset so the SDPA query never
       exceeds KV coverage); pieces starting ON a 2048 boundary are the
       reference's own shape and are never padded.
     - kiln-models: `CausalLm`/`Attention` honor pads on the paged AND
       contiguous paths — pad rows ride the trunk matmuls, front-pad the
       SDPA query (causal mask is bottom-right aligned, so real rows keep
       their exact spans), are refilled as zero lanes for o_proj, are
       never written to KV, and are never sampled. generate.rs's prefill
       applies the same rule, keeping engine == contiguous by
       construction (batching.rs now pins that with a deliberately ragged
       prompt).
     - Hazard tests (pipeline-discard analogy, per instruction):
       kiln-engine/tests/prefill_pad.rs — checksum-mock contract: pads
       flagged exactly per rule and only on lone prefill pieces; step
       input ids and every K/V write run cover REAL rows only; never
       sampled/emitted; containment rerun and extension resume warm==cold
       with re-padded canonical shapes. kiln-models/tests/prefill_pad.rs —
       real weights: bit-exact vs fixture THROUGH a padded piece,
       containment rerun (hit 137) and extension resume (hit 128) exact.
  4. **New llama fixture** `raw-tiny-remainder` (138 prompt tokens →
     9-row ragged piece, below every dispatch threshold on every listed
     GPU class): generated via scripts/gen-golden.py on the B1-verified
     stack (mlx.core 0.31.1 / mlx-lm 0.31.2); the 5 existing llama
     fixtures regenerated byte-identical in the same run (cmp-verified) —
     no reference drift. Case added to gen-golden.py CASES with a sizing
     note. This is what gives B1 committed CI coverage.
- Result against the corrected bars:
  - **Single-stream: 16/16 fixtures bit-exact** — llama 6 (incl. the new
    tiny-remainder), qwen2.5 5, qwen3 5; the two 261-token qwen fixtures
    that diverged at generated token 10 before B1 are now exact.
  - **Width 16: llama 6/6 token-exact. qwen 6/10** — 4 flips, all decode-
    side (trunk M=16 crosses the mlp/lm_head qmv/qmm threshold; knife-
    edge argmax positions flip; B1 is prefill-only and cannot help):
    ```
    qwen2.5/chat-basic        div @33, EOS @28 -> post-EOS only (the
                              entire user-visible answer is identical)
    qwen2.5/raw-long-prefill  div @10  (pre-EOS)
    qwen3/chat-basic          div @28  (pre-EOS, think-mode text)
    qwen3/chat-code           div @42  (pre-EOS, think-mode text)
    ```
  - warm==cold under B1: prefix_cache + prefix_multiturn green; both pad
    tests assert warm==cold through padded pieces explicitly.
- Acceptance (real outputs, trimmed):
  ```
  $ cargo test -p kiln-models --test golden -- --nocapture  (commit shape)
  llama-3.2-1b-4bit: 6 fixtures — exact match (batched/paged engine) x6,
  token-id match at decode width 16 x6 -> ok
  $ cargo test -p kiln-models --test golden  (with local qwen fixture dirs)
  qwen2.5: 5/5 exact single-stream; FAILS at width-16 chat-basic per the
  ADR 0002 token-equality bar (matrix above; qwen3 same pattern)
  $ cargo test -p kiln-engine --test prefill_pad -> ok (pad contract)
  $ cargo test -p kiln-models --test prefill_pad -> ok
    "padded piece: cold == fixture, rerun hit 137, extension hit 128"
  $ cargo test -p kiln-models --test batching -- --nocapture
    "solo engine == contiguous path for 5 jobs" (incl. padded ragged job)
  $ cargo test --workspace --no-fail-fast -> 35/36 targets ok (36th =
    golden with the held qwen fixtures present locally)
  $ cargo fmt --check / clippy --all-targets (both CI shapes) -> clean
  $ cargo build --workspace --no-default-features -> clean
  $ uv run --project tests/e2e pytest tests/e2e -q -> 21 passed
  $ pytest python/kiln_worker_py/tests -> 28 passed ; ruff -> clean
  ```
- Decisions: new fixture case via the sanctioned generator with existing
  fixtures byte-verified untouched; PREFILL_PAD_MIN_ROWS=32 rationale in
  the ADR and const docs.
- Deviations: none.
- Committed: kiln-mlx clip; ADR 0002 + engine pad plumbing + contract
  test; qwen2/qwen3 models over shared CausalLm; worker AnyModel dispatch;
  golden/batching reframing + tiny-remainder coverage; this entry. HELD
  uncommitted pending the residual ruling below: tests/golden/qwen*/
  fixture dirs and the fetch-test-model.sh qwen2.5 pin — committing them
  turns the width-16 golden rounds red on main (4/10 qwen flips).
- DECISION NEEDED (residual, narrow): the ADR 0002 batched bar (token-id
  equality at width 16) measurably does not hold for qwen2/qwen3 at this
  kernel pin, while llama holds on all fixtures. The instruction's
  "full golden parity ... at width-16" is therefore not satisfiable for
  qwen by any correct implementation at this pin. Options:
  A) Per-arch batched enablement, encoded in the harness: land the qwen
     fixtures; width-16 rounds assert token equality for batched-enabled
     archs (llama) and assert the RECORDED nonconforming status for
     qwen2/qwen3 (strict xfail: a kernel bump that fixes dispatch flips
     the round to "must promote"). worker="auto" (Task 4) then gains a
     principled input: qwen routes to rust for single-stream-safe use
     only after the PM enables it, or to python. My read: this matches
     ADR 0002's "batched-decode enablement per architecture is gated on
     the width-16 golden rounds" — but it encodes an allowed failure, so
     it needs your explicit approval.
  B) Redefine the batched bar as token equality through the first EOS:
     rescues only qwen2.5/chat-basic (1 of 4); qwen3's think-mode flips
     are pre-EOS. Not recommended.
  C) Cap qwen decode width below the smallest dispatch threshold
     (12 here, 6 on some GPU classes): restores bit-equality wholesale
     but forfeits the batch-16 throughput target for qwen. Not
     recommended as a default.
- Next: residual ruling (A/B/C) -> land held fixtures + per-arch gating,
  close Task 1, then Task 2 (Gemma2/3, fixtures first).

## [2026-07-04] Phase 6 / Task 1 — batched-bar investigation (PM-directed) — REPORT
- Scope: investigation only, per instruction. No fixture, harness, or
  source changes; all probes were scratch tests, removed after
  measurement. Working tree still holds only the previously-HELD items
  (qwen fixture dirs + fetch pin).
- 1) Precision mitigation — no narrow high-precision path can close this
  at the pin, for two independent reasons:
  a. The root cause is not accumulation precision but ARGMAX TIES.
     Reference-side top-2 logprob gaps at every observed flip position
     (measured on the pinned mlx-lm stack, f32-read):
     ```
     qwen2.5/chat-basic  @33  gap = 0.015625  (one f16 ulp)
     qwen2.5/raw-long    @10  gap = 0.000000  (EXACT f16 tie)
     qwen3/chat-basic    @28  gap = 0.000000  (EXACT f16 tie)
     qwen3/chat-code     @42  gap = 0.125
     ```
     At an exact tie the reference argmax picks by index order. ANY
     perturbation — including a MORE accurate fp32 head — resolves the
     tie by magnitude instead, which need not match that index-order
     choice. Higher precision therefore cannot reproduce the reference
     token except by being bit-identical; "compute-in-fp32-then-cast"
     narrows nothing here. Llama is not structurally safer: its
     chat-basic stream carries an exact 0.0 tie at position 24 and
     simply has not flipped (verified through width 24) — fixture luck,
     as ADR 0002 already frames it.
  b. MLX v0.31.1 exposes NO dispatch control: the qmv/qmm split
     (quantized.cpp, `M >= get_qmv_batch_limit(K, N, device)`) and the
     SDPA vector/full split (`q_len <= 8`) are hard-coded; no env var or
     API parameter reaches either. The only adjacent knob is
     MLX_METAL_GPU_ARCH, which spoofs the GPU arch string — it can only
     shift thresholds within the table's 6..32 range while perturbing
     unrelated kernel selections; unusable. `mx.quantized_matmul` has no
     kernel/mode argument at this pin; `env::enable_tf32` affects only
     fp32 GEMMs. This is a real "no such control" answer.
- 2) Empirical characterization (flip onset per decode width, fillers
  pinning the batch; widths 1,2,4,6,8,10,12,14,16,20,24):
  ```
  qwen2.5/chat-basic        w<=8 ok | @33 from w10 (stable through w24)
  qwen2.5/raw-long-prefill  w<=8 ok | @10 from w10
  qwen3/chat-basic          w<=8 ok | @52 at w10, @28 from w12
  qwen3/chat-code           w<=8 ok | @103 at w10, @42 from w12
  llama/chat-basic          ok at every width tested
  ```
  - The cliff is exactly the dispatch ladder measured earlier on this
    GPU: lm_head (limit 10) crosses first at w10; the MLP shapes
    (limit 12) cross at w12 and MOVE the flip position (two independent
    noise sources, visible in the qwen3 rows); attention projections
    (limit 18) cross later and add nothing new on these fixtures. Below
    the lowest threshold the trunk rows are BIT-identical to M=1 (qmv
    row-stability, probe-verified) — token equality at w<=8 is
    guaranteed by construction, not marginal.
  - Weights vs structure: the flips are near-tie statistics, not qwen op
    structure. Class-crossing exists identically for llama; whether a
    fixture flips depends only on whether its greedy stream visits a
    knife-edge position. A different qwen checkpoint/quantization would
    flip elsewhere or not at all. Per-arch conformance inferred from
    fixture passes is therefore statistically weak; the honest system
    property is a per-DEVICE deterministic width bound.
  - The bound is hardware-dependent: flips start at w10 here (safe
    verified <= 8; <= 9 implied by the limit table's 10 minimum);
    M1/M2-class large-shape limit is 6 (safe <= 5); 'd'-class is 12
    (safe <= 11). A ~100ms startup calibration (row-stability probe per
    weight shape) derives it robustly on unknown GPUs, vs replicating
    MLX's device table.
- 3) Cost of a load-bearing max-safe-batch-width:
  - Proto: one additive WorkerInfo field (e.g.
    `max_deterministic_decode_width`, 0 = "greedy determinism under
    batching not guaranteed at this build") — additive fields are
    explicitly allowed post-Phase-2; gateway plumbs it through for
    observability/routing. Cheap (~50 lines both workers + gateway).
  - Enforcement, two shapes:
    (a) cap concurrent DECODING sequences at W: ~20 scheduler lines, but
        aggregate throughput is then W-way batching, remainder queues.
    (b) sub-batched decode: admit any width, split each decode forward
        into <=W-row chunks (prefill already runs as its own forward;
        the pipelined path folds in). This makes batched greedy ==
        single-stream BIT-exact at any admitted width for every arch —
        the whole fixture-luck problem disappears and llama's margin
        stops being load-bearing. Est. 150-250 lines (run_iteration,
        pipelined path, tests).
  - Measured cost of (b) at width 16, qwen2.5-0.5B, release micro-bench
    (per-op medians; absolutes inflated by per-op sync — ratios are the
    signal): lm_head 2x(M=8) = 2.85x its qmm(M=16) time (~+5ms/step —
    dominant: the 78MB packed head streams twice), gate/up +62%,
    down +20%, attention projections neutral-to-faster (already qmv at
    M=16 on this GPU). Net trunk-matmul delta ~+8% in the micro-bench;
    on 14B-class production models the MLP share grows, trending the
    penalty toward +20-60% at width 16 with W=8, and roughly doubling on
    M1-class (W=5 -> 4 chunks). The Phase-4 3x batch-16 gate still
    clears on the pinned test models (~8x single-stream equivalent at 2
    chunks); the 14B target needs an engine-level bench before (b)
    becomes the default.
- Revised DECISION NEEDED:
  A') Sub-batched deterministic decode (3b) as the default for all
      archs: width-16 golden rounds become a hard BIT bar for every
      model, qwen fixtures land green, and ADR 0002's batched token-id
      bar is guaranteed by construction rather than empirical. First
      implementation step is the engine-level width-16 throughput bench
      on a large model, with (a)-style capping as the fallback if the
      penalty misses the perf gate. Calibrated W per device (5
      conservative constant; 9 measured here; 11 on 'd'-class).
  B') A' scoped to deterministic traffic only: greedy and seeded
      requests decode in <=W sub-batches, unseeded temperature>0 traffic
      keeps full-width steps (its draws are not reproducible
      run-to-run anyway). Keeps peak throughput for sampling workloads;
      costs step-builder complexity (partitioning by determinism class).
  C') The prior entry's options (per-arch xfail status / EOS-scoped bar /
      blanket width cap) — all now strictly weaker: the first encodes
      fixture luck, the second rescues 1 of 4 flips, the third is A'
      with worse throughput at equal bandwidth.
  My read: A' (with B' as a follow-on refinement if sampling throughput
  matters before Phase 9): it is the only option that makes the SPEC
  §6.6 invariant true by construction, on every arch and every GPU, and
  the measured cost on the pinned test models is acceptable.
- Next: PM ruling (A'/B'/fallback) -> implement + engine bench, land the
  held qwen fixtures, close Task 1, then Task 2 (Gemma, fixtures first).

## [2026-07-04] Phase 6 / Task 1 — B' design answers (recorded before implementation)
- Ruling implemented: B' — sub-batched deterministic decode, scoped
  per-request to `temperature == 0` or an explicit client seed. Two design
  questions answered per instruction:
- Q1 (mixed-batch mechanics): **selective partitioning, implemented as
  multiple forwards per step — not whole-step sub-batching.** A row's
  bits are a property of the FORWARD it rides in (MLX picks the kernel
  per dispatch from the total row count M), so "selective treatment
  inside one forward" does not exist: selectivity means partitioning the
  step's decode rows into separate forwards. That is structurally
  natural here because `StepBatch` already describes exactly one forward
  and `run_iteration` already issues several per step (prefill runs
  separately — the B1 padding relies on it). Decode rows partition into
  deterministic groups of <= W rows (bit-identical to M=1 by qmv row
  M-invariance) plus ONE unrestricted full-width group for
  non-deterministic rows. Correctness: sequences only gather their own
  history within a step (per-seq gather), so grouping and group order
  are value-neutral for other sequences; post-preemption replay rows of
  deterministic sequences ride <= W groups too, which is what keeps
  resumed streams bit-exact (any group size <= W reproduces M=1 bits,
  independent of the original group's size). The pipelined decode path
  partitions identically (its Lazy input is a concat of per-seq [1,1]
  arrays — trivially groupable). Trade-off accepted and stated:
  non-deterministic rows pay NOTHING (one full-width forward, no weight
  re-reads); deterministic rows pay ceil(det_rows/W) weight reads —
  the price of the guarantee; mixed steps cost one extra dispatch chain
  per group vs today's single batch. Whole-step sub-batching was
  rejected precisely because it would tax non-deterministic rows with
  the same re-read factor for no benefit.
  - Corollary the design must also close: **prefix-cache donation.**
    Generated-token rows of NON-deterministic sequences are computed at
    arbitrary M (ulp-off from M=1); donating them would let a later
    deterministic request reuse non-canonical KV and silently break its
    bit guarantee (SPEC: prefix caching must not change greedy outputs).
    Non-deterministic sequences therefore donate only their
    prefill-covered blocks (prefill is its own forward — canonical at
    any load); deterministic sequences keep donating everything, since
    their decode rows ARE M=1 bits. No regression for the multi-turn
    reuse story (greedy chat traffic is deterministic traffic).
- Q2 (threshold portability): **startup calibration, no hardcoding.**
  `AnyModel::calibrate_deterministic_width(stream)`: for each distinct
  linear shape in the LOADED model (attention/MLP projections + the
  (tied) lm_head, deduped by (K, N)), compute row 0 of the projection at
  M=1 and at rising M = 2..=32 over a realistic activation row
  (dequantized embedding row tiled to K); the first M whose row-0 bytes
  differ from M=1 is that shape's dispatch threshold on THIS device;
  W = min(thresholds) - 1, capped at 32 when nothing diverges. This
  measures the exact property the guarantee relies on (row-bit
  M-invariance) instead of replicating MLX's device table, so it is
  robust on GPUs the table doesn't describe, and probing through the
  same `Linear::forward`/`as_linear` code covers quantized and dense
  (BF16) weights alike. Engines built WITHOUT calibration default to
  W = 4 (safe under the smallest threshold, 6, anywhere in MLX's table)
  so an uncalibrated engine is conservative-but-correct; the worker and
  every real-model harness calibrate at load (expected: 9 on this
  machine, 5 on M1/M2-class, 11 on 'd'-class). Test: calibration
  self-consistency (rows bit-stable at W, divergent at W+1 — the
  hardware-independent definition of a correct answer) plus the value
  printed and recorded per run.

## [2026-07-05] Phase 6 / Task 1 — B' implemented; Task 1 CLOSED except one perf ruling
- What (implements the approved B' with the two recorded design answers):
  1. **Selective partitioning** (Q1 answer, as recorded): deterministic
     rows (temperature == 0 or explicit client seed —
     `SamplingOptions::explicit_seed`, worker maps proto `seed != 0`)
     decode in sub-batches of at most the calibrated width; NON-
     deterministic rows ride one unrestricted full-width forward in the
     same step. Both the synchronous and pipelined paths partition; the
     prefix cache is kept coherent by capping non-deterministic donors at
     their prefill-covered blocks (their decode rows carry arbitrary-
     width bits and must never serve a deterministic warm run).
  2. **Startup calibration** (Q2 answer, as recorded):
     `AnyModel::calibrate_deterministic_width` probes each distinct
     projection shape's row-bit stability boundary at model load — no
     hardware table. Measured: **9** on this machine (llama-1B, qwen-0.5/
     0.6B, and qwen3-8B all calibrate to 9; probe cost 100-480 ms).
     tests/calibration.rs pins the result against an independent raw-ops
     re-derivation (first divergence at M=10 -> width 9) — the hardware-
     independent correctness definition, valid on CI's different GPUs.
  3. WorkerInfo gains additive `max_deterministic_decode_width = 13`
     (diagnostics; python worker reports 0 = not guaranteed); python
     bindings regenerated.
  4. kiln-engine/tests/deterministic_partition.rs (checksum mock): det
     forwards <= W and closed-form through partitioning incl. pipelined
     Lazy groups; non-det forwards full-width; mixed steps decompose as
     det chunks + one full non-det forward; donation cap enforced (det
     extension of a non-det donor resumes at the prefill bound,
     warm == cold; det donors still donate generated rows). One-#[test]
     convention after a parallel-thread SIGSEGV flake (two live engines
     race the Metal stream) — fixed and re-run stable.
  5. Held qwen fixtures + the fetch-test-model.sh qwen2.5 pin LANDED.
- **Correctness result: the corrected bar holds everywhere.** All 16
  fixtures (llama 6, qwen2.5 5, qwen3 5) are bit-exact single-stream AND
  at decode width 16, all three architectures — the width-16 rounds are
  now guaranteed by construction, not argmax margin. Full suite: 40/40
  test targets, e2e 21, python worker 28, fmt/clippy/ruff + both CI
  shapes clean. Non-deterministic traffic verified unregressed
  (sampled-16: 334.1 pre-B' vs 331.6 tok/s under B' — noise).
- **Throughput gate (the ordered mixed-load batch-16 >= 3x): MISSES under
  B' for any load containing deterministic rows.** Attribution
  (llama-3.2-1b-4bit, release, single-stream 123.8 tok/s):
  ```
  greedy16  unpartitioned  410.4 tok/s  3.31x   (the historical gate)
  greedy16  B' (chunks 9+7) 259.9 tok/s 2.10x   B' cost -37%
  mixed16   single-forward  346.6 tok/s 2.80x   (pre-B' equivalent)
  mixed16   B' ([8]+[8])    222.3 tok/s 1.80x
  sampled16 (non-det path)  331.6 tok/s 2.68x   (sampler-op cost, pre-existing)
  ```
  8B-class scaling point (qwen3-8b-4bit @545dc425, ~/.kiln/bench-models/,
  loaded 10.2s, calibrated W=9 in 480ms, single-stream 19.5 tok/s):
  ```
  greedy16  unpartitioned   67.5 tok/s  3.46x
  greedy16  B'              44.8 tok/s  2.30x   B' cost -34%
  mixed16   B'              45.7 tok/s  2.34x
  sampled16 (non-det)       62.1 tok/s  3.18x   >= 3x
  ```
  Reading: the B' penalty is ~-35% at width 16 at BOTH scales (the
  MLP-share fear did not compound at 8B), non-deterministic loads clear
  3x at 8B (the 1B sampled miss is a small-model sampler-op artifact),
  and deterministic-containing loads sit at ~2.1-2.3x — the 3x gate is
  structurally out of reach for them at W=9: two sub-batches per step
  means every deterministic token pays roughly double the trunk weight
  bandwidth. That is the physical price of bit-exact batched greedy under
  this kernel pin, not an implementation inefficiency (the partition adds
  exactly the predicted 2x weight reads and nothing else measurable).
- **14B-class bench: DEFERRED — dev machine is 16GB.** The 14B-4bit
  weights (7.7GB) plus KV pool fit only with a truncated pool and no
  headroom; the download was additionally network-bound (~1MB/s shaped).
  The number must come from an M4 Pro/Max-class box (>= 24GB); CI cannot
  provide it either (GitHub macOS runners have less RAM than this
  laptop). The 8B point above is the stand-in scaling datum; the 14B run
  is a standing item for the first deployment-class hardware session.
- Deviations: none beyond the above.
- DECISION NEEDED (narrow, perf-only — correctness is closed): the
  batch-16 >= 3x gate as written cannot hold for deterministic-heavy
  traffic under B' (measured ~2.1-2.3x at 1B and 8B). Options:
  A) Re-scope the gate: >= 3x for non-deterministic/mixed-majority load
     (holds: 3.18x at 8B), with the deterministic-load number documented
     as the B' price (~2.3x) in the ADR — batched greedy buys bit-exact
     reproducibility with ~35% aggregate cost. RECOMMENDED: single-stream
     latency and non-det throughput are untouched, and greedy batch
     throughput still scales 2.3x.
  B) Per-op chunking (only the shapes whose threshold < batch width
     sub-batch; attention projections ride full width): recovers an
     estimated ~10-15% of the loss for ~150 lines + per-shape calibration
     plumbing. Can stack on (A) later; not a gate-saver alone.
  C) Revisit at the quarterly mlx-c bump (dispatch may change; the
     calibration adapts automatically).
- Next: perf ruling (A/B/C) -> record in ADR 0002 and close Task 1
  formally; then Task 2 (Gemma2/3, fixtures first). Qwen3-8B-4bit kept
  under ~/.kiln/bench-models/ for future phase benches.

## [2026-07-05] Phase 6 / Task 1 — op-level sub-batching investigation (PM-directed) — REPORT
- Scope: investigation only, per instruction. No source, fixture, or test
  changes; one scratch probe (crates/kiln-models/tests/opsplit_probe.rs)
  written, run, and removed after measurement, like the 2026-07-04
  batched-bar probes. Question: can splitting ONLY the threshold-crossing
  matmuls (MLP gate/up/down, lm_head) inside one full-width deterministic
  forward — attention, norms, and other threshold-safe ops shared at
  width 16 — match whole-forward B' bit-exactness at lower cost?
- 1) Threshold safety of the would-be full-width ops (measured):
  - SDPA never sees decode width at all: the paged decode path
    (nn.rs `Attention::forward_step`) slices per sequence and runs
    qk-norm/RoPE/gather/SDPA per-seq at q_len=1 regardless of batch
    width — width-invariant by construction, on every device. Its
    two-class q_len structure matters only for prefill (closed by B1
    padding). The "neutral" figure in the 2026-07-04 table was about
    the attention PROJECTIONS' dispatch cost, not SDPA.
  - Every other full-width non-matmul op is row-stable through M=32 on
    this machine (probe value 33 = never diverged): rms_norm=33,
    residual+swiglu elementwise=33, embed gather+dequant=33, both models.
  - Attention projections are shape-dependent, not device-constant:
    llama-1B shapes (K,N <= 2048) threshold 18 -> bit-safe at 16;
    qwen3-8B shapes (4096x4096, 4096x1024) threshold 12 -> NOT safe at
    width 16 even on this machine. MLP and lm_head: threshold 10 at both
    scales (consistent with the calibrated W=9).
- 2) Measured cost (release; each timing = median eval over one lazy
  graph of 16-64 independent reps — no per-op sync; us/dispatch):
  ```
  llama-1b     thr    M=16      9+7     11+5
  attn q/o      18   153.7    153.9    156.2   split-neutral
  attn k/v      18    43.5     42.3     42.9   split-neutral
  mlp gate/up   10   312.7    646.3   (11+5 inadmissible, thr 10)
  mlp down      10   312.4    634.4   (      "        )
  lm_head tied  10  4847.8   9952.1   (      "        )
  qwen3-8b
  attn q/o      12   338.0    647.4    596.9   NOT neutral; thr < 16
  attn k/v      12    87.7    152.8    152.9
  mlp gate/up   10   996.8   1963.7   (11+5 inadmissible, thr 10)
  mlp down      10  1005.4   1929.9   (      "        )
  lm_head       10 12326.1  23202.9   (      "        )
  ```
  Composed trunk time per width-16 step: 1B unpart 26.2 / whole-forward
  47.1 / op-level 47.1 ms; 8B 151.0 / 291.7 / 288.1 ms. Cross-check
  against the engine-measured B' deltas validates the attribution:
  probe 20.9 vs engine 22.6 ms/step at 1B (93% — the ~1.7ms residual is
  the doubled norm/embed/dispatch-chain overhead a single forward would
  also save); probe 140.7 vs engine 120.1 at 8B (probe overestimates
  there; conclusions below only get stronger).
  **Op-level recovery: 1B -0.03 ms/step kernel-side, ~+3% total with the
  residual overhead credited (predicted <= ~267 vs B' 259.9 tok/s);
  8B +3.6 ms/step, ~+1% (predicted 45.3 vs 44.8 tok/s).**
- 3) Why the intuition fails (and the earlier ~10-15% option-B estimate
  was wrong):
  a. Whole-forward splitting never RECOMPUTES attention — each sub-batch
     computes only its own rows. The only duplication is per-dispatch
     weight re-streaming plus small per-forward overhead; there is no
     shared attention output for op-level splitting to reuse.
  b. Where attention is split-neutral (1B: small, latency-bound
     matrices), unsplitting it saves nothing. Where splitting attention
     is expensive (8B: bandwidth-bound 4096-wide shapes, 1.9x), its own
     threshold (12) is below 16, so bit-exactness forces it to chunk
     anyway. The premise "attention is threshold-safe at width 16" fails
     on production-scale shapes on THIS machine, before M1/M2-class
     (lower thresholds still) even enters.
  c. MLP + lm_head — 80%+ of streamed bytes — sit at threshold 10 at
     both scales and chunk identically (<= 9 rows, 2 chunks at width 16)
     under either scheme. The dominant cost is untouchable.
  d. On M1/M2-class (W=5), all shapes compress toward the same bound:
     op-level converges to whole-forward exactly where B' hurts most.
  The B' penalty is the qmv-vs-qmm kernel-class efficiency gap on the
  bandwidth-dominant matmuls; it attaches to the kernel class the bit
  guarantee pins, not to how rows are grouped around it. No partition
  arrangement at this mlx-c pin can close it. The only structural lever
  is a kernel that streams weights once across row groups while keeping
  qmv row bits — an upstream-MLX question for the quarterly bump, not a
  Kiln scheduling question.
- Decisions: none required within latitude (no changes made).
- Deviations: none.
- Acceptance (real output of the scratch probe, trimmed; probe deleted
  after the run, `git status` clean save this entry):
  ```
  $ KILN_TEST_MODELS=~/.kiln/test-models cargo test -p kiln-models \
      --test opsplit_probe --release -- --nocapture
  === llama-3.2-1b-4bit ===
  full-width ops row-stability (33 = stable through M=32):
    rms_norm=33 elementwise=33 embed_gather_dequant=33
  trunk estimate/step @16: unpart 26.16ms | whole-forward 47.06ms |
    op-level 47.09ms
  cross-check: probe B' delta 20.90ms vs engine-measured 22.58ms
  op-level recovery -0.03ms/step -> predicted 259.8 tok/s (B' 259.9)
  === qwen3-8b-4bit ===
  full-width ops row-stability: rms_norm=33 elementwise=33 embed=33
  trunk estimate/step @16: unpart 150.95ms | whole-forward 291.69ms |
    op-level 288.05ms
  cross-check: probe B' delta 140.74ms vs engine-measured 120.11ms
  op-level recovery 3.64ms/step -> predicted 45.3 tok/s (B' 44.8)
  test result: ok. 1 passed
  ```
- DECISION NEEDED (supersedes the A/B/C perf ruling in the B' closeout
  entry): option B (per-op chunking) is measured dead — 0% recovery at
  1B, ~1% at 8B, best case ~3% crediting every avoidable overhead —
  against ~200-300 lines of per-shape calibration plumbing plus a per-op
  M-sensitivity classification that every future architecture (Gemma is
  next) would have to maintain. Whole-forward B' is the honest floor;
  ~2.1-2.3x at width 16 is the real, physical price of bit-exact batched
  greedy at this kernel pin. Remaining options:
  A) Re-scope the batch-16 >= 3x gate to non-deterministic/mixed-majority
     load (holds: 3.18x at 8B) and record the deterministic-load figure
     (~2.3x) in ADR 0002 as the documented B' price. RECOMMENDED.
  C) Additionally record in ADR 0002 that the only structural lever is
     kernel-level (a batched qmv streaming weights once across row
     groups), to be re-evaluated at each quarterly mlx-c bump — the
     startup calibration adapts automatically if dispatch changes. No
     code change now. Compatible with A.
  My read: A + C together, and strike B permanently.
- Next: PM ruling -> record in ADR 0002, close Task 1 formally; then
  Task 2 (Gemma2/3, fixtures first, SPEC §12 order).

## [2026-07-05] Phase 6 / Task 1 — perf ruling recorded (ADR 0003) — Task 1 CLOSED
- What (implements the PM ruling A + C on the two open DECISION NEEDED
  items — the B' closeout entry and the op-level investigation entry;
  both are hereby CLOSED):
  1. **ADR 0003**
     (docs/decisions/0003-throughput-bar-under-deterministic-sub-batching.md):
     records the finding (B' costs ~2.1-2.3x vs unpartitioned at width 16,
     attributable to qmv-vs-qmm weight re-streaming at the kernel-class
     boundary — a property of the pinned dispatch table, not a defect;
     op-level investigation measurements + the 93% methodology
     cross-check cited as evidence it is not recoverable by
     re-partitioning); amends the SPEC §11.3 batch-16 >= 3x target to
     non-deterministic/mixed-majority load; records ~2.1-2.3x as the
     deterministic-load measured FLOOR (not a target) with regressions
     below it failing phase gates like any bench regression; sets the
     revisit trigger at every mlx-c/core-MLX bump per ADR 0001's C1
     process, figures stale the moment the pin moves.
  2. New ADR vs appended ADR 0002 section (the call was delegated):
     new ADR — docs/decisions/ is agent-read-only once landed (the
     ADR 0001 addenda were explicit one-off PM instructions), and
     ADR 0002 defines correctness bars while this amends a performance
     acceptance target with its own revisit trigger; cross-referenced
     both directions in the ADR text instead.
  3. SPEC §11.3 perf bullet: doc-only amendment referencing ADR 0003
     (acceptance clause only; section otherwise untouched).
- Decisions: bench.sh phase runs to report det/non-det batch numbers
  separately (recorded as an ADR consequence; implementation rides the
  next bench.sh touch, no code change now).
- Deviations: none.
- Acceptance (doc-only change):
  ```
  $ ls docs/decisions/
  0001-mlx-c-pin.md
  0002-parity-bars-under-mlx-kernel-dispatch.md
  0003-throughput-bar-under-deterministic-sub-batching.md
  $ git diff --stat HEAD
  PROGRESS.md | docs/SPEC.md (1 line) | docs/decisions/0003-... (new)
  ```
- **Phase 6 / Task 1 (Qwen2.5/Qwen3 + parity/throughput bars) is CLOSED.**
- Next: Task 2 — Gemma2/Gemma3 impls, fixtures first (SPEC §12 Phase 6),
  width-16 parity verified against the ADR 0002/0003 bars.

## [2026-07-05] Phase 6 / Task 2 — Gemma2 + Gemma3 — DONE
- What (fixtures first, then implementation, per CLAUDE.md):
  1. **Golden fixtures** (sanctioned generator, pinned stack mlx.core
     0.31.1 / mlx-lm 0.31.2): tests/golden/gemma-3-1b-it-4bit/ (6 cases)
     and tests/golden/gemma-2-2b-it-4bit/ (6 cases), standard CASES list.
     New pinned test model `gemma-2-2b-it-4bit`
     (mlx-community @ 2c715097) — the smallest gemma-2 checkpoint that
     exists; gemma3 uses the already-pinned gemma-3-1b-it-4bit.
  2. **Shared-trunk extension** (nn.rs `TrunkOptions` — one parity-proven
     CausalLm for all five archs): `NormStyle::OnePlus` (gemma `1+w`
     RMSNorm, folded into the weights at load — same add, same bits,
     fewer ops), `Activation::GeluApprox` (`mlx.nn.gelu_approx`
     op-for-op in the reference's evaluation order), gemma sandwich
     `Block` (post-attention/pre-/post-feedforward norms on sublayer
     OUTPUTS), `clip_residual` (gemma3: f32 residual adds clipped to the
     f16 range), embedding scaling (gemma3: bf16(sqrt(hidden)) cast
     in-graph = f16 34.0 verified against the reference; gemma2: weak
     f32 scalar), final logit softcapping (gemma2), per-layer
     `mk_rope(layer)` (gemma3 local 10k / global 1M+scaling), qk-norm
     with the trunk's norm style (gemma3), `scale_override` computed
     with each arch's own f64 formula, and **manual softcapped
     attention** for gemma2 (`tanh(scores/cap)*cap`, boolean causal mask
     via `where`, `softmax(precise)` — `mx.fast` SDPA is unusable under
     softcapping, so the reference matmuls scores/probs explicitly) on
     BOTH decode paths. kiln-mlx gains `ops::tanh` (bindgen surface
     already had it). `weak_scalar` helper mirrors MLX's Python-float
     promotion (f32 value cast to the tensor dtype at the op) so scalar
     ops run in the reference's dtype.
  3. **Arch modules + dispatch**: gemma3.rs / gemma2.rs, Gemma2Config /
     Gemma3Config (defaults mirror the reference ModelArgs; gemma3 tie
     decided from the checkpoint like the reference's `sanitize` — the
     1B ships an lm_head, so it runs untied), ArchConfig/AnyModel
     variants, SUPPORTED_ARCHITECTURES += gemma2, gemma3_text
     (multimodal "gemma3" stays python-routed).
  4. **Parity envelopes, worker-enforced** (existing CTX_OVERFLOW path):
     - gemma3: total context (prompt+generated) <= sliding_window (512).
       The reference's RotatingKVCache(keep=0) is bit-for-bit a plain
       temporal cache BELOW the window and its masks are plain causal;
       ABOVE it the buffer rotates (ring order) and serving without
       ring-order gather would silently diverge from the reference and
       from the model's own window semantics. modelinfo.rs min's the
       advertised max_context_len with the window.
     - gemma2: prompt <= 2048 (one mlx-lm `prefill_step_size` chunk) +
       the single-tail prefill schedule (worker engine_main and the
       golden harness set `prefill_fine_chunk = prefill_chunk` when the
       model declares `monolithic_prefill_required`). Manual-attention
       score/prob matmul row bits depend on the key-axis length and the
       reference's bool mask exists only at offset 0 (KVCache has no
       `make_mask` at the pin), so only reference-shaped monolithic
       pieces are parity-defined. New `StaticInfo.max_prompt_len` backs
       the admission check.
- **Result: 28/28 fixtures bit-exact** — llama 6, qwen2.5 5, qwen3 5,
  gemma2 6, gemma3 6 — single-stream AND at decode width 16 under B'
  (both gemmas calibrate deterministic width 9 on this machine, probing
  through the untied 262k-vocab gemma3 head and gemma2's tied embedding
  alike). Both models were bit-exact on the FIRST harness run.
- Decisions (within latitude):
  - gemma3 window-crossing support deferred behind a hard cap rather
    than shipped semantically wrong (unwindowed attention past 512).
    Lifting it = ring-order decode gather (slot = closed-form rotation
    of the reference buffer) + reference-shaped >window prefill masks +
    a dedicated >512-token fixture; recorded as the follow-up item.
  - gemma2 fine grid off is the Phase-5-B2 tradeoff scoped to one arch
    (forfeits sub-2048 multi-turn prefill reuse for gemma2 only) —
    required for mask semantics, not a perf preference.
  - qk-norm slot: the reference norms after the head transpose, Kiln
    norms before it (identical row content either way, both reduce over
    head_dim) — kept Kiln's existing slot; verified empirically by the
    bit-exact fixtures.
- Deviations: fetch-test-model.sh gained the gemma-2-2b-it-4bit pin (new
  entry; existing pins untouched) — same shape as the Phase 6 qwen2.5
  addition, unavoidable for gemma2 coverage. Flagging per protocol.
- Known issue (pre-existing, NOT addressed here):
  `cargo test -p kiln-engine --test sampler --release` dies with
  SIGBUS/SIGSEGV — reproduced on clean HEAD cce71d7 (pre-gemma), debug
  mode passes, and the CI gates run debug, which is why it was never
  seen. Discovered via an exploratory --release suite run. Needs its own
  session; test not weakened, nothing skipped.
- Acceptance (real outputs, trimmed):
  ```
  $ uv run ... python scripts/gen-golden.py --model gemma-3-1b-it-4bit ...
  wrote 6 fixtures  (chat-basic 21 tok ... raw-long-prefill 242 tok)
  $ uv run ... python scripts/gen-golden.py --model gemma-2-2b-it-4bit ...
  wrote 6 fixtures  (prompts 6..241 tok; all within both parity caps)
  $ cargo test -p kiln-models --test golden --release -- --nocapture
  == gemma-3-1b-it-4bit: model_type=gemma3_text, 6 fixture(s), det width 9
  6x "exact match (batched/paged engine)" + 6x "exact match at decode
  width 16 (B' deterministic sub-batching)"
  == gemma-2-2b-it-4bit: model_type=gemma2, 6 fixture(s), det width 9
  6x exact + 6x exact at width 16
  (llama 6, qwen2.5 5, qwen3 5 all still exact both rounds)
  test result: ok
  $ cargo test --workspace --no-fail-fast   (debug = CI shape)
  38/38 test targets ok (golden, calibration, deterministic_partition,
  batching, prefill_pad, prefix suites, both leak gates, worker rpc)
  $ cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings
  clean (also clean with --no-default-features; linux compile-check builds)
  $ uv run --project tests/e2e pytest tests/e2e -q -> 21 passed
  $ KILN_TEST_MODELS=... pytest python/kiln_worker_py/tests -> 28 passed
  $ ruff check / ruff format --check python/ tests/e2e -> clean
  ```
- Next: Task 3 — 8-bit and BF16 dtype matrix (SPEC §12 Phase 6 order);
  standing items: gemma3 window-crossing (ring-order gather + fixture),
  release-mode sampler crash (separate session).

## [2026-07-05] Phase 6 — release-mode sampler SIGBUS root-caused + fixed (pre-Task-3 blocker) — DONE
- Scope: the crash flagged in the Task 2 entry
  (`cargo test -p kiln-engine --test sampler --release`: SIGBUS/SIGSEGV,
  release-only, reproduced on clean pre-gemma HEAD). PM-directed:
  backtrace-first diagnosis, soundness verdict on kiln-mlx, fix,
  release-exercised regression coverage, close the CI debug-only blind
  spot. (A chip session was independently started on this bug; this
  session's fix is the authoritative one — the two will conflict if both
  land.)
- Diagnosis (evidence chain, no guessing):
  1. lldb: fault inside MLX `ArrayDesc::~ArrayDesc` — an atomic refcount
     op through a garbage control-block pointer. Downstream symptom, not
     the source (unwinder could not walk further).
  2. Guard Malloc (`libgmalloc`) moved the FIRST fault to
     `mlx::core::random::split` reading the key array's descriptor: the
     key handle's C++ object was freed memory. Use-after-free, not
     misalignment.
  3. Section bisect (temporary markers): sampler A finishes its 16
     seeded draws; the crash is sampler B's FIRST draw. Reduced to a
     ~30-line kiln-mlx-only reproducer: struct-held `Option<Array>` key
     chain, sampler passed BY VALUE into a closure, two samplers drawn
     back to back. kiln-engine exonerated.
  4. Env-gated Drop logging under gmalloc (no address reuse there): the
     fatal split consumed exactly the box freed by sampler A's own —
     correct — drop. Sampler B, freshly constructed with `key: None`,
     `take()`-returned `Some(<A's dangling key>)`.
  5. lldb at the closure's two entries: call 1 argument slot =
     `{0, 0}` (proper None); call 2, SAME slot = `{1, <dangling>}` —
     the second by-value temporary was never re-initialized.
  6. `--emit=mir` (stable): `KeyOnly { key: None }` is built ONCE
     (`_22`/`_21`); call 1 receives `copy _21`, call 2 `move _21`.
     rustc's MIR GVN merged the two identical constant argument
     temporaries; under the indirect by-value ABI the callee mutates and
     drops IN the caller's slot, so call 1 tears the memory that call
     2's `move` then reads.
- **Verdict (the soundness call, per instruction): NOT a Kiln soundness
  hole and NOT a latent flaw an existing SAFETY comment failed to cover.**
  Every corrupted state transition (`Option::take`, `Some(next)`
  assignment, drop) is safe Rust; every FFI call received pointers valid
  at call time (kiln-mlx handle ownership re-reviewed against mlx-c's
  vendored `mlx_array_new_/set_/free_` — holds). This is a **rustc
  1.96.1 (aarch64-apple-darwin) miscompilation**, reproducible at every
  opt-level >= 1 and absent at opt-0 — which is exactly why the
  debug-only CI lane never saw it. Trigger shape: a
  constant-constructible aggregate with drop glue and no niche
  (`Option<Array>` behind a raw-pointer handle) passed by value twice
  from one frame.
- Fix (workaround in Kiln; the compiler bug itself is upstream):
  1. **`Sampler.key` is now eager** — `key: Array` created from the seed
     at construction (`Sampler::new`/`greedy` return `Result`); the
     opaque FFI call makes the constructed value unprovable-identical
     (GVN cannot merge) and deletes the stale-discriminant layout
     entirely. Sampling behavior is bit-identical: the first draw still
     splits `key(seed)` exactly as before (same-seed/different-seed
     tests pin it; full determinism suite green).
  2. `Engine::submit`: sampler construction failure emits the proto's
     in-band `Finished{error}` pre-admission (no engine resources exist
     yet).
  3. Exposure survey: `Sampler` was the workspace's only
     constant-constructible by-value-passable `Option<Array>` aggregate
     (kv_cache/Linear-bias/Block-norms/TrunkOptions are load-time or
     heap state, never constant argument temps). Production `submit`
     builds samplers from runtime options (not GVN-mergeable), so
     serving output was almost certainly unaffected — the landmine was
     test-shaped today, refactor-shaped tomorrow; it is gone either way.
  4. ADR 0003's release-measured throughput figures stand: the engine
     bench paths never construct constant sampler temporaries, and the
     entire workspace suite now passes in release, including golden
     parity (a stronger statement than was ever true before).
- Regression coverage (release-only failure => release-exercised test):
  `same_seed_same_tokens` IS the trigger shape and crashed in release
  before the fix; it now carries a REGRESSION SHAPE doc contract (keep
  the constant options + by-value closure structure) and runs under the
  new CI release lane, where it fails on any reintroduction of a
  constant-constructible sampler on an affected toolchain.
- CI blind spot closed: new `test-macos-release` job runs the full
  `cargo test --workspace --release` on Apple Silicon (job comment
  records why: optimized builds are what ship and what every ADR 0003
  benchmark measured).
- Upstream: minimal-repro recipe + MIR evidence recorded here for a
  rust-lang/rust report; a kiln-free minimization needs care (naive
  reductions were defeated by niche layouts and IPSCCP — the reproducer
  needs a no-niche Drop aggregate, opaque callees, and constant
  construction). Standing item: file upstream + re-test the workaround at
  every toolchain bump.
- Deviations: none. All scratch diagnostics (two scratch test files;
  temporary eprintlns in random.rs/array.rs/sampler.rs; the Drop
  backtrace hook) removed — the committed diff is the fix, the test
  updates, and the CI lane only.
- Acceptance (real outputs, trimmed):
  ```
  $ cargo test -p kiln-engine --test sampler --release   (was: SIGBUS 100%)
  test sampler_behavior ... ok        (x3 consecutive runs + debug run ok)
  $ cargo test --workspace --no-fail-fast                (debug, CI shape)
  zero "failures:" / "test result: FAILED" across all targets; exit 0
  $ cargo test --workspace --release --no-fail-fast      (NEW bar)
  zero failures; exit 0 — golden parity, leak gates, engine suites all
  green in release for the first time
  $ cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings
  clean (also --no-default-features clippy + build: clean)
  $ uv run --project tests/e2e pytest tests/e2e -q -> 21 passed
  $ KILN_TEST_MODELS=... pytest python/kiln_worker_py/tests -> 28 passed
  ```
- Next: Task 3 — 8-bit and BF16 dtype matrix (SPEC §12 Phase 6 order).
  Standing items: gemma3 window-crossing (ring-order gather + fixture),
  rustc miscompile upstream report.

## [2026-07-05] Phase 6 — yarn freq unit test red on CI (1-ulp) — root-caused + bar corrected — DONE
- What:
  - `nn::tests::yarn_freqs_match_reference_bit_for_bit` (kiln-models) was
    red on main in BOTH `test-macos` and `test-macos-release` (run
    28736232673): `freq[3] = 0x3FF49A1A` on CI vs fixture `0x3FF49A1B`,
    identical in debug and release — deterministic per machine, never
    reproducible locally.
  - Diagnosis (measured, not guessed): the divergence is **MLX/Metal
    computation, NOT Rust host-side floating point**. freq[3]'s dataflow
    provably excludes every host-side float: `low`=23.596/`high`=39.651
    are ~0.4 from their floor/ceil boundaries (libm ulp noise is ~1e-16
    relative), the ramp clips to exactly 0 at index 3 so `freq_mask`=1.0
    exactly, and the host-f64 `mscale` assertion passed byte-for-byte on
    CI. A scratch provenance test (removed) isolated the op: the
    mul/div tail `(4e·e)/(4e)` is bit-transparent for both candidate
    bits (strict-f32 emulation AND MLX on both streams), so the
    divergent op is `ops::power` itself. Measured pow(1e6, 0.046875),
    true value 1.91095297497…: local Metal `0x3FF49A1B` (0.41 ulp),
    CI macos-14 paravirtual GPU `0x3FF49A1A` (1.41 ulp), MLX CPU
    backend on this same machine `0x3FF49A1C` (0.59 ulp), host Rust
    powf f32/f64 `0x3FF49A1B`. Three faithful, per-device-deterministic
    pow implementations spanning 2 ulp. Metal transcendentals carry no
    correctly-rounded or cross-device guarantee — a cross-machine
    bit-exact bar on raw f32 intermediates was never achievable
    (ADR 0002's per-device/kernel-class observation at unit-test scale).
  - Fix: test renamed `yarn_freqs_match_reference_within_ulp_tol`;
    per-element ulp bound with a failure path that reports EVERY
    offending index + worst ulp, so one CI run yields the whole
    spectrum. That reporting immediately earned its keep: a first PR
    run at tolerance 2 (the pow spread at index 3) was red on CI with
    the full spectrum revealed — all 64 elements within 2 ulp EXCEPT
    freq[39] = 15407.383 vs 15407.387: 4 ulp (2.6e-7 relative).
    Index 39 is the deepest interpolation-blend element (ramp = 16/17,
    mask = 1/17), where the (I*E)/(I*m + E*(1-m)) divide chain
    AMPLIFIES the pow spread instead of passing it through
    bit-transparently as at mask-saturated indices — consistent with
    ~2x amplification of a ±1.5-ulp pow difference, and hundreds of
    thousands of ulp away from any real defect. Final bound:
    YARN_FREQ_ULP_TOL = 8 (2x the observed worst, ~1e-6 relative —
    the rel-epsilon band anticipated for f32). `mscale` assertion
    stays bit-exact (host f64; f64->f32 rounding absorbs libm ulp
    noise; CI agrees). Test doc states explicitly why bit-exact was
    wrong HERE and why the golden bar (bit-exact token ids, same-device
    reference) is different and must never be relaxed citing this.
    `Rope::new` doc amended: freq tables match the Python reference
    bit-for-bit *on the generating device*.
- Decisions:
  - Tolerance over recomputing freqs host-side: nn.rs mirrors mlx-lm's
    MLX graph op-for-op so same-device golden parity holds; moving the
    table to host math would break that for a non-goal (cross-machine
    float bit-equality). Test-bar correction only; zero model-code
    change.
  - Assertion weakened only because the bar was unattainable-by-
    construction, and under explicit human direction (task prompt);
    recorded here per the never-weaken-a-test rule.
- Deviations: none.
- Acceptance:
  ```
  $ cargo test -p kiln-models --lib yarn_freqs
  test nn::tests::yarn_freqs_match_reference_within_ulp_tol ... ok
  $ KILN_TEST_MODELS=~/.kiln/test-models cargo test -p kiln-models --test golden
  test greedy_parity_is_exact_for_every_fixture_model ... ok (217.30s)
    (all fixtures, all archs, single-stream + width-16 rounds per
     ADR 0002/0003 — unchanged, as expected: NO committed fixture model
     exercises the yarn path (qwen3/qwen2.5/gemma-2/gemma-3 configs have
     rope_scaling: None; llama-3.2 uses the llama3 branch), so this unit
     test is the only guard on yarn freqs; propagation into golden token
     ids is structurally impossible for the committed set. Note: CI's
     `cargo test --workspace` runs before models are fetched, so the
     golden harness skips on CI and has only ever run on dev machines.)
  $ cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings
  clean (also clippy --no-default-features: clean; ruff check + format: clean)
  $ cargo test -p kiln-models -> 11 passed lib + integration green
  CI (PR #7): run 28736761483 at tol=2 -> test-macos red, full-spectrum
  report "worst 4: freq[39] ... 4 ulp" (the measurement above);
  final run at tol=8 -> all four jobs green (run id + confirmation in
  the PR thread; merged to main only after green).
  ```
- Next: Task 3 — 8-bit and BF16 dtype matrix (SPEC §12 Phase 6 order),
  unchanged. Standing items unchanged (gemma3 window-crossing fixture,
  rustc miscompile upstream report).

## [2026-07-05] Phase 6 — CI never runs the model-gated Rust test tier — analysis + options — BLOCKED
- What:
  - Confirmed the gap flagged in the yarn-freq entry above, and it is much
    wider than the golden harness. In `.github/workflows/ci.yml`,
    `test-macos` runs `cargo test --workspace` (line 83) BEFORE the model
    cache/fetch steps (lines 100–108), and `KILN_TEST_MODELS` is exported
    only inside the python-worker and e2e pytest steps (lines 109–117) —
    never for any cargo step. `test-macos-release` has no model steps at
    all. Every `KILN_TEST_MODELS`-gated Rust test therefore skips silently
    in CI and has only ever run on dev machines: 14 test files across
    kiln-models (golden, batching, preemption, prefix_cache,
    prefix_multiturn, prefill_pad, calibration, leak, leak_batched,
    throughput, prefill_schedule_bench), kiln-worker (rpc), and
    kiln-tokenize (tokenizer, detok — pure-CPU; skipped purely for lack of
    the env var).
  - SPEC mandates this coverage. §11.2: the golden harness — "Every
    model-impl PR runs this." §11.3: the integration tier (worker gRPC
    black-box, preemption, prefix-cache hit counters) is explicitly
    assigned to the "macOS runner". §3 CI row: "macOS-14 arm64 runner for
    Metal tests". The current workflow is a SPEC gap, not a deliberate
    scope choice; nothing in git history or PROGRESS records deciding to
    keep these dev-only.
  - Classified the gated tier by ADR 0002 cross-device exposure:
    - Device-independent bars (assert same-device self-consistency or
      structural invariants; valid on any GPU): batching (engine vs
      contiguous, concurrent vs solo), prefix_cache*, prefix_multiturn
      (warm vs cold), preemption* (preempted vs unpreempted), leak,
      leak_batched (live-object counter), calibration (its doc already
      anticipates CI: "CI's M1-class GPUs have different thresholds than
      the dev machine; both must satisfy it"), rpc (its READY_TIMEOUT
      comment anticipates "a cold CI runner"), tokenizer, detok.
      (*prefix_cache also asserts >= 5x TTFT — wall-clock on a shared
      paravirtual runner, a flake risk independent of the bar question;
      *preemption includes one committed-fixture round, see below.)
    - Committed-fixture comparisons (cross-device-SENSITIVE — the
      ADR 0002 caveat): golden (5 fixture models x 6 cases, single-stream
      + width-16 rounds, bit-exact token ids vs fixtures generated by
      mlx-lm ON THE DEV MACHINE), prefill_pad (raw-tiny-remainder
      fixture), preemption's golden-fixture scenario.
    - Already `#[ignore]`d perf gates, unaffected by any option here:
      throughput, prefill_schedule_bench.
  - Marginal-signal analysis (what CI enablement would actually add):
    the e2e suite ALREADY runs cross-worker greedy parity on the CI
    runner (test_cross_worker_parity.py: Rust worker vs mlx-lm python
    worker, identical greedy text, same device) and it is green on main —
    so "Rust == mlx-lm on the same device" is already CI-covered
    end-to-end (text-level, small prompt set). What CI has NEVER probed
    is whether the committed dev-machine fixture token ids reproduce on
    the macos-14 paravirtual GPU at all — for either worker (python
    worker tests are protocol-shape only; no CI test compares anything
    against tests/golden fixtures). A CI golden failure would therefore
    mean "the fixtures are device-specific", not "Kiln has a bug" — and
    every remedy is ADR-level: SPEC §11.2's relaxed bar "requires an
    ADR", the never-weaken rule applies, and ADR 0002 already records
    that a foreign-device token-id pass is argmax margin, "fixture-,
    model-, and GPU-dependent". Failure risk is real, not hypothetical:
    the yarn incident measured the CI GPU computing pow() 1.41 ulp off
    (4 ulp after one blend), and ADR 0002 records qwen fixtures flipping
    a token at position 28–33 under same-magnitude ulp noise. If goldens
    go red on main, no agent can fix it: CI is hostage until a PM ruling.
  - Cost side: run 28737060728 (current main, warm caches): test-macos
    9m19s, test-macos-release 3m53s, macOS billed at 10x linux. Golden
    harness alone: 217s debug on the M4-class dev machine; macos-14
    runners are ~3-core M1 paravirtual, so expect ~1.5–3x that, plus the
    batching/preemption/prefix/leak suites (unmeasured; each drives
    10^2–10^3 engine steps on real weights — plausibly comparable in sum
    to golden again). Rough estimate: debug lane grows from ~9 min to
    ~25–40 min per PR; doubled again if the release lane gets models too.
    Model payload ~3.6 GB, already cached in test-macos under
    `test-models-${hashFiles('scripts/fetch-test-model.sh')}`; a second
    job restores the same entry at no extra cache cost.
- Decisions: none — enabling the fixture-comparing suites on a second GPU
  class is a parity-bar question (below), and the right workflow mechanics
  (job-level env export vs per-step split) depend on its answer, so no
  workflow change was made.
- Deviations: none (analysis only; no code, workflow, or test change).
- Acceptance: (analysis task — evidence, not build gates)
  ```
  $ grep -c KILN_TEST_MODELS .github/workflows/ci.yml        -> 3
    (lines 111, 116: pytest steps only; line 104: cache path; no cargo
     step sees it; fetch at line 106 runs after cargo test at line 83)
  $ gh run view 28737060728 --json jobs ...
  test-macos: success 10:01:41 -> 10:11:00   (9m19s)
  test-macos-release: success 10:01:41 -> 10:05:34   (3m53s)
  $ grep -rl KILN_TEST_MODELS crates/*/tests/ -> 14 files (list above)
  $ ls tests/golden/ -> 5 model dirs x 6 fixtures
  ```
- Next: Task 3 — 8-bit and BF16 dtype matrix (SPEC §12 Phase 6 order),
  unchanged; the CI change executes immediately once the ruling below
  lands. Standing items unchanged (gemma3 window-crossing fixture, rustc
  miscompile upstream report).
- DECISION NEEDED: does the SPEC §11.2 bit-exact golden bar apply on
  devices other than the fixture-generating one? Concretely: which suites
  may gate CI on the macos-14 paravirtual GPU?
  - **Option A — full enablement.** Move cache/fetch above the cargo
    steps, export KILN_TEST_MODELS at job level in both macOS jobs.
    Pro: SPEC-conformant ("every model-impl PR runs this"); the whole
    Phase 4–6 integration surface finally gates PRs; closes the exact
    blind-spot class that let the yarn regression reach main. Con:
    silently extends the §11.2 bit-exact bar to a GPU class it has never
    been validated on; one legitimate cross-device argmax flip turns main
    red with no agent-fixable path (relaxed bar requires an ADR;
    never-weaken applies); +15–30 macOS-billed min/PR; prefix_cache's 5x
    TTFT wall-clock assertion now runs on shared runners.
  - **Option B — split enablement.** Keep `cargo test --workspace`
    env-less; after fetch, add a blocking step running the
    device-independent suites by explicit `--test` target with the env
    exported, and a `continue-on-error: true` advisory step running the
    fixture-comparing suites (golden, prefill_pad, preemption). Pure
    workflow change, zero Rust/test edits, no bar weakened (the bar's
    device scope is left undecided; advisory red blocks nothing). Pro:
    most of the coverage risk-free NOW, and the advisory lane gathers the
    exact evidence the bar ruling needs (do dev fixtures reproduce on
    macos-14 at all?). Con: advisory lanes rot; the workflow carries an
    explicit --test list a future test file must be added to (recreating
    this gap one level down unless a workspace-level guard is added);
    "keystone gates every PR" still not literally true until promoted.
  - **Option C — status quo, documented.** Golden stays a dev-machine
    phase-acceptance gate recorded in PROGRESS. Pro: zero cost/risk now.
    Con: leaves the §11.2/§11.3 CI mandate unmet; the silent-skip trap
    stays armed for every future integration test.
  - Sub-question if A or B: debug lane only, or release lane too? The
    release lane exists because of the release-only rustc GVN miscompile
    (PROGRESS 2026-07-05) — model-path coverage there has independent
    value, at roughly double the added minutes.

## [2026-07-05] Phase 6 — CI Option B implemented: model-gated tier blocking, fixture parity advisory — DONE
- What:
  - `.github/workflows/ci.yml` `test-macos`: model cache/fetch moved ABOVE
    the cargo steps (fixing the ordering bug — `cargo test --workspace`
    previously ran before models existed, so every gated suite silently
    skipped in every CI run to date). Two new steps after the env-less
    workspace run: a BLOCKING "Model-gated suites (device-independent)"
    step (kiln-tokenize tokenizer+detok, kiln-models batching/calibration/
    leak/leak_batched/preemption/prefill_pad/prefix_cache/prefix_multiturn,
    kiln-worker rpc; `KILN_TEST_MODELS` exported, `KILN_FIXTURE_PARITY=skip`)
    and a `continue-on-error` ADVISORY "Golden parity vs committed
    fixtures" step (golden harness + `KILN_FIXTURE_PARITY=only` runs of
    preemption/prefill_pad; both halves run regardless of which fails).
    `test-macos-release` untouched per the ruling.
  - `KILN_FIXTURE_PARITY` (unset|empty = everything, `skip` =
    device-independent only, `only` = committed-fixture comparisons only;
    anything else panics) added to `tests/preemption.rs` (scenario 4 is
    the fixture side; 1–3, 5–8 the device-independent side) and
    `tests/prefill_pad.rs` (cold-vs-fixture assert is the fixture side;
    the cold run itself always executes as the local reference).
  - `prefill_pad` rerun/extension now compare against the same-process
    cold run instead of the fixture ids (assertion-equivalent when the
    cold-vs-fixture assert holds, and the actual containment invariant —
    valid on any GPU); the extension prompt extends with the cold run's
    tokens for the same reason.
- Decisions:
  - Env-var scenario split over splitting into multiple `#[test]`s: the
    files are single-`#[test]` by design (process-global live-object
    counter; libtest would run sibling tests in parallel threads).
    Unset = run everything, so the dev-machine bar is unchanged; every
    assertion still runs by default and in exactly one CI step. No bar
    weakened.
  - Helper duplicated across the two test files (integration tests can't
    share code without a common-mod file; two 15-line copies beat that
    machinery). Cross-referenced keep-in-sync comments.
- Deviations: none.
- Acceptance:
  ```
  Local (dev machine), both split suites, all three modes:
  $ cargo test -p kiln-models --test prefill_pad            -> ok (5.43s)
  $ KILN_FIXTURE_PARITY=skip ... --test prefill_pad         -> ok; "cold-vs-fixture
    compare deferred"; rerun hit 137, extension hit 128 — warm == cold exact
  $ KILN_FIXTURE_PARITY=only ... --test prefill_pad         -> ok; "cold == committed
    fixture, exact"; rerun/extension skipped
  $ cargo test -p kiln-models --test preemption             -> ok (17.86s, all 8 scenarios)
  $ KILN_FIXTURE_PARITY=skip ... --test preemption          -> ok (scenarios 1-3,5-8)
  $ KILN_FIXTURE_PARITY=only ... --test preemption          -> ok (scenario 4 only)
  $ cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings
  clean (also --no-default-features clippy, ruff check + format: clean)

  CI: PR #8, run 28753659372 (commit e5b3f47) -> conclusion SUCCESS.
  Blocking step: ALL GREEN on the clean macos-14 runner — 15 tests / 11
  binaries: detok 2, tokenizer 4, batching 20.1s, calibration 1.2s,
  leak 15.7s, leak_batched 104.3s, preemption 30.0s, prefill_pad 4.9s,
  prefix_cache 10.5s (incl. the >=5x TTFT assert), prefix_multiturn
  38.3s, rpc 2 tests 6.7s. Step ~4m21s; test-macos total 11m03s
  (was 9m19s on main). Model cache hit (fetch skipped).
  ```
- FIRST CROSS-DEVICE EVIDENCE (advisory step, run 28753659372 — the
  golden harness's first-ever CI execution): **FAILED**, exactly the
  ADR 0002 class. Detail:
  - gemma-2-2b-it-4bit: all 12 rounds EXACT on the paravirtual GPU
    (6 fixtures single-stream + 6 at width 16, calibrated width 9).
  - gemma-3-1b-it-4bit/chat-basic, single-stream: identical for the
    first 49 generated tokens, diverges at token 50 of 64 — CI sampled
    188797 where the fixture has 195597 (golden.rs:284). Divergence
    position is BEYOND token 48, i.e. within the position clause of the
    SPEC §11.2 relaxed bar; the logprob-delta clause is unmeasured (the
    harness has no logprob instrumentation). The harness fails fast, so
    the remaining gemma-3 fixtures and all llama/qwen2.5/qwen3 golden
    rounds are still UNPROBED cross-device.
  - Partial llama evidence via the `only`-mode runs, both EXACT on CI:
    preemption scenario 4 (chat-code fixture, preempt+resume path) and
    prefill_pad cold-vs-fixture (raw-tiny-remainder).
  - Advisory-step red is visible only in the step log/annotation — the
    job and run stay green (continue-on-error). Anyone assessing golden
    status on CI must read the step log, not the run conclusion.
  This is evidence FOR the open golden-bar device-scope ruling (the
  DECISION NEEDED in the analysis entry above): a real single-stream
  argmax flip from per-device Metal ulp differences, on the committed
  fixtures, with no Kiln code change implicated (same commit is 12/12
  exact on gemma-2 and exact on both llama probes). Resolution options
  are ADR-level per SPEC §11.2; no test or bar was touched in response.
- Next: Task 3 — 8-bit and BF16 dtype matrix (SPEC §12 Phase 6 order),
  unchanged. The golden-bar device-scope ruling remains open, now with
  concrete evidence. Standing items unchanged (gemma3 window-crossing
  fixture — note the cross-device flip is also gemma-3 — rustc miscompile
  upstream report).

## [2026-07-05] Phase 6 — independent re-verify: GVN exposure sweep confirms Sampler-only; gemma-3 divergence logprob delta measured — DONE
- What (task A — coverage check on the sampler-UAF exposure survey):
  - Independently re-swept the workspace for the GENERAL trigger shape
    (struct passed by value + two provably-identical constant call sites
    + callee mutates a field dropping the old value), not just the
    `Option<Array>` scoping of the original survey. Method: (1) every
    `.take()`/`mem::replace`/`mem::take`/`mem::swap`/`Option::replace`
    site in the workspace — 22 sites; (2) every `Option<Array>` field and
    every `Option<handle>` of other kiln-mlx types; (3) every `impl Drop`
    type; (4) every by-value `mut`-bound struct parameter.
  - Result: **original claim CONFIRMED — Sampler was the only exposure.**
    All 22 extract-and-replace sites sit behind `&mut self`, heap
    indexing (`self.nodes[..]`, `self.pools`, `self.running[j]`), TLS
    (`RefCell` in kiln-mlx error slot), or child-process handles — zero
    by-value receivers. `KvCache` is the one OTHER constant-constructible
    no-niche double-`Option<Array>` aggregate with the interior
    take+reassign shape (kv_cache.rs:54), but it is only ever reached via
    `&mut KvCache`/`&mut [KvCache]`/`Vec<KvCache>` — never a by-value
    param anywhere in src or tests (the Vec-collect construction stores
    resourceless `{None,None,0}` bytes to distinct heap slots — sound
    even if const-merged). Remaining by-value params fail the other
    conditions: `SamplingOptions`/`PenaltyOptions` are all-scalar (no
    drop glue — merging Copy data is sound); `supervise(mut ctx:
    SuperviseCtx)` has one call site, all-runtime niched fields
    (Arc/Vec/watch channels), and no field-drop; `finish()` takes
    `seq: &mut Seq` (the by-value move at engine.rs:606 is a reborrow
    into `donate`). `Array`/`Stream`/`VectorArray` are FFI-only
    constructible (unprovable-identical). The one intentional instance of
    the trigger shape is tests/sampler.rs's REGRESSION SHAPE canary, as
    documented. Nothing was left unchecked by the original survey; its
    `Option<Array>` scoping was the right cut (per the recorded MIR
    evidence the merge requires a no-niche constant aggregate, and
    Option<Array> is the workspace's only such layout).
- What (task B — logprob delta at the gemma-3-1b/chat-basic divergence):
  - CI logit access impractical (no logprob instrumentation in the
    harness); reproduced locally on the generating device instead, per
    instruction, replaying the EXACT fixture-generation path
    (mlx-lm 0.31.2 generate_step, mlx.core 0.31.1, greedy, gen-golden.py
    prompt encoding, pinned date_string; scratch probe, not committed).
    Full 64-token replay matches the fixture — and since CI matched
    tokens 1–49 too, the input state at the divergence position is
    identical across devices; only accumulated activation rounding
    differs.
  - **Measured, dev machine, step 49 (token 50 of 64):
    logprob[195597] = -2.40625 (top-1, the fixture token),
    logprob[188797] = -2.46875 (top-2, the token CI sampled).
    DELTA = 6.25e-2 — FAILS the < 1e-3 clause by ~62x.**
    So this instance satisfies SPEC §11.2's relaxed bar on POSITION ONLY
    (divergence at token 50 > 48); the full relaxed bar is NOT met as
    measured on the generating device. (CI-side delta has the opposite
    sign by construction; its magnitude is unmeasured without CI
    instrumentation.)
  - Structural observation for the ruling (flagged, not acted on): the
    logprob vector is bf16 — all top-5 values sit on the 1/128 grid
    (-2.40625, -2.46875, -2.71875, -2.796875, -3.2265625). At this
    magnitude a nonzero bf16 logprob delta is >= 2^-7 ≈ 7.8e-3, so the
    1e-3 clause is UNSATISFIABLE for this model class except for exact
    ties — and a 1-2 ulp bf16 kernel-order difference at raw-logit
    magnitude ~16-32 is exactly 0.06-0.125, the size of the observed
    flip. The relaxed bar's delta threshold appears calibrated for f32
    logits; whether to recalibrate it is part of the open golden-bar
    device-scope ruling (ADR-level, not mine to change).
- Decisions: none — both tasks were verification/measurement only; no
  code, test, or workflow change.
- Deviations: none. Probe script lives in the session scratchpad only.
- Acceptance:
  ```
  Task A greps (workspace, vendor excluded): 22 extract-replace sites, all
  classified above; Option<Stream|Closure|VectorArray>: 0 hits; impl Drop:
  Worker(test)/SsdStore/Stream/Array/VectorArray; by-value mut params:
  Pin<&mut Self>, supervise(SuperviseCtx), 2x median(Vec<f64>).
  Task B probe output:
    mlx.core 0.31.1, mlx_lm 0.31.2 / prompt tokens: 21
    step 49: sampled 195597
    logprob[195597] = -2.4062500000, logprob[188797] = -2.4687500000
    DELTA = 6.2500000000e-02  (>= 1e-3)
    top-5: 195597 -2.40625 | 188797 -2.46875 | 224816 -2.71875
           | 4631 -2.796875 | 20861 -3.2265625
    full 64-token replay == fixture: True
  ```
- Next: Task 3 — 8-bit and BF16 dtype matrix (SPEC §12 Phase 6 order),
  unchanged. The golden-bar device-scope ruling now has the delta number
  it was waiting on: position clause met, delta clause failed 62x, and
  the delta clause is structurally unsatisfiable under bf16 logprobs.
  Standing items unchanged.

## [2026-07-05] Phase 6 — ADR 0004: golden-bar device scope RULED + dtype-aware relaxed bar — DONE (closes the DECISION NEEDED)
- What:
  - **The golden-bar device-scope DECISION NEEDED (opened in the
    2026-07-05 CI-gap analysis entry) is CLOSED by PM ruling**, recorded
    as `docs/decisions/0004-golden-parity-device-scope-and-dtype-aware-delta.md`:
    (1) golden-token bit-exactness is a SAME-DEVICE guarantee bound to
    the fixture-generating device class; the CI golden step is
    permanently advisory, never promoted to blocking; dev-machine golden
    red remains a correctness bug with no relaxation. (2) The relaxed
    bar's delta clause is recalibrated dtype-aware: <= 4 ULPs of the
    logit compute dtype at the divergence candidates' raw-logit
    magnitude (replaces the fixed < 1e-3; tightens f32, makes fp16/bf16
    satisfiable-in-principle); position clause unchanged; invoking the
    relaxed bar still requires a further ADR naming model + reason.
  - SPEC §11.2 amended to state the same-device scope and reference
    ADR 0004 (PM-directed; the dtype-aware clause supersedes the fixed
    1e-3 in place). ci.yml advisory-step comment updated from "open PM
    ruling" to the ADR 0004 permanent-advisory contract.
  - **CORRECTION to the previous entry** (append-only, so noted here):
    the gemma-3 logit dtype is **float16, not bf16** — measured:
    `generate_step` logprobs and raw logits are `mlx.core.float16`
    (config.json declares `torch_dtype: bfloat16`; the loaded pipeline
    computes fp16). The candidates' raw logits sit at ~16.72 (fp16
    binade [16,32), ULP 2^-6 = 1.5625e-2), so the measured 6.25e-2
    delta is exactly **4 fp16 ULPs**, and the minimum nonzero delta at
    that magnitude is 1.5625e-2 — still >= 1e-3, so the
    "clause unsatisfiable for half-precision logits" conclusion stands
    with the corrected mechanism (the earlier "bf16 / 1/128 grid /
    7.8e-3" framing was imprecise).
  - New measurement that unifies the finding with ADR 0002: recomputing
    the same 70-token divergence state in ONE prefill pass (M=70, tiled
    kernel class) on the generating device yields an EXACT fp16 tie —
    both candidates' logits = 16.71875 — and argmax breaks to 188797,
    the CI outcome. The "cross-device" flip reproduces on the dev
    machine by kernel class alone; device change is kernel-class change.
    This is the decisive evidence line in the ADR.
- Decisions: all within the PM directive. My call per the directive: the
  dtype-aware formulation is a 4-ULP-at-logit-magnitude bound rather
  than waiving the delta clause for half-precision — a waiver would
  accept ANY-magnitude divergence past token 48 (a real distribution
  difference would pass), which guards nothing; the ULP bound keeps the
  clause meaningful at every dtype and covers the observed legitimate
  flip (exactly 4 fp16 ULPs, an exact tie one kernel class over) with
  no headroom for genuine defects (orders of magnitude larger).
- Deviations: none. (ci.yml change is comment-only; no step semantics
  touched. docs/decisions/ 0001–0003 untouched; 0004 is a new file
  created under explicit PM direction.)
- Acceptance:
  ```
  $ ls docs/decisions/ -> 0001..0003 + 0004-golden-parity-device-scope-
    and-dtype-aware-delta.md (new)
  $ grep -c "ADR 0004" docs/SPEC.md -> 2 (§11.2 scope + clause)
  dtype probe (dev machine, worker venv):
    config torch_dtype: bfloat16
    generate_step logprobs dtype: mlx.core.float16 / delta 0.0625
    raw logits dtype: mlx.core.float16
    decode path: logprob delta 6.25e-2 = 4 fp16 ULPs @ |logit| 16.72
    one-pass (M=70) recompute: logit[195597] = logit[188797] = 16.71875
    (exact tie), argmax -> 188797 (the CI token, on the dev machine)
  ci.yml: comment-only edit; YAML parse + step list unchanged (verified)
  ```
- Next: Task 3 — 8-bit and BF16 dtype matrix (SPEC §12 Phase 6 order).
  No open DECISION NEEDED remains. Standing items: gemma3
  window-crossing fixture, rustc miscompile upstream report, and (per
  ADR 0004) read the advisory lane for pattern changes at each mlx bump.

## [2026-07-10] Phase 6 — PR #8 red runs: e2e metrics flake deflaked; Actions billing block (owner action needed) — DONE
- What: two unrelated failures behind the PR #8 red checks, diagnosed
  separately:
  1. **Run 28754975315 (docs-only commit 4d60fa6): a genuine e2e flake,
     not a regression.** `test_metrics.py::
     test_worker_stats_reexported_with_model_label[rust]` failed with
     `assert 0.0 > 0` on `kiln_worker_tokens_generated_total`, having
     passed on the two prior runs of identical code the same day. Root
     cause: the test polled /metrics (10s deadline) until
     `engine_steps_total > 0` only, then asserted `requests_total` and
     `tokens_generated_total` on that SAME snapshot. The gateway
     re-exports worker Stats on a 1s cadence, so a poll landing
     mid-request captures steps>0/tokens=0, and /metrics serves that
     snapshot until the next tick — the CI dump shows exactly that state
     (plus earlier crash-recovery tests had reset the worker-lifetime
     gauges, narrowing the window). Fix: the poll predicate now covers
     ALL THREE asserted counters, same 10s deadline, deadline failure
     reports all three values. Not a weakening: the same assertions must
     hold within the same deadline — the poll condition was simply
     incomplete for an eventually-consistent export (the test's own
     comment already said "allow a few ticks").
     Also in that run's log: the advisory golden step reproduced the
     gemma-3-1b/chat-basic divergence identically (same fixture, same
     position) — per-device deterministic, as ADR 0002/0004 predict.
  2. **Runs 28755209357 and later (incl. all 4 jobs "failed" in 2-3s):
     GitHub Actions account-level billing block** — every job annotated
     "The job was not started because recent account payments have
     failed or your spending limit needs to be increased", zero steps
     executed, Linux jobs included. The repo is PUBLIC (standard runners
     are free), so this is a failed-payment/account issue, not minutes
     from the new CI tier. NOT fixable in-repo: needs the account owner
     in GitHub Settings -> Billing & plans. Until then every push
     fail-fasts (including this entry's); after it clears, rerun via
     `gh run rerun 28755209357` (its tree is this branch minus this fix)
     or just let the next push run.
- Decisions: deflake shape (poll-all-asserted-counters) chosen over
  raising the deadline or re-reading once — it encodes the actual
  contract (eventual consistency of every re-exported counter) and fails
  with the full counter state.
- Deviations: none.
- Acceptance:
  ```
  $ uv run --project tests/e2e pytest tests/e2e/test_metrics.py -v
  4 passed in 13.34s   (both worker kinds, real stack)
  $ uv run --project tests/e2e pytest tests/e2e -q
  21 passed in 28.82s
  $ ruff check tests/e2e + ruff format --check tests/e2e -> clean
  CI verification deferred: Actions is billing-blocked account-wide
  (see above); local full-stack run is the acceptance until it clears.
  ```
- Next: Task 3 — 8-bit and BF16 dtype matrix (SPEC §12 Phase 6 order).
  Blocked externally only on CI: Actions billing needs owner action.

## [2026-07-10] Phase 6 — deferred CI acceptance fulfilled: PR #8 green — DONE
- What: the Actions billing block cleared account-side; run 29106462796
  (deflake commit f4c9aa2) completed with ALL FOUR checks green — the
  previously flaky e2e metrics test passed on the runner under the new
  poll-all-counters predicate. The advisory golden step reproduced the
  gemma-3-1b/chat-basic divergence identically for the third consecutive
  run (fresh runner, five days later) and the job stayed green — the
  ADR 0004 permanent-advisory contract behaving exactly as specified.
- Decisions / Deviations: none.
- Acceptance:
  ```
  $ gh run view 29106462796 -> run: success
    test-macos / test-macos-release / lint / compile-linux: all success
    advisory step log: gemma-3 chat-basic divergence, same fixture,
    same position (per-device deterministic); only-mode probes ok
  ```
- Next: Task 3 — 8-bit and BF16 dtype matrix (SPEC §12 Phase 6 order).
  No blockers.
## [2026-07-05] Phase 6 — sampler SIGBUS: independent verification (chip session) + kiln-free upstream repro — DONE
- Scope: this is the independently-started chip session anticipated in
  the 2026-07-05 sampler-SIGBUS entry ("the two will conflict if both
  land"). The authoritative fix (495bc3c, eager PRNG key) had already
  merged to main via PR #6 when this session ran, so this session lands
  NO code change: this branch was fast-forwarded onto main (4c74379)
  and this entry records an independent re-derivation of the diagnosis
  from scratch — plus the kiln-free minimization the original entry
  left as a standing item for the upstream report.
- What (verification, all evidence re-derived without consulting the
  original session's intermediate artifacts):
  1. Reproduced the crash at the pre-fix base: restored
     `crates/kiln-engine` to cce71d7 inside the main checkout —
     `cargo test -p kiln-engine --test sampler --release` SIGSEGVs
     deterministically; lldb: fault in `mlx::core::array::~array()` via
     `mlx_array_free` inside `Sampler::sample`, called from the
     `same_seed_same_tokens` closure, freeing a garbage handle
     (`0x573b0916…`). Matches the original entry's frames 1-3. Checkout
     restored to HEAD afterwards (`git status` clean).
  2. **Kiln-free zero-unsafe reproducer** (closes the original entry's
     upstream-report standing item — their naive reductions were
     defeated by niche layouts/IPSCCP; this shape survives both):
     `Sampler { options: Options, key: Option<Box<Key>> }` where
     `Options` mirrors SamplingOptions' field mix, `Key` carries a Drop
     impl with new/drop counters, `new()` returns the constant
     `Self { options, key: None }`, a closure takes the sampler BY
     VALUE and advances the key chain 16x via `take()`/`Some(...)`
     reassignment, and is called twice with `Sampler::new(opts)` from
     one constant local. rustc 1.96.1 (31fca3adb, aarch64-apple-darwin),
     no unsafe, no FFI: opt-level 0/1 pass; **opt-level 2/3 die in
     libsystem_malloc `mfm_free.cold.4` (double free)** — safe Rust
     double-freeing is a compiler bug by definition. (Note: the kiln
     shape triggered at >= 1 per the original entry; the kiln-free
     Box-niche shape needs >= 2 — inlining-threshold difference, same
     defect.) Source preserved below for the rust-lang/rust filing.
  3. **Pass-level isolation** (new evidence, not in the original
     chain): `RUSTC_BOOTSTRAP=1 rustc -C opt-level=3
     -Zmir-enable-passes=-GVN` on the identical source runs clean
     ("ok: streams match, 34 keys minted, 34 dropped"). Disabling the
     single MIR GVN pass removes the double free — the miscompiling
     pass is pinned, not inferred.
  4. MIR confirmation on the kiln-free shape (`--emit=mir`, opt 3):
     one construction `_6 = Sampler { options: copy _1, key: const
     Option::<Box<Key>>::None }` feeds both closure calls through one
     tuple local — call 1 `copy _5`, call 2 `move _5` — exactly the
     merged-argument-temporary structure the original entry read out of
     the kiln reproducer's MIR (`_22`/`_21`).
  5. Fix verification at HEAD (4c74379): the previously-crashing test
     passes in release x3 consecutive runs (plus the earlier pre-revert
     run: x4 total). Test NOT weakened — confirmed the committed test
     preserves the trigger structure (constant options, by-value
     closure, back-to-back draws per the REGRESSION SHAPE contract) and
     the same assertions, and the new `test-macos-release` CI lane
     executes it in the miscompiling configuration.
- Verdict: the 495bc3c diagnosis is CONFIRMED from an independent
  angle — rustc 1.96.1 MIR-GVN miscompilation of safe code; kiln-mlx
  FFI surface exonerated (wrapper audit of array/ops/random/stream/error
  found the SAFETY contracts sound). The eager-key workaround is the
  right shape: an opaque FFI call in the constructor is exactly what
  GVN cannot prove identical.
- Upstream-report payload (standing item now unblocked; file against
  rust-lang/rust with: rustc 1.96.1 aarch64-apple-darwin, opt-level>=2,
  disappears under -Zmir-enable-passes=-GVN):
  ```rust
  use std::sync::atomic::{AtomicUsize, Ordering};
  static NEWS: AtomicUsize = AtomicUsize::new(0);
  static DROPS: AtomicUsize = AtomicUsize::new(0);
  struct Key(u64);
  impl Key {
      fn mint(v: u64) -> Box<Key> {
          NEWS.fetch_add(1, Ordering::SeqCst);
          Box::new(Key(v))
      }
  }
  impl Drop for Key {
      fn drop(&mut self) { DROPS.fetch_add(1, Ordering::SeqCst); }
  }
  #[derive(Clone, Copy)]
  struct Options { temperature: f32, top_p: f32, top_k: u32,
                   min_p: f32, seed: u64, explicit_seed: bool }
  struct Sampler { options: Options, key: Option<Box<Key>> }
  impl Sampler {
      fn new(options: Options) -> Self { Self { options, key: None } }
      fn sample(&mut self) -> u64 {
          let key = match self.key.take() {
              Some(key) => key,
              None => Key::mint(self.options.seed),
          };
          let v = key.0;
          self.key = Some(Key::mint(v + 1));
          v
      }
  }
  fn main() {
      let opts = Options { temperature: 1.0, top_p: 0.95, top_k: 0,
                           min_p: 0.0, seed: 42, explicit_seed: false };
      let draw = |mut sampler: Sampler| -> Vec<u64> {
          (0..16).map(|_| sampler.sample()).collect()
      };
      let a = draw(Sampler::new(opts));
      let b = draw(Sampler::new(opts)); // double free at opt-level >= 2
      assert_eq!(a, b);
      assert_eq!(NEWS.load(Ordering::SeqCst), DROPS.load(Ordering::SeqCst));
      println!("ok");
  }
  ```
- Decisions: fast-forwarded this session branch onto main instead of
  re-implementing on the stale cce71d7 base — avoids the double-land
  conflict the original entry predicted; PROGRESS stays append-only.
- Deviations: none (no code touched; diagnosis used the main checkout
  read-only plus one temporary `git restore --source=cce71d7` of
  crates/kiln-engine, reverted and verified clean).
- Acceptance (real output, trimmed):
  ```
  $ git -C ~/KILN checkout cce71d7 -- crates/kiln-engine
  $ cargo test -p kiln-engine --test sampler --release
  process didn't exit successfully: (signal: 11, SIGSEGV)
  lldb: frame #0 mlx::core::array::~array() +36
        frame #1 mlx_array_free  frame #2 kiln_engine::sampler::Sampler::sample
        frame #3 sampler::same_seed_same_tokens::{{closure}}
  $ git -C ~/KILN restore --source=HEAD --staged --worktree crates/kiln-engine
  $ rustc -C opt-level=3 repro.rs && ./repro          # kiln-free, no unsafe
  EXC_BREAKPOINT libsystem_malloc mfm_free.cold.4 (double free); opt 0/1 ok
  $ RUSTC_BOOTSTRAP=1 rustc -C opt-level=3 -Zmir-enable-passes=-GVN repro.rs && ./repro
  ok: streams match, 34 keys minted, 34 dropped
  $ cargo test -p kiln-engine --test sampler --release   (HEAD 4c74379, x3)
  test sampler_behavior ... ok   |   test result: ok. 1 passed (x3)
  ```
- Next: Task 3 — 8-bit and BF16 dtype matrix (SPEC §12 Phase 6 order),
  unchanged. Standing item updated: rustc miscompile upstream report is
  ready to file (payload above); re-test the eager-key workaround at
  every toolchain bump.

## [2026-07-05] Phase 6 — rustc MIR-GVN miscompile: upstream report FILED — DONE
- What (closes the "file upstream" standing item from the two 2026-07-05
  sampler-SIGBUS entries):
  1. **Filed https://github.com/rust-lang/rust/issues/158830** — "MIR GVN
     merges by-value argument temporaries with drop glue, causing double
     free in safe code (1.95/1.96, aarch64-apple-darwin)". Payload: the
     kiln-free zero-unsafe reproducer (previous entry), the
     `-Zmir-enable-passes=-GVN` clean-run isolation, the MIR excerpt
     (`_6 = Sampler {...}` built once; `_5 = (copy _6,)`; call 1
     `copy _5`, call 2 `move _5`), and the opt-level threshold finding
     stated as an inlining artifact of one defect (kiln shape — no-niche
     `Option<FfiHandle>` — triggers at >= 1; minimized niched
     `Option<Box>` shape needs >= 2), target rustc 1.96.1
     aarch64-apple-darwin. Labels applied via rustbot: T-compiler,
     C-bug, I-unsound, A-mir-opt, A-mir-opt-GVN (+ auto I-prioritize).
  2. New pre-filing evidence gathered for the report:
     - rustc 1.95.0 (59807616e) also crashes (opt 2/3) — NOT a 1.96
       regression; introduction unbisected, stated as such.
     - nightly 1.98.0 (c397dae80 2026-07-02) does NOT crash **but emits
       the identical merged MIR** — the report asks upstream to
       determine fixed-needs-backport vs masked-latent.
     - duplicate search (gh, several phrasings incl. broad "GVN"
       sanity check): no existing report of this defect.
  3. **rust-toolchain.toml created** (did not previously exist) with the
     issue link + workaround/bump-protocol comment, `channel = "stable"`.
- Decisions: `channel = "stable"`, not a 1.96.1 pin — CI already floats
  on dtolnay/rust-toolchain@stable so this changes no behavior anywhere,
  the landed eager-key workaround makes 1.96.1 safe for Kiln, and
  freezing onto a known-miscompiling compiler would invert the intent;
  the comment (not the pin) carries the warning, and the
  test-macos-release lane is the enforcement at every bump.
- Deviations: none (no Rust code touched).
- Acceptance (real output, trimmed):
  ```
  $ gh issue create -R rust-lang/rust --title "MIR GVN merges by-value
    argument temporaries with drop glue, causing double free in safe
    code (1.95/1.96, aarch64-apple-darwin)" --body-file issue-body.md
  https://github.com/rust-lang/rust/issues/158830
  $ gh issue view 158830 -R rust-lang/rust --json number,state,labels
  158830 [OPEN] | labels: T-compiler, I-unsound, C-bug, A-mir-opt,
  I-prioritize, needs-triage, A-mir-opt-GVN
  $ rustup run 1.95.0 rustc -C opt-level=2 main.rs && ./repro -> exit 133
  $ rustup run nightly rustc -C opt-level=3 main.rs && ./repro -> exit 0
    (nightly --emit=mir: same single _6/_5 construction, copy/move pair)
  ```
- Next: Task 3 — 8-bit and BF16 dtype matrix (SPEC §12 Phase 6 order),
  unchanged. Standing item updated: upstream report FILED (#158830);
  at every toolchain bump re-run the release suite and check #158830
  for the fix/backport landing.

## [2026-07-05] Phase 6 — MSRV history audit + retroactive toolchain pin (1.96.1) — DONE
- History audit (PM-directed; whether "MSRV 1.96.1" was ever enforced):
  - `git log --all -- rust-toolchain.toml` (+ `rust-toolchain*` glob,
    `--follow`): the file exists in exactly ONE commit — d726468
    (2026-07-05, this session), which created it with `channel =
    "stable"`. No pin was ever committed before, on any ref.
  - Ruled out every disappearance mechanism: not reverted (nothing to
    revert), not history-rewritten (all 7 `git fsck` unreachable
    commits scanned — none contain the file), not stashed (no stashes),
    not gitignored (no matching pattern ever), not sitting uncommitted
    in the primary checkout (absent).
  - Origin of the gap: Phase 0 entry (2026-07-02), Decisions: "MSRV
    left unset — SPEC §14 lists it as an open pre-Phase-0 PM decision.
    Non-blocking; flagging for review." The flag was never picked up:
    zero MSRV mentions in PROGRESS between that line and 2026-07-05,
    no ADR touches it, no `rust-version` in any manifest, all four CI
    jobs floated on dtolnay/rust-toolchain@stable.
  - **Plainly: MSRV was an unenforced assumption from Phase 0 through
    this pin landing — ~3 days, Phases 0 through 6 — coincidentally
    stable at 1.96.1 only because the project is young and upstream
    stable did not move in that window, not because anything enforced
    it.** 1.96.1 is simply the stable that was current when the Phase 0
    session installed rustup on this machine.
- What (the pin, now formalized in all three layers):
  1. rust-toolchain.toml: `channel = "1.96.1"` (hard pin, replacing
     "stable"); comment records the retroactive formalization date +
     this entry, the rust-lang/rust#158830 context, and the lockstep
     rule.
  2. Cargo.toml `[workspace.package] rust-version = "1.96.1"` +
     `rust-version.workspace = true` in all 8 crate manifests — cargo
     itself now gates the floor (verified via `cargo metadata`: all 8
     packages report 1.96.1). Belt-and-suspenders: rust-toolchain.toml
     controls what rustup fetches; rust-version is what the crates
     declare they need.
  3. ci.yml: explicit `toolchain: 1.96.1` input on all four
     dtolnay/rust-toolchain steps. Confirmed the action does NOT read
     rust-toolchain.toml (toolchain comes from the action ref or the
     `toolchain:` input); rustup's directory override would still have
     enforced the pin on every cargo invocation once the file exists,
     but the explicit input removes the dead-weight stable install and
     makes the pinned version visible in the action step log.
- Standing practice (bump protocol, same tier as ADR 0001's C1 quarterly
  process): toolchain bumps are deliberate — (a) update the pin in all
  three places (rust-toolchain.toml, workspace rust-version, 4 CI
  steps), (b) re-run the full workspace suite in debug AND release
  (test-macos-release lane), (c) check rust-lang/rust#158830 and its
  resolution/backport status before trusting a new stable on the
  GVN-affected shape.
- Deviations: none.
- Acceptance (local, real output trimmed; CI matrix run + 1.96.1
  visibility verified on this branch's PR — run ids and confirmation in
  the PR thread, per the PR #7 precedent):
  ```
  $ rustup show active-toolchain        (in repo, after pin)
  1.96.1-aarch64-apple-darwin (overridden by '.../rust-toolchain.toml')
  $ cargo metadata --no-deps | ... rust_version
  all 8 kiln crates -> 1.96.1
  $ cargo fmt --all --check -> clean
  $ cargo build --workspace --no-default-features -> Finished dev 21.70s
  $ cargo clippy --workspace --all-targets --no-default-features -- -D warnings -> clean
  $ uv run ... ruff check python/ tests/e2e -> All checks passed!
  $ uv run ... ruff format --check python/ tests/e2e -> 16 files already formatted
  ```
- Next: Task 3 — 8-bit and BF16 dtype matrix (SPEC §12 Phase 6 order),
  unchanged.

## [2026-07-10] Phase 6 / Task 3 — 8-bit + BF16 quantization paths (SPEC §7.3) — DONE
- What:
  1. **ADR 0001 B1 alignment re-verified before generating** (per task
     instruction): worker venv mlx.core 0.31.1 / mlx-lm 0.31.2; mlx-c
     submodule at 0726ca922 (v0.6.0 → MLX v0.31.1). gen-golden.py's
     version gate passed.
  2. **New pin** `qwen3-0.6b-8bit` = mlx-community/Qwen3-0.6B-8bit @
     11de96878523501bcaa86104e3c186de07ff9068 (fetch-test-model.sh;
     sha256-verified). Uniform affine {group_size: 64, bits: 8}, no
     per-module overrides (verified in the fetched config — checkpoints
     with overrides stay rejected by the Phase 3 fail-loud check in
     config.rs). Same base model as the qwen3-0.6b-4bit pin, so the
     4-bit/8-bit matrix cells differ only in quantization. BF16 is the
     existing smollm2-135m-bf16 pin (weights verified pure-BF16
     safetensors; llama arch, tied embeddings → dense as_linear head).
  3. **12 fixtures generated** (explicitly instructed):
     tests/golden/qwen3-0.6b-8bit/ and tests/golden/smollm2-135m-bf16/
     (6 each, full CASES list).
  4. **Code audit: the dtype stack was already generic** — config.rs
     gates bits 4|8 / group 32|64|128; nn.rs Linear/Embedding resolve
     quantized-vs-dense per tensor (`.scales` presence, mlx-lm
     class_predicate rule); weights.rs maps BF16 safetensors; paged-KV
     pools adopt the model dtype at first write; B' calibration probes
     through Linear::forward (quantized and dense alike). **8-bit:
     green with zero code changes** (W = 9, same as 4-bit).
  5. **BF16 initially FAILED single-stream** on raw-tiny-remainder
     (divergence at generated token ~44, 260 vs fixture 284). Root
     cause, established by pure-python bisect against the reference
     stack (no Rust suspect left standing):
     - an mlx-lm replica of the reference schedule (single 132-row
       prefill piece + M=1 step) reproduces the fixture; a replica of
       Kiln's fine-grid schedule (128 + 4-row tail padded to 32 per
       ADR 0002) flips the same token the Rust engine flips → **the
       ADR 0002 kernel-class pad does not make fine-grid pieces
       bit-reproduce the reference's single-piece pass for dense bf16
       trunks at this pin**; the Rust implementation is exonerated.
     - op bisect on real tensors: layer-1 q/k/v bit-equal; SDPA output
       differs for (Lq=5, Lk=133) vs (Lq=133) — real-data only
       (synthetic probes, incl. outlier-heavy, false-negative: kernel
       classes differ but benign values round identically). Real-data
       SDPA query-class boundary here is 9, so the padded-32 piece IS
       reference-class at layer 1 — yet the full padded schedule still
       flips: some deeper layer's op crosses class on its own data.
       Per-op auditing is not a viable guarantee for dense trunks.
     - **Fix: dense (unquantized) checkpoints take the
       gemma2-precedented monolithic prefill override.**
       AnyModel::monolithic_prefill_required is now gemma2-softcap OR
       `quantization.is_none()`; engine builders already honor it as
       prefill_fine_chunk = prefill_chunk. Reference-shaped pieces by
       construction (identical to mlx-lm's own prefill loop), and the
       ADR 0002 pad rule never triggers (every piece starts on a
       prefill_chunk boundary). Quantized models keep the fine grid —
       behavior unchanged, all previously-green rounds re-verified.
  6. **Dense-bf16 deterministic width is 1** (calibration: dense
     gemv/gemm row-stability boundary at M=2, vs qmv thresholds ~10).
     Under B', deterministic rows on this model decode fully serial;
     the width-16 golden rounds pass by construction. Recorded as an
     ADR 0003 bar-(2) datapoint (measured floor framework; W adapts
     per model/device). WorkerInfo.max_deterministic_decode_width
     reports it truthfully.
  7. Comment corrections at the three override sites (model.rs,
     golden.rs, kiln-worker engine.rs) and generate.rs (the Phase-3
     contiguous path always uses the fine grid → documented as
     parity-meaningful for quantized checkpoints only; all its callers
     are llama-4bit tests/benches).
- Decisions:
  - 8-bit model choice (task latitude "at least one"): Qwen3-0.6B-8bit
    — smallest uniform-quant 8-bit candidate; isolates the bits axis
    against the existing 4-bit pin of the same model.
  - Monolithic-prefill keyed on `quantization.is_none()`, not on bf16:
    the fine grid was never validated on ANY dense trunk, and
    reference-shaped prefill is parity-safe by construction for
    fp16-dense too. Cost is bounded: the schedule equals mlx-lm's own;
    prefix-cache boundary granularity coarsens to prefill_chunk for
    dense models (same posture gemma2 already has).
  - **ADR 0002 scope note (proposing via PROGRESS per CLAUDE.md;
    docs/decisions is agent-read-only):** bar (3)'s pad mechanism is
    now measured insufficient for dense trunks; dense models bypass
    the fine grid entirely instead. An ADR 0002 addendum recording
    this scope limit would keep the ADR accurate.
- Deviations: none.
- Acceptance (real output, trimmed; dev machine = fixture-generating
  device, the binding scope per ADR 0004):
  ```
  $ uv run --project python/kiln_worker_py python -c "import mlx.core..."
  mlx.core 0.31.1 / mlx_lm 0.31.2   (B1 holds; submodule 0726ca922 v0.6.0)
  $ ./scripts/fetch-test-model.sh --only qwen3-0.6b-8bit
  ==> qwen3-0.6b-8bit (mlx-community/Qwen3-0.6B-8bit @ 11de96878523)
      fetching model.safetensors (633.4 MB) ... done
  $ uv run ... python scripts/gen-golden.py --model qwen3-0.6b-8bit ...
  wrote 6 fixtures to tests/golden/qwen3-0.6b-8bit
  $ uv run ... python scripts/gen-golden.py --model smollm2-135m-bf16 ...
  wrote 6 fixtures to tests/golden/smollm2-135m-bf16
  $ KILN_TEST_MODELS=... cargo test -p kiln-models --test golden -- --nocapture
  == qwen3-0.6b-8bit: model_type=qwen3, 6 fixture(s), deterministic width 9
  golden qwen3-0.6b-8bit/*: exact match (batched/paged engine) x6
  golden qwen3-0.6b-8bit/*: exact match at decode width 16 (B') x6
  == smollm2-135m-bf16: model_type=llama, 6 fixture(s), deterministic width 1
  golden smollm2-135m-bf16/*: exact match (batched/paged engine) x6
  golden smollm2-135m-bf16/*: exact match at decode width 16 (B') x6
  (all 5 pre-existing models: exact match, both rounds, unchanged)
  test result: ok. 1 passed; finished in 292.28s
  $ cargo test --workspace                  -> exit 0 (38 result blocks, 0 failed)
  $ cargo test -p kiln-tokenize --test tokenizer --test detok
    cargo test -p kiln-models --test batching --test calibration --test leak
      --test leak_batched --test preemption --test prefill_pad
      --test prefix_cache --test prefix_multiturn   (dev posture: fixture
      comparisons INCLUDED)
    cargo test -p kiln-worker --test rpc    -> all green, 0 failed
  $ cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
    -> clean
  $ uv run ... ruff check python/ tests/e2e scripts/ && ruff format --check ...
    -> All checks passed! / 16 files already formatted
  ```
  CI matrix verification (real CI shapes) recorded on this branch's PR
  thread per the PR #7/#8 precedent; the advisory golden lane's pattern
  change (two new fixture dirs) to be read there and recorded in a
  follow-up PROGRESS note per ADR 0004.
- Next: Phase 6 Task 4 — gateway worker="auto" routing (SPEC §12
  Phase 6 order).

## [2026-07-10] Phase 6 / Task 3 — CI verification on PR #10 — DONE
- Real CI shapes (PR #10, run 29113687692, head 53b8b06): all four
  jobs green — lint, compile-linux, test-macos, test-macos-release.
  - Blocking lanes green with the new pin; the test-model cache
    rebuilt from the updated fetch-test-model.sh hash as designed
    (one-time ~650 MB fetch, then re-cached).
  - test-macos-release green → the rustc 1.96.1 MIR-GVN standing
    check (rust-lang/rust#158830) holds on this change.
- Advisory golden lane (ADR 0004, permanently non-blocking) reading,
  recorded per the ADR's consequence clause:
  - gemma-2-2b: all 12 rounds token-exact on the foreign GPU.
  - gemma-3-1b/chat-basic: diverges at generated token 50 of 64,
    runner 188797 vs fixture 195597 — byte-identical to the pattern
    ADR 0004 records from run 28753659372 (the 4-fp16-ULP
    kernel-class coin toss). NO pattern change accompanies this
    change.
  - The harness fail-fasts at the first divergence, so the two new
    fixture dirs (qwen3-0.6b-8bit, smollm2-135m-bf16) received no
    cross-device datapoint this run; their BINDING bar (same-device,
    previous entry) is green. If per-model cross-device readings
    become wanted, the harness needs per-model fail-isolation —
    noted, not done (advisory-only value; no bar depends on it).
  - Advisory preemption + prefill_pad committed-fixture comparisons:
    exact on the foreign device this run.
- Next: Phase 6 Task 4 — gateway worker="auto" routing (SPEC §12
  Phase 6 order).

## [2026-07-10] Phase 6 / Task 4 — worker="auto" routing + /v1/completions — DONE
- What:
  1. **worker="auto" routing (SPEC §10/§12).** registry.rs `resolve_worker`
     replaces the Phase-3 placeholder (auto→python unconditionally): auto
     resolves to RUST iff `kiln_models::ArchConfig::from_model_dir` accepts
     the checkpoint — implemented arch (llama/qwen2/qwen3/gemma2/
     gemma3_text), known rope_scaling, uniform affine 4/8-bit @ group
     32/64/128 or unquantized bf16/f16 (SPEC §7.3) — else PYTHON with the
     reason logged. The routing predicate IS the loader's own fail-loud
     validation, so the decision is exactly "would the rust worker load
     this". An unloadable tokenizer.json downgrades auto to python (the
     rust route requires gateway-side tokenization); explicit
     worker="rust" keeps hard-failing at startup.
  2. **Stale explicit-rust validation fixed in passing:** worker="rust"
     was still validated with `LlamaConfig::from_model_dir` (Phase 3
     vintage) — a qwen/gemma model explicitly pinned to rust failed
     gateway startup with UnsupportedArchitecture despite full support.
     Now ArchConfig; regression-guarded by a unit test.
  3. **/v1/completions (SPEC §8.1).** New completions.rs: OpenAI legacy
     text-completions endpoint, stream + non-stream, both worker kinds.
     Raw prompt (no chat template) encoded WITH special tokens (raw-prompt
     BOS contract = mlx-lm raw generate). max_tokens defaults to 16
     (OpenAI legacy semantics; chat keeps fill-remaining-context).
     Rejected by name: multi-prompts, token-id prompts, echo, suffix,
     best_of>1, n>1, logprobs. Downstream of validation it reuses the chat
     machinery — ready_entry/encode_prompt extracted; TextPipeline/
     CompletionCtx/classify_finished shared pub(crate); CompletionCtx
     parameterized over the endpoint's counter — so the chat.rs
     stop-string-precedence and usage semantics apply verbatim. New metric
     kiln_completions_total{model,outcome}; sampling-range validation
     deduped into openai.rs validate_sampling so the endpoints cannot
     drift.
  4. **Python worker detok contract bug — found by the new e2e, fixed.**
     First e2e run: the python-routed twin's /v1/completions text lacked
     the leading space (" a device..." vs "a device...", token ids
     IDENTICAL). Root cause: mlx-lm's streaming detokenizers trim the
     first segment's leading space (`_maybe_trim_space`: `elif not
     self.text: return current_text[1:]`) — CLI display behavior that
     violates the proto contract (TokenChunk.text = exact detokenization
     of token_ids) and cross-worker text parity; chat never noticed
     because templated responses rarely open with a space-prefixed token.
     Fix: kiln_worker_py/detok.py — a Kiln-owned port of kiln-tokenize's
     two-offset StreamingDecoder (same algorithm the gateway uses for rust
     workers; decode with specials included, no cleanup), presented behind
     mlx-lm's reset/add_token/finalize/last_segment interface; engine.py
     now uses it instead of tokenizer.detokenizer. No monkey-patching —
     mlx-lm is untouched. 7 tokenizer-only unit tests (leading space,
     UTF-8 holdback incl. ZWJ/flag sequences, specials not skipped,
     concat == full decode).
  5. **E2E (the task's required test).** test_auto_routing.py: doctored
     copy of the pinned llama-3.2-1b-4bit — weights symlinked, config.json
     quantization gains a per-module override entry whose params EQUAL the
     uniform block: a rejected quantization form for the rust matrix
     (mixed-precision shape), loaded bit-identically by mlx-lm via its
     class_predicate. One gateway, both models under worker="auto":
     asserts the supported twin got a rust worker process and the doctored
     twin a python worker process (matched per-model via the socket-path
     hash), the routing decisions are logged, and the python-routed model
     serves BOTH endpoints with output byte-identical to the rust twin
     (transparent fallback; the greedy cross-worker parity invariant is
     the "serves correctly" oracle). test_completions.py: /v1/completions
     against both worker kinds — non-stream, stream==non-stream (greedy),
     stop strings, default max_tokens=16, rejections by name, 404,
     kiln_completions_total. conftest running_stack now accepts per-model
     paths (3-tuples).
  6. Registry unit tests for the matrix (unsupported arch / per-module
     override / mxfp4 mode / bits=3 / group=16 / longrope → python; all
     five archs + dense-bf16 accepted; explicit-rust hard error;
     explicit-python skips validation). kiln.toml.example comment updated
     to name the quant-format condition.
- Decisions:
  - Auto downgrades to python when tokenizer.json is unloadable even if
    config.json is servable: the rust route REQUIRES gateway
    tokenization; python owns its tokenizer. Explicit rust stays loud.
  - /v1/completions default max_tokens=16 per OpenAI legacy docs, not
    chat's fill-remaining-context — a deliberate, documented divergence
    between the two endpoints.
  - Unsupported-config e2e uses an equal-params per-module override
    rather than a genuinely mixed checkpoint: same weights ⇒ greedy
    parity doubles as the serving-correctness oracle, and no new pin.
  - The python worker's first-chunk space trim was treated as a worker
    bug, not a test to relax: the proto comment and the gateway's
    fuzzed-against-full-decode rust path both define text as the exact
    detokenization. (mlx-lm's own server inherits the trim; Kiln now
    matches OpenAI/HF-decode semantics instead.)
- Deviations: none.
- Task 5 / folded-items audit (PM-instructed): the SPEC §12 Phase 6 list
  maps to the ledger as 1=Qwen2.5/3, 2=Gemma2/3, 3=8-bit+BF16,
  4=routing+/v1/completions (this entry). The remaining item,
  "rope_scaling variants", never needed its own session — folding
  confirmed accurate: default/linear/llama3 landed with Phase 3 (llama
  config parsing + nn.rs Rope; llama3 exercised end-to-end by the
  llama-3.2 fixtures), yarn landed in Task 1
  (yarn_freqs_match_reference_within_ulp_tol + the qwen pins), gemma3's
  local/global dual rope in Task 2. All four variants dispatch in nn.rs
  (config.rs resolve_rope_scaling → Rope). Nothing outstanding; with this
  task every Phase 6 item is done.
- Acceptance (dev machine; KILN_TEST_MODELS set):
  ```
  $ cargo test -p kiln-gateway --lib
  registry::tests::auto_routes_unsupported_configs_to_python ... ok
  registry::tests::auto_prefers_rust_but_downgrades_without_tokenizer ... ok
  registry::tests::explicit_rust_on_unservable_model_is_a_startup_error ... ok
  registry::tests::explicit_rust_accepts_every_supported_architecture ... ok
  registry::tests::explicit_python_skips_validation_entirely ... ok
  (+ completions validation/serialization tests in openai::tests)
  test result: ok. 33 passed; 0 failed
  $ cargo test --workspace                    -> exit 0
  $ cargo test -p kiln-models --test golden -- --nocapture   (explicit re-run)
  all 7 pinned models × all fixtures: exact match, BOTH rounds
  (batched/paged engine + decode width 16 B'), incl. qwen3-0.6b-8bit and
  smollm2-135m-bf16
  test result: ok. 1 passed; finished in 295.20s
  $ uv run --project python/kiln_worker_py pytest python/kiln_worker_py/tests -q
  35 passed   (28 prior + 7 new detok)
  $ uv run --project tests/e2e pytest tests/e2e -v
  test_auto_routing.py::test_auto_resolves_by_the_support_matrix PASSED
  test_auto_routing.py::test_unsupported_model_serves_transparently_from_python PASSED
  test_auto_routing.py::test_unsupported_model_serves_completions_too PASSED
  test_completions.py (6 tests) [python] PASSED  [rust] PASSED
  (chat/crash/metrics/cross-worker-parity suites unchanged, all PASSED)
  36 passed in 33.68s
  NOTE: the FIRST e2e run failed test_unsupported_model_serves_completions_too
  — rust " a device..." vs python "a device...", identical token ids; that
  failure IS the detok bug in item 4. Fixed, full suite re-run green.
  $ cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
  clean
  $ cargo build --workspace --no-default-features
    cargo clippy --workspace --all-targets --no-default-features -- -D warnings
  clean (linux CI shape)
  $ ruff check python/ tests/e2e scripts/ && ruff format --check python/ tests/e2e scripts/
  All checks passed! / 21 files already formatted
  ```
  CI matrix verification (real CI shapes) to be recorded on this branch's
  PR thread per the PR #10 precedent, incl. the advisory golden lane
  reading (ADR 0004).
- Next: PM phase gate on Phase 6 (SPEC §13.4) — all Phase 6 items are
  done — then Phase 7: llguidance structured output, tool-call parsers,
  /v1/messages, paged-attention kernel (SPEC §12).

## [2026-07-10] Phase 6 / Task 4 — CI verification on PR #12 — DONE
- Real CI shapes (PR #12, run 29122429155, head d139ab3): all four
  blocking jobs green — lint, compile-linux, test-macos,
  test-macos-release.
  - The task's required test verified on the foreign GPU: e2e 36 passed
    incl. test_auto_routing (unsupported-quant config transparently
    routed to + served from the python worker, byte-identical to the
    rust twin on both endpoints) and the full /v1/completions suite over
    both worker kinds. Python worker 35 passed (incl. the 7 new detok
    tests).
  - test-macos-release green → the rustc 1.96.1 MIR-GVN standing check
    (rust-lang/rust#158830) holds on this change.
- Advisory golden lane (ADR 0004, permanently non-blocking) reading,
  recorded per the ADR's consequence clause:
  - gemma-2-2b: all rounds token-exact on the foreign GPU (visible in
    the log before the fail-fast).
  - gemma-3-1b/chat-basic: diverges at generated token 50 of 64, runner
    188797 vs fixture 195597 — byte-identical to the pattern ADR 0004
    records (the 4-fp16-ULP kernel-class coin toss). NO pattern change
    accompanies this change.
  - Advisory preemption + prefill_pad committed-fixture comparisons:
    exact on the foreign device this run (1 passed each).
- **Phase 6 / Task 4 is CLOSED; every Phase 6 item is done.**
- Next: PM phase gate on Phase 6 (SPEC §13.4), then Phase 7 —
  llguidance structured output, tool-call parsers, /v1/messages,
  paged-attention kernel (SPEC §12).

## [2026-07-10] Phase 6 — PM gate review (SPEC §13.4 / §13.1) — CRITERIA MET; one DECISION NEEDED
- Scope: PR #12 merged to main (f6284c3; tree bit-identical to the
  CI-verified PR head). Adversarial review of the whole phase
  (baseline 268d4f6 = Phase 5 closeout → f6284c3) against SPEC
  §7.2/§7.3/§8.1/§10/§11/§12; dynamic gates re-run on the dev/PM
  machine against merged main.
- Static review (git-diff audit over the phase span) — ALL CLEAN:
  - proto freeze: one additive WorkerInfo field
    (max_deterministic_decode_width = 13, diagnostics); no renumber/
    retype/repurpose. Allowed under the post-Phase-2 additive clause.
  - docs/decisions/: exactly the three PM-ruled ADRs (0002/0003/0004)
    added; 0001 untouched.
  - SPEC.md: only the two recorded amendments (ADR 0003 perf clause,
    ADR 0004 golden bar).
  - Dependencies: only the rust-version=1.96.1 MSRV pin (GVN incident,
    recorded); zero new runtime deps in Cargo.toml/pyproject.
  - Golden fixtures: additions only (new model dirs + the sanctioned
    llama raw-tiny-remainder case); zero modifications to committed
    fixtures.
  - Modified pre-existing test files (9): each maps to a recorded ledger
    entry (B'/ADR 0002-0004 harness work, sampler-SIGBUS fix, CI Option
    B advisory mode, e2e metrics deflake, task 6.4 conftest). Leak-gate
    diffs are API adaptations only; 0→0 assertions intact. No silent
    weakening found.
  - unwrap()/expect() outside test modules in library code: none.
    Monkey-patching idioms in the python worker: none. dead_code allows:
    none.
- Phase 6 acceptance criteria (SPEC §12):
  1. "Golden parity exact for all fixture models × dtypes" — PASS.
     Fresh same-device run on this tree: 7 models (llama/qwen2.5/qwen3/
     gemma2/gemma3 4-bit, qwen3 8-bit, smollm2 bf16) × all fixtures ×
     both rounds (batched/paged + width-16 B') exact (295s). CI advisory
     lane: only the known ADR 0004 gemma-3 pattern.
  2. "An unsupported-arch model transparently serves via python worker"
     — PASS via the task-6.4 e2e (unsupported-QUANT config end-to-end,
     output byte-identical to the rust twin; the PM's task prompt
     sanctioned the quant example). Arch rejection is covered at
     routing-unit level; no unsupported-arch model is pinned, and the
     post-resolution serving path is identical. Noted; no action.
- Dynamic gates on merged main (this machine):
  - e2e suite: 36 passed (both worker kinds + auto-routing +
    /v1/completions + parity).
  - scripts/bench.sh still does not exist (flagged since Phase 4;
    Phase 10 tooling item) — the release throughput test is the stand-in.
  - **Throughput: the committed gate test FAILS as written; the
    measurements PASS every PM-approved bar.** Numbers (llama-3.2-1b-
    4bit, release, W=9; two runs, medians):
    ```
    single-stream        122.6-125.0 tok/s   (recorded post-B': 123.8)
    all-sampled batch-16 350.9 tok/s -> 2.81x (recorded: 331.6 -> 2.68x)
    mixed 8+8 batch-16   233.4-234.0 -> 1.87x (ADR 0003 floor: 1.80x) OK
    all-greedy batch-16  274.7-276.3 -> 2.20x (ADR 0003 floor: 2.10x) OK
    ```
    Every lane is at or above its recorded post-B' value — NO regression
    (>10% bar, SPEC §11.3); the non-deterministic 1B lane improved ~5%.
- FINDING (the one executable-vs-ADR inconsistency in the phase):
  crates/kiln-models/tests/throughput.rs still asserts the PRE-ruling
  bar — ≥3x on BOTH the mixed 8+8 and all-greedy lanes. It was authored
  with the B' landing (ed32598), before the ADR 0003 ruling; the ruling
  task then executed as "doc-only" and never re-aimed the test. Today's
  gate was the first recorded execution of the amended test: mixed 1.87x
  and greedy 2.20x fail its ≥3x asserts while clearing their ADR 0003
  floors. Per CLAUDE.md (never adjust a test without saying so), the
  test is untouched; measurements came from a temporary uncommitted
  probe copy, deleted after the run (working tree clean).
- DECISION NEEDED (test re-aim; picking nothing):
  A) Rewrite throughput.rs to the ADR 0003 two-bar split — assert
     bar (1) on the all-sampled lane as no-regression vs the recorded 1B
     number (the absolute ≥3x holds at 8B per the ADR; 1B is the known
     sampler artifact), and bar (2) as no-regression-below-floor for the
     greedy/mixed lanes (2.10x/1.80x recorded). Keeps an executable gate
     aligned with the ruled bars. RECOMMENDED.
  B) Strip the asserts (report-only) and defer gating to bench.sh
     (Phase 10) — loses the executable gate for two phases.
- Verdict: **Phase 6 acceptance criteria are MET.** The stale test is a
  Task 6.1 closure leftover, not a Phase 6 functional or performance
  regression; recommend ruling A/B before Phase 7's paged-attention
  kernel work, which needs this gate trustworthy.
- Next (on the ruling): Phase 7 — llguidance structured output,
  tool-call parsers, /v1/messages, paged-attention kernel (SPEC §12).

## [2026-07-10] Phase 6 — throughput gate re-aimed to the ADR 0003 two-bar split — DONE (PM ruling A on the gate-review DECISION NEEDED)
- What: crates/kiln-models/tests/throughput.rs rewritten from the
  superseded single ≥3x bar to the bars ADR 0003 actually established;
  test renamed batch16_aggregate_is_3x_single_stream →
  batch16_aggregate_meets_adr0003_bars. Harness (two single-stream
  denominators, run_engine, medians-of-3) unchanged.
  1. **Lane → bar mapping, confirmed by measurement:** all-sampled (new
     lane in the test) is the bar-1 non-deterministic load — one
     full-width forward per step, B' uninvolved — and measures at the
     single-forward scale (2.8x). Mixed 8+8 and all-greedy are bar-2
     deterministic-containing lanes (~1.9x/2.2x, the two-forwards-per-
     step scale) — matching ADR 0003's own floor table, which lists the
     mixed number under the deterministic-containing floor.
  2. **Bar 1 assert:** ratio ≥ 3x (SPEC §11.3, still in force for this
     lane) OR ≥ recorded-1B-reference (2.68x) × 0.90. The absolute 3x
     is out of reach on the tiny model for pre-B' reasons (recorded
     pre-/post-B' 334.1/331.6 tok/s ≈ 2.7x — the small-model sampler
     artifact named in ADR 0003; 3x is certified at 8B-class, 3.18x),
     so on this harness the lane gates as no-regression against its
     recorded reference while the OR keeps the SPEC bar self-documenting.
  3. **Bar 2 asserts:** ratio ≥ recorded floor × 0.90 per lane (greedy
     2.10x, mixed 1.80x — the ADR 0003 dev-machine W=9 table), floors
     as no-regression bounds, not targets, per the ADR.
  4. **Tolerance = 10% (× 0.90), stated in the test:** exactly SPEC
     §11.3's bench-regression threshold, which ADR 0003 adopts for
     floor breaches; it also covers the ~9% run-to-run spread the
     ledger records for this harness (Phase 4/5 gate runs 3.05x–3.34x).
     The floors are single measured samples — asserted bare they would
     flake on ordinary run noise (mixed's recorded 1.80x IS a typical
     value, not a lower envelope).
  5. **Floors NOT ratcheted to today's numbers** (the PM's
     floor-vs-headroom question): today's three measurement sets
     (sampled 2.81/2.84x, greedy 2.20/2.21/2.22x, mixed 1.87/1.87/1.90x)
     sit +4–6% over the recorded values — inside the historical ~9% run
     spread, single machine, single day. Treated as headroom; recorded
     values stand. Ratcheting floors is an ADR 0003 revisit action (and
     floors go stale at the next mlx-c pin bump); the test constants
     cite the ADR and say exactly that.
- Decisions:
  - "Mixed-majority" (ADR 0003 bar-1 wording) gets no asserted ≥3x lane:
    any deterministic admixture pays the extra full weight pass under B'
    (50/50 measures ~1.9x; a 15-sampled+1-greedy load has the same
    two-forward step shape), so a mixed-majority ≥3x lane would fail
    structurally at W=9. The ADR's own floor table already places mixed
    under bar 2; the test follows the table. The pin-bump revisit
    re-opens this if dispatch changes.
- Deviations: none.
- Acceptance (release, dev machine, KILN_TEST_MODELS set):
  ```
  $ cargo test -p kiln-models --release --test throughput -- --ignored --nocapture
  prompt: 27 tokens, decode: 128 tokens, deterministic width 9
  single-stream decode: 123.9 tok/s (phase-3 pipelined path), 123.1 tok/s (engine batch 1)
  batch-16 aggregate vs the stricter single-stream rate:
    all-sampled 352.5 tok/s -> 2.84x (bar 1),
    mixed 235.1 tok/s -> 1.90x (bar 2, floor 1.8x),
    all-greedy (B' sub-batched) 275.2 tok/s -> 2.22x (bar 2, floor 2.1x)
  test batch16_aggregate_meets_adr0003_bars ... ok
  test result: ok. 1 passed; finished in 75.47s
  $ cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
  clean
  $ cargo build --workspace --no-default-features
    cargo clippy --workspace --all-targets --no-default-features -- -D warnings
  clean (linux CI shape)
  ```
  CI matrix verification recorded on the PR thread per precedent (the
  test is #[ignore]d in CI; the blocking lanes gate its compilation via
  clippy --all-targets).
- **Phase 6 is CLOSED** (gate-review criteria met + this ruling
  implemented). Phase 7 not started per instruction.
- Next: Phase 7 — llguidance structured output, tool-call parsers,
  /v1/messages, paged-attention kernel (SPEC §12).

## [2026-07-10] Phase 6 — CI verification on PR #13 + one flake datapoint — DONE; PHASE 6 CLOSED
- PR #13 (throughput gate re-aim) merged; real CI shapes green — lint,
  compile-linux, test-macos, test-macos-release (run 29127458930).
- Flake datapoint (pre-existing, unrelated to the diff — the change is
  a single #[ignore]d perf test): first test-macos attempt failed in
  kiln-worker rpc `cancel_and_drain_rpc_semantics` — the graceful-drain
  phase asserts a 40-token request finishes inside deadline_ms=2500,
  a >=16 tok/s decode assumption; the hosted runner decoded at ~4 tok/s
  under shared-GPU contention (Timings: 3431ms for 15 tokens), so the
  deadline escalated the short request to Cancelled. Same flake class
  as the e2e metrics race deflaked earlier this phase (PR #8 entry).
  Job re-run: green. Deflake filed as a follow-up task (size the
  deadline from a measured per-token rate, or adjust the short-request/
  deadline ratio for worst-case CI speeds — without weakening the
  deadline-escalation semantics under test).
- **Phase 6 is CLOSED**: all §12 Phase 6 items done (tasks 6.1–6.4 +
  folded rope_scaling variants), gate review criteria met, the one
  gate-review finding (stale throughput bar) ruled and implemented,
  all bars green on the dev machine and CI.
- Next: Phase 7 — llguidance structured output, tool-call parsers,
  /v1/messages, paged-attention kernel (SPEC §12). Not started in this
  session per instruction.

## [2026-07-10] Phase 6 — rpc.rs graceful-drain deflake (PR #13 red run 29127458930) — DONE
- What: `cancel_and_drain_rpc_semantics` (crates/kiln-worker/tests/rpc.rs)
  flaked on PR #13's CI: the graceful-drain phase hardcoded a 40-token
  "short" request against `deadline_ms: 2500` — a ≥16 tok/s decode
  assumption. The hosted runner decoded at ~4 tok/s under shared-GPU
  contention (Finished.timings: decode_ms 3431 for 15 tokens), so the
  deadline escalated the short request to CANCELLED and the Length
  assert failed. Fix, following the PR #8 poll-all-counters precedent
  (encode the actual contract instead of tuning constants): the
  escalation contract is about ORDERING, not absolute speed — the
  deadline must outlive the short request yet expire well before the
  long one could finish, and both bounds scale with the decode rate.
  The cancel phase (which runs first on the same worker) now reads one
  warmup chunk then times 12 generated tokens; the drain phase sizes
  `deadline_ms = max(8 x 16-token short request x measured period,
  2500ms floor)` and the long request's max_tokens to 5x whatever the
  deadline can decode. Ratios are rate-independent: short finishes at
  ~16p << deadline (>=128p) << long (5x deadline). Failure messages and
  an eprintln now carry the measured period + computed sizing for
  future CI-log diagnosis.
- Decisions: measured-rate sizing chosen over widening fixed constants —
  any fixed deadline/token pair embeds a bounded rate window (a faster
  future machine would let the long request FINISH before the deadline,
  silently un-testing escalation; a slower runner re-flakes the short
  side). Short request shrunk 40 -> 16 tokens to bound worst-case wall
  time on slow runners (deadline = 128 x per-token period). 2500ms
  floor keeps fast-machine behavior identical to the old constant.
- Deviations: none. Escalation is still exercised on every run (long
  request must end CANCELLED); no assertion was weakened — the short
  request still must finish Length with exact completion_tokens.
- Acceptance:
  ```
  $ cargo clippy --workspace --all-targets -- -D warnings   -> clean (exit 0)
  $ cargo fmt --check                                       -> clean
  $ KILN_TEST_MODELS=~/.kiln/test-models cargo test -p kiln-worker --test rpc -- --nocapture
  running 2 tests
  measured decode 12.2 ms/token -> drain deadline 2500 ms, long request 1025 tokens
  test prefix_cache_stats_and_ssd_restart ... ok
  worker 1: cancel + graceful drain (deadline escalation) ok
  worker 2: immediate drain ok
  test cancel_and_drain_rpc_semantics ... ok
  test result: ok. 2 passed; 0 failed; ... finished in 5.16s
  (second run: measured 15.0 ms/token -> long request 832 tokens; 2 passed)
  $ ruff check python/ tests/e2e -> All checks passed!  ruff format --check -> clean
  ```
  On this dev machine the 2500ms floor dominates (short ~200ms, long
  would need ~12.5s), so the local run exercises the floor regime; the
  measured regime (deadline 128p > 2500ms) engages exactly on runners
  slower than ~78 tok/s — the flake population.
- Next: land this as its own PR to main — PR #13 merged (after a
  manual job re-run went green) while this fix was in flight, so this
  is the deflake follow-up filed in the phase-close entry above. Then
  Phase 7 (SPEC §12).

## [2026-07-10] Phase 6 follow-up — drain deflake v2: long side made structural (PR #14 red run 29139947140) — DONE
- What: the measured-rate deflake's FIRST CI execution failed the
  OPPOSITE assert (rpc.rs:329 "drain deadline must escalate
  stragglers"): the long request finished Length (all 640 tokens)
  before the 24097ms deadline. Log: measured 188.3 ms/token during the
  cancel phase, but the long request decoded at 17.3 ms/token
  (Timings: decode_ms 11045 for 640; 57.85 tok/s) — an 11x
  intra-run swing. Cause: contention on the shared runner is
  NON-STATIONARY — the sibling test in the same binary
  (prefix_cache_stats_and_ssd_restart) held the GPU through our
  measurement window (its "ok" logged at 04:47:06, mid-way through our
  16.9s run), then finished, and the drain phase ran uncontended. Any
  predicted-rate multiple on the long side loses to an
  order-of-magnitude swing in that direction.
- Fix (v2): the long request is no longer sized from the measured rate.
  Its max_tokens is a constant 12_000, chosen against the two hard
  bounds: (a) admission — engine rejects prompt + max_tokens over the
  KV-pool capacity (512 blocks x 32 = 16384; 8 + 12000 fits, and
  runtime usage before escalation stays far under the pool); (b)
  escalation coverage — finishing 12000 tokens inside the deadline
  needs a >=37x measurement-to-execution swing (observed worst: 11x),
  or >=2400 tok/s in the floor regime (~20x current hardware). The
  deadline still scales from the measured period — now 20x the short
  request's decode time, floored at 5000ms (was 8x/2500) — because on
  the short side over-estimating the period only adds margin; a
  post-measurement SLOWDOWN now needs >23x to break it.
- Reconciliation: PR #14 IS the deflake follow-up task filed in the
  phase-close entry above ("Deflake filed as a follow-up task", commit
  a7aa48a) — one effort, not two; no other drain-deflake work is
  outstanding once PR #14 merges.
- Deviations: none. Same semantics as v1: short must finish Length
  with exact completion_tokens under GRACEFUL drain; long must be
  escalated to Cancelled; DRAINING rejection and Health checks
  unchanged.
- Acceptance:
  ```
  $ cargo fmt --check -> clean; cargo clippy -p kiln-worker --all-targets -- -D warnings -> clean
  $ KILN_TEST_MODELS=~/.kiln/test-models cargo test -p kiln-worker --test rpc -- --nocapture
  measured decode 14.8 ms/token -> drain deadline 5000 ms
  worker 1: cancel + graceful drain (deadline escalation) ok
  worker 2: immediate drain ok
  test result: ok. 2 passed; 0 failed (8.19s)
  contention stress (two staggered concurrent instances of the rpc
  binary, exercising both swing directions): exit A=0 B=0, both
  "2 passed; 0 failed"
  ```
  CI on PR #14 is the acceptance for the contended-runner regime; the
  dev GPU cannot reproduce CI-grade contention swings.
- Next: PR #14 green on CI -> merge -> Phase 7 (SPEC §12). Phase 7 not
  started until PR #14 is resolved and merged.

## [2026-07-10] Phase 6 follow-up — PR #14 merged: drain deflake CI-verified; follow-up CLOSED — DONE
- PR #14 (drain deflake v1+v2) merged to main (c5eb22c) with all four
  checks green on run 29141387062 — lint, compile-linux, test-macos,
  test-macos-release. `cancel_and_drain_rpc_semantics` executed THREE
  times on the shared runner within the run (workspace pass +
  model-gated suite in test-macos, plus test-macos-release): all ok.
  (Passing tests' stdout is cargo-captured, so the measured-rate
  eprintln only appears in logs on failure — by design, it's a
  diagnosis aid for red runs.)
- This closes the deflake follow-up filed in the Phase 6 close entry
  (a7aa48a): that filed task, the PR #13 flake datapoint, and PR #14
  are ONE effort — reconciled explicitly in the v2 entry above. No
  drain-deflake work remains outstanding.
- Verification history for the record: v1 (measured-rate sizing both
  sides) failed its first CI execution on the long side under a
  non-stationary 11x contention swing (run 29139947140); v2 (long side
  structural at max_tokens 12000, deadline-only rate scaling) is what
  merged. Both attempts and the root-cause analysis are in the two
  entries above.
- Next: Phase 7 — llguidance structured output, tool-call parsers,
  /v1/messages, paged-attention kernel (SPEC §12). Not started in this
  session per instruction.

## [2026-07-13] Phase 7 / Task 1 — llguidance structured output (json_schema + regex) — DONE
- What: grammar-constrained decoding per SPEC §12 Phase 7 / §6.2 step 3.
  New `kiln-engine/src/grammar.rs`: `GrammarEnv` (llguidance ParserFactory
  over a token trie built from the model's tokenizer.json, padded to the
  model's `vocab_size`, model EOS ids wired in) and `Grammar` (per-request
  matcher: `allowed_tokens` mask + `commit`). Engine: `EngineRequest.grammar`;
  mask computed host-side at plan time (per-seq in-band error on fault),
  applied in `sample_from_row` as `where_cond(mask, logits, -inf)` after
  penalties and before logprob normalization; `settle_sampled` commits each
  sampled token and finishes `Stop` on grammar completion; grammar seqs are
  excluded from the async_eval pipeline (their next mask needs this step's
  token host-side — same reason as penalties). Worker: engine thread builds
  `GrammarEnv` after model load (~0.4s, degrades to no-capability on
  failure), advertises `CAPABILITY_GRAMMAR`; Submit compiles GrammarSpec on
  the handler task — `json_schema`/`regex` supported, `lark` →
  GRAMMAR_UNSUPPORTED, uncompilable → GRAMMAR_COMPILE, both in-band.
  Gateway: `response_format` `json_schema`/`json_object` → GrammarSpec on
  the frozen proto field; 400 when the worker lacks CAPABILITY_GRAMMAR
  (SPEC §5 gating). Python worker unchanged (scoped out per SPEC §5: no
  GRAMMAR capability in v1; its in-band rejection + unit test already
  existed). Tests: kiln-worker/tests/grammar.rs (100/100 acceptance,
  CI-blocking — schema validity is device-independent), kiln-engine/tests/
  grammar.rs (host-level mask/commit/compile-error), gateway response_format
  unit tests, e2e test_structured_output.py (openai SDK, both stacks).
  ci.yml model-gated step now runs the two new suites.
- Decisions: llguidance 1.7.6 + toktrie_hf_tokenizers 1.7.6 added
  (SPEC §3 names llguidance for structured output; MIT, pure Rust; optional
  deps behind kiln-engine's `metal` feature so the Linux compile-check
  stays lean; toktrie_hf_tokenizers internally uses tokenizers 0.21 beside
  the workspace's 0.23 — no cross-version type sharing, the trie is built
  straight from tokenizer.json). Mask at plan time (before the forward):
  state-correct since it depends only on committed tokens, and host cost is
  ~2µs/step (measured in a probe; first mask ~0.8ms at compile). Grammar
  completion finishes `Stop` even without an EOS sample (covers ignore_eos).
  `lark` deliberately not wired despite llguidance supporting it — Phase 7
  task text scopes json_schema + regex; proto error code semantics used:
  UNSUPPORTED = grammar kind, COMPILE = bad grammar text. Acceptance
  schemas set llguidance's `x-guidance: {whitespace_flexible: false}` so
  compact output gives a hard token-length bound — a Length finish is a
  real failure, not a sizing artifact (PR #13/#14 deflake lesson applied
  in advance).
- Deviations: none. Proto untouched (GrammarSpec/capability/error codes
  were already defined and are now honored as specified).
- Acceptance:
  ```
  $ KILN_TEST_MODELS=~/.kiln/test-models cargo test -p kiln-worker --test grammar -- --nocapture
  100/100 schema-valid generations (per shape: [34, 33, 33])
  test grammar_constrained_decoding ... ok (19.63s)   [release run: ok, see below]
  $ KILN_TEST_MODELS=~/.kiln/test-models cargo test -p kiln-models --test golden -- --nocapture
  ... exact match for every fixture model (llama/qwen2.5/qwen3 x 4bit/8bit/bf16/gemma) — masking
  is a verified no-op on unconstrained requests. test result: ok (293s)
  $ batching/preemption/prefix_cache/prefix_multiturn/leak/leak_batched/calibration/prefill_pad/rpc: all ok
  $ cargo test --workspace -> 40 targets, 0 failures;  cargo test -p kiln-engine --test grammar -> 4 passed
  $ cargo clippy --workspace --all-targets -- -D warnings -> clean (metal + --no-default-features)
  $ cargo fmt --all --check -> clean;  ruff check/format -> clean
  $ pytest python/kiln_worker_py/tests -> 35 passed (grammar-unsupported test intact)
  $ uv run --project tests/e2e pytest tests/e2e -> 42 passed, 2 skipped (rust-only structured-output
    tests skip on the python stack; python stack asserts the 400 capability gate)
  ```
  One local red herring during the run: the rpc suite timed out once because a
  concurrent `cargo build --no-default-features` (Linux-shape check) replaced
  target/debug/kiln-worker with the metal-less stub while the suite was
  spawning it — not a code fault; re-run green. CI builds each shape in its
  own job, so the shapes cannot collide there.
- Next: Phase 7 continues — tool-call streaming parsers (Hermes/Llama3/Qwen)
  in kiln-tokenize with unit fixtures (SPEC §12). Stopped before them per
  instruction.

## [2026-07-13] Phase 7 / Task 1 — PR #15 merged: structured output CI-verified — DONE
- PR #15 (llguidance structured output) merged to main (b00f8bb) with all
  four checks green on run 29267386981 — lint 48s, compile-linux 47s,
  test-macos 16m43s, test-macos-release 4m59s. First CI execution, no
  re-runs needed.
- Verified in the test-macos log (not assumed): the model-gated blocking
  step ran both new suites — `grammar_constrained_decoding` passed on the
  shared runner in ~40s (16:50:13 → 16:50:52), kiln-engine's host-level
  grammar tests passed — and the e2e structured-output tests passed on
  both stacks (`[rust]`: schema-valid + streaming + json_object;
  `[python]`: the 400 capability gate; `test_grammar_is_unsupported`
  intact in the python worker suite). The workspace (env-less) lanes ran
  the grammar test binaries as skips, as designed.
- Next: Phase 7 continues — tool-call streaming parsers (Hermes/Llama3/
  Qwen) in kiln-tokenize with unit fixtures (SPEC §12).

## [2026-07-13] Phase 7 / Task 2 — tool-call streaming parsers (Hermes/Llama/Qwen-XML) — DONE
- What: SPEC §8.2 tool-call extraction. New `kiln-tokenize/src/toolcall.rs`:
  three streaming parsers behind one `ToolCallParser` (events: Content /
  CallStart / CallArgs / CallEnd) — Hermes `<tool_call>{json}</tool_call>`
  (Qwen 2.5/3), Llama 3.x (optional `<|python_tag|>` + bare JSON,
  `;`-separated multi-call, and the `{"type":"function","function":{...}}`
  wrapper real Llama-3.2 output mirrors back), Qwen3-Coder XML-ish
  (`<function=`/`<parameter=`, values coerced via the request's tool
  schemas). Format selected from model metadata: `ToolCallFormat::detect`
  over the chat template source (`<function=` → XML, `<tool_call>` →
  Hermes, `Environment: ipython`/`<|python_tag|>` → Llama), exposed as
  `ChatTemplate::tool_call_format()`. Parsers consume the StreamingDecoder
  text stream; argument bytes stream verbatim (never re-serialized).
  Gateway: `tools`/`tool_choice` (`auto`/`none`; forced choice is a clear
  400), assistant `tool_calls` + `tool` role messages render through the
  template (`ChatMessage.tool_calls`, `render_with_tools`); parser output
  becomes OpenAI `tool_calls` deltas (stream) / `message.tool_calls`
  (non-stream), `finish_reason: "tool_calls"`, ids `call_<uuidv7>`.
  Fixtures: scripts/gen-tool-fixtures.py captures REAL greedy completions
  from the pinned models (llama-3.2-1b, qwen3-0.6b — 11 generation
  fixtures incl. `<think>` + call, two-call, multibyte args, python-tag
  call, post-tool-result turns) plus template-derived serializations for
  formats no pinned model emits (Qwen3-Coder XML from the official
  template; Llama bare-JSON) → crates/kiln-tokenize/tests/fixtures/toolcall/.
  Tests: toolcall unit tests (18) + fixture suite — chunk-split invariance
  (whole vs 1/2/3/5/11-char chunks), token-level replay through the real
  StreamingDecoder (1/2/3 tokens per chunk), and byte-for-byte prompt
  render parity vs the captured transformers rendering. e2e
  test_tool_calls.py: openai SDK round trip on BOTH worker kinds (llama
  stack ×2) + a rust-worker qwen3 stack for Hermes; streaming deltas must
  reassemble to exactly the non-streaming response. ci.yml model-gated
  blocking step now also runs `--test toolcall_fixtures`. test_chat.py's
  validation-shape probe swapped its obsolete premise (tools 400'd in
  Phase 2) for a still-unsupported feature (`tool_choice: "required"`) —
  same guarantee under test, current feature set.
- Decisions: parser events are gateway-agnostic (kiln-tokenize stays
  HTTP-free). Text runs outside calls: whitespace-only runs dropped,
  substantive runs verbatim (keeps inter-call `\n` separators out of
  content without eating think-blocks). Malformed blocks degrade — before
  the name is known the block replays as content; after (CallStart already
  emitted) the call closes with what streamed; truncation (length) flushes
  partial arguments then CallEnd, mirroring OpenAI under
  `finish_reason: "length"`. Real capture drove two shape extensions the
  docs don't mention: Llama-3.2 emits `<|python_tag|>` for CUSTOM tools
  under the default template (the "JSON format" docs imply bare JSON), and
  after a tool result it mirrors the OpenAI wrapper shape back — both are
  committed fixtures. Qwen XML values coerce by declared schema type
  (template serializes Python-style `True`; reference parser behavior),
  unknown params fall back to JSON-parse-else-string. Feature flags
  `preserve_order` on serde_json + minijinja (no new crates; indexmap was
  already in the graph): tool JSON renders into the PROMPT, so key order
  is model-visible; with a Kiln-owned Python-style `tojson` filter
  (json.dumps separators/indent semantics) our render is byte-identical
  to the transformers reference — asserted per fixture. `tool_choice`
  forcing ("required"/named) deliberately 400s: honest forcing needs
  grammar-coupled decoding (natural later marriage with llguidance).
- Deviations: none.
- Acceptance:
  ```
  $ KILN_TEST_MODELS=~/.kiln/test-models cargo test -p kiln-tokenize
  lib 31 passed (18 toolcall unit incl. chunk-split invariance on edge inputs)
  toolcall_fixtures 3 passed: 16 fixtures x {whole, 1/2/3/5/11-char chunks};
    11 generation fixtures x token-level replay through StreamingDecoder
    (1/2/3 ids per chunk); prompt render parity vs transformers byte-for-byte
  detok/model_dir/tokenizer suites unchanged, green
  $ KILN_TEST_MODELS=... cargo test --workspace  -> 23 targets, exit 0
    (incl. golden: exact match every fixture — preserve_order + the
     python-style tojson filter perturb nothing outside tool rendering)
  $ cargo build/clippy --workspace --no-default-features -> clean (linux shape)
  $ cargo fmt --all --check; cargo clippy --workspace --all-targets -- -D warnings -> clean
  $ ruff check / format --check python/ tests/e2e -> clean
  $ pytest python/kiln_worker_py/tests -> 35 passed
  $ uv run --project tests/e2e pytest tests/e2e -> 53 passed, 2 skipped
    (test_tool_calls.py: 11 new — non-stream + stream-reassembly + round
     trip + tool_choice none/required on BOTH worker kinds vs llama-3.2-1b;
     Hermes via a rust-worker qwen3-0.6b stack, think-block as content;
     2 skips = pre-existing rust-only structured-output tests on python stack)
  ```
- Next: Phase 7 part 3 — `POST /v1/messages` (Anthropic API adapter,
  thinking passthrough), then the paged-attention Metal kernel (SPEC §12).

## [2026-07-13] Phase 7 / Task 2 — PR #16 merged: tool-call parsers CI-verified — DONE
- PR #16 (tool-call streaming parsers) merged to main (f1c6b45) with all
  four checks green on run 29273399967 — lint, compile-linux, test-macos,
  test-macos-release. First CI execution, no re-runs needed.
- Verified in the test-macos log (not assumed): the model-gated blocking
  step ran `--test toolcall_fixtures` with the models present — the
  decoder-fed replay (`fixtures_reassemble_from_streaming_decoder_segments`)
  executed for ~13s (real run, not an env-less skip) and the
  transformers render-parity test passed on the runner; the e2e suite
  passed all 11 tool tests on BOTH stacks (`[python]` and `[rust]`:
  non-stream, stream-reassembly equality, round trip, tool_choice
  none/400) plus `test_hermes_tool_call_streaming` on the rust-worker
  qwen3-0.6b stack. test-macos-release ran the fixture suite green in
  release as well.
- Next: Phase 7 part 3 — `POST /v1/messages` (Anthropic API adapter,
  thinking passthrough as `thinking` content blocks), then the
  paged-attention Metal kernel (SPEC §12). Stopped before them per
  instruction.

## [2026-07-13] Phase 7 / Task 3 — `POST /v1/messages` (Anthropic Messages API) — DONE
- What: SPEC §8.1 Anthropic adapter over the SHARED chat pipeline — a second
  API framing, not a new serving path. kiln-tokenize: new `think.rs`
  streaming `<think>` extractor (`ThinkParser` → Thinking/Text events;
  chunk-split invariance like the toolcall parsers; partial-tag holdback;
  tag-boundary whitespace trimmed, interior verbatim; unclosed block =
  truncated thinking), thinking-model detection from the chat template
  source (`ChatTemplate::emits_think_tags`, `</think>` marker — same
  pattern as tool-format detection), and `render_full` (render_with_tools
  + extra JSON template vars). Gateway: `anthropic.rs` (wire types +
  validation into the ValidatedChat-equivalent internal shape: system
  string/blocks, content blocks incl. tool_use/tool_result round trip,
  Anthropic→OpenAI tool-shape conversion for the templates/parsers,
  temperature [0,1] + top_k wired to SamplingParams) and `messages.rs`
  (handler reusing ready_entry/encode_prompt/TextPipeline/ToolRoute/
  classify_finished/CompletionCtx; non-stream JSON + named-event SSE:
  message_start, content_block_start/delta/stop with thinking_delta /
  text_delta / input_json_delta, message_delta, message_stop, mid-stream
  `event: error`). Auth accepts `x-api-key` (the anthropic SDK's only
  header) everywhere; /v1/messages gets the Anthropic error envelope
  (`{"type":"error","error":{type,message}}`, status-keyed types).
  `kiln_messages_total` metric. `stop_sequence` attribution: TextPipeline
  now records WHICH stop fired (rust path); python path checks
  `Finished.matched_stop` membership in the request's stop_sequences
  (worker reports EOS token text on natural stop — membership filters it).
  e2e: `test_messages.py` drives the real `anthropic` SDK against BOTH
  worker kinds (llama stack ×2: non-stream shape, stream-reassembly
  equality, system-prompt steering, stop_sequence attribution, tool use
  non-stream + stream + round trip, forced-choice 400, error envelopes)
  plus the rust-worker qwen3-0.6b stack (thinking blocks correctly shaped
  AND separated from text: [thinking, text] non-stream, streamed
  block/delta sequence reassembling byte-equal, [thinking, tool_use] with
  tools, thinking-disabled → [text] only). qwen stack fixture moved to
  conftest (session-scoped, shared with test_tool_calls — one model load).
- Decisions: think extraction is /v1/messages-only per SPEC §8.1 (the
  OpenAI endpoint keeps `<think>` as plain content — test_tool_calls
  asserts that); the parser is gated on template detection so a
  non-thinking model echoing "<think>" is never misclassified.
  `thinking: {"type":"disabled"}` renders the template with
  `enable_thinking=false` (real disable on Qwen3; lenient-undefined
  no-op elsewhere); enabled/adaptive are the models' native behavior and
  budgets are unenforceable → accepted, ignored. Request-history thinking
  blocks are dropped (the thinking-trained templates strip prior-turn
  reasoning themselves); `signature` is always "" (open weights — nothing
  to sign; SDK requires the field). stop upgrade precedence mirrors the
  OpenAI adapter: completed calls → `tool_use`, else matched request
  sequence → `stop_sequence` + value, else `end_turn`. Non-streaming
  tool_use whose args never became valid JSON (length truncation) is
  dropped with a debug log — `input` is an object, partial JSON is
  unrepresentable; streaming shows the partial input_json_delta bytes
  (same stream/non-stream divergence as the reference API).
  `tool_choice` any/tool → 400 (forcing needs grammar-coupled decoding,
  same ruling as task 7.2); `disable_parallel_tool_use: true` → 400
  (can't bound what a model emits; reject-don't-silently-drop). New
  test-only dep `anthropic` in tests/e2e (SPEC §11.3 mandates conformance
  via the real client SDKs).
- Deviations: none.
- Acceptance:
  ```
  $ cargo test -p kiln-tokenize --lib   -> 43 passed (12 new think:: incl.
    chunk-split invariance at sizes 1/2/3/5/7/11/whole, multibyte, truncation)
  $ cargo test -p kiln-gateway --lib    -> 42 passed (7 new: anthropic
    validation/wire shapes + anthropic error envelope)
  $ KILN_TEST_MODELS=... cargo test --workspace -> 41 suites, all ok
    (incl. golden: exact match every fixture — adapter changes touch no
    model/engine path)
  $ cargo build --workspace --no-default-features -> clean (linux shape)
  $ cargo fmt --all --check; cargo clippy --workspace --all-targets -- -D warnings -> clean
  $ ruff check / format --check python/ tests/e2e -> clean
  $ pytest python/kiln_worker_py/tests -> 35 passed
  $ uv run --project tests/e2e pytest tests/e2e -> 77 passed, 2 skipped
    (24 new in test_messages.py via the anthropic SDK: 10 tests x both
     worker stacks + 4 thinking tests on the rust qwen3 stack; the SDK's
     pydantic models validate every response/stream-event shape; 2 skips =
     pre-existing rust-only structured-output tests on python stack)
  ```
- Next: Phase 7 part 4 — custom Metal paged-attention kernel behind a flag
  + parity test vs the gather path (SPEC §12). Own session; not started
  here per instruction. This entry closes Phase 7's API surface.

## [2026-07-13] Phase 7 / Task 3 — PR #17 merged: /v1/messages CI-verified — DONE
- PR #17 (Anthropic Messages API adapter) merged to main (f9961cd) with all
  four checks green on run 29278581375 — lint (42s), compile-linux (35s),
  test-macos (14m55s), test-macos-release (3m42s). First CI execution, no
  re-runs needed.
- Verified in the test-macos log (not assumed): the e2e step ran all 24
  test_messages.py tests through the real `anthropic` SDK and every one
  PASSED on the runner — the 10-test conformance set on BOTH stacks
  (`[python]` and `[rust]`: non-stream shape, stream-reassembly equality,
  system-prompt steering, stop_sequence attribution, tool use non-stream/
  stream/round-trip, forced-choice 400, Anthropic error envelopes, typed
  NotFoundError) plus the 4 thinking tests on the rust-worker qwen3-0.6b
  stack (thinking blocks shaped and separated: [thinking, text],
  streamed block/delta sequence reassembling byte-equal to non-stream,
  [thinking, tool_use] with tools, thinking-disabled → [text] only).
  Suite total on the runner: 77 passed, 2 skipped (the pre-existing
  rust-only structured-output skips on the python stack); python worker
  35 passed; the think-parser unit tests ran in the env-less
  `cargo test --workspace` step.
- This closes Phase 7's API surface (SPEC §12 Phase 7: structured output
  7.1, tool parsers 7.2, Anthropic API 7.3 — all merged and CI-verified).
- Next: Phase 7 part 4 — custom Metal paged-attention kernel behind a
  flag + parity test vs the gather path, ≥15% decode throughput at 8k
  context or the flag stays off (SPEC §12). Stopped before it per
  instruction; it is its own session.

## [2026-07-13] Phase 7 — scoping: mlx_compile experiments fold into the kernel session
- Question (PM): is the §12 Phase 7 `mlx_compile` item still in scope for
  this phase, or does it fold into/after the paged-attention kernel session?
- Ruling: in scope for Phase 7, folded INTO the kernel session as its
  profiling-gated rider — not standalone, not deferred past the phase.
  Basis: SPEC conditions the item on profiling in both places it appears
  (§7.2 "where profiling shows wins (Phase 7 optimization task, not
  before)"; §12 "experiments where profiled") and Phase 7's acceptance
  criteria attach no bar to it — the phase's only remaining gate is the
  kernel item (parity-exact + ≥15% decode at 8k, else flag off,
  documented). The kernel session must produce decode-step profiles at 8k
  to prove that bar; those same profiles are the "where profiled" evidence
  for compile experiments. Running them earlier would baseline against a
  step function the kernel changes; later would re-derive the profiles.
- Verified at the pin: mlx-c v0.6.0 exposes the compile API
  (mlx/c/compile.h: mlx_compile, mlx_detail_compile, compile modes) — no
  submodule bump needed if experiments proceed; no DECISION NEEDED.
- Exit condition for the item (kernel session records one of): (a) profile
  shows no fusion/dispatch-bound hotspots → experiments not warranted,
  item closed with the profile as evidence; (b) hotspots → experiments
  run, adopted or rejected with numbers. Either way, greedy outputs must
  stay bit-identical (golden/parity gates apply to any adoption).
- Next: Phase 7 part 4 — paged-attention Metal kernel session, now
  including the mlx_compile rider above.
## [2026-07-13] Phase 7 / Task 4 — custom Metal paged-attention kernel (SPEC §7.4/§12) — DONE
- What: block-table-aware paged attention via `mlx_fast_metal_kernel_new`,
  replacing the gather copy for decode steps, behind
  `EngineConfig::paged_attention_kernel` / `--paged-attention-kernel` /
  `[defaults] paged_attention_kernel` (DEFAULT OFF — not flipped in this
  session, per instruction, even though both acceptance bars passed).
  - kiln-mlx: `fast::MetalKernel` safe wrapper (custom-kernel handle +
    per-call config, RAII/leak-counted), `device::gpu_architecture()`;
    `VectorArray` out-param support; custom-kernel smoke test in wrappers.
  - kiln-engine `paged_attn.rs`: ports of the pinned MLX v0.31.1
    `sdpa_vector` / `sdpa_vector_2pass_1/2` kernels with ONLY the K/V
    addressing changed (pool + u32 block table instead of contiguous seq
    strides); the reference's variant-dispatch predicate and device-class
    `blocks` table replicated exactly (keyed on the architecture string's
    last char, from `mlx_device_info`). `PagedKv::paged_sdpa` +
    `enable_attention_kernel`; per-seq per-step inputs (`PagedAttnInputs`)
    prepared once in `build_seq_step` (covers sync + pipelined decode) and
    shared across all layers' calls.
  - kiln-models nn.rs: decode-shaped segments (`len==1`, `pad==0`,
    non-softcap) route to the kernel when inputs are prepared; everything
    else (prefill pieces, ADR 0002 padded pieces, gemma2 softcap) stays on
    gather+SDPA. gemma2 therefore never uses the kernel (documented).
  - Plumbing: worker `EngineOptions` (renamed from `CacheOptions`) +
    CLI flag; gateway `EngineDefaults.paged_attention_kernel` + supervisor
    argv; kiln.toml.example.
- Parity guarantee (stated per session instruction BEFORE writing the
  kernel, recorded here): NOT assumed bit-exact — designed bit-exact BY
  CONSTRUCTION and then MEASURED. Basis: (1) the kernels are ports of the
  exact kernels the gather path executes, same iteration order, same
  f32-accumulator/`fast::exp`/`simd_sum` algebra — reduction order
  preserved, which is precisely what ADR 0002 says defines a kernel class;
  (2) both compile paths are non-fast-math at the pin (builtin metallib
  `-fno-fast-math`, custom-kernel JIT `setFastMathEnabled(false)`);
  (3) the one unresolvable-from-source risk — offline metallib compiler vs
  runtime JIT codegen (e.g. fma contraction) — was named up front and
  measured directly; (4) variant dispatch (1-pass vs 2-pass at
  `((devc=='d'|'s')&&N>=1024)||(GQA&&N>=4096)`, plus the devc x N `blocks`
  quantization) is replicated so the port never sits in a different
  variant than the reference at any (device, N). Fallback bar if any bit
  divergence appeared: token-id equality per ADR 0002, characterized,
  tests not weakened, DECISION NEEDED. Outcome: bit-exactness HELD
  everywhere measured (below), so the fallback was not needed. Scope
  unchanged by this session: same-device claims only (ADR 0004); B'
  untouched (attention was always per-sequence — no M-dependence).
- Measured results (dev machine, M4-class, W=9):
  - Kernel-vs-gather BIT equality (new `kiln-engine/tests/paged_attn.rs`):
    raw output bytes identical across {f16, bf16} x {32/8/64, 16/8/128,
    4/1/256 (H/HK/D)} x 14 context lengths {1..8193} straddling every
    dispatch boundary, outlier-heavy values, rotated (non-identity) block
    table, ragged tails. The offline-vs-JIT compiler risk did NOT
    materialize at this pin on this device.
  - Golden (flag ON and OFF, every fixture model): exact token-id match,
    single-stream AND width-16 (corrected bars: ADR 0002 B' + ADR 0004
    same-device scope). golden.rs now runs the full round set on BOTH
    attention paths.
  - 8k-context decode throughput (new `#[ignore]`d release gate
    `kiln-models/tests/paged_attn_gate.rs`, llama-3.2-1b-4bit, 8064-token
    prompt + 128 decode, single-stream, median of 3): gather 42.3 tok/s
    (23.65 ms/step) -> kernel 61.8 tok/s (16.19 ms/step) = **1.461x**,
    vs the SPEC §12 >=15% bar. Greedy tokens identical between paths at 8k
    (kernel engagement proven end-to-end). Batched 8k deferred: 16GB dev
    machine; the bar is single-stream-measurable and the 14B-class bench
    machine records its own numbers per ADR 0003's deferral pattern.
- mlx_compile rider (PROGRESS 2026-07-13 scoping): CLOSED under exit
  condition (a) — no fusion/dispatch-bound hotspot; experiments not
  warranted at this pin. Evidence (decode step @8k, kernel path,
  16.19 ms): isolated per-layer attention 0.667 ms x 16 layers = 66% of
  the step, a single fused kernel either way (gather variant costs
  1.431 ms/layer; engine-level delta reconciles at 61% of 16x the
  isolated delta — isolated timing pays per-rep eval, engine overlaps,
  so the isolated delta is an upper bound). The 5.52 ms non-attention
  residual is trunk qmv + sampler + dispatch, which the 2026-07-05
  op-level-split investigation (ADR 0003) already measured as ~93%
  kernel(weight-streaming)-time at 1B — dispatch is not a hotspot, and
  every hot op is already a single fused kernel (qmv, SDPA/paged-SDPA,
  fast::rms_norm, fast::rope). mlx_compile fuses elementwise chains only;
  its recoverable surface here is the small elementwise slice of the
  residual — far below a phase-gate-visible win. Greedy bit invariants
  untouched (nothing adopted).
- Decisions:
  - Kernel scope is decode-shaped segments only (qL=1). Multi-row
    attention (prefill/padded pieces) keeps gather+SDPA: that is where
    the reference runs tiled/padded classes, and the gather cost there is
    amortized over the piece's rows. This is the vLLM-shaped split and
    keeps the ADR 0002 pad machinery untouched.
  - `blocks` table + 2-pass predicate transcribed as pure functions with
    pin-referenced unit tests (paged_attn.rs) so a future mlx-c bump that
    changes dispatch fails loudly in review, not silently in parity.
  - Block tables are zero-padded to >=8 entries: mlx-c's custom-kernel
    signature generator switches an input between `constant`/`device`
    address spaces at size 8, which would otherwise flip the generated
    source (and force a JIT rebuild) between short and long contexts.
  - Template args (D/V/GQA/HK/BS/BLOCKS + dtype) specialize per model at
    JIT; per-step runtime inputs are only the block table + context
    length (built once per seq per step in build_seq_step, shared across
    layers) and a cached scale array on Attention — no per-layer host
    allocation on the hot path.
  - The bit probe is BLOCKING in `cargo test --workspace` (it asserts a
    same-device two-implementation invariant, not a committed-fixture
    comparison — ADR 0004's advisory carve-out does not apply). Known
    residual: a CI runner whose Xcode/driver compiler pairing bit-diverges
    the JIT from the runner-built metallib would fail this lane; that is
    a REAL kernel-class finding on that device class, wanted loudly.
    Advisory-izing it preemptively would be a unilateral scope call —
    if it fires on CI, options (characterize + explicit advisory ruling
    vs fix) come back here for a ruling. Golden flag-ON rounds ride the
    existing PERMANENTLY-advisory golden lane unchanged (ADR 0004); the
    8k gate is `#[ignore]`d like the ADR 0003 throughput gate (CI never
    runs either).
  - Worker `CacheOptions` renamed `EngineOptions` (it now carries a
    non-cache engine switch); argv contract extended additively.
- Deviations: none. (SPEC §12's "else flag stays off" branch not taken —
  both bars passed; the default nevertheless stays OFF per the session
  instruction. Flipping it is a PM decision.)
- Acceptance:
  ```
  $ cargo test -p kiln-engine --test paged_attn      # kernel-vs-gather BIT probe
  test kernel_output_is_bit_identical_to_gather_sdpa ... ok   (5.16s)
  $ cargo test -p kiln-engine --lib paged_attn       # dispatch-table transcription tests
  test paged_attn::tests::two_pass_blocks_matches_the_pin ... ok
  test paged_attn::tests::two_pass_predicate_matches_the_pin ... ok
  $ KILN_TEST_MODELS=... cargo test -p kiln-models --test golden
  test greedy_parity_is_exact_for_every_fixture_model ... ok  (579.90s;
    every fixture model x {single-stream, width-16} x {gather, kernel})
  $ KILN_TEST_MODELS=... cargo test -p kiln-models --release --test paged_attn_gate -- --ignored --nocapture
  decode @8k: gather 42.3 tok/s (23.65 ms/step), kernel 61.8 tok/s (16.19 ms/step) -> 1.461x (bar 1.15x)
  profile @8k: isolated per-layer attention gather 1.431 ms vs kernel 0.667 ms;
    step delta 7.46 ms vs 16 x attention delta 12.23 ms (composition 61%);
    non-attention residual 5.52 ms/step
  test paged_attention_kernel_meets_the_8k_bar ... ok
  $ KILN_TEST_MODELS=... cargo test --workspace      -> 46 suites, all ok
    (incl. golden both paths 585s; 3 ignored = the perf gates)
  $ cargo build --workspace --no-default-features    -> clean (linux shape)
  $ cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings  -> clean
  $ cargo clippy --workspace --all-targets --no-default-features -- -D warnings       -> clean
  $ ruff check / format --check python/ tests/e2e    -> clean (23 files)
  $ pytest python/kiln_worker_py/tests               -> 35 passed
  $ uv run --project tests/e2e pytest tests/e2e      -> 77 passed, 2 skipped
    (identical to the pre-change baseline; default config = flag off)
  ```
- Next: Phase 7 is functionally complete (7.1–7.4 + compile rider). Phase
  gate review per SPEC §13.4 (PM runs bench.sh + e2e on their hardware),
  then Phase 8 — speculative decoding. Open PM decisions parked here:
  (1) flip `paged_attention_kernel` default ON (both §12 bars passed on
  the dev machine; suggest observing the CI bit-probe lane on a few runs
  first); (2) nothing else — no DECISION NEEDED items.

## [2026-07-13] Phase 7 / Task 4 — PR #18 merged: paged-attention kernel CI-verified — DONE
- PR #18 (block-table-aware paged-attention Metal kernel, flag default
  OFF) merged to main (314d490) with all four checks green on run
  29289929704 — lint, compile-linux, test-macos, test-macos-release.
- Verified in the run logs (not assumed):
  - `kernel_output_is_bit_identical_to_gather_sdpa` EXECUTED (no
    Metal-skip message in either lane) and PASSED in BOTH macOS lanes
    (debug 22:30:45Z, release 22:30:11Z). This is the first
    foreign-device datapoint for the offline-metallib-vs-runtime-JIT
    codegen question: the runner's Xcode/driver pairing also produces
    bit-identical kernel-vs-gather output. CI bit-probe green run #1.
  - Advisory golden lane (permanently advisory per ADR 0004): gemma-2-2b
    all 24 rounds exact (6 fixtures x {single-stream, width-16} x
    {gather, kernel}); gemma-3-1b/chat-basic diverged on the GATHER path,
    single-stream — the exact recorded ADR 0004 baseline (4-ULP fp16
    kernel-class coin toss, cf. run 28753659372), on the path this PR
    does not modify, aborting the remaining models' rounds as before.
    NO pattern change accompanying this code change; nothing to
    investigate. The dispatch-table unit tests also ran green both lanes.
- PM ruling (2026-07-13, recorded verbatim in intent): the default stays
  OFF at merge. The blocking bit-probe lane is the deliberate safety net:
  each CI run on GitHub's runners is free continuous validation on a
  device/compiler pairing different from the dev machine. Flip the
  default only in a small DEDICATED follow-up commit after the probe has
  stayed green across a handful of real CI runs. If the probe ever fires
  on CI, that is the DECISION NEEDED it was built to surface — bring it
  back for a ruling; never pre-emptively soften the gate.
- Next: Phase 7 complete (7.1-7.4 + mlx_compile rider closed). SPEC §13.4
  phase-gate review (PM: bench.sh + e2e on their hardware), then Phase 8 —
  speculative decoding. Parked: the default-flip follow-up above, gated
  on accumulated green CI bit-probe runs.

## [2026-07-13] Phase 8 / Part 1 — Drafter abstraction + draft-model loading — DONE
- What:
  - `kiln-engine/src/drafter.rs` (new): the SPEC §6.5 `Drafter` trait —
    `memory` / `begin` / `propose` / `release`, sequences keyed by the
    engine's arrival numbers — plus `DraftError`, `DrafterMemory`, and
    the §6.5 defaults (`DEFAULT_GAMMA = 4`, `DEFAULT_SPEC_MAX_BATCH = 4`).
    `Engine` owns an `Option<Box<dyn Drafter>>` (`set_drafter`,
    `drafter_memory`) because spec decode is scheduler-native; the step
    loop does NOT consult it yet — this session is shape + loading only.
  - `kiln-models/src/draft.rs` (new): `DraftModel` — a second `AnyModel`
    with its OWN `BlockManager` + `PagedKv` (pools lazy, 0 bytes until
    first write, exactly like the target's) and weights accounted by the
    `StaticInfo.weights_bytes` `.safetensors` convention. Implements
    `Drafter`: real memory numbers, real seq-lifecycle tracking, and a
    placeholder `propose` returning the EMPTY proposal ("no speculation
    this round" — a legal answer under the trait contract) until the
    part-2 decode loop lands.
  - `kiln-worker`: additive argv `--draft-model <dir>`; the draft loads
    on the engine thread after target calibration, sharing the device/
    stream; a configured draft that fails to load marks the worker
    UNHEALTHY (silently serving without requested speculation would hide
    a misconfiguration). `MemoryReport.weights_bytes`/`kv_pool_*` are
    now explicitly worker TOTALS (target + draft summed in
    `memory_report`; per-model gauges kept separate in `Shared`).
    `CAPABILITY_SPECULATIVE` is deliberately NOT advertised yet.
  - Tests: `kiln-engine/tests/drafter.rs` (scripted stub exercises the
    trait-object contract incl. committed-feed-through and re-begin
    reset); `kiln-models/tests/draft.rs` (coexistence, see Acceptance);
    `kiln-worker/tests/rpc.rs::draft_model_loads_alongside_target`
    (proto-level totals + identical greedy stream vs a draft-less
    worker + SPECULATIVE not advertised). CI blocking model-gated step
    gains `--test draft` (same-device invariants → device-independent
    tier per the Option B split).
- Decisions:
  - Draft pool geometry defaults mirror the target pool (same
    block_size × num_blocks ⇒ same token capacity): a target-admitted
    sequence never needs a draft-side capacity decision; auto-disable
    heuristics remain a later Phase 8 part.
  - `propose(committed)` feed-through makes rollback implicit: the
    drafter reconciles (truncates its speculated KV) from the committed
    tokens before proposing again — O(1) via block release, same as the
    target side. `begin` on a known seq doubles as the
    preemption-resume reset.
  - Proto untouched (frozen): draft bytes fold into the existing
    `MemoryReport` totals — the fields keep their "whole worker" §2.3
    semantics, which is what gateway budget math wants. `kv_blocks_*`
    gauges stay target-pool-only (mixing pools with different
    bytes-per-block would corrupt the gauge's meaning).
  - NO tokenizer/vocab compatibility check at load, deliberately: this
    session's mandated pair (qwen3-0.6b draft under the larger pinned
    llama-3.2-1b target) is cross-family by construction of the pinned
    model set — part 1 proves loading isolation. The compat check
    belongs to the verify-loop session, where drafting actually starts.
  - No deterministic-width calibration for the draft: proposals are
    target-verified, so draft numerics never bind greedy correctness.
- Deviations: none.
- Acceptance:
  ```
  $ cargo test -p kiln-engine --test drafter        # trait stub, boxed
  test drafter_contract_via_trait_object ... ok
  $ KILN_TEST_MODELS=... cargo test -p kiln-models --test draft -- --nocapture
  coexistence: target weights 695283921B + draft weights 335450584B
    + target kv 536870912B + draft kv 234881024B = 1802486441B;
    budget 13743895347B; mlx active 1265752240B
  test draft_model_coexists_with_target ... ok
    (target greedy BIT-IDENTICAL alone / beside resident draft pool /
     with drafter attached; draft-pool sentinel bytes survive target
     generation; pool accounting geometry-exact; live-object leak gate
     back to baseline)
  $ KILN_TEST_MODELS=... cargo test -p kiln-worker --test rpc draft_model_loads_alongside_target
  draft coexistence over RPC ok: weights 695283921 -> 1030734505 bytes,
    24 identical tokens                  (= target + draft file bytes, exact)
  test draft_model_loads_alongside_target ... ok
  $ KILN_TEST_MODELS=... cargo test --workspace     -> exit 0, all suites ok
    (45 test-result lines incl. golden both paths 601.75s, rpc 3/3;
     3 ignored = the perf gates, as before)
  $ cargo build --workspace --no-default-features   -> clean (linux shape)
  $ cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings -> clean
  $ cargo clippy --workspace --all-targets --no-default-features -- -D warnings      -> clean
  $ ruff check / format --check python/ tests/e2e   -> clean (23 files)
  $ pytest python/kiln_worker_py/tests              -> 35 passed
  $ uv run --project tests/e2e pytest tests/e2e     -> 77 passed, 2 skipped
    (identical to the pre-change baseline)
  ```
- Next: PR opened from `claude/p8-drafter-loading`; CI verification (all
  four checks on the real runners) recorded per the established protocol
  once the run completes. Then Phase 8 part 2 — the batched draft/verify
  decode loop per §6.5: draft decode inside `DraftModel::propose`,
  verify forward in the batch step, O(1) rollback via block release, and
  the greedy-invariance test (speculation on vs off) — followed by
  acceptance-rate metrics, auto-disable, and gateway config wiring
  (`[model.speculative]` → `--draft-model` argv; `SpeculativeConfig`
  parsing already exists). Only then advertise CAPABILITY_SPECULATIVE.

## [2026-07-13] Phase 8 / Part 1 — PR #19 CI fix: draft RPC test placement — DONE
- What: PR #19 run 29294575202 failed ONE lane (test-macos, 16m39s);
  lint, compile-linux, and test-macos-release passed, and every NEW
  suite passed on the runner — including
  `draft_model_loads_alongside_target` and the kiln-models coexistence
  suite. The failure was the PRE-EXISTING
  `rpc.rs::cancel_and_drain_rpc_semantics`: my draft test initially
  lived in rpc.rs, and cases inside one test binary run CONCURRENTLY.
  The drain test measured 2055.3 ms/token while this test's two workers
  held the GPU, sized its escalation deadline (657s) from that, then
  contention vanished and the long request decoded at 29 tok/s — a ~60x
  rate swing, past the 37x design margin that test documents (its
  hardening was calibrated for the two historical siblings). The long
  request finished (Length, 12000 tokens, 413s) before escalation.
- Fix: restored `tests/rpc.rs` byte-for-byte to its pre-change state
  and moved the draft coexistence test to its own binary,
  `kiln-worker/tests/draft.rs` (self-contained harness — the
  established grammar.rs pattern; test BINARIES run sequentially under
  cargo test, so the drain test's contention profile is exactly its
  calibrated one again). CI worker line gains `--test draft`. No
  assertion anywhere was weakened; the drain test is untouched.
- Acceptance (dev machine):
  ```
  $ KILN_TEST_MODELS=... cargo test -p kiln-worker --test draft -- --nocapture
  draft coexistence over RPC ok: weights 695283921 -> 1030734505 bytes,
    24 identical tokens
  test draft_model_loads_alongside_target ... ok
  $ KILN_TEST_MODELS=... cargo test -p kiln-worker --test rpc
  test result: ok. 2 passed  (restored suite, original calibration)
  $ cargo fmt --all --check; clippy --all-targets (both shapes) -D warnings -> clean
  ```
- Next: PR #19 CI re-run on the new head; record verification once the
  four checks complete, then Phase 8 part 2 (draft/verify decode loop).

## [2026-07-13] Phase 8 / Part 2 — PARTIAL: verify loop built and measured; greedy-invariance gate red on qwen2.5 — DECISION NEEDED
- What: the batched draft/verify decode loop per SPEC §6.5, end to end:
  - `DraftModel::propose` really drafts now (kiln-models/src/draft.rs):
    reconcile committed context (speculated-tail truncation = O(1) block
    release), catch-up prefill in 2048 chunks, gamma greedy tokens with
    lazily chained argmax feeds, one eval per round. Draft numerics stay
    deliberately uncalibrated — the target verifies everything.
  - Engine verify rounds (kiln-engine/src/engine.rs): an eligible request
    (greedy, penalties off, no grammar, batch ≤ `spec_max_batch`, gamma
    clamped under the ADR 0002 deterministic width and the remaining
    token budget) swaps its 1-token decode segment for a gamma+1-slot
    verify segment run as its OWN forward group; every position sampled
    (`SeqStep.sample` → `sample_rows`); longest agreeing prefix + the
    target's own next token commit through the normal settle path
    (stops/cancel/max_tokens honored per token); rejected slots roll back
    via new `BlockTable::truncate` — block release only, no data motion.
    Drafter faults disable speculation for that request only, in-band.
    The async_eval pipeline yields to speculation (`spec_would_engage`).
  - Vocab/tokenizer compat gate deferred from part 1
    (`kiln_models::check_draft_compat`): tokenizer.json `model.vocab` +
    `added_tokens` must agree and draft logits width must fit the
    target's; the worker runs it before attach and an incompatible pair
    is UNHEALTHY at load ("draft model rejected: ... incompatible ...").
    The part-1 pair (qwen3-0.6b draft under llama-1b target) is REJECTED
    by construction — see Deviations for the acceptance-metrics fallout.
  - Metrics: engine `SpecStats` (rounds/proposed/accepted/rollback
    tokens+nanos/draft errors), per-request `FinishSummary.spec_tokens_*`
    → proto `Timings` fields 6/7, worker totals → `WorkerStats` 14/15
    (fields existed; no proto change). CAPABILITY_SPECULATIVE stays
    un-advertised per the part-1 plan (config wiring + auto-disable are
    later parts).
  - Tests: kiln-engine/tests/rollback_cost.rs (O(1) measured, no Metal);
    kiln-models/tests/spec_decode.rs (the invariance gate: every golden
    fixture × {self-draft, adversarial scripted drafter}, plus compat
    matrix, spec_max_batch gating, prefix-cache composition, acceptance
    metrics, in-situ rollback scaling; `KILN_FIXTURE_PARITY=skip`
    switches baselines from committed fixtures to a live speculation-off
    run for foreign devices per ADR 0004); kiln-models/tests/spec_probe.rs
    (characterization instrument for the finding below);
    kiln-worker/tests/draft.rs rewritten (incompatible pair → UNHEALTHY
    over RPC; compatible qwen3-0.6b-8bit + qwen3-0.6b-4bit pair → READY,
    verify loop live, stream identical to draft-less worker, Timings and
    Stats metrics populated); kiln-models/tests/draft.rs lifecycle updated
    for real proposals (its final case is now a live adversarial-drafter
    invariance check on the cross-tokenizer pair, engine-level).
- FINDING (the reason this is PARTIAL): greedy output under speculation
  diverges from speculation-off for qwen2.5-0.5b-4bit at gamma=4, and the
  full standard was applied — characterized, not weakened, not patched:
  - Which fixture/token: qwen2.5-0.5b-4bit/chat-basic, generated index 33
    (engine 2585 vs fixture 9645), right after a hallucinated
    `<|im_end|>\n<|endoftext|>Human:` turn — and
    qwen2.5-0.5b-4bit/raw-long-prefill index 10 under the adversarial
    drafter. Every other fixture model × {self-draft, adversarial} is
    token-exact (gemma2's manual-softcap path included).
  - Which op, measured: the divergence position is a 1-fp16-ULP argmax
    race on the plain path (raw logits 16.765625 vs 16.75; ULP at
    [16,32) = 2^-6). A 5-row verify-shaped forward from the IDENTICAL
    plain-built KV state shifts both lanes ~2-3 ULPs and flips the
    argmax; 2-, 3-, and 4-row shapes are BIT-IDENTICAL to plain on those
    lanes. Root cause in the pinned MLX (v0.31.1) fused-SDPA dispatch
    (mlx/backend/metal/scaled_dot_product_attention.cpp):
    `supports_sdpa_vector` requires `qL <= 8 && qL * gqa_factor <= 32`,
    and `supports_sdpa_full` requires `qL > 8` — qwen2.5-0.5b has
    gqa_factor 7 (14 Q heads / 2 KV heads), so a gamma=4 verify (qL 5,
    5×7=35) satisfies NEITHER and silently takes the UNFUSED composed-op
    attention: a different kernel class than the qL=1 vector kernel of
    plain decode (ADR 0002's phenomenon, on a new dispatch axis). All
    passing models have gqa_factor ≤ 4 (5 rows ≤ 32 lanes). The trunk is
    exonerated: matmuls at M=5 ≤ W=9 are the same shapes the B' width-16
    goldens already prove bit-stable for qwen2.5.
  - Stability: deterministic per process layout (4/4 identical fresh-
    process probe runs) but NOT across allocation histories — the same
    binary produced the flip in one suite layout and none in another
    (suite runs 1 vs 2). There is no stable fixture bar for these shapes;
    "rerun until green" must not be applied to this gate.
  - Second boundary of the same family (not fixture-exercised): the
    vector 1-pass/2-pass split keys on `k.shape(2) >= 1024` (device class
    'd'/'s') or `>= 4096` (GQA); a verify segment reaches those kv
    lengths up to gamma tokens earlier than plain decode does.
  - The engine's deterministic-width clamp does not cover this dispatch
    axis (it calibrates linear projections only); the limit is documented
    in the engine module docs pending resolution.
- Decisions:
  - Speculation eligibility is greedy-only (plus penalties-off,
    grammar-free): argmax consumes no PRNG draws, so variable commits per
    round cannot desync seeded key chains; penalties/grammar need
    host-visible per-position state a batched verify does not have.
    Seeded-sampling speculation (true rejection sampling) is future work.
  - Verify segments never evict or preempt: if the pool cannot cover
    gamma+1 slots the request falls back to a plain 1-token step.
  - Draft pool geometry mirrors the target pool (part-1 decision stands).
  - Rejected-row KV hygiene rides the existing discarded-row invariant
    (gathers trim to table length; donation stops at settled rows; the
    sequence's own next append rewrites stale rows) — no new mechanism.
  - Compat check compares tokenizer id→token MEANING (model.vocab +
    added_tokens) and logits-width embeddability, not byte equality —
    merges/normalizers may differ without changing what an id means.
  - spec_decode is NOT wired into CI yet: the gate is red/layout-unstable
    for qwen2.5 pending the decision below, and a flaky blocking lane is
    worse than none. rollback_cost runs everywhere in the plain workspace
    suites automatically.
- Deviations:
  - The session prompt asked for "acceptance-rate metrics recorded for
    the qwen3-0.6b/llama-1b pair" AND for the compat check that rejects
    exactly that cross-tokenizer pair. The check wins (part 1 recorded
    that the pair is cross-family by construction and deferred the check
    to this session). Recorded outcomes: the qwen3/llama pair → loud
    rejection (worker UNHEALTHY; engine-level it is the adversarial
    invariance case); acceptance-rate metrics → the compatible same-
    family pair qwen3-0.6b-8bit (target) / qwen3-0.6b-4bit (draft),
    per SPEC §11.3's "same-family draft" bar.
  - No other deviations from SPEC §6.5; the invariance bar itself is
    exactly SPEC's and stays unweakened (the gate is red on one model).
- Acceptance:
  ```
  $ cargo test -p kiln-engine --test rollback_cost --release -- --nocapture
  rollback of 5 slots against a 128-token table:   22 ns/cycle (mean of 20000)
  rollback of 5 slots against a 1024-token table:  31 ns/cycle
  rollback of 5 slots against a 8192-token table:  24 ns/cycle
  rollback of 5 slots against a 65536-token table: 23 ns/cycle
  rollback of 1 block: 23 ns/cycle; of 16 blocks: 40 ns/cycle (64k table)
  test result: ok. 2 passed     (O(1) MEASURED: flat across a 512x span)

  $ KILN_TEST_MODELS=... cargo test -p kiln-models --test spec_decode -- --nocapture
  compat gate rejects qwen3-draft/llama-target: "draft logits width 151936
    exceeds the target's 128256" (loud, structured)
  gemma-2-2b-it-4bit:  self-draft 356/356 accepted (100.0%);
    adversarial 0/1728, 441 rollback rounds        - all fixtures exact
  gemma-3-1b-it-4bit:  self-draft 356/356 (100.0%); adversarial 0/1728 - exact
  llama-3.2-1b-4bit:   self-draft 356/357 (99.7%);  adversarial 7/1706 - exact
  qwen3-0.6b-4bit:     self-draft 303/313 (96.8%);  adversarial 8/1450 - exact
  qwen3-0.6b-8bit:     self-draft 356/358 (99.4%);  adversarial 4/1712 - exact
  smollm2-135m-bf16:   speculation disabled by the width-1 clamp (dense
    trunk protection) - plain-path outputs verified exact
  qwen3-0.6b-8bit target / qwen3-0.6b-4bit draft: 46/66 accepted (69.7%)
    over 17 rounds on English prose (SPEC 11.3 >50% bar MET); warm-prefix
    rerun reused 24 prompt tokens WITH speculation active - both exact;
    per-request Timings mirror engine totals
  spec_max_batch gate: width 6 never consulted the drafter; narrowed
    batch did (90 consultations); outputs bit-identical
  in-situ rollback: 1174 ns/round at 8-token context vs 1359 ns/round at
    6000-token context (flat - O(1) in the live engine)
  FAILED - greedy output moved under speculation on 2 case(s):
    qwen2.5-0.5b-4bit/chat-basic [self-draft]:   index 33: 2585 vs 9645
    qwen2.5-0.5b-4bit/chat-basic [adversarial]:  index 33: 2585 vs 9645
  (THE finding - bar deliberately unweakened; see DECISION NEEDED)

  $ KILN_TEST_MODELS=... cargo test -p kiln-models --test spec_probe -- --nocapture
  qwen2.5-0.5b-4bit/chat-basic:       adversarial g4 div=Some(33);
    oracle g4 div=Some(33); oracle g2 div=None; oracle g1 div=None
  qwen2.5-0.5b-4bit/raw-long-prefill: adversarial g4 div=Some(10) (this
    layout; div=None in the suite layout - allocation-history dependent)
  qwen3-0.6b-4bit / qwen3-0.6b-8bit / smollm2: div=None everywhere
  (identical across 4 fresh-process runs)
  plain path at position 33: fixture 9645 logit 16.765625, spec 2585
    logit 16.75 - raw delta 0.015625 = EXACTLY 1 fp16 ULP at [16,32);
  5-row verify-shaped forward from the IDENTICAL KV state: row-0 argmax
    flips to 2585 (lanes shift ~2-3 ULPs);
  2-, 3-, 4-row shapes: row-0 bits MATCH plain exactly.

  $ KILN_TEST_MODELS=... cargo test -p kiln-worker --test draft -- --nocapture
  incompatible pair rejected as UNHEALTHY: "draft model rejected:
    draft/target tokenizers are incompatible: ..."
  draft/verify over RPC ok: weights 633442994 -> 968893578 bytes (exact
    worker total), 24 identical tokens vs the draft-less worker,
    16/27 draft tokens accepted; Stats totals mirror Timings
  test result: ok. 1 passed

  $ KILN_TEST_MODELS=... cargo test --workspace
  every suite green - golden 585s (all fixture models, gather + kernel
  paths, single-stream AND width-16, leak gate to baseline), batching,
  calibration, draft coexistence (now with a LIVE adversarial cross-
  tokenizer drafter attached: bit-identical), preemption, prefill_pad,
  prefix_cache, prefix_multiturn, leak, leak_batched, engine suites incl.
  deterministic_partition/pipeline_discard on the renamed sample_rows
  contract, gateway 42 unit tests - EXCEPT kiln-models/spec_decode:
  FAILED (the qwen2.5 finding above). Binaries alphabetically after the
  failing one (cargo fail-fast) re-run explicitly:
  kiln-models spec_probe (2 ok) + throughput (perf, ignored),
  kiln-proto (43 ok), kiln-tokenize (14 ok across binaries),
  kiln-worker rpc + grammar (3 ok, original calibration untouched).
  $ cargo build --workspace --no-default-features           -> clean (linux shape)
  $ cargo clippy --workspace --all-targets -- -D warnings   -> clean
  $ cargo clippy --workspace --all-targets --no-default-features -- -D warnings -> clean
  $ cargo fmt --all --check                                 -> clean
  $ ruff check / ruff format --check python/ tests/e2e      -> clean (23 files)
  $ pytest python/kiln_worker_py/tests                      -> 35 passed
  $ uv run --project tests/e2e pytest tests/e2e             -> 77 passed, 2 skipped
    (identical to the pre-change baseline)
  ```
- Next: PM ruling on the DECISION below; then (per ruling) the speculation
  envelope ADR + gate re-run to green, then the remaining Phase 8 parts
  (acceptance-rate auto-disable heuristics, gateway `[model.speculative]`
  config wiring, CAPABILITY_SPECULATIVE advertisement). Branch
  `claude/p8-verify-loop` holds this work; NO PR opened — the invariance
  gate must be green (or its bar PM-redefined) before merge per the
  standing protocol.
- DECISION NEEDED: where may speculation run at this MLX pin, given the
  fused-SDPA dispatch envelope? Options:
  - A) Dispatch-envelope gamma clamp (recommended): clamp verify rows to
    `gamma+1 <= min(deterministic_width, 8, floor(32 / gqa_factor))`,
    derived from the pinned dispatch predicate — qwen2.5-0.5b lands at
    gamma=3 (its fixtures pass there, deterministically across
    processes; the 4-row shape measured bit-identical to plain), every
    other pinned model keeps gamma=4. Plus a surgical guard for the
    1-pass/2-pass kv-length boundaries (disable speculation for a
    request while its kv length is within gamma of 1024/4096), OR accept
    those as a documented residual à la ADR 0002. Needs an ADR naming
    the envelope + the revisit-at-pin-bump trigger; the spec_decode gate
    (then green) becomes its enforcement.
  - B) Fixture-evidence allowlist: speculation ON per architecture only
    where the full gate is green at gamma=4 (qwen2 family OFF entirely).
    Blunter than A, ignores the found predicate, still needs the
    kv-boundary story.
  - C) Hold speculation at this pin (feature stays dark until an MLX bump
    makes SDPA dispatch query-length-invariant; quarterly ADR 0001
    process). Forfeits Phase 8 at this pin.
  A is recommended: it is measurement-grounded, keeps the invariant
  absolute, and costs only gamma 4→3 on high-GQA models. Picking nothing,
  stopping here per protocol.

## [2026-07-13] Phase 8 / Part 2 (resolution) — DONE: ADR 0005 dispatch-envelope clamp; gate green everywhere
- What: the PM-directed resolution of this morning's DECISION NEEDED
  (option A), with the clamp formula verified against the pinned MLX
  SOURCE before implementation, per the ADR 0002/0003 standard:
  - Predicate verified from `mlx/backend/metal/scaled_dot_product_
    attention.cpp` (`use_fallback`): the fused vector kernel requires
    `qL <= 8 && qL <= kL && head_dim in {64,96,128,256} (Q==V) &&
    qL * gqa_factor <= 32`. UNIFORM at the pin — no device or dtype
    branching (dtype only selects the kernel template), so dtype does
    NOT enter the clamp; head_dim DOES, as a set-membership
    precondition. Two further source facts shaped the envelope:
    (a) inside the 1-pass vector kernel, per-row bits are invariant to
    query count AND key length by kernel construction (fixed stride-32
    key->simdgroup assignment, index-ordered online softmax, fixed
    32x32 reduction tree, bottom-right causal key sets identical to
    plain decode's) — the source-level proof behind the measured
    2/3/4-row bit-identity; (b) the 2-pass variant is OUT of any
    certificate: its partition count depends on `n_simds = gqa * qL`
    and on N, with device-class thresholds (kL >= 1024 on 'd'/'s'
    GPUs, >= 4096 with GQA elsewhere) — so the envelope also needs a
    key-length bound, not just the gamma formula.
  - Implementation: `AnyModel::speculative_gamma_bound()` (config-
    derived per checkpoint, never hardcoded per model: fused-SDPA path
    required — gemma2's manual softcapped attention excluded; quantized
    trunk required — dense excluded per the ADR 0002 addendum
    precedent; head_dim set; `gamma+1 <= min(8, 32/gqa)`); the worker
    clamps `EngineConfig::gamma` at drafter attachment and REJECTS
    (UNHEALTHY, "outside the ADR 0005 speculation envelope") a target
    with no envelope — as loud as an incompatible tokenizer; the
    engine adds the per-round key-length bound
    (`VERIFY_MAX_KEY_LEN = 1023`: verify kL = offset+gamma+1 stays in
    the 1-pass region on EVERY device class; gamma shrinks at the
    boundary, then speculation stops and plain decode continues —
    the pipeline re-engages past it via `spec_would_engage`).
  - ADR 0005 written (docs/decisions/): the fourth instance of the
    0002/0003/0004 kernel-dispatch pattern; records the formula, its
    source derivation, the key-length bound (device-aware refinement to
    4096 via `mlx_device_info` noted, not implemented), and — as an
    explicit documented precondition, not an implicit consequence —
    that any NEW architecture/geometry needs an envelope review plus a
    green full spec_decode gate on the generating device before
    speculation is enabled for it.
  - Tests: spec_decode applies the same envelope the worker applies
    (gemma2 joins smollm2 in the structurally-disabled branch: stats
    must be zero, outputs exact); new kv-envelope crossing case (1000-
    token prompt, 64 generated: speculation runs below the cap, stops
    at it, output exact vs a live plain baseline); in-situ rollback
    scaling re-scoped to the envelope's real operating range (8 vs 900
    tokens); worker test gains the envelope loud-rejection case
    (gemma-2-2b with itself as draft: tokenizer-compatible, envelope
    None -> UNHEALTHY naming ADR 0005). spec_probe stays in-tree as the
    pinned evidence: it deliberately builds the UNCLAMPED gamma=4 shape
    on qwen2.5 and keeps printing the divergence for as long as the pin
    dispatches this way (re-run at pin bumps per ADR 0001).
  - CI: spec_decode joins the BLOCKING model-gated step (previously
    deferred). Under the step's `KILN_FIXTURE_PARITY=skip` it compares
    speculation-ON against a live speculation-OFF run — the
    device-independent SPEC §6.5 invariant, now certifiable on any
    device because the envelope keeps every verify inside the
    source-proven 1-pass class.
- Answering the session's explicit questions:
  - The predicate is uniform across the pinned MLX for all attention
    head configurations; the only additional dispatch factors are
    head_dim (envelope precondition) and the 1-pass/2-pass key-length
    thresholds (envelope kv bound). dtype does not affect dispatch.
  - The qwen3-0.6b pair (gqa_factor 16/8 = 2, head_dim 128): a gamma=4
    verify is 5x2 = 10 <= 32 — ALREADY inside the envelope, not
    coincidentally exempt; its acceptance bar re-verified unchanged.
- Decisions:
  - Universal `VERIFY_MAX_KEY_LEN = 1023` rather than device-aware
    thresholds: mlx-c does expose the architecture string
    (`mlx_device_info`), but the deployment bench class (M4 Pro/Max,
    likely 'd') has the 1024 boundary anyway, and avoiding new FFI in a
    correctness change is worth the conservatism. Recorded in ADR 0005
    as the refinement path.
  - A configured draft on an out-of-envelope target is UNHEALTHY, not
    silently-inert speculation (same policy as the tokenizer gate).
- Deviations: none. SPEC §6.5's default gamma=4 is qualified per model
  by the ADR 0005 envelope (qwen2.5-0.5b runs gamma 3; gemma2/dense do
  not speculate) — recorded in the ADR, SPEC text untouched.
- Acceptance:
  ```
  $ KILN_TEST_MODELS=... cargo test -p kiln-models --test spec_decode -- --nocapture
  == gemma-2-2b-it-4bit:  envelope None    -> speculation disabled; plain-path exact
  == gemma-3-1b-it-4bit:  envelope Some(7) -> gamma 4; self 356/356 (100.0%);
       adversarial 0/1728, 441 rollbacks               - all fixtures exact
  == llama-3.2-1b-4bit:   envelope Some(7) -> gamma 4; self 356/357 (99.7%);
       adversarial 7/1706                              - exact
  == qwen2.5-0.5b-4bit:   envelope Some(3) -> gamma 3 (THE clamp); self
       286/289 (99.0%); adversarial 3/1113, 375 rollbacks - ALL FIXTURES
       EXACT, including chat-basic which diverged at gamma 4
  == qwen3-0.6b-4bit:     envelope Some(7) -> gamma 4; self 303/313 (96.8%) - exact
  == qwen3-0.6b-8bit:     envelope Some(7) -> gamma 4; self 356/358 (99.4%) - exact
  == smollm2-135m-bf16:   envelope None (dense) -> disabled; plain-path exact
  qwen3-0.6b-8bit target / qwen3-0.6b-4bit draft: 46/66 accepted (69.7%)
    over 17 rounds on prose (SPEC 11.3 >50% bar, unchanged by the clamp —
    the pair was already inside the envelope at gqa_factor 2);
    warm-prefix rerun reused 24 prompt tokens WITH speculation active
  spec_max_batch gate: width 6 never consulted; narrowed batch did;
    outputs bit-identical
  in-situ rollback: 1621ns/round @ 8-token vs 1013ns/round @ 900-token
    context (flat across the envelope's operating range)
  kv-envelope crossing (1000-token prompt): 22 rounds below the cap,
    clean stop at VERIFY_MAX_KEY_LEN, output exact vs live plain baseline
  test result: ok. 1 passed  (116.93s — the gate is GREEN on every model)

  $ KILN_TEST_MODELS=... cargo test -p kiln-worker --test draft -- --nocapture
  incompatible pair rejected as UNHEALTHY: ... tokenizers are incompatible ...
  out-of-envelope target rejected as UNHEALTHY: draft model rejected:
    target architecture is outside the ADR 0005 speculation envelope
    (no certified verify kernel class)                 [gemma-2 self-pair]
  draft/verify over RPC ok: weights 633442994 -> 968893578 bytes,
    24 identical tokens, 16/27 draft tokens accepted
  test result: ok. 1 passed

  $ KILN_TEST_MODELS=... cargo test --workspace
  every suite green: golden 587s (all fixture models, both attention
  paths, single-stream AND width-16 — speculation never engages above
  spec_max_batch, so width-16 shapes are the plain B' path by design),
  batching, preemption, prefix suites, leak gates, spec_decode 119s,
  spec_probe (merged to one #[test] after the first workspace pass
  caught its two GPU cases running CONCURRENTLY and segfaulting — the
  same single-engine-thread discipline every Metal suite follows; the
  probes themselves are unchanged and still print the gamma-4
  divergence), worker rpc/grammar/draft, gateway units.
  $ cargo build --workspace --no-default-features           -> clean (linux shape)
  $ cargo clippy --workspace --all-targets -- -D warnings   -> clean (both shapes)
  $ cargo fmt --all --check                                 -> clean
  $ ruff check / ruff format --check python/ tests/e2e      -> clean
  $ pytest python/kiln_worker_py/tests                      -> 35 passed
  $ uv run --project tests/e2e pytest tests/e2e             -> 77 passed, 2 skipped
  ```
- Next: PR opened from `claude/p8-verify-loop`; record CI verification
  (all four checks on the real runners) per the established protocol
  once the run completes. Then the remaining Phase 8 parts: throughput
  measurement against the SPEC §12 Phase 8 bar (>= 1.6x at acceptance
  > 60% — NOT claimed here), acceptance-rate auto-disable heuristics,
  gateway `[model.speculative]` config wiring, and only then
  CAPABILITY_SPECULATIVE advertisement.

## [2026-07-13] Phase 8 / Part 2 — PR #20 CI verification recorded — DONE
- What: PR #20 (`claude/p8-verify-loop`, both task-8.2 commits) run
  29304244457: ALL FOUR checks pass on the real runners — lint 36s,
  compile-linux 37s, test-macos-release 3m3s, test-macos 23m44s.
- The newly BLOCKING spec_decode lane passed on the foreign GPU under
  `KILN_FIXTURE_PARITY=skip` (live speculation-off baselines): the
  device-independent SPEC §6.5 on-vs-off invariant held on a different
  device class, which is exactly the certification the ADR 0005
  envelope was built to give. rpc/grammar/draft and every other
  blocking suite green; the reworked worker draft suite (tokenizer +
  envelope loud rejections, compatible-pair identity + metrics) passed
  on the runner.
- Advisory golden lane (ADR 0004, permanently non-blocking): the ONLY
  divergence is the known gemma-3-1b-it-4bit/chat-basic flip — the same
  fixture and position class recorded in ADR 0004 (4-ULP fp16 race,
  kernel-class coin toss on the foreign device). No pattern change →
  no action per the ADR; noted here per its protocol.
- Next: remaining Phase 8 parts — throughput measurement against the
  SPEC §12 Phase 8 bar (≥1.6x at acceptance >60%, not yet claimed),
  acceptance-rate auto-disable heuristics, gateway `[model.speculative]`
  → `--draft-model` config wiring, then CAPABILITY_SPECULATIVE
  advertisement.

## [2026-07-14] CI infra — stalled-download hardening for fetch-test-model.sh — DONE
- What:
  - Incident: CI run 29305594680 attempt 1, "Fetch pinned test models"
    ran 04:20:02→06:02:43 (1h42m, manually cancelled) after a cache miss
    hit a stalled HF connection. Root cause: the embedded stdlib
    downloader called `urllib.request.urlopen` with NO timeout, so a
    dead socket blocked `read()` forever.
  - `scripts/fetch-test-model.sh`: every request now carries a 30 s
    socket stall timeout and a 4-attempt retry budget with linear
    backoff (5/10/15 s); non-retryable HTTP codes fail immediately.
    Large files stream to a `.part` file (no longer read whole into
    memory) and resume across attempts/invocations via HTTP Range; the
    pinned lfs sha256 is verified on the `.part` before an atomic
    rename, and a sha mismatch discards the partial rather than
    resuming onto a corrupt prefix. Worst-case dead-connection cost is
    now ~150 s per file, measured, vs unbounded.
  - `.github/workflows/ci.yml`: `timeout-minutes: 30` on the fetch step
    as the second line of defense — it bounds what the socket-level
    detector can't see (a connection trickling a few bytes per timeout
    window). Sizing: the only two observed real cold-cache fetches on
    the runner took 3m56s (run 29116444273) and 12m20s (29113687692);
    30 min is >2x the slowest observed and far below the 6 h job limit.
  - Pins and sha256s untouched (diff touches zero PINS lines);
    interface unchanged (`--only`, `--list`, `KILN_TEST_MODELS` layout,
    `.kiln-revision` marker).
- Decisions:
  - Kept the stdlib-Python downloader instead of switching to curl: the
    semantics map 1:1 (socket timeout ≈ `--speed-time` for a dead
    connection, Range resume ≈ `-C -`, bounded attempts ≈ `--retry`)
    and it avoids depending on curl behavior differences across hosts.
  - Honor `HF_ENDPOINT` (the standard HF hub override env var, default
    unchanged) so the stall tests can aim the real script at a local
    misbehaving server. Additive knob, not an interface change.
  - `resp.read1()` instead of `resp.read(n)` in the stream loop — found
    by the stall simulation, not by inspection: `read(n)` blocks until
    the full chunk buffers, so a mid-chunk stall discarded the partial
    bytes and every retry restarted from byte 0 (the test server logged
    four Range-less requests). With read1 the bytes bank as they arrive
    and the retry resumed from the true offset.
- Deviations: none.
- Acceptance:
  ```
  $ KILN_TEST_MODELS=<tmp> ./scripts/fetch-test-model.sh --only smollm2-135m-bf16
      fetching model.safetensors (269.1 MB) ... done   (1m43s, real HF, sha256 gate passed)
  rerun -> all "ok"; flip 1 byte at offset 1000000 (size unchanged), rerun
      -> "fetching model.safetensors" (sha256 re-check caught it), re-fetch clean
  SIGKILL mid-download at 30.4 MB, rerun against real HF CDN:
      "resuming model.safetensors at 30.4 MB" -> 206 resume, full-file sha256 passed
  stall simulations (local deliberately-misbehaving server via HF_ENDPOINT):
    A: TCP accepted, server never sends a byte  -> 4x "timed out", exit=1 in 150s
    B: headers arrive, body never does          -> 4x "timed out", exit=1 in 150s
    C: 512 KiB then silence; Range honored      -> stall detected in 30s, resumed
       at 524288, exit=0; downloaded sha256 == server payload sha256 (fbbab289...)
    D: C plus a size-0 file in the tree         -> empty file created, no request,
       no .part leftovers, marker written
  (pre-hardening baseline on scenario A/B shapes: hangs indefinitely — the incident)
  $ bash -n scripts/fetch-test-model.sh; ./scripts/fetch-test-model.sh --list
      -> OK, pins print byte-identical
  $ cargo fmt --all --check && cargo clippy --workspace -- -D warnings  -> clean
  $ ruff check / ruff format --check python/ tests/e2e                  -> clean
  ```
- Next: unchanged from the previous entry — remaining Phase 8 parts
  (throughput bar, auto-disable heuristics, gateway speculative config
  wiring, CAPABILITY_SPECULATIVE advertisement).

## [2026-07-14] Phase 8 / Part 3 — auto-disable heuristics + throughput acceptance run — DONE (heuristics); §12 speedup bar MEASURED, NOT MET on pinned pairs
- What:
  - kiln-engine: SPEC §6.5's batch-width auto-disable is now a stand-down
    RAMP, not a cliff (`drafter::spec_gamma_at_width`, pure fn,
    unit-tested): per-round gamma runs full single-stream and shrinks
    linearly as the admitted batch approaches `spec_max_batch`
    (4→3→2→1 at widths 1..4 with the defaults), off strictly ABOVE the
    threshold — SPEC's "auto-disables when batch size > spec_max_batch"
    is preserved exactly at the boundary. `spec_would_engage` and
    `collect_proposal` compute the ramp in lockstep, so pipeline-yield
    and proposal decisions cannot diverge.
  - Acceptance auto-disable (the heuristic this ledger has carried as
    "acceptance-rate auto-disable" since part 1): new
    `EngineConfig::spec_min_acceptance` (default
    `DEFAULT_SPEC_MIN_ACCEPTANCE = 0.125`; `0.0` disables). Once a
    request has had `SPEC_ACCEPTANCE_WARMUP_PROPOSED = 16` proposed
    tokens verified, a verified acceptance rate below the floor stands
    it down for the rest of its life: draft-side state is released
    immediately (blocks back to the draft pool), plain decode continues,
    and — because speculation no longer demands the synchronous path —
    the request re-enters the async_eval pipeline (asserted: 43
    pipelined steps after stand-down in the new check). New
    `SpecStats::standdowns_total` counter.
  - ADR 0005 boundaries stay HARD cutoffs applied under the heuristics,
    per the task directive: per-model envelope gamma (attachment-time
    `speculative_gamma_bound` clamp), deterministic width, and
    `VERIFY_MAX_KEY_LEN` bound the ramp's output; heuristics only ever
    shrink a round, never widen the envelope. Documented in the engine
    module docs and config docs.
  - spec_decode.rs grows `check_width_ramp` (a gamma-recording drafter
    observes the exact per-round gamma at every width 1..spec_max_batch)
    and `check_acceptance_standdown` (total-rejection adversary stood
    down right after warmup, output exact, pipeline resumed). The
    self-draft and cross-quant arms now run the production-default
    heuristic and assert it never fires on healthy pairs
    (standdowns_total == 0); the adversarial arms set
    `spec_min_acceptance = 0.0` explicitly so their full rollback
    pressure is preserved (the heuristic would correctly stand them
    down — its own checks cover that behavior).
  - NEW tests/spec_throughput.rs: the Phase 8 throughput acceptance run
    (#[ignore]d perf measurement, release-only, throughput.rs
    conventions — median of 5 after a discarded warm-up, decode-window
    rate, fresh engine per run, prefix cache off). Structural gates:
    ON == OFF token equality per lane, SPEC §11.3 >50% same-family
    acceptance, speculation really engaged, no false stand-down. Ratios
    are recorded here, deliberately NOT gated (see verdict).
- Decisions:
  - Ramp formula `ceil(gamma·(spec_max_batch+1−W)/spec_max_batch)`:
    full gamma at W=1, ≥1 through W=spec_max_batch (SPEC §6.5 disables
    strictly above), 0 beyond. Chosen over the previous cliff per the
    task directive ("stand down as batch size approaches
    spec_max_batch").
  - Acceptance floor 0.125: break-even acceptance ≈ draft/target cost
    ratio (a round spends γ·c_draft + c_verify to commit 1+accepted
    tokens vs c_target for 1), so 1-in-8 never clips a properly-sized
    pair (deployment ratio ≈ 0.05) while a noise draft stops costing
    after one warmup window (16 proposed ≈ 4 rounds at γ4).
  - Stand-down is permanent per request: content that drafts badly stays
    that content; requests are bounded; periodic re-probe is future work
    if metrics ever justify it.
  - Throughput lanes are measured, not speedup-gated: codifying a floor
    under a measured net LOSS would be meaningless; the correctness and
    acceptance gates stay.
- Deviations: none from SPEC text. SPEC §12 Phase 8's "≥1.6× baseline at
  acceptance >60%" is NOT MET and, on the evidence below, NOT MEETABLE
  on the pinned tiny pairs — recorded plainly, not reinterpreted; see
  DECISION NEEDED.
- Throughput verdict (M-series dev machine, release, single-stream,
  256-token decode window, median of 5):
  - qwen3-0.6b-8bit target / qwen3-0.6b-4bit draft (certified pair,
    envelope bound 7): OFF 104.9 tok/s; ON γ4 74.1 tok/s = 0.71×
    (acceptance 83.8%, 196/234 over 59 rounds); ON γ3 71.0 tok/s =
    0.68× (acceptance 89.9%).
  - qwen2.5-0.5b-4bit self-pair at the ADR 0005 clamp γ3: OFF 224.6
    tok/s; ON 140.5 tok/s = 0.63× at 100.0% acceptance (191/191).
  - Plainly: at this model scale speculation is a NET LOSS single-stream
    on every pair the pins can form, despite acceptance far above the
    >60% qualifier and exact outputs. The physics: the draft:target
    weight ratio is ~0.65 (633,442,994 / 968,893,578 bytes), so γ draft
    forwards + a verify cost more than the plain steps they replace, and
    a speculating request additionally forfeits the async_eval pipeline.
    Speculation's value case rests entirely on the size-gap pair SPEC
    §12 names (0.6B drafting 8–14B, cost ratio ≈ 0.05–0.1), which no
    pinned CI model provides.
  - The γ4→γ3 clamp itself costs ~3 points of ratio on the healthy pair
    (0.71× → 0.68×): the ADR 0005 clamp is a minor erosion next to the
    draft-cost economics — clamping is NOT what makes tiny pairs
    unprofitable, so no model class loses a win it otherwise had.
  - The acceptance heuristic CANNOT catch this loss (acceptance is high;
    the loss is cost-ratio-driven and invisible to the engine).
    Guarding it is a deployment-configuration concern; candidate future
    work: an attachment-time weights-byte-ratio warning in the worker,
    which knows both byte counts. Not implemented, recorded here.
- Acceptance:
  ```
  $ cargo test -p kiln-engine --lib drafter
  3 passed (spec_gamma_at_width ramp: defaults, monotonicity+bounds over
  gamma/smb grids, edges incl. the qwen2.5 gamma-3 clamp shape)

  $ KILN_TEST_MODELS=... cargo test -p kiln-models --test spec_decode -- --nocapture
  every fixture model exact under speculation, both drafters (self 99-100%,
  adversarial 0.2-0.6%); envelope-disabled models (gemma-2, smollm2-bf16)
  exact on the plain path; qwen3-0.6b-8bit/4bit cross-quant 46/66 (69.7%),
  warm-prefix composition intact
  spec_max_batch gate: width 6 never consulted; narrowed batch did
  width ramp: 1 requests -> gamma bound 4 over 14 consultations
  width ramp: 2 requests -> gamma bound 3 over 28 consultations
  width ramp: 3 requests -> gamma bound 2 over 42 consultations
  width ramp: 4 requests -> gamma bound 1 over 56 consultations
  acceptance stand-down: 4 rounds (0/16 accepted), stood down after warmup,
    43 pipelined steps afterwards, output exact
  in-situ rollback: 1191ns/round @ 8-token vs 1262ns/round @ 900-token
  kv-envelope crossing: 22 rounds, clean stop, output exact
  test result: ok. 1 passed (128.86s)

  $ KILN_TEST_MODELS=... cargo test -p kiln-models --release --test spec_throughput -- --ignored --nocapture
  == qwen3-0.6b-8bit target / qwen3-0.6b-4bit draft: prompt 39 tokens,
     decode 256, deterministic width 9, ADR 0005 envelope Some(7)
     speculation OFF: 104.9 tok/s
     ON gamma 4 (envelope: unclamped): 74.1 tok/s -> 0.71x OFF,
       acceptance 196/234 (83.8%) over 59 rounds, 15 rollback rounds
     ON gamma 3 (the clamp shape, priced on a real pair): 71.0 tok/s
       -> 0.68x OFF, acceptance 186/207 (89.9%) over 69 rounds
  == qwen2.5-0.5b-4bit self-pair: ADR 0005 envelope Some(3)
     speculation OFF: 224.6 tok/s
     ON gamma 3 (ADR 0005 clamp; self-draft): 140.5 tok/s -> 0.63x OFF,
       acceptance 191/191 (100.0%) over 64 rounds, 0 rollback rounds
  test result: ok. 1 passed (77.79s)

  $ KILN_TEST_MODELS=... cargo test -p kiln-worker --test draft -- --nocapture
  incompatible pair rejected as UNHEALTHY ... out-of-envelope target
  rejected as UNHEALTHY ... draft/verify over RPC ok: weights 633442994
  -> 968893578 bytes, 24 identical tokens, 16/27 draft tokens accepted
  test result: ok. 1 passed (16/27 = 59% >> the 12.5% floor: no stand-down)

  $ KILN_TEST_MODELS=... cargo test --workspace
  all 50 test binaries ok, exit 0 (golden 641s, spec_decode 128s incl.
  the new heuristic checks, batching/preemption/prefix/leak/worker/
  gateway suites green).
  NOTE for future sessions: a first workspace run failed spuriously
  ("worker did not settle in 180s" in kiln-worker/tests/draft.rs)
  because a concurrent `cargo build --workspace --no-default-features`
  in the same checkout overwrote target/debug/kiln-worker with the
  Metal-less shape mid-run — CARGO_BIN_EXE spawns resolve to that path.
  Never share a target dir between a running test suite and another
  cargo invocation. Serial rerun: green.
  $ cargo build --workspace --no-default-features          -> clean (linux shape)
  $ cargo clippy --workspace --all-targets -- -D warnings  -> clean (both shapes)
  $ cargo fmt --all --check                                -> clean
  $ ruff check / ruff format --check python/ tests/e2e     -> clean
  $ pytest python/kiln_worker_py/tests                     -> 35 passed
  $ uv run --project tests/e2e pytest tests/e2e            -> 77 passed, 2 skipped
  ```
- Next: PR opened from `claude/p8-autodisable-throughput`; record CI
  verification per protocol. Then the remaining Phase 8 parts: gateway
  `[model.speculative]` → `--draft-model` config wiring, then
  CAPABILITY_SPECULATIVE advertisement (the measured verdict above
  should inform whether tiny-ratio draft configs warn at attach).
- DECISION NEEDED: disposition of the SPEC §12 Phase 8 speedup bar
  ("≥1.6× at acceptance >60%"). The bar is measured NOT MET on every
  pinned pair, and the arithmetic says no pinned pair can meet it (at
  weight-cost ratio ~0.65, even 100% acceptance caps below ~1.4× before
  overheads; measured reality is 0.63–0.71×). Options:
  A) Amend the bar via ADR to name the deployment shape (0.6B drafting
     8–14B on the operator's hardware) as the measurement that gates
     CAPABILITY_SPECULATIVE advertisement, keeping tiny-pair CI lanes
     correctness-only. Honest, but the bar is then unverifiable in CI.
  B) Pin one mid-size model (e.g. a 7–8B 4-bit, ~4.5 GB) solely for a
     dev-machine/perf-lane measurement of the real pair. Verifiable, but
     grows the pinned set and the perf lane's runtime materially.
  Per protocol: picking nothing; the heuristics and measurement stand
  regardless of the disposition.

## [2026-07-14] Phase 8 / Part 3 — PR #22 CI verification recorded — DONE
- What: PR #22 (`claude/p8-autodisable-throughput`, commit 67fc8f4) run
  29321876048: ALL FOUR checks pass on the real runners — lint 44s,
  compile-linux 55s, test-macos-release 3m20s, test-macos 25m13s.
- The blocking spec_decode lane (`KILN_FIXTURE_PARITY=skip`, live
  speculation-off baselines) passed on the foreign GPU with the part 3
  heuristics live: width ramp, acceptance stand-down, and the
  production-default no-false-fire assertions all held on a different
  device class — the heuristics are policy over the device-independent
  SPEC §6.5 invariant, exactly as designed.
- Advisory golden lane (ADR 0004, permanently non-blocking): the ONLY
  divergence is the known gemma-3-1b-it-4bit/chat-basic flip — the same
  fixture and class recorded in ADR 0004 (4-ULP fp16 race, kernel-class
  coin toss on the foreign device). No pattern change → no action per
  the ADR; noted here per its protocol.
- Next: PM ruling on the DECISION NEEDED above (disposition of the SPEC
  §12 Phase 8 speedup bar), then the remaining Phase 8 parts — gateway
  `[model.speculative]` → `--draft-model` config wiring, then
  CAPABILITY_SPECULATIVE advertisement.

## [2026-07-14] CI infra — serialize GPU-worker cases in the rpc suite — DONE
- What:
  - Incident: run 29364413227 (the PR #22 merge push) attempt 1 failed
    `prefix_cache_stats_and_ssd_restart` — the ssd-restart worker died
    mid-stream (h2 BrokenPipe) while `cancel_and_drain_rpc_semantics`'s
    worker was concurrently live on the macos-14 runner's shared
    paravirtual GPU. Attempt 2 of the identical commit passed, and the
    suite passed 6/6 locally on real Metal → environment flake of the
    same class as the spec_probe segfault (PROGRESS 2026-07-13), at
    process level instead of in-process.
  - `crates/kiln-worker/tests/rpc.rs`: the two `#[tokio::test]` fns are
    now plain async fns invoked sequentially from a single
    `worker_rpc_semantics` test (old in-file order), so only one GPU
    worker process runs at a time — the spec_probe remedy applied at the
    worker-process level. Zero assertion changes; both phases' skip
    guards intact.
  - Sweep for a third instance (this is the second of this class): every
    other MLX-driving integration binary is already a documented single
    `#[test]` ("Single #[test] because the kiln-mlx live-object counter
    is process-global" — batching, preemption, prefix_cache/multiturn,
    draft, spec_decode/probe/throughput, leak*, golden, paged*,
    pipeline_discard, prefill_pad, sampler, deterministic_partition,
    wrappers, rollback_cost+grammar(engine) are explicitly no-MLX/no-GPU,
    tokenize suites are CPU). In-crate unit tests: the only GPU-driving
    unit test in the workspace is nn.rs's single YarnRoPE parity test
    (config.rs/paged_attn.rs/openai.rs matches are comments/CPU). NO
    third instance exists; rpc.rs was the last multi-test GPU binary.
- Decisions: merged-#[test] over `--test-threads=1` in the CI step:
  the repo convention already encodes the constraint in the test files
  themselves (spec_probe precedent), it holds for local `cargo test
  --workspace` runs too, and it needs no workflow change.
- Deviations: none.
- Acceptance:
  ```
  $ KILN_TEST_MODELS=~/.kiln/test-models cargo test -p kiln-worker --test rpc -- --nocapture
  running 1 test
  worker 1: cancel + graceful drain (deadline escalation) ok
  worker 2: immediate drain ok
  stats + prefix cache over RPC ok: WorkerStats { requests_total: 2, ...
    prefix_tokens_reused_total: 63, ssd_blocks_stored: 2, ssd_writes_total: 2, ... }
  worker restart served the prefix from SSD: PrefixCacheHit { tokens_reused: 63, from_ssd: true }
  test worker_rpc_semantics ... ok
  test result: ok. 1 passed; 0 failed; 0 ignored; ... finished in 16.21s
  $ cargo fmt --all --check                                            -> clean
  $ cargo clippy --workspace --all-targets -- -D warnings              -> clean
  $ cargo clippy --workspace --all-targets --no-default-features ...   -> clean
  $ ruff check / ruff format --check python/ tests/e2e                 -> clean
  ```
- Next: unchanged — PM ruling on the Phase 8 speedup-bar DECISION
  NEEDED, then gateway `[model.speculative]` config wiring and
  CAPABILITY_SPECULATIVE advertisement.

## [2026-07-14] Phase 8 — ADR 0006: speedup-bar DECISION NEEDED closed (PM ruling: option A) — DONE
- What:
  - The 2026-07-14 DECISION NEEDED (disposition of the SPEC §12 Phase 8
    "≥1.6× at acceptance >60%" bar, measured NOT MET on every pinned
    pair) is CLOSED by PM ruling: option A. Recorded as
    `docs/decisions/0006-speculative-throughput-bar-deployment-shape.md`.
  - ADR 0006 states the bar as originally written implicitly assumed a
    deployment shape (small draft, large target, meaningful cost
    asymmetry) that the sub-1B pinned CI fleet cannot produce, citing
    this session's `spec_throughput.rs` measurements as evidence:
    certified qwen3-0.6b-8bit/4bit pair 0.71× (γ4) / 0.68× (γ3),
    clamped qwen2.5-0.5b self-pair 0.63× at 100% acceptance, with the
    cost-ratio arithmetic (draft weighs ~0.65× target, so γ proposal
    forwards + verify exceed the plain steps replaced; ceiling <~1.4×
    even at 100% acceptance) establishing the loss as structural at
    these pins, not tunable.
  - The amendment: CAPABILITY_SPECULATIVE's correctness gate (golden
    parity under speculation, the ADR 0005 envelope, the auto-disable
    heuristics) remains a permanent BLOCKING CI requirement — unchanged.
    The ≥1.6× throughput claim is decoupled from CI and becomes a
    documented deployment-shape precondition: expected for draft ≤1B vs
    target ≥7–8B, unverified in CI until such a pair enters the pinned
    fleet, and operators enabling small/small pairs should expect a
    measured loss (0.63–0.71×), not a gain.
  - SPEC §12 Phase 8 Accept line amended to reference ADR 0006 (doc-only;
    correctness clauses lead, throughput clause restated as the
    precondition). Option B (pinning a 7–8B model for a perf-lane-only
    measurement) declined for pinned-fleet cost; revisit trigger in the
    ADR re-arms the measurement if such a pair is ever pinned.
  - The attachment-time weights-byte-ratio guard (flagged in the part 3
    entry as candidate future work) is now a TRACKED backlog item: a
    `> BACKLOG:` block in SPEC §6.5 (same convention as the §8.3
    rate-limits item) plus a Consequences bullet in ADR 0006 — warn or
    reject at drafter attachment when the byte ratio predicts a loss,
    because the acceptance heuristic structurally cannot catch
    cost-ratio losses.
- Decisions: within the ruling's latitude — ADR numbered 0006 following
  0005; revisit triggers mirror ADR 0003's convention (pin bumps stale
  the figures; a size-gap pin lifts "unverified").
- Deviations: none (SPEC §12 edit and new ADR are both PM-directed; the
  docs/decisions read-only rule applies to EXISTING files, and 0006 is
  new).
- Acceptance: docs-only change —
  ```
  $ ls docs/decisions/ | tail -1
  0006-speculative-throughput-bar-deployment-shape.md
  $ grep -c "ADR 0006" docs/SPEC.md
  2   (Phase 8 Accept line + §6.5 BACKLOG block)
  $ cargo fmt --all --check && ruff format --check python/ tests/e2e  -> clean
  ```
- Next: gateway `[model.speculative]` → `--draft-model` config wiring
  (operator docs must state the ADR 0006 shape precondition), then
  CAPABILITY_SPECULATIVE advertisement gated on correctness only.

## [2026-07-14] Phase 8 / final part — [model.speculative] config wiring + CAPABILITY_SPECULATIVE — DONE
- What (commit 74eb417, PR #25):
  - Gateway registry resolves `[model.speculative].draft` to a local model
    directory (tilde-expanded, config.json required — the model-path
    locality contract) and stores it on the ModelEntry; a speculative
    block on a python-routed model (explicit or auto-downgraded) is a
    STARTUP ERROR — speculation is a rust-worker capability, and silently
    dropping the block would be the gateway-side twin of the "requested
    speculation silently inert" state ADR 0005 forbids the worker.
  - Supervisor appends `--draft-model <resolved dir> --draft-gamma <n>`
    to the rust worker argv (kiln.toml schema: draft + gamma, SPEC §10).
  - Worker grew `--draft-gamma` (>= 1 enforced at parse; rejected without
    `--draft-model`): sets the engine's per-round proposal count BEFORE
    the ADR 0005 envelope clamp, which still applies on top.
  - CAPABILITY_SPECULATIVE is now advertised in WorkerInfo, exactly when
    a compatible draft is attached: the gate is correctness only — the
    tokenizer-compat check plus the ADR 0005 envelope, both enforced at
    attach — never the ADR 0006 throughput precondition. A draft-less
    worker, and any UNHEALTHY-rejected pair, never advertises it.
  - Operator docs (kiln.toml.example) state the ADR 0006 deployment-shape
    precondition plainly: correctness-safe on any certified pair; only a
    small-draft/large-target shape (~<=1B drafting >=7-8B) is expected to
    gain; similar-size pairs are a measured LOSS (0.63-0.71x at sub-1B
    scale) regardless of observed acceptance.
  - Tests: NEW tests/e2e/test_speculative.py — a real kiln.toml with
    [model.speculative] through the real gateway; asserts the spawned
    worker's live command line carries the resolved --draft-model path
    and --draft-gamma, the gateway log shows the draft attach + envelope
    clamp, and the stack serves greedy completions (integration, not
    config parsing). draft.rs flips its capability assertion (advertised
    with a compatible draft via the full gateway argv shape incl.
    --draft-gamma 4; absent on the draft-less baseline) and pins the
    --draft-gamma argv rejections (0; gamma without draft) — parse-level,
    pre-MLX. Registry unit tests: python rejection (explicit + auto),
    non-local draft rejection, local-draft resolution.
- Decisions: flag named `--draft-gamma` (pairs with `--draft-model`;
  gamma is meaningless without a draft, enforced). Worker-kind gate
  checked before draft locality (the more fundamental config
  contradiction reports first). Capability advertised at attach and never
  withdrawn at runtime: the auto-disable heuristics are per-request
  stand-downs (policy), not capability loss.
- Deviations: none.
- Acceptance (local, Metal dev machine):
  ```
  $ cargo test -p kiln-gateway --lib                       -> 45 passed (3 new resolve_draft tests)
  $ KILN_TEST_MODELS=... cargo test -p kiln-worker         -> draft 1 passed (argv rejections ok;
      incompatible + out-of-envelope UNHEALTHY; SPECULATIVE advertised with draft, absent without;
      24 identical greedy tokens, 16/27 accepted); grammar 1 passed; rpc 1 passed
  $ uv run --project tests/e2e pytest tests/e2e            -> 79 passed, 2 skipped
      (test_speculative.py::test_speculative_config_reaches_the_worker_argv PASSED,
       ::test_speculative_stack_serves_greedy_completions PASSED)
  $ cargo fmt --all --check; clippy --all-targets (default + --no-default-features) -D warnings;
    ruff check + format --check python/ tests/e2e          -> all clean
  ```
- CI verification (PR #25, commit 74eb417, run 29374672707 — ALL FOUR
  checks pass on the real runners): lint 44s, compile-linux 35s,
  test-macos-release 3m9s, test-macos 28m28s. On the foreign GPU:
  draft_verify_over_rpc_with_compat_gate ok in both the workspace pass
  and the blocking model-gated lane; e2e 79 passed, 2 skipped with both
  test_speculative.py cases PASSED. Advisory golden lane (ADR 0004,
  permanently non-blocking): the ONLY divergence is the known
  gemma-3-1b-it-4bit/chat-basic flip — same fixture and class as recorded
  in the ADR; no pattern change -> no action, noted per its protocol.
- This closes Phase 8 entirely (SPEC §12): Drafter abstraction, draft
  loading, batched draft/verify, O(1) rollback, acceptance metrics,
  auto-disable heuristics, and config wiring are all landed; correctness
  clauses are permanent blocking CI gates, and the >=1.6x throughput
  clause stands as the ADR 0006 deployment-shape precondition (unverified
  in CI until a size-asymmetric pair enters the pinned fleet).
- Next: Phase 9 — multi-model supervision + memory governance (LRU
  eviction with drain, pinning, TTL; budget accounting from heartbeats;
  per-worker admission; INTERACTIVE/BATCH priorities; crash-loop backoff;
  the §8.3 BACKLOG rate-limit/timeout enforcement lands here too).

## [2026-07-14] Phase 9 / part 1 — machine memory budget + model lifecycle supervision — DONE
- What (commit 9729fd6, PR #26):
  - NEW crates/kiln-gateway/src/lifecycle.rs — machine-level memory
    governance (SPEC §2.3): budget = total unified memory (shelled-out
    `sysctl hw.memsize`; unsafe stays confined to kiln-mlx, same
    rationale as the supervisor's /bin/kill) × `memory.budget_fraction`,
    or the NEW explicit `[memory] budget_bytes` override. Each worker is
    charged `max(mlx_active, weights + kv_pool_allocated) + mlx_cache`
    from its real heartbeat MemoryReport — every MLX byte counted once,
    weights+pool as the floor under a runtime that can't report active.
    Load-time projection = weight bytes on disk (target + draft
    safetensors); the first post-READY heartbeat replaces it with the
    measured footprint BEFORE the machine-wide load permit releases, so
    the next load budgets against real bytes, never stale reservations.
  - supervisor.rs restructured into a command-driven per-model state
    machine: idle (Unloaded) → Load → budget acquisition → spawn → READY
    → monitor. A load that would exceed the budget first evicts the LRU
    model that is loaded, unpinned, and outside its TTL keep-alive lease
    — per the SPEC §2.2 ladder: Drain GRACEFUL (30s bound, polled until
    the worker is empty) → SIGTERM → SIGKILL after 5s grace, all
    process-group signaled; with no evictable candidate the load is
    rejected (Unloaded{over budget}). The monitor records memory every
    heartbeat, counts in-flight work as use, and self-unloads once idle
    past `ttl_seconds`. Phase 2 crash semantics preserved (exponential
    backoff, max 3 attempts, STABLE_RESET forgiveness, Failed = manual
    reset; Failed refuses Load commands) and respawns re-pass the budget
    gate. Startup loads are sequenced in config order by a bootstrap
    task, so boot-time eviction is deterministic (LRU clock starts at
    READY; the load permit alone would leave ordering to task-spawn
    races).
  - Lifecycle surfaced end-to-end: WorkerStatus grew Draining +
    Unloaded{evicted | idle ttl | over budget}; /readyz treats Unloaded
    as settled (200 — deliberate and reversible); ready_entry touches
    the LRU/TTL clock on every routed request, 503s `model_unloading`
    on Draining, and on Unloaded triggers the on-demand background
    reload while returning the retriable `model_loading` — what makes
    TTL unload and eviction reversible before the Phase 10 admin API.
  - Metrics: kiln_memory_budget_bytes, kiln_memory_used_bytes,
    kiln_worker_memory_bytes + MemoryReport mirrors (weights, kv_pool,
    mlx_active, mlx_cache, process_rss), kiln_worker_unloads_total
    {model, reason}. kiln.toml.example documents the budget/eviction/
    pinning/ttl semantics.
  - NEW tests/e2e/test_lifecycle.py — 3 real-stack scenarios, budgets
    placed between measured bounds (dev machine: gemma-3-1b rust worker
    766MB idle / 1176MB after traffic — the 512-block KV pool allocates
    lazily on first use; qwen2.5-0.5b 300MB / 486MB; weights 733MB /
    278MB): (1) three gemma models vs a 2.08GB budget — startup evicts
    the LRU (alpha); touching charlie-then-bravo and reloading alpha
    then evicts charlie, NOT the older worker bravo (request recency,
    not load order), and reloaded alpha serves; (2) a pinned qwen model
    that is the machine-wide LRU survives eviction pressure that takes
    the unpinned model instead; (3) a ttl_seconds=10 model auto-unloads
    on schedule (process gone, gauges zeroed, readyz stays 200) and
    reloads on demand. All memory assertions read real heartbeat bytes
    via /metrics — nothing mocked.
- Decisions:
  - `ttl_seconds` reads as a keep-alive lease (the task's "unpinned,
    non-TTL-protected" eviction set): within ttl of last use a model is
    protected from LRU eviction; past it the supervisor auto-unloads it
    anyway, and in-flight work holds the lease open (busy heartbeats
    touch). Pinned = permanent lease.
  - Budget enforcement is load-time only (SPEC §2.3 "gateway rejects
    load()"): measured usage can drift over budget as lazily-allocated
    KV pools and caches grow; continuous-pressure eviction is left for
    part 2 alongside admission control.
  - One load at a time machine-wide (lifecycle load permit): budget
    acquisition can't double-spend headroom, and one GPU only loads one
    weight set usefully anyway.
  - Eviction racing a crash resolves cleanly: the backoff wait accepts
    Unload and cancels the pending restart. The python worker's
    UNIMPLEMENTED Drain is best-effort — escalate straight to SIGTERM.
  - Gateway shutdown remains SIGKILL (Phase 2 semantics); the graceful
    ladder is the eviction contract, not the shutdown contract.
- Deviations: SPEC §2.3's budget formula says "minus a fixed floor"; no
  separate floor knob was added — the 1−fraction headroom (20% default)
  is the floor, and `budget_bytes` covers operators needing an exact
  cap. Flagged here rather than silently dropped.
- Acceptance (local, Metal dev machine):
  ```
  $ cargo test -p kiln-gateway --lib
  test result: ok. 52 passed; 0 failed  (new: lifecycle victim selection —
    pinned/TTL-lease/LRU/Ready-only; footprint math; budget_bytes override;
    weights-on-disk projection; memory gauges record/clear)
  $ KILN_TEST_MODELS=... uv run --project tests/e2e pytest tests/e2e
  82 passed, 2 skipped in 147.29s
    test_lifecycle.py::test_lru_eviction_order_and_on_demand_reload PASSED
    test_lifecycle.py::test_pinned_model_survives_eviction_pressure PASSED
    test_lifecycle.py::test_ttl_idle_model_auto_unloads_and_reloads_on_demand PASSED
  $ cargo test --workspace                                  -> green (exit 0)
  $ cargo fmt --all --check                                 -> clean
  $ cargo clippy --workspace --all-targets -- -D warnings   -> clean
  $ cargo clippy --workspace --all-targets --no-default-features -- -D warnings -> clean
  $ ruff check / ruff format --check python/ tests/e2e      -> clean
  ```
- CI verification (PR #26, commit 9729fd6, run 29380692612 — ALL FOUR
  checks pass on the real runners): lint 43s, compile-linux 49s,
  test-macos-release 4m1s, test-macos 27m30s. On the foreign runner the
  full e2e suite is 82 passed, 2 skipped in 256s with all three
  lifecycle scenarios PASSED — the measured-bounds budgets (2.08GB /
  730MB) held on the CI machine exactly as on the dev machine,
  confirming the footprint components (packed weights + fixed 512-block
  KV pool) are device-stable. Advisory golden lane (ADR 0004,
  permanently non-blocking): the ONLY divergence is the known
  gemma-3-1b-it-4bit/chat-basic flip — same fixture and class as
  recorded in the ADR and the P8 closeout; no pattern change -> no
  action, noted per its protocol.
- Next: Phase 9 part 2 — INTERACTIVE/BATCH priority classes + preemption
  ordering, admission control, the §8.3 BACKLOG rate-limit/timeout
  enforcement, python worker batching upgrade (mlx-lm batch API) if
  straightforward, the interactive-vs-batch acceptance (flood BATCH,
  send INTERACTIVE -> interactive TTFT p95 unaffected >2×), and the
  30-min full-stack soak leak gate. Consider continuous-pressure
  eviction (usage drift over budget from lazy pools) alongside the
  admission work.

## [2026-07-14] Phase 9 / part 2 — priority classes + per-request admission control — DONE
- What (commit 1f2dd80, PR #27):
  - INTERACTIVE/BATCH priorities wired end-to-end (SPEC §6.1/§12): all
    three completion endpoints accept a `priority` request field
    ("interactive" default | "batch"; anything else is a named 400),
    validated into the frozen proto's `SubmitRequest.priority` — which the
    gateway previously hardcoded to INTERACTIVE, making the field accepted
    but ignored. The engine now consults priority at ADMISSION, not just
    plan-time victim choice: the WAITING queue is class-ordered
    (INTERACTIVE ahead of BATCH, FIFO by arrival within a class, preempted
    requests keep class seniority — engine.rs enqueue_waiting), and
    admit() escalates from prefix-cache eviction to preempting
    strictly-lower-priority RUNNING requests in SPEC §6.1 deservingness
    order, so an INTERACTIVE arrival claims blocks from BATCH decodes
    instead of waiting for a natural finish. Same-class arrivals never
    preempt (the newcomer is the least deserving of its class), so
    uniform-priority traffic schedules exactly as before — golden,
    batching, and determinism suites see no change.
  - Per-request memory admission (SPEC §2.3 second level, the §8.2
    "admission check" step) — closes part 1's continuous-drift gap. The
    rust worker publishes its paged-pool geometry in GetInfo via two
    ADDITIVE WorkerInfo fields (`kv_bytes_per_block` = 14,
    `kv_pool_blocks` = 15; additive changes are explicitly allowed on the
    frozen proto, no renumber/retype/reuse, SubmitRequest's reserved range
    untouched; python worker reports 0 = not gated). Bytes/block is
    computed at load from KvDims with the 16-bit element size every
    rust-servable checkpoint computes in (SPEC §7.3), single-sourced in
    KvSpec::bytes_per_block (PagedKv::bytes_per_block now delegates). Per
    request, the gateway projects the pool growth serving could
    materialize (full-pool commitment − heartbeat-materialized bytes,
    saturating: heartbeat pool totals include draft pools) against LIVE
    headroom (budget − Σ measured footprints) and refuses with a
    retriable 503 `insufficient_memory` + kiln_admission_rejects_total
    {model}. The drift part 1 permitted — lazily-materialized KV pools
    blowing past the budget after every load-time check passed — is now
    refused at the request that would cause it; a fully-materialized pool
    projects zero growth and is never gated (its bytes are already in the
    footprint ledger). ready_entry's LRU touch still precedes the gate, so
    recency semantics are unchanged. kiln.toml.example documents both.
  - NEW tests/e2e/test_priority.py — the SPEC §12 Phase 9 acceptance
    scenario on a real stack: 12 BATCH streams (unique ~1150-token
    prompts, greedy, 512 max_tokens) saturate llama-3.2-1b's 512-block
    pool (verified via kv_blocks gauges), then INTERACTIVE probes measure
    TTFT. Stated threshold: worst flooded TTFT ≤ 2× worst unloaded
    baseline + 500 ms floor (SPEC's ">2×" bar plus timer-noise floor).
    Also asserts preemptions occurred and every preempted BATCH stream
    still finished with `length`.
  - NEW tests/e2e/test_admission.py — three real-stack scenarios with
    budgets between part 1's measured bounds (qwen2.5-0.5b: 300 MB idle /
    486 MB warm / 278 MB weights / 201 MB pool commitment): (1) budget
    450 MB — the LOAD passes, the first request 503s `insufficient_memory`,
    the worker stays READY, the pool stays unmaterialized, ledger ≤
    budget (the per-request check, distinct from part 1's load-time
    check); (2) budget 850 MB — the identical request serves, the pool
    materializes, ledger ≤ budget (the gate refuses over-commitment, not
    traffic); (3) two models under 900 MB — warming one model's pool
    consumes the headroom, the OTHER model's first request is then
    refused: per-request admission pricing in another model's
    post-load drift.
  - kiln-models/tests/preemption.rs grew scenarios 9 (admission-time
    preemption: BATCH-saturated pool, late INTERACTIVE's first token ≤ 8
    steps after submit — measured 2 — with the most recent BATCH request
    preempted, everything bit-exact vs solo) and 10 (WAITING-queue class
    order: an INTERACTIVE arrival is served before an earlier-queued BATCH
    request without preempting the same-class runner).
- Decisions:
  - Priority surfaces as a request-body extension field on all three
    endpoints (OpenAI ignores unknown fields, so clients can send it
    unconditionally); OpenAI's `service_tier` was not overloaded — its
    semantics (uptime tiers) are not preemption classes.
  - The admission gate lives in the gateway: it owns the budget and the
    live footprint ledger, and SPEC §8.2 places an "admission check" in
    the request path. Worker-local admission (queue on free blocks,
    in-band OOM_REJECTED for never-fits) is unchanged Phase 4 behavior.
  - Over-headroom requests are REJECTED (retriable 503), not queued or
    eviction-triggering — per the task ("reject or queue"); rejection
    keeps the gate O(1) and the retry loop client-visible. Residual gap,
    documented: cache-only drift on fully-materialized pools has no
    admission lever (growth = 0); that remains for continuous-pressure
    eviction, still open alongside §8.3.
  - Pool-growth projection is the materialization delta, not
    tokens×bytes: MLX pools materialize per-layer on first write, so any
    first request materializes the whole pool — a token-proportional
    projection would be fiction. The e2e asserts the projected number
    against real post-traffic heartbeat bytes.
  - Two pre-existing test scenarios asserted behavior this task was
    explicitly assigned to change; updated with rationale, NOT silently
    weakened (CLAUDE.md rule): (a) preemption.rs scenario 5 pinned the
    pre-Phase-9 thrash dynamics — a preempted BATCH request resumed at
    the first free block and was re-preempted by each next INTERACTIVE
    arrival (`preemptions >= 2`); class-ordered admission makes it yield
    once and queue behind the arrivals (now asserts `preemptions == 1`);
    its bit-exactness and liveness clauses are unchanged and still pass.
    (b) test_lifecycle.py's LRU and pinned flows deliberately over-packed
    budgets and completed requests by silently drifting past them (the
    recorded part 1 gap); those exact requests now assert the 503
    `insufficient_memory` where the drift used to begin, plus ledger ≤
    budget throughout. Eviction-order, pinning, TTL, and recency
    assertions are all unchanged.
- Deviations: none. (Out of this part's scope, still open in Phase 9:
  the §8.3 rate-limit/timeout BACKLOG, python-worker batching upgrade,
  continuous-pressure eviction for materialized-pool cache drift.)
- Acceptance (local, Metal dev machine):
  ```
  $ KILN_TEST_MODELS=... uv run --project tests/e2e pytest tests/e2e
  86 passed, 2 skipped in 287.15s
    test_priority.py::test_interactive_ttft_survives_batch_flood PASSED
      baseline TTFTs: [2.462, 2.423, 2.471, 2.453]  (worst 2.471s)
      flooded  TTFTs: [3.162, 4.377, 2.908, 3.846]  (worst 4.377s)
      -> 1.77x worst-vs-worst, within the stated 2x + 500ms bar
         (limit 5.441s); preemptions observed; all 12 BATCH streams
         finished "length"
    test_admission.py::test_request_rejected_when_pool_growth_exceeds_headroom PASSED
    test_admission.py::test_request_admitted_when_headroom_allows PASSED
    test_admission.py::test_drift_from_one_model_gates_anothers_requests PASSED
    test_lifecycle.py (all 3, recalibrated for the closed drift gap) PASSED
  $ KILN_TEST_MODELS=... cargo test -p kiln-models --test preemption
  ok (scenarios 1-10; new: admission preemption — interactive first token
     2 steps after submit into a saturated BATCH pool, all bit-exact;
     waiting-queue class order — interactive served before earlier batch)
  $ cargo test -p kiln-gateway --lib      -> 55 passed (new: priority
     field validation; admit_request growth/headroom math incl. the
     cross-model drift case)
  $ KILN_TEST_MODELS=... cargo test --workspace          -> green (exit 0)
  $ uv run --project python/kiln_worker_py pytest ...    -> 35 passed
  $ cargo fmt --all --check                              -> clean
  $ cargo clippy --workspace --all-targets -- -D warnings                 -> clean
  $ cargo clippy --workspace --all-targets --no-default-features -- -D warnings -> clean
  $ ruff check / ruff format --check python/ tests/e2e   -> clean
  ```
- CI verification (PR #27, commit 1f2dd80, run 29390381637 — ALL FOUR
  checks pass on the real runners): lint 49s, compile-linux 46s,
  test-macos-release 3m19s, test-macos 30m52s. On the foreign GPU the
  full e2e suite is 86 passed, 2 skipped in 450s, with all three
  test_admission.py scenarios AND
  test_priority.py::test_interactive_ttft_survives_batch_flood PASSED —
  the TTFT bar, the measured-bounds budgets (450/850/900 MB), and the
  recalibrated lifecycle budgets (2.08 GB / 730 MB) all held on the CI
  machine, confirming the pool-commitment projection
  (kv_bytes_per_block × kv_pool_blocks) matches real materialization
  device-independently. Advisory golden lane (ADR 0004, permanently
  non-blocking): the ONLY divergence is the known
  gemma-3-1b-it-4bit/chat-basic flip — same fixture and class as
  recorded in the ADR and the P8/P9p1 closeouts; no pattern change ->
  no action, noted per its protocol.
- Next: Phase 9 part 3 — the 30-minute full-stack soak leak gate that
  closes the phase. (Also still open in Phase 9, per the part-1 Next
  list: §8.3 rate-limit/timeout enforcement, python-worker batching
  upgrade if straightforward, continuous-pressure eviction for
  materialized-pool cache drift.)

## [2026-07-15] Phase 9 / Part 3 — DONE (closes Phase 9)
- What:
  - tests/e2e/test_soak.py: the SPEC §11.3 30-minute full-stack mixed-load
    soak, run via scripts/soak.sh (the CLAUDE.md-documented interface) and
    gated by KILN_SOAK_MINUTES so the regular e2e sweep skips it. One
    gateway, five models, thirteen concurrent traffic classes exercising
    Phases 4-9 together under sustained load for the first time: pinned
    llama-3.2-1b (interactive+batch priorities, 12-stream floods, grammar,
    prefix warm/cold, /v1/messages, client-abort cancellations, greedy
    canary), qwen2.5-0.5b self-draft speculative pair at gamma 3,
    TTL-leased qwen2.5 (75 s) and gemma-3-1b (90 s, touched in ~5-min
    pressure bursts), and the python worker on smollm2-135m-bf16. Budget
    3.9 GB deliberately over-subscribes the fleet (all-warm ≈ 4.2 GB) so
    LRU eviction, TTL cycles, and per-request admission 503s run all
    30 minutes; warmup retries ride the system's own TTL recovery instead
    of assuming dev-machine margins.
  - Leak gates tracked THROUGHOUT, not endpoint-compared: 10 s memory
    samples; quiesced checkpoints every ~6 min asserting the mlx
    live-object counter back at its drained floor per worker generation
    (bounded interior excursions only — see below), mlx_active flat
    (±2 MB), mlx_cache capped; gateway RSS gated on its working set
    (160 MB cap = 2× measured plateau + bounded second-half delta),
    worker RSS on one-sided final-third slopes; committed bytes
    (weights+pools) ≤ budget at every sample.
  - Correctness gates held throughout: fixed greedy canaries on llama
    (~60 s) and the speculating model (~75 s) bit-identical every time;
    zero crash-restarts; pinned model never unloaded/rejected; grammar
    100% schema-valid; every 503 structured (insufficient_memory |
    model_loading); interactive < 90 s mid-flood; every gemma burst
    recovers to a 200.
  - Wire/metrics support: additive proto field
    MemoryReport.mlx_live_objects = 9 (the CLAUDE.md debug-build leak
    counter; 0 in release and the python worker) filled by kiln-worker
    from kiln_mlx::debug::live_objects() and re-exported as
    kiln_worker_mlx_live_objects; also re-exported the two WorkerStats
    speculation counters the gateway polled but dropped
    (kiln_worker_spec_tokens_{proposed,accepted}_total). Python gen code
    regenerated.
  - ci.yml: scripts/soak.sh --minutes 30 as a BLOCKING step at the end of
    test-macos — in the existing matrix, not a nightly; the ~30 min/PR is
    the accepted tradeoff per this task's ruling (the ci.yml comment says
    so and forbids quietly demoting it).
- Decisions:
  - Soak speculation pair: qwen2.5-0.5b-4bit self-draft at gamma 3, not
    the e2e suite's qwen3 8bit←4bit pair. Measured (probe, this date):
    the qwen3 pair fully warm is 5.03 GB — two 1.88 GB 512-block KV pools
    (28 layers × 8 KV heads × head_dim 128) on top of 969 MB weights —
    which cannot coexist with a multi-model fleet inside a CI-runner-
    sized budget. qwen2.5-0.5b@gamma 3 is inside the ADR 0005 envelope
    (the ADR itself records its clamp: gqa_factor 7 ⇒ gamma+1 ≤ 4;
    quantized trunk, head_dim 64, fused SDPA) and self-draft is the
    standard spec_decode gate shape; fully warm it is 0.98 GB.
    Speculation is greedy-requests-only by design (Phase 8), so the
    self-pair's 100% acceptance is expected; draft/verify/rollback
    machinery still runs on every greedy spec request (4,334 proposals
    in the acceptance run).
  - Ledger gate = committed bytes (Σ weights + materialized pools) ≤
    budget — the invariant load/pool-growth admission actually enforces —
    NOT raw used ≤ budget. The first soak iteration proved the raw ledger
    oscillates past budget through MLX cache growth on ~1k-token
    prefills: exactly the residual gap parts 1-2 recorded (cache-only
    drift has no admission lever; continuous-pressure eviction still
    open). The soak quantifies the gap instead of mis-gating it.
  - Three gate calibrations, each from a diagnosed failure, none a
    product bug (full narrative in commits 99eaf56/998df53):
    (a) CI run 29398308457 refused the third warmup with a CORRECT
    insufficient_memory 503 — the runner's idle footprints are fatter
    than the dev box's — so warmup retries up to 240 s, riding TTL
    expiry (gemma's 90 s lease frees ~0.75 GB); 503 model_loading
    (on-demand reload) and empty-text/finish="stop" (EOS-first) are
    legitimate outcomes, classified as such.
    (b) Gateway RSS: two 30-min runs converged on the same ~80 MB
    saturating working set (+19 MB/3min early collapsing to +1-2), but
    a macOS page-reclaim dip (-26 MB mid-run) and its re-climb blew a
    fixed-window slope gate (+667 KiB/min) — so the gateway is gated on
    the working set itself (160 MB absolute + ≤20 MB second-half
    delta); worker slope gates were stable (>2.5× margin) and stay.
    (c) One quiesced checkpoint read llama's live objects at 439 vs the
    437 floor, returning to 437 at every later checkpoint: engine-thread
    SSD-flush maintenance (engine.rs flush_entries → read_block_bytes)
    holds exactly 2 handles (K+V gather) per block mid-copy and drains
    after traffic stops; the heartbeat thread can sample mid-drain. The
    gate now requires return-to-floor at each group's last checkpoint
    with interior excursions ≤ 8 handles — a real leak still fails the
    floor check. Spec counters are accumulated across worker
    generations in the sampler (the final generation alone under-counted
    against 17-21 evictions per run).
- Deviations: none. SPEC §11.3 "RSS slope ~0 + MLX allocation counter
  stable" is implemented as the composite above; on the MLX side
  (bit-flat live objects and mlx_active at drained quiesce) it is
  strictly stronger than "not obviously growing".
- Acceptance (local, Metal dev machine — 30-minute acceptance run,
  PASS all gates; third 30-min run of the session, on the final gates):
  ```
  $ ./scripts/soak.sh --minutes 30
  duration: 1814s   PASSED (1836.49s total)
  -- quiesced checkpoints (live objects / mlx_active / gen) --
    llama-int    437 / 1232.1 MB at ALL SIX checkpoints (t+8s ->
                 t+1814s), generation 0 throughout — bit-flat.
    spec-qwen25  1401 / 965.0 MB at every warm checkpoint across
                 generations g0 -> g17 (17 evict/reload cycles,
                 identical drained baseline every time).
  -- RSS (report window last 2/3; gates: working-set / final-third) --
    gateway   74.8 -> 82.2 MB final (cap 160 MB), late delta +3.0 MB
              (cap +20); slopes +346 (2/3) / +281 (1/3) KiB/min reported
    llama-int 64.1 -> 69.7 MB, gated slope +114.4 KiB/min (cap +1024)
    py-smollm gated slope negative (page reclaim; one-sided gate)
  -- ledger vs budget --
    committed (weights+pools) peak 3.67 / 3.90 GB — never exceeded
    raw used worst +539 MB over budget (35/177 samples above) — the
    recorded continuous-pressure-eviction gap, quantified
  -- correctness --
    canary llama 31/31 identical; canary spec 20/20 identical
    restarts 0 on all five models; pinned llama: 0 unloads, 0 rejects
    evictions: spec 17, ttl-qwen25 1, burst-gemma 1, py-smollm 24 (all
    unpinned); idle_ttl unloads: ttl-qwen25 14, burst-gemma 5
    admission rejects (insufficient_memory): 52, never the pinned model;
    on-demand reload retries (model_loading): 95; gemma bursts 6/6
    recovered; preemptions 5; worker-side cancellations 86
    prefix tokens reused 120,936; SSD writes 2,848
    speculation run total 4,334 proposed / 4,334 accepted (self-draft
    greedy accept-all, per ADR 0005/0006 expectations)
    interactive latency: normal p50 0.29s p95 0.48s; during-flood
    p50 3.74s p95 8.38s max 8.49s (gate 90s)
  $ cargo test --workspace                     -> green (exit 0)
  $ uv run --project python/kiln_worker_py pytest ... -> 35 passed
  $ cargo fmt --all --check                    -> clean
  $ cargo clippy --workspace --all-targets -- -D warnings           -> clean
  $ cargo clippy --workspace --all-targets --no-default-features -- -D warnings -> clean
  $ ruff check / ruff format --check python/ tests/e2e -> clean
  ```
- CI verification (PR #28): two runs, both part of the record.
  - First run (commit 99eaf56, run 29398308457): the soak step FAILED at
    warmup — the CI runner's fatter idle footprints made the third warm
    request a CORRECT insufficient_memory refusal (the admission gate
    working as built); lint/compile-linux/test-macos-release all passed,
    and every pre-soak step of test-macos (workspace tests, model-gated
    suites, python, full e2e with test_soak properly SKIPPED env-less)
    was green. Diagnosed as harness calibration, fixed in 998df53
    (retry warmup + working-set gateway gate + cross-generation spec
    accounting; Decisions above).
  - Second run (commit 998df53, run 29404814236): ALL FOUR checks pass —
    lint 1m13s, compile-linux 46s, test-macos-release 5m09s, test-macos
    57m53s with the 30-minute soak BLOCKING and green on the runner:
    ```
    duration: 1815s   PASS: all gates held
    live objects: llama-int 437 at ALL SIX checkpoints (gen 0);
      spec-qwen25 1401 at every warm checkpoint across g0 -> g15 —
      the SAME drained baselines as the dev machine, device-independent
    mlx_active: 1232.1 / 965.0 MB flat — identical to dev to 0.1 MB
    gateway working set: 63.5 MB final (cap 160), late delta +0.1 MB
    committed peak 3.71 / 3.90 GB; raw worst +601 MB over (same
      cache-drift gap class as dev: +539..+711 MB across all runs)
    canaries 28/28 and 17/17 bit-identical; restarts 0; pinned llama
      0 unloads / 0 rejects; evictions spec 15, py-smollm 14, gemma 1;
      idle_ttl 20; admission rejects 133 (fatter footprints -> more
      pressure, all structured, never the pinned model); gemma bursts
      6/6 recovered; preemptions 3; cancellations 97; prefix reused
      103,109; SSD writes 2,249; spec run total 3,826/3,826
    interactive: normal p95 1.39s; during-flood p95 16.25s (slower
      paravirtual GPU; gate 90s)
    ```
    Threshold-margin note: the ttl class landed exactly at its >=8
    success minimum on CI (slower reloads convert more touches into
    model_loading retries) — a margin to watch, not a failure.
    Advisory golden lane (ADR 0004, permanently non-blocking): the ONLY
    divergence remains gemma-3-1b-it-4bit/chat-basic — same fixture and
    class as recorded in the ADR and the P8/P9 closeouts; no pattern
    change -> no action, noted per its protocol.
- Phase 9 closeout: parts 1-3 delivered machine memory budgeting from
  real heartbeats, LRU eviction with drain + pinning + TTL leases,
  crash-loop backoff, INTERACTIVE/BATCH priority classes with
  preemption-ordered admission, per-request pool-growth admission, and
  this soak as the standing full-stack leak/correctness gate. Still open
  (recorded, not silently dropped): §8.3 rate-limit/timeout enforcement
  BACKLOG; python-worker batching upgrade; continuous-pressure eviction
  for materialized-pool cache drift — now with a measured magnitude
  (+539 to +711 MB worst across three 30-min runs under this scenario).
  With this task the Phase 4-9 correctness arc closes: batching, paging,
  prefix cache, SSD tier, speculative decoding, structured output,
  priorities, and memory governance have now run COMPOSED under
  sustained concurrent multi-tenant load, holding greedy determinism
  bit-exact and returning every leak counter to its baseline.
- Next: Phase 10 (jobs, admin UI, packaging, docs) — not started in this
  session per the task instruction.

## [2026-07-15] Phase 9 / Part 3 addendum — BLOCKED (soak outcome contract; admission race needs a ruling)
- What: the CI run of the previous entry's own recording commit
  (283749a — docs-only, code identical to the run that passed) FAILED
  the soak, and the fix cycle surfaced two further findings on the next
  CI run. Every failure across six 30-min runs (5 local + 2 CI soaks)
  has been taken to root cause. Harness contract fixes shipped
  (fe877d9 + follow-up); ONE product-side finding needs a PM ruling
  before Phase 9's closure stands — see DECISION NEEDED.
- Finding 1 — bounded-drain severed tail (CI run 29408682271 → fixed
  contract in fe877d9): one in-flight spec-qwen25 request died with a
  retriable 502 worker_crashed while restarts stayed 0 (8 deliberate
  evictions that run). Mechanism pinned in code: routing refuses
  Draining workers (503 model_unloading) so this was no routing race;
  tonic Unknown on a streaming RPC = stream severed MID-FLIGHT;
  eviction's drain is bounded (supervisor.rs DRAIN_DEADLINE 30 s →
  SIGTERM → SIGKILL, the SPEC §2.2 contract), so a request outliving
  its evicted worker's drain window is severed, and the dead stream
  maps to §2.2's retriable 502. Composition of two specced behaviors;
  once in ~20k requests, slowest runner only. Soak contract: a 502
  worker_crashed is tolerated ONLY correlated (±60 s) with an unload
  of that model, never the pinned model, ≤3/run, every instance
  printed; uncorrelated → hard failure; restarts==0 still gates.
- Finding 2 — worker_draining, the race's structured leg (CI run
  29436961038): a request routed while Ready, refused worker-side
  after its Drain RPC, surfaces 503 worker_draining (gateway
  error.rs:152). Correct, structured, retriable — added to the counted
  expected-outcome classes (insufficient_memory, model_loading,
  model_unloading, worker_draining).
- Finding 3 — ADMISSION TOCTOU: committed bytes exceeded budget
  (CI run 29436961038; the subject of the DECISION NEEDED):
  - Observed: Σ(weights + materialized pools) = 3,906,848,078 >
    budget 3,900,000,000 (+6.8 MB) for 3 consecutive 10 s samples
    (~t+1491..1511s), then self-healed by the next unload. State:
    llama warm (1232.2 MB) + gemma warm (1168.8) + spec warm (958.3)
    + ttl idle-resident (278.1) + smollm (269.1).
  - Why this is a real race and not measurement noise: used (ledger) ≥
    committed always, and every admission requires used + growth ≤
    budget at decision time — so NO sequence of correctly-priced
    admissions can produce committed > budget. The state above can
    only arise because admission priced against HEARTBEAT-LAGGED
    footprints (1 s cadence, plus load/materialization latency before
    the next heartbeat lands): rapid admissions across workers jointly
    overshot. Worst-case schedule bound: one pool commitment + one
    load footprint (~0.9 GB with this fleet), not the 6.8 MB observed.
  - This contradicts part 2's recorded acceptance ("ledger stays <=
    budget throughout" — measured there under sequential scenarios
    only) and was invisible until this soak measured committed-vs-
    budget continuously under concurrency. The soak's committed gate
    remains STRICT pending the ruling: weakening it to tolerate the
    race would be weakening a test to make it pass (CLAUDE.md).
- Harness gate corrections (design errors mine, same fix cycle):
  - py-smollm RSS slope → 700 MB working-set cap: cross-generation
    slope on an EVICTABLE python worker (11-24 evictions/run, each
    fresh process ramping ~50 → ~470 MB) measures churn phase, not
    leaks (read -17,490 and +4,716 KiB/min on adjacent clean runs).
    Pinned llama keeps its slope gate: one process generation.
  - ttl class: retries through 503 model_loading (a slow CI runner's
    reloads ate 26 attempts leaving 3 successes) but BACKS OFF on
    insufficient_memory — local run D proved retrying through pressure
    keeps the TTL lease alive and removes the lease-expiry release
    valve gemma-burst recovery depends on (one burst starved its full
    120 s window; ttl rejects 2 → 50).
- Corroborations across the fix-cycle runs, all machines: live objects
  drained to exactly 437/1401 at final checkpoints everywhere (CI run 4
  showed two interior 439s that DRAINED — the SSD-flush transient
  reproduced and returned to floor on a second machine, validating that
  characterization); canaries bit-identical in every run (llama 27-31
  samples, spec 17-21); zero crash-restarts in ~120k total requests;
  pinned model never unloaded/rejected; victim selection unpinned-only
  throughout.
- Acceptance (local run E, 30 min, PASS all gates, on the shipped
  contract):
  ```
  $ ./scripts/soak.sh --minutes 30
  duration: 1822s   PASSED (0:30:36)
  live objects: llama-int 437 at ALL SIX checkpoints (gen 0);
    spec-qwen25 1401 at every warm checkpoint through g11
  gateway 59.2 MB final (cap 160), late delta -11.4 MB (cap +20)
  llama-int gated slope -288.6 KiB/min; py-smollm working set ~457 MB
    (cap 700; its window slope read +2,077 KiB/min — would have
    false-failed the old gate)
  committed peak 3.84 / 3.90 GB (no overshoot this run); raw used
    worst +512 MB over (cache-drift gap; six-run envelope +479..+711)
  canaries 29/29 and 21/21 identical; restarts 0; severed 0;
  evictions spec 11 + smollm 11; idle_ttl 19; rejects 87;
  preemptions 8; cancellations 93; prefix 114,441; SSD 2,736;
  spec totals 4,685/4,685
  $ ruff check / ruff format --check python/ tests/e2e -> clean
  ```
- CI verification: run 29404814236 (998df53) remains the fully-green
  matrix proof (all four checks, soak blocking, 57m53s test-macos).
  Run 29436961038 (fe877d9) passed lint/compile-linux/
  test-macos-release and failed ONLY the soak on findings 2 and 3
  above; finding 2's classification is shipped, finding 3 is the open
  ruling. The next CI run reds or greens with the race's dice until
  the ruling lands.
- Deviations: none beyond the open item below.
- DECISION NEEDED: how to close the admission TOCTOU (finding 3) —
  Phase 9's "closed" status from the previous entry is qualified until
  one of these is picked:
  - Option A (fix product-side): reservation-based ledger — admission
    atomically reserves the priced growth/load against the budget and
    releases the reservation when the heartbeat reflects it
    (lifecycle.rs; moderate change to the part 2 machinery, needs its
    own tests). Soak gate stays strict and becomes the regression
    proof. Cost: new engineering on closed part 2 code; benefit: the
    recorded "ledger <= budget" guarantee becomes true under
    concurrency.
  - Option B (accept bounded transience): record the staleness window
    as a designed property alongside the cache-drift gap; the soak
    gates SUSTAINED committed-overshoot (leak-shaped, e.g. > 60 s)
    and counts+reports transients with their magnitude. Cost: the
    part 2 acceptance line is weakened from "always" to "except
    ~heartbeat-window transients bounded by one admission's
    footprint"; benefit: no new code, CI deterministic.
  - Option C (strict + tolerate red): keep the strict gate and accept
    occasional CI failures until Option A is scheduled (e.g. as the
    Phase 10 opener). Honest but makes a blocking gate
    non-deterministic, which invites alert fatigue.
  No option is picked. The branch is otherwise complete: PR #28 holds
  the soak harness, the outcome contract, and this ledger.
- Next: the ruling above; then Phase 10 (jobs, admin UI, packaging,
  docs). Also still open from earlier parts: §8.3 rate-limit/timeout
  BACKLOG, python-worker batching upgrade, continuous-pressure
  eviction for cache drift (envelope now +479..+711 MB measured).

## [2026-07-15] Phase 9 / Part 3 ruling — DONE (Option A: reservation ledger; Phase 9 closed)
- Ruling (PM, this date): Option A from the previous entry's DECISION
  NEEDED — fix the admission TOCTOU product-side with a
  reservation-based ledger; the soak's strict committed-bytes gate
  stands and becomes the regression proof. This entry closes that
  DECISION NEEDED and, with the deterministic CI pass below, closes
  Phase 9.
- What (commit 5c7d73d):
  - Red first: tests/e2e/test_admission.py::
    test_concurrent_admissions_cannot_jointly_overshoot fires two cold
    models' first requests simultaneously under the existing
    DRIFT_BUDGET — the run-29436961038 race as a deterministic repro
    (sub-ms concurrent admissions vs the 1 s heartbeat cadence).
    Pre-fix: both admitted (200/200), joint pool materialization
    overshot the budget. The red run is part of this ledger.
  - lifecycle.rs reservation ledger: admit_request now RESERVES the
    projected growth at decision time under an admission lock
    (non-blocking arithmetic only, never held across await), and every
    admission decision — request growth and load alike — prices against
    charged_bytes() = measured usage + outstanding reservations. A
    racing admission sees the winner's obligation immediately; refusals
    reserve nothing; a second request on the same still-cold pool rides
    the outstanding reservation (the pool materializes once). A
    reservation whose request never materializes (cancel/error) lingers
    conservatively — it only under-reports headroom — and reconciles on
    the next served request or unload.
  - Heartbeat reconciliation (record_pool_materialized, same lock):
    confirmed bytes shift to the measured footprint and the reservation
    shrinks to the still-outstanding growth (commitment −
    materialized), releasing over-reservation difference exactly as the
    ruling prescribed. Materialization that NO reservation covered is
    the alertable condition: tracing::warn +
    kiln_admission_uncovered_bytes_total{model} (bytes), never silent.
    kiln_memory_reserved_bytes gauges the in-flight total; the soak
    reports its peak and gates uncovered == 0.
  - Complete projections, so actual > reserved is truly exceptional:
    - WorkerInfo.kv_pool_commitment_bytes (additive field 16; python
      regen included): the WHOLE worker's lazily-materialized pool cost
      — target + attached draft. The gateway prefers it over the
      target-only kv_bytes_per_block × kv_pool_blocks product, which
      under-projected draft-carrying workers by an entire draft pool
      (201 MB on the soak's qwen2.5 self-pair; the likely main
      contributor to the observed CI overshoot).
    - Load projections carry LOAD_OVERHEAD_MARGIN_BYTES = 64 MiB over
      raw on-disk weight bytes (idle footprints measured 17-33 MB over
      weights across the pinned fleet), so an admission racing a load
      window cannot consume unprojected bytes; the first measured
      heartbeat replaces the whole projection before the load permit
      releases — reserve high, reconcile down.
- Deviations: none.
- Acceptance:
  ```
  RED (pre-fix, this session):
    test_concurrent_admissions_cannot_jointly_overshoot FAILED:
    "exactly one concurrent admission must win: {'right': 200,
     'left': 200}" — the TOCTOU, reproduced in 17 s.
  GREEN (post-fix, local):
    $ uv run --project tests/e2e pytest tests/e2e/test_admission.py
    4 passed in 44.19s   (race repro: one 200 / one 503; committed
    <= budget at every 200 ms sample; reservations drain to 0;
    kiln_admission_uncovered_bytes_total absent on both models)
    $ cargo test -p kiln-gateway --lib -> 58 passed (new: joint-
      overshoot arithmetic, heartbeat reconciliation incl. no-double-
      count, uncovered-growth alerting byte-accuracy)
    $ cargo test --workspace -> green (exit 0, 50 suites)
    $ uv run --project tests/e2e pytest tests/e2e (minus soak)
      -> 87 passed, 2 skipped in 271 s — the measured-bounds
      lifecycle/admission budgets all held under the new projections
      (no recalibration needed: the 64 MiB margin sits inside every
      calibrated boundary)
    $ uv run --project python/kiln_worker_py pytest -> 35 passed
    $ fmt / clippy (default + --no-default-features) / ruff -> clean
  30-minute soak (local run F, the reservation ledger live), PASS all
  gates:
    reservation ledger: peak in-flight 201 MB; uncovered growth 0.0 MB
    committed (weights+pools) peak 3.64 / 3.90 GB — LOWER than every
      pre-fix run (3.71-3.84): admissions now price complete
      obligations, and pool-side races no longer contribute to the raw
      cache-drift overshoot either (worst +384 MB vs +479..+711 across
      the six pre-fix runs; that residual is pure mlx-cache drift, the
      separately-recorded open item)
    live objects 437 at all six checkpoints (llama, gen 0); spec model
      flat per generation across 26 evict/reload cycles (churn is up
      from ~11-17: spec re-warms now honestly need 402 MB, so pressure
      refuses/evicts them more — sane, and every cycle recovered)
    canaries 29/29 and 20/20 bit-identical; restarts 0; severed 0;
    bursts 6/6 recovered; preemptions 8; cancellations 93; prefix
    reused 114,489; SSD writes 2,713; spec totals 4,073/4,073
  ```
- Post-ruling calibration (commit 11cc55e), from the ledger's first CI
  run (29449124438 — the ledger itself was flawless there: peak
  in-flight 403 MB, uncovered 0, committed 3.67/3.90; the soak step
  failed on scenario dynamics):
  - The spec class stands down 20-30 s after an insufficient_memory
    refusal: under honest pricing a spec re-warm needs its whole 402 MB
    double pool, and its 4-8 s retry cadence beat the gemma burst to
    every headroom slot the TTL valve opened — a harness livelock that
    starved one burst on the slow runner. Burst windows widened to
    180 s (a slow-runner recovery legitimately chains eviction ack
    45 s + TTL expiry ≤75 s + load 10-30 s).
  - RSS gates are now absolute working-set caps ONLY (gateway 160 MB,
    pinned worker 1.2 GB, python 700 MB): ten 30-min runs established
    that every derivative measure aliases macOS page-reclaim timing —
    fixed-window slopes read -47,308..+3,370 KiB/min and a mid-run-
    referenced delta -46..+23 MB across runs with identical workloads
    and bit-flat mlx counters; the pinned worker's RSS breathes
    36..846 MB with page cache over its 695 MB weight mmap. Slopes and
    deltas remain computed and reported; fine-grained leak detection
    is owned by the bit-exact live-object gate, flat mlx_active, the
    committed/reservation ledgers, and the 1k-iteration leak suites.
- CI verification (PR #28, commit 11cc55e, run 29456964198): ALL FOUR
  checks pass — test-macos 61m15s with the 30-minute soak BLOCKING and
  green on the runner, the race-repro e2e in the blocking sweep, and
  the reservation ledger live:
  ```
  duration: 1816s   PASS: all gates held
  reservation ledger: peak in-flight 201 MB, uncovered growth 0.0 MB
  committed (weights+pools) peak 3.67 / 3.90 GB — never exceeded
  live objects: llama-int 437 at ALL SIX checkpoints (gen 0);
    spec-qwen25 flat per generation through g11
  raw used worst +575 MB over (mlx-cache drift, the separate open item)
  canaries 28/28 and 16/16 bit-identical; restarts 0; bursts 5/5
  one severed-by-drain 502 at t+724.7s ACCEPTED by its unload-counter
    correlation — the part 3 outcome contract exercised end-to-end on
    real hardware and classified correctly
  interactive during-flood p95 14.77s (slow runner; gate 90 s)
  Advisory golden lane: the ONLY divergence remains
    gemma-3-1b-it-4bit/chat-basic (ADR 0004 pattern; no change).
  ```
  This is the deterministic pass the ruling demanded: the TOCTOU is
  structurally impossible (reservations serialize admission decisions),
  the race repro is a permanent blocking e2e test, and every remaining
  soak outcome class is contract-classified rather than
  timing-dependent.
- Phase 9 closeout (superseding the qualification in the addendum
  entry): parts 1-3 delivered machine memory budgeting, LRU eviction
  with drain + pinning + TTL leases, crash-loop backoff,
  INTERACTIVE/BATCH priorities with preemption-ordered admission,
  per-request admission now backed by a reservation ledger that holds
  "committed <= budget" under real concurrency, and the 30-minute
  full-stack soak as the standing blocking gate proving it all
  composes. Still open, recorded: §8.3 rate-limit/timeout BACKLOG;
  python-worker batching upgrade; continuous-pressure eviction for
  mlx-cache drift (envelope +384 MB worst under the soak scenario with
  pool races eliminated). The Phase 4-9 correctness arc stands closed.
- Next: Phase 10 (jobs, admin UI, packaging, docs) — not started, per
  the task instruction.

## [2026-07-15] Phase 10 / Part 1 — kiln-jobs + gateway admin jobs API — DONE
- What:
  - `crates/kiln-jobs` implemented (was a stub): `download <hf_repo>`
    (resumable HF download, progress as JSON lines on stdout), `quantize
    <path> --bits N --group-size N` (wraps `python -m mlx_lm convert` in the
    new jobs venv `python/kiln_jobs_py` — quantization is not reimplemented),
    `serve --socket <uds>` (long-running job server). Job state in SQLite
    (`~/.kiln/jobs.sqlite` default), per SPEC §9.1.
  - The downloader is a Rust port of the fetch-test-model.sh hardened logic
    with semantics preserved: 30s per-read stall timeout, 4 attempts with
    linear backoff, retryable set {408,425,429,500,502,503,504} vs fatal
    statuses, `.part` resume via HTTP Range with banked bytes (read1
    semantics), restart on Range-ignoring 200, 416 discard, LFS sha256
    verification that never retries atop a corrupt prefix, verified skip of
    present files, `.kiln-revision` marker, HF_ENDPOINT override.
    Extensions: ref→commit-sha resolution up front (interrupt/resume stays
    revision-coherent) and Link-header tree pagination.
  - New additive proto `proto/kiln/v1/jobs.proto` (Jobs: SubmitDownload/
    SubmitQuantize/GetJob/ListJobs) compiled into kiln-proto alongside
    worker.proto — worker.proto wire semantics untouched.
  - Gateway admin API, bare minimum for part 2's UI: POST
    /admin/jobs/download, POST /admin/jobs/quantize, GET /admin/jobs,
    GET /admin/jobs/{id}. Admin bearer auth activates the previously
    parsed-but-unused `auth.admin_token_hash`; kiln-jobs is spawned on
    demand (`server.jobs_argv`, `server.jobs_db`; socket under runtime_dir)
    and proxied over gRPC/UDS with the existing channel plumbing.
- Decisions:
  - Gateway↔jobs IPC = gRPC over UDS via a NEW proto file: SPEC §3 names
    tonic gRPC/UDS as THE IPC layer and protobuf as the IPC serialization;
    zero new gateway dependencies. jobs.proto follows the worker.proto
    freeze discipline (additive only) once shipped.
  - New deps, justified in the commit: rusqlite 0.40 `bundled` (SPEC §9.1
    SQLite state; no system lib), reqwest 0.13 rustls-only (per-read stall
    timeout via read_timeout + Range + redirects; no openssl). Jobs venv
    pins mlx==0.31.1/mlx-lm==0.31.2 — identical to the worker venv pins
    (2026-07-03 option B1) so converter outputs come from the same MLX core.
  - Admin surface fails CLOSED (403 `admin_disabled`) when no
    admin_token_hash is set — deliberate departure from the API-key
    warn-and-stay-open precedent: admin endpoints trigger downloads and
    subprocesses, and SPEC §8.1 says bearer-token gated. API keys never
    grant admin.
  - Jobs execute sequentially (single runner task). Crash recovery: store
    open marks `running` jobs failed ("interrupted; resubmit to resume"),
    `queued` jobs re-enqueue at serve start; resume is file-level via .part.
- Deviations: none.
- Acceptance:
  ```
  Real download, deliberately interrupted + resumed (pinned repo):
    $ kiln-jobs download mlx-community/Qwen3-0.6B-4bit --dest /tmp/kiln-accept-dl
      kill -9 mid-safetensors; banked .part = 53,107,712 bytes
    resubmit, JSON lines:
      {"event":"skip","path":"config.json"}  (+2 more verified skips)
      {"event":"file","path":"model.safetensors","size":335450584,"resume_from":53107712}
      {"event":"done","dest":"/tmp/kiln-accept-dl"}   exit=0
    sha256(model.safetensors) bit-identical to the pinned fetch-script copy
    (main still resolves to the pinned 73e3e38d9813; LFS oid also verified
    in-band before rename). Job store: run1 failed{"interrupted...resubmit
    to resume"}; run2 succeeded{"event":"done"}.
  Real quantization (BF16 -> 4-bit) + loadable:
    $ kiln-jobs quantize ~/.kiln/test-models/smollm2-135m-bf16 --bits 4 --group-size 64
      {"event":"log","line":"[INFO] Quantized model with 4.503 bits per weight."}
      {"event":"done","dest":"/tmp/kiln-accept-quant"}   exit=0
    config.json: model_type=llama, quantization {group_size:64, bits:4}
    (parses via kiln_models::ArchConfig — the gated quantize test);
    served by the REAL stack: gateway + rust worker on the converted dir,
    /readyz 200, POST /v1/chat/completions -> 200, 24 completion tokens,
    finish_reason "length".
  Suites (local, M-series):
    cargo test --workspace -> exit 0 (54 suites, incl. the 8-case stub-hub
      download suite: interrupt+Range-resume, Range-ignored restart, 416
      discard, sha-mismatch discard, retryable-vs-fatal statuses, verified
      skip; store crash recovery; gateway admin auth + proxy translation)
    uv run --project tests/e2e pytest tests/e2e -> 90 passed, 3 skipped
      (new test_admin_jobs.py: gateway spawns kiln-jobs, download job runs
      end-to-end against a local stub hub, files land, queued->succeeded)
    uv run --project python/kiln_worker_py pytest -> 35 passed
    cargo build --workspace --no-default-features -> clean
    fmt / clippy (default and --no-default-features, --all-targets) /
      ruff check + format -> clean
  CI (PR #29, run 29467795313): ALL FOUR checks pass —
    lint 1m37s, compile-linux 1m59s, test-macos-release 6m16s,
    test-macos 1h1m20s with the new steps live on the runner:
      stub-hub download suite in the workspace step (8 tests ok)
      "kiln-jobs quantize (real mlx_lm convert, BF16 -> 4-bit)":
        [INFO] Quantized model with 4.503 bits per weight. -> 1 passed
      E2E incl. test_admin_jobs.py 3/3 PASSED -> 90 passed, 3 skipped
      30-min soak: duration 1812s, PASS all gates
  ```
- Next: Phase 10 part 2 — minimal admin UI (SvelteKit static, embedded via
  rust-embed: models table, load/unload/pin, live stats via SSE, job
  launcher calling /admin/jobs/*), per SPEC §12 Phase 10.

## [2026-07-16] Phase 10 / Part 2 — admin UI + models/stats API + admin-hash hard-fail — DONE
- What:
  - Task A (behavior correction): a present-but-malformed
    `auth.admin_token_hash` is now a hard startup `ConfigError::Invalid`
    (was: warn + silently disable admin), matching the existing
    malformed-config rejections (non-power-of-two block_size, unsupported
    quantization). Unset/empty (the kiln.toml.example placeholder) is
    unchanged: fail-closed 403 `admin_disabled` naming the fix. The unit
    test asserting warn-and-disable now asserts the startup failure — a
    deliberate correction of the tested contract, NOT a weakened test; the
    missing/empty fail-closed assertions are retained unchanged. Auth is
    built before `Supervisor::start`, so the failure spawns no workers.
  - Admin models/stats API (SPEC §8.1): `GET /admin/models` (table +
    machine memory ledger), `POST /admin/models/{id}/load|unload|pin`,
    `GET /admin/stats` (1s SSE snapshots = lifecycle ledger + live worker
    Health/Stats RPCs over the existing UDS channels). Runtime-mutable
    pinning (AtomicBool on the lifecycle slot; not persisted — kiln.toml
    stays boot-time truth); `UnloadReason::Admin`; Failed models refuse
    load/unload with 409 naming the manual reset (admin reset stays out
    of part 2 scope).
  - Admin UI (SPEC §12 Phase 10, §3): `admin/` — one prerendered
    SvelteKit page (static adapter, base `/ui`, no framework beyond
    SvelteKit): models table with load/unload/pin, memory ledger, live
    per-model stats via the SSE stream (streaming fetch — EventSource
    cannot send Authorization), download/quantize job launcher on the
    part 1 API. API 401/403 messages render VERBATIM — the fail-closed
    admin_disabled 403 already names the fix, so "admin token not
    configured" is a visible state, not a silent failure. Served by the
    gateway at `/ui` via rust-embed: debug builds read `admin/build/`
    from disk, release builds embed (single static binary, SPEC §1.1);
    kiln-gateway build.rs creates the gitignored folder so cargo-only
    checkouts compile, `/ui` then 503s naming the npm build command. The
    /ui shell is unauthenticated static code; all data rides the
    bearer-gated /admin API.
  - Browser e2e (`tests/e2e/test_admin_ui.py`, playwright, Chrome
    channel with chromium fallback; `KILN_E2E_REQUIRE_BROWSER=1` in CI
    forbids the skip path): full operator flow through the ACTUAL UI —
    connect → ready → an API completion moves the UI token counter with
    no reload (live-SSE proof, exact-count match) → unload → "unloaded
    (admin)" → load → ready → pin/unpin (cross-checked via API) →
    download job launched in the UI runs to succeeded against the local
    stub hub, files verified on disk. Plus verbatim-403 rendering and an
    embedded-shell check.
  - Found & fixed by that test's first run: the never-ending
    /admin/stats SSE stream wedged axum's graceful shutdown — an open
    dashboard turned SIGTERM into the 20s hard kill, leaking the worker
    process group (caught by the conftest leaked-worker guard). AppState
    now carries an http-shutdown watch flipped when graceful shutdown
    begins; the stats stream ends on it. Verified: SIGTERM with an open
    stream exits rc=0 in <1s; the e2e keeps the dashboard open through
    stack teardown, pinning the regression.
  - CI (test-macos): setup-node (npm cache) + `npm ci && npm run build`
    for admin/ before the e2e step; guarded `playwright install
    chromium` only if the runner image lacks Chrome; e2e step exports
    KILN_E2E_REQUIRE_BROWSER=1.
- Decisions:
  - Empty-string admin_token_hash ≡ unset (disabled fail-closed), only
    non-empty unparseable is the hard error: kiln.toml.example ships
    `admin_token_hash = ""` as the placeholder, and an empty value is
    the TOML idiom for "not set" — the dangerous case is a typo'd real
    hash, which is exactly what now fails loudly.
  - Model actions as POST subroutes (`/admin/models/{id}/load` etc.)
    within SPEC §8.1's "GET/POST /admin/models" latitude; 202 +
    observe-via-status rather than blocking the HTTP call on a ~35s
    drain ladder.
  - Stats SSE assembles live Health/Stats RPCs per tick (500ms per-RPC
    timeout) instead of re-reading prometheus gauges: simpler, truly
    live, and only costs RPCs while a dashboard is open.
  - rust-embed default mode (debug = disk, release = embed): UI
    iteration without gateway recompiles in dev, single binary in
    release. Embedding named by SPEC §3 — not a discretionary dep.
  - Browser = installed Chrome first (present on dev machines + GitHub
    macOS runners; zero download), playwright chromium fallback; skip
    only outside CI.
- Deviations: none.
- Acceptance:
  ```
  Task A:
    $ kiln-gateway --config bad-admin.toml   (admin_token_hash = "not-a-phc")
      kiln-gateway: invalid configuration: auth.admin_token_hash is not a
      valid PHC string (password hash string missing field); hash a token
      with `kiln-gateway hash-key`        exit=1, before any worker spawn
    unset case unchanged: GET /admin/jobs -> 403
      {"code":"admin_disabled","message":"The admin API is disabled: set
       auth.admin_token_hash in kiln.toml (hash a token with `kiln-gateway
       hash-key`)."}
  Operator flow through the real UI (browser e2e, run twice locally):
    tests/e2e/test_admin_ui.py::test_admin_ui_full_operator_flow PASSED
    tests/e2e/test_admin_ui.py::test_admin_ui_surfaces_disabled_admin_verbatim PASSED
    tests/e2e/test_admin_ui.py::test_ui_shell_is_served_embedded PASSED
    (first run of the flow test failed ONLY in teardown — the leaked-worker
     guard caught the SSE/graceful-shutdown bug above; green after the fix)
  Shutdown regression check (manual, debug build):
    SIGTERM with an open /admin/stats stream -> "gateway exited rc=0
    after 0s with SSE stream open"
  Suites (local, M-series):
    cargo test --workspace (KILN_TEST_MODELS set, model-gated tiers incl.)
      -> exit 0, all suites ok (73 gateway lib tests incl. new
         admin_models/ui/auth coverage)
    uv run --project tests/e2e pytest tests/e2e -> 93 passed, 3 skipped
      (was 90 passed in part 1; +3 = the new admin-UI browser tests)
    uv run --project python/kiln_worker_py pytest -> 35 passed
    cargo build --workspace --no-default-features -> clean
    fmt / clippy (default + --no-default-features, --all-targets) /
      ruff check + format -> clean
    npm ci + npm run build (admin/) -> 2.8s warm
  CI (PR #30): ALL FOUR checks pass (run 29477773950) —
    lint 1m6s, compile-linux 1m56s, test-macos-release 5m26s,
    test-macos 1h4m18s (part 1 baseline 1h1m20s: the UI adds ~3 min,
    dominated by browser startup inside the new e2e tests, not the build):
      "Build admin UI": npm ci 50 packages in 1s + vite build ~2s
      "Ensure a browser": runner Chrome present, no chromium download
      E2E with KILN_E2E_REQUIRE_BROWSER=1: test_admin_ui.py 3/3 PASSED
        -> 93 passed, 3 skipped in 9m42s
      30-min soak: PASS (blocking gate)
  ```
- Next: Phase 10 part 3 — packaging + docs (Homebrew formula + launchd
  plist, `kiln` CLI, README/config/API docs), the phase and build close.

## [2026-07-16] Phase 10 / Part 3 — packaging, CLI, docs — DONE; **BUILD CLOSEOUT** (final phase per SPEC §12)
- What:
  - `kiln` CLI (`crates/kiln-cli`, binary `kiln`; zero new external deps —
    reqwest/tokio/serde_json were already workspace deps): `kiln serve`
    execs the sibling `kiln-gateway` with resolved config (`--config` >
    `$KILN_CONFIG` > `./kiln.toml` > `<prefix>/etc/kiln/kiln.toml`);
    `kiln models` is a CLI view of the admin API's `GET /admin/models` —
    the models table (id/worker/status/pinned/ttl/memory) plus the machine
    memory ledger, bearer token from `$KILN_ADMIN_TOKEN`, host/port via
    kiln-gateway's own config loader (new `KilnConfig::load_env_only`) so
    resolution can't drift; `kiln bench` execs `scripts/bench.sh`, args
    passed through verbatim.
  - `scripts/bench.sh` + `bench.py` — the SPEC §11.3 load harness
    CLAUDE.md has referenced since Phase 4 (flagged missing at every gate
    review since): stdlib-only HTTP lanes (single-stream TTFT p50/p95 +
    decode tok/s; batch aggregate tok/s at configurable width) against a
    self-spawned release stack or `--url` a running gateway, results JSON
    to `bench/results/` (gitignored); `--engine` reruns the committed
    ADR 0003 release gate (`kiln-models --test throughput`). Prompts embed
    the request index so the radix cache can't fake prefill; token counts
    come from the server's own usage blocks.
  - `Formula/kiln.rb` + launchd: rustup-pinned toolchain build (the
    rust-toolchain.toml 1.96.1 pin is load-bearing — rust-lang/rust#158830
    — so brew's floating `rust` is deliberately NOT used), admin UI
    npm-built before cargo so rust-embed embeds it, `cargo install
    --locked` for all four binaries off one shared target dir, worker+jobs
    venvs `uv sync --frozen --no-dev` from the repo's committed uv.lock
    pins into libexec, `mlx.metallib` installed next to the binaries,
    starter `etc/kiln/kiln.toml` with venv-pointed argvs, `service do`
    block (brew services) + `packaging/dev.kiln.gateway.plist` template
    for non-brew installs (plutil-linted in CI).
  - User docs: `README.md` (what/install/quickstart-to-completion/CLI/
    admin/doc map), `docs/CONFIGURATION.md` (every kiln.toml field with
    defaults; ADR cross-refs for the non-obvious knobs — kernel flag ↔
    ADR 0002 context, speculative gamma ↔ ADR 0005, deployment-shape
    precondition ↔ ADR 0006; §8.3 rate-limit gap stated plainly),
    `docs/API_COMPAT.md` (per-endpoint supported surface + the recorded
    gaps: forced tool_choice, logprobs, /v1/embeddings, rpm/tpm
    parsed-not-enforced, structured output rust-only, determinism scope).
  - Three real bugs found ONLY by executing the packaging end to end:
    (1) `jobs_argv = ["kiln-jobs", "--venv", <dir>]` — documented in
    kiln.toml.example since part 1 — never parsed (first arg was consumed
    as the subcommand); flags now parse before the subcommand, both orders
    unit-tested. (2) Installed workers died on their first array
    (`mlx_array_new_data returned a null array`): MLX resolves
    mlx.metallib colocated-with-executable then via a compiled-in path
    into the (deleted) build tree; checkout builds never see this because
    target/mlx-c-build/ persists. Formula now installs the metallib into
    bin/. (3) `kiln bench` missed the keg: brew links bin/ (symlinks) but
    not libexec/, and current_exe() is the unresolved symlink;
    exe_relative() now tries raw then canonicalized paths.
  - CI: packaging lint (`HOMEBREW_DEVELOPER=1 brew style` + `plutil
    -lint`) added to test-macos-release; ruff gates extended to scripts/
    (bench.py lives there); CLAUDE.md gates + repo map updated.
- Decisions:
  - `kiln models` surfaces the ADMIN API's list (status/memory ledger),
    not `/v1/models` (ids only) — the operator CLI wants operator data;
    the 403-until-configured admin discipline (part 1) applies unchanged.
  - `kiln bench` wraps scripts/bench.sh (per the task instruction) with a
    dual search path so it works from a checkout and from the installed
    libexec copy; bench.py is stdlib-only so an installed kiln needs no
    venv to benchmark.
  - Formula stable `url` tracks `main` (no tagged release exists;
    version = workspace 0.0.1); a tagged release should pin
    `tag:`/`revision:`. Homebrew ≥4.6 refuses path formulas outside taps
    unless `HOMEBREW_DEVELOPER=1` (HOMEBREW_FORBID_PACKAGES_FROM_PATHS
    defaults true) — the acceptance command runs in developer mode, and
    README documents both that and the `brew tap` route.
  - SPEC §4's layout gains crates/kiln-cli, Formula/, packaging/ — §12
    Phase 10 names the CLI/formula without placing them; recorded here as
    within-latitude, SPEC text untouched.
  - launchd = the formula `service` block (brew services generates and
    loads the plist); the standalone template plist is provided for
    non-brew installs rather than being the primary path.
- Deviations: none in this part. Flagged for the record (SPEC-internal
  inconsistency, no action): §8.1 lists `POST /v1/embeddings` in the API
  surface but no §12 phase ever scheduled it (Phase 11 stretch =
  "embeddings-native"); it is documented as not-implemented in
  API_COMPAT.md rather than silently absent.
- Acceptance (the §12 Phase 10 gate, executed for real on this machine):
  ```
  $ HOMEBREW_DEVELOPER=1 brew install --build-from-source ./Formula/kiln.rb
    (formula = the committed file with url branch: main substituted to this
     PR's branch — the only delta, verified by diff — since the code was
     not yet on main; brew clones the repo, so the local tree is unused)
    ==> rustup toolchain install            (1.96.1 per rust-toolchain.toml)
    ==> npm ci --prefix admin && npm run build --prefix admin
    ==> cargo install --locked x {kiln-cli, kiln-gateway, kiln-worker, kiln-jobs}
    ==> uv sync --frozen --no-dev x {kiln_worker_py, kiln_jobs_py}  (mlx 0.31.1 pins)
    🍺  /opt/homebrew/Cellar/kiln/0.0.1: 11,008 files, 625.9MB, built in 8 minutes
  $ brew test kiln -> pass (kiln --version, hash-key argon2 round-trip, jobs usage)
  Real kiln.toml (pristine etc/kiln/kiln.toml + admin hash + one [[model]]
  block: llama-3.2-1b -> ~/.kiln/test-models/llama-3.2-1b-4bit, worker auto):
  $ kiln serve                              (config resolved from <prefix>/etc)
    /readyz -> 200 {"status":"ready","models":{"llama-3.2-1b":"ready"}}
    POST /v1/chat/completions (greedy, max_tokens 48) -> 200:
      "A kiln is a type of oven or furnace that uses high heat to
       transform and harden various materials..." finish_reason stop,
      usage 46+39 — byte-identical across both install acceptance runs
      (greedy determinism holding through a from-scratch rebuild).
  $ KILN_ADMIN_TOKEN=... kiln models
      ID            WORKER  STATUS  PINNED  TTL  MEMORY
      llama-3.2-1b  rust    ready   no      -    1.2 GiB
      memory: 1.2 GiB used / 12.8 GiB budget (0 B reserved, 16.0 GiB machine)
  GET /ui -> 200 text/html (rust-embed release embed, no admin/ on disk)
  $ kiln bench --model llama-3.2-1b --url http://127.0.0.1:8080 (installed
    libexec harness): single-stream decode 123.8 tok/s — matching the
    Phase 6 recorded post-B' single-stream number exactly; results JSON
    written.
  launchd cycle: brew services start kiln -> launchctl list shows
    homebrew.mxcl.kiln exit-code 0 -> /readyz 200 -> completion "ok"
    served by the service -> brew services stop -> label gone, port
    closed. var/log/kiln.log populated.
  Suites (local, M-series):
    cargo test --workspace (KILN_TEST_MODELS set) -> exit 0
    uv run --project tests/e2e pytest tests/e2e -> 93 passed, 3 skipped
    uv run --project python/kiln_worker_py pytest -> 35 passed
    fmt / clippy (default + --no-default-features, --all-targets) clean;
    cargo build --workspace --no-default-features clean;
    ruff check/format (python/ tests/e2e scripts) clean;
    brew style --formula Formula/kiln.rb -> no offenses; plutil -lint OK
  CI (PR #31, head f944f7c): ALL FOUR checks pass —
    lint 42s, compile-linux 1m2s, test-macos-release 3m57s with the
    new packaging-lint step live on the runner, test-macos 1h6m52s
    (workspace + model-gated suites + quantize + e2e incl. admin-UI
    browser flow + 30-min soak, all green; run 29536246532).
  ```
- **BUILD CLOSEOUT** (SPEC §12: Phase 10 is the last phase; Phase 11 is
  stretch, separate approval — NOT started):
  - Phases 0-10 all closed: 0 scaffold/contract; 1 python worker E2E;
    2 gateway tracer bullet; 3 rust Llama worker + golden harness;
    4 paged KV + continuous batching; 5 radix prefix cache + SSD tier;
    6 Qwen/Gemma + routing (ADR 0002/0003 determinism+throughput bars);
    7 structured output, tools, /v1/messages, paged-attention kernel
    (passed both bars, ships default-off by ruling); 8 speculative
    decoding (ADR 0005 envelope, ADR 0006 deployment shape); 9 memory
    governance + priorities + reservation-ledger admission + standing
    30-min soak gate; 10 jobs + admin UI + packaging + docs (this entry).
  - The keystone guarantees, as they stand at closeout: golden parity
    exact same-device for every fixture model incl. batched/width-16
    (ADR 0004 scope; the sole standing advisory divergence remains
    gemma-3-1b-it-4bit/chat-basic on the FOREIGN-device CI lane, the
    ADR 0004 pattern); greedy determinism invariant under batching,
    prefix cache, preemption, and speculation — enforced by blocking CI
    suites and 28/28+16/16 soak canaries; leak gates (RSS slope + mlx
    live-object counter) blocking on every PR via the 30-min soak.
  - Standing open items, all recorded, none blocking the §12 gate:
    §8.3 rate limits/timeouts parsed-not-enforced (BACKLOG since Phase 2;
    now user-visible in API_COMPAT.md); python-worker batching upgrade
    (SPEC §9.2 "Phase 9 improvement", never scheduled); continuous-
    pressure eviction for mlx-cache drift (Phase 9 soak envelope
    +384 MB worst); /v1/embeddings (Phase 11); ADR 0006 BACKLOG
    weights-byte-ratio guard at drafter attachment; Formula bottling +
    tagged release when the repo cuts one.
- Next: nothing — the SPEC §12 build plan is complete. Phase 11
  (embeddings-native, VLM via python worker, MTP self-draft, distributed
  exploration) requires separate human approval per SPEC.

## [2026-07-21] Post-closeout / Add-Model flow (admin API + UI) — DONE
- What:
  - One "Add Model" flow, HF repo id → loaded and servable with ZERO
    gateway restart. New `POST /admin/models` registers a model into the
    live Phase 9 machinery (registry entry, lifecycle slot, supervision
    task — the exact code paths a configured model boots through, so
    load/unload/pin/TTL/eviction apply immediately) AND appends the
    equivalent `[[model]]` block to kiln.toml in the same locked flow.
  - Runtime-add plumbing: `Registry` and the lifecycle slot map are now
    interior-mutable (RwLock; entries only ever added), entry build
    extracted to `registry::build_entry`, supervision-task spawn
    extracted to a `ModelSpawner` handle shared by boot and runtime
    paths; gateway shutdown awaits runtime-spawned tasks too. New
    `UnloadReason::Registered`: an added model parks as "unloaded
    (registered)" — /readyz counts it settled; the existing load
    endpoint (or a request for it) starts the load.
  - kiln.toml persistence (kiln-gateway/src/config_write.rs): toml_edit
    in-place append — comments, formatting, and unrelated entries
    preserved. Safety: the file is re-read from disk at write time
    (hand edits made while the service runs survive), any parse failure
    refuses loudly before touching disk, a duplicate id already in the
    file is a 409 (never an overwrite), and the edited text must
    re-validate through the real KilnConfig parser before an atomic
    tmp+rename replace (original permissions kept).
  - `GET /admin/models/estimate?path=…`: the plain memory answer before
    a large download — weights bytes (local dir, or the hub tree
    listing; HF_ENDPOINT honored) plus the Phase 9 load-overhead
    margin, against the live ledger (budget/charged/headroom/fits).
  - Admin UI: "add model" section — repo/path + id + worker/pin/TTL,
    "check size" ("needs ~X GB (hub weights) — you have ~Y GB free of Z
    budget"), submit; a not-downloaded model auto-runs the existing
    Phase 10 download job (progress shown inline AND in the jobs
    table), auto-registers on success, then a "load now" button — one
    continuous flow. Docs: CONFIGURATION.md [[model]] intro + README.
- Decisions:
  - Not-downloaded models answer a structured 409 `model_not_downloaded`
    carrying `{download: {repo, dest}}` (dest = kiln-jobs' derivation
    `<model_dir>/<org>--<name>`); the UI orchestrates download → retry.
    Chosen over server-side background orchestration: no new job-watcher
    state, a gateway restart mid-download loses nothing, and the retried
    add resolves naturally (a repo already at its default dest registers
    directly, so the API is also two curl calls for a human).
  - kiln.toml records the RESOLVED local path, not the repo id — the
    registry requires local dirs at boot, so the persisted block must be
    bootable as written. Persist-then-insert ordering: a failed disk
    write leaves the gateway unchanged; a crash between write and
    insert costs a restart-time entry, never a lost one.
  - `[model.speculative]` stays config-file-only (out of add scope).
  - SPEC §8.1 words the admin surface as "GET/POST /admin/models
    (list/load/unload/pin)"; register + estimate recorded here as an
    additive extension within the Phase 10 admin latitude (SPEC text
    untouched).
  - Deps (both already in the build graph): toml_edit 0.22 direct in
    kiln-gateway (was transitive via figment's toml backend; needed for
    the format-preserving edit API; MIT/Apache-2.0) and reqwest (the
    existing workspace dep kiln-jobs uses; rustls-only unchanged) for
    the hub size probe.
- Deviations: none.
- Acceptance:
  ```
  cargo fmt --check                                             clean
  cargo clippy --workspace --all-targets -- -D warnings         clean
  cargo clippy --workspace --all-targets --no-default-features  clean
  cargo build --workspace --no-default-features                 clean
  cargo test --workspace                     exit 0 (238 passed, 0 failed)
  ruff check + format --check (python/ tests/e2e scripts)       clean
  uv run --project tests/e2e pytest tests/e2e
      97 passed, 3 skipped in 366.95s   (was 93+3; the 4 new tests pass)
  New coverage:
  - test_admin_ui.py::test_admin_ui_add_model_full_flow (real browser,
    stub hub serving the REAL qwen3-0.6b-4bit dir from disk): estimate
    rendered from the hub listing, download progress observed, auto-
    registered, "load now" -> status ready over SSE, completion 200 on
    'qwen-added', gateway.poll() is None (zero restarts), and the
    kiln.toml line-diff is exactly one inserted [[model]] block naming
    the downloaded dest (whole flow 22s locally).
  - test_add_model.py::test_add_local_model_live_with_hand_edit_preserved:
    a comment hand-written into kiln.toml while the gateway runs
    survives the concurrent add-write (difflib opcodes: all equal + one
    insert); duplicate id -> 409 model_exists with file untouched;
    file-only duplicate -> 409 config_conflict naming the restart;
    added model load -> ready -> completion 200 on the same process.
  - test_add_model.py::test_add_model_download_flow_over_the_api:
    estimate source=hub matching the stub listing bytes exactly;
    structured 409 with {repo, dest}; standard download job succeeded;
    retried add 201 persisting path == dest, worker preserved.
  - Rust unit: config_write (comment preservation, duplicate-in-file,
    unparseable + re-validation refusals with file untouched, no-model
    file), admin_register (register/persist, both duplicate shapes,
    structured 409 + post-download resolution, 400s, local estimate
    arithmetic, repo-shape detection). Gateway suite 86 tests green.
  ```
- Next: nothing scheduled — the SPEC §12 plan remains complete; this was
  a post-closeout UX task (live add-model). Phase 11 still requires
  separate human approval.

## [2026-07-21] Post-closeout / System-memory-aware admission (field bug fix) — DONE
- What:
  - Fixes a confirmed field finding: on this 16 GB dev machine under
    daily-use load, Qwen2.5-Coder-14B-4bit was admitted under the
    total-RAM budget (budget 13.74 GB, used 11.57 GB) while the OS ran
    ~4.4 GB of active swap — generation measured ~0.44 tok/s, 150-200x
    under the machine's benchmark. The budget is cut from INSTALLED RAM
    and cannot see memory other processes hold.
  - New `kiln-gateway/src/sysmem.rs`: live probe of what the machine can
    actually grant — availability = free+speculative+inactive pages
    (`vm_stat`; disjoint queues), swap-used (`sysctl vm.swapusage`), and
    the kernel's own live pressure signal
    (`kern.memorystatus_vm_pressure_level`: 1/2/4). Shelled out like the
    existing `hw.memsize` read (unsafe stays confined to kiln-mlx).
  - Load admission (supervisor `run_once`): after the budget/eviction
    loop settles, the load is priced against a FRESH probe — refused when
    it would leave less than `memory.min_available_bytes` (new config key,
    default 1 GiB, 0 = gate off) of real availability, or when the kernel
    already reports pressure ≥ warning. Structured refusal: status
    `unloaded (system memory pressure)` (settled for /readyz, retried on
    the next request), `kiln_load_rejects_total{model,constraint}`
    (budget | system_available | system_pressure — the budget-reject path
    now counts too), and an ERROR log carrying every number the decision
    used. No eviction on a system refusal: the budget check owns
    Kiln-caused contention; a system shortfall is external, and evicting
    our own fleet against a laggy kernel signal would spiral.
  - Request admission (`admit_request`): effective headroom is now
    min(budget − charged, available − floor − outstanding reservations),
    from a cached snapshot the supervisors' 1s health polls keep ≤2s old
    (background `spawn_blocking`, one probe in flight); pool growth is
    refused outright under kernel pressure. Warm pools (zero growth) are
    never gated — live traffic keeps serving. `MemoryDenial` gained a
    `constraint` field; the 503 `insufficient_memory` message names which
    bound refused. Missing snapshot fails OPEN to budget-only (never a
    refusal on absent data).
  - Observability: gauges `kiln_system_available_bytes`,
    `kiln_system_swap_used_bytes`, `kiln_system_pressure_level`; the
    admin memory ledger and stats SSE carry a `system` object; the
    startup budget log line includes the live snapshot;
    `GET /admin/models/estimate` now answers `fits_budget`/`fits_system`
    with `fits` the conjunction (admin UI message distinguishes "unload
    something" from "the machine itself is low on free memory").
  - Deliberately NOT gated on swap-allocated: macOS keeps swap around
    long after pressure passes (this machine idles at ~2.5 GB used at
    pressure level normal, and grew it to ~4 GB during cargo builds while
    still normal). Gating on it would refuse loads on machines with
    gigabytes genuinely free; the pressure level is the honest version of
    "swap is active", and low availability catches the rest.
  - Tests: 4 sysmem parser/probe unit tests; 4 lifecycle system-gate unit
    tests via injected snapshots (availability bound with reservations,
    request-path min() bound + recovery, pressure refusal incl. the
    swap-active shape + warm-pool passthrough, fail-open); new e2e
    tests/e2e/test_system_memory.py — a real 3 GiB touched-page hog makes
    the boot load refuse (structured status + counter, nothing charged,
    /readyz settled 200) and the IDENTICAL config admits and serves the
    moment the hog exits; plus the estimate/system-gate surface.
  - Also fixed en route: admin_models' test helper named temp dirs by
    SystemTime nanos — µs clock granularity let concurrent tests collide
    on one dir, and Lifecycle::new's new ~20 ms probe widened the window
    until one test's teardown deleted another's dir mid-setup (flaked
    ~1-in-3 full-lib runs). Now a process-wide atomic counter.
- Decisions:
  - Availability formula free+speculative+inactive (psutil's macOS
    "available"): slightly overstates (dirty anon pages in the inactive
    queue still need compression) — `min_available_bytes` is the guard
    band. Purgeable excluded (overlaps the active/inactive queues).
  - Fresh probe per load (loads are rare, serialized, and worth 2 process
    spawns); cached ≤2s snapshot per request (fork/exec has no place on
    the request path). Probe failure fails open with a warning — an odd
    environment degrades to exactly the pre-fix behavior, never an
    outage.
  - The e2e hog starts FIRST and the floor anchors to hog-resident
    availability: the first version measured before the hog and proved
    macOS absorbs big allocations partly by compressing other pages
    (availability fell < 1.5 GiB for a 3 GiB hog → no refusal). Only the
    hog-resident baseline and the release direction (~3 GiB back to the
    free queue at exit) are deterministic.
  - SPEC §2.3 text untouched (same additive-latitude treatment as the
    add-model entry); it already gestures at "minus a fixed floor", and
    the system gate is that floor made real against the live machine.
- Deviations: none.
- CI shapes (investigated per acceptance): the full e2e suite runs on
  GitHub `macos-14` runners — 7 GB VMs with run-to-run memory variance.
  The hog test needs hog (3 GiB) + projection + 2 GiB measured available
  and SELF-SKIPS below that with a message naming this entry, so it is
  effectively dev-hardware-only; the estimate/gauge e2e and all unit
  tests run everywhere. What CI cannot simulate at all — the kernel
  actually reporting pressure ≥ warning — is pinned by injected-probe
  unit tests and covered by the real-hardware verification below. The
  existing suite runs with the default 1 GiB floor ON (runners need
  ~1.4-1.8 GB available per tiny-model load; the
  kiln_system_available_bytes gauge and the startup log line now give CI
  logs the diagnostic if a runner ever runs that tight).
- Manually verified on the real hardware that found the bug (16 GB dev
  machine, swap already 2.7 GB allocated — the "admission while swap is
  active" scenario): a real gateway on the ACTUAL
  Qwen2.5-Coder-14B-Instruct-4bit checkpoint with pure default config
  refused the exact load the old code admitted:
  ```
  budget_bytes: 13743895347  (the finding's 13.74 GB, fraction 0.8 of 17.18 GB)
  load rejected: ... projected_bytes: 8376603097,
    system_available_bytes: 8285700096, min_available_bytes: 1073741824,
    swap_used_bytes: 2715612610, pressure_level: 1,
    constraint: "system_available"
  /readyz: {"status":"ready","models":{"qwen2.5-coder-14b-4bit":"unloaded (system memory pressure)"}}
  kiln_load_rejects_total{constraint="system_available",...} 1
  kiln_memory_used_bytes 0
  ```
- Acceptance:
  ```
  cargo fmt --check                                             clean
  cargo clippy --workspace --all-targets -- -D warnings         clean
  cargo clippy --workspace --all-targets --no-default-features  clean
  cargo build --workspace --no-default-features                 clean
  cargo test --workspace                    246 passed, 0 failed (was 238)
  cargo test -p kiln-gateway --lib          8 consecutive runs green
                                            (temp-dir flake fixed + verified)
  ruff check + format --check (python/ tests/e2e scripts)       clean
  uv run --project tests/e2e pytest tests/e2e
      99 passed, 3 skipped in 699s   (was 97+3; both new tests pass —
      every Phase 9 lifecycle/admission/priority scenario green under
      the new gate)
  tests/e2e/test_system_memory.py::test_load_refused_under_real_memory_pressure_and_recovers PASSED
  tests/e2e/test_system_memory.py::test_estimate_reports_the_system_gate PASSED
  ./scripts/soak.sh   (30-minute Phase 9 leak/correctness gate)
      PASS: all gates held — 1 passed in 1828s (30:28).
      committed peak 3.67 GB of 3.90 GB budget; reservation ledger peak
      403 MB in flight, uncovered growth 0.0 MB (gated at 0); interactive
      p50 0.29s / p95 0.44s (during floods p95 8.01s); spec acceptance
      1.00; 46 evictions + 22 idle-TTL unloads + 27 request-admission
      503s exercised, 0 crashes, 0 over_budget, 0 system_memory refusals
      (availability ~8 GB against the 1 GiB floor throughout) — the new
      gate changed no soak behavior.
  ```
- Next: nothing scheduled — SPEC §12 remains complete; this was a
  field-bug fix inside Phase 9's admission machinery. Phase 11 still
  requires separate human approval.
