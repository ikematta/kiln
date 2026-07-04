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
