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
