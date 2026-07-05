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
