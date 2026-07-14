# KILN — Technical Specification & Build Plan

**A production-grade LLM inference server for Apple Silicon.**
Rust control plane and data plane over MLX (via `mlx-c` FFI), with process-isolated model workers, continuous batching, paged + SSD-tiered KV cache, first-class speculative decoding, and OpenAI/Anthropic-compatible APIs.

Working name: **Kiln** (rename freely; used consistently below so the agent has a stable identifier).

Document version: 1.0 — 2026-07-02
Intended use: this document is the single source of truth for an AI coding agent (Claude Fable 5 via Claude Code) building the system in a task loop. Section 12 defines the loop protocol. Every phase task has explicit acceptance criteria the agent must satisfy before marking a task done.

---

## 1. Goals & Non-Goals

### 1.1 Goals
1. Serve multiple LLMs concurrently on a single Apple Silicon machine with continuous batching, matching or exceeding mlx-lm/oMLX-class throughput while eliminating Python from the token loop for supported architectures.
2. Process-per-model isolation: a crashing model worker never takes down the gateway or other models; model eviction returns memory to the OS deterministically (process exit, not GC).
3. Two-tier worker system: native Rust workers (mlx-c FFI) for the high-traffic architectures (Llama-family, Qwen2/2.5/3, Gemma2/3), plus a thin Python worker wrapping mlx-lm as a compatibility escape hatch — both behind the identical worker protocol.
4. Paged KV cache with prefix sharing (radix tree), copy-on-write, and an SSD cold tier with persistence across restarts.
5. Speculative decoding (draft-model and self-drafting/MTP-style) as a scheduler-native capability that composes with batching and paging — not a separate engine.
6. OpenAI (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`) and Anthropic (`/v1/messages`) compatible APIs with SSE streaming, tool calling, and structured output.
7. Standard observability: Prometheus metrics endpoint, OpenTelemetry traces, structured JSON logs.
8. Single static gateway binary + worker binaries; installable via Homebrew; runnable as a launchd service. No venv gymnastics for the core path.

### 1.2 Non-Goals (v1)
- No training/finetuning. Inference only.
- No multi-node distributed serving (design the worker protocol so it doesn't preclude it; do not build it).
- No macOS GUI app in v1 (CLI + web admin only; SwiftUI supervisor is a v2 item).
- No image *generation*; VLM (image input) support is Phase 11 stretch, via the Python worker only.
- No Linux/CUDA support. macOS 14+ on Apple Silicon only. CI may compile-check on Linux but tests requiring Metal run on macOS.

### 1.3 Explicit lessons from oMLX (anti-goals)
- **No runtime monkey-patching** of third-party libraries. Model-specific behavior lives in Kiln-owned code (Rust model impls, or adapter code inside the Python worker). If mlx-lm needs a fix, vendor the file or pin+patch via a git patch queue applied at build time — never mutate classes at import time.
- **No single-process monolith.** Gateway, workers, and job runner are separate OS processes.
- **No feature silos.** Speculative decoding, paging, prefix cache, and batching must compose. Any feature that "manages its own cache" and "falls back to another engine" is a design rejection.
- **No rlimit-only "sandboxes."** Untrusted code execution (eval harness, later) uses real isolation (out of scope for v1; the eval harness only computes metrics, never executes model-generated code).

---

## 2. System Architecture

```
                                   ┌─────────────────────────────┐
   OpenAI / Anthropic clients ───► │        kiln-gateway          │
   (HTTP + SSE)                    │  axum · auth · rate-limit    │
                                   │  API adapters · tokenizer*   │
                                   │  router · admission control  │
                                   │  metrics · traces · logs     │
                                   └──────┬───────────┬──────────┘
                                          │ gRPC/UDS  │ gRPC/UDS
                              ┌───────────▼───┐   ┌───▼──────────────┐
                              │ kiln-worker-rs │   │ kiln-worker-py   │
                              │ (per model)    │   │ (per model)      │
                              │ kiln-engine:   │   │ mlx-lm wrapper   │
                              │  batch sched   │   │ same proto       │
                              │  paged KV      │   └──────────────────┘
                              │  prefix radix  │
                              │  spec decode   │        ┌────────────┐
                              │ kiln-mlx FFI   │        │ kiln-jobs  │
                              │  → mlx-c/Metal │        │ downloads, │
                              └───────┬────────┘        │ quantize   │
                                      │                 └─────┬──────┘
                              ┌───────▼────────┐              │
                              │ SSD block store │        gateway API
                              │ (per model dir) │
                              └────────────────┘
   Admin SPA (static) ──────► gateway /admin/* API
```

### 2.1 Process model
- **kiln-gateway** (one process): terminates HTTP, translates external APIs to the internal request form, tokenizes/detokenizes (Rust `tokenizers` crate) for workers that request gateway-side tokenization, routes requests to workers, enforces auth/quotas, supervises worker lifecycle (spawn, health-check, kill/evict), aggregates metrics.
- **kiln-worker-rs / kiln-worker-py** (one process per loaded model): owns the model weights, the batching engine, KV block manager, prefix cache, and SSD tier for that model. Speaks the worker protocol (Section 5) over a Unix domain socket.
- **kiln-jobs** (one process, on demand): Hugging Face downloads, quantization jobs, model card generation. Talks to the gateway's admin API; never linked into the serving path.

Rationale: batching decisions require intimate knowledge of the model's KV layout and memory, so the batching engine lives **in the worker** (as a reusable crate, `kiln-engine`, embedded by the Rust worker; the Python worker delegates batching to mlx-lm's batch generator behind the same protocol). The gateway does routing, queueing, admission, and supervision only. This is the TGI shape adapted to per-model process isolation.

### 2.2 Model lifecycle
Gateway maintains a `ModelSupervisor`:
- `load(model_id)` → spawn worker process with model path + config; wait for `Ready` health status; register socket.
- LRU eviction when projected memory exceeds budget: send `Drain` (stop admitting, finish in-flight), then `SIGTERM`, then `SIGKILL` after grace. Memory reclamation is guaranteed by process exit.
- Pinning (never evict) and TTL (auto-unload after idle) per model via config.
- Crash handling: on unexpected worker exit, in-flight requests get 502 with a retriable error code; model marked `Crashed` with exponential backoff before auto-reload (max 3 attempts, then requires manual reset).

### 2.3 Memory accounting
Worker reports, in every heartbeat: weights bytes, KV pool bytes (allocated/used), Metal cache bytes (`mlx_get_cache_memory` equivalents via mlx-c), peak bytes. Gateway keeps a machine-level budget = `total_unified_memory × configurable_fraction (default 0.80)` minus a fixed floor. Admission control at two levels: gateway rejects `load()` that would exceed budget; worker rejects/queues requests whose prefill projection exceeds its local headroom (Section 6.4).

---

## 3. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Language (core) | Rust, stable toolchain, edition 2024 | workspace: gateway, engine, worker, mlx FFI, proto, jobs |
| MLX binding | `kiln-mlx` crate: hand-rolled FFI over **mlx-c** | Do NOT depend on third-party `mlx-rs` for the core; we need exact control over streams, eval, custom kernels. `mlx-c` is fetched + built via `cmake` in `build.rs` (vendored as git submodule, pinned commit). |
| HTTP | `axum` + `tower` (+ `tower-http` for CORS, trace, limits) | SSE via `axum::response::sse` |
| IPC | `tonic` (gRPC) over Unix domain sockets | Streaming RPCs for token delivery. UDS path: `$KILN_RUNTIME_DIR/worker-<model_hash>.sock` |
| Serialization | protobuf (`prost`) for IPC; `serde_json` for HTTP | Proto files are the contract; both workers generate from the same `.proto` |
| Tokenization | `tokenizers` (HF Rust crate) | Loaded from `tokenizer.json`; chat templates via `minijinja` (Jinja2-compatible) |
| Model files | `safetensors` crate, mmap loading | quantized weights per mlx-lm conventions (see 7.3) |
| Structured output | `llguidance` | logit masking inside the worker decode step |
| Async runtime | `tokio` | gateway + workers |
| Python worker | Python 3.12, `uv`-managed venv, `grpcio`, `mlx-lm` (pinned) | thin; no FastAPI, no HTTP — gRPC only |
| Metrics | `prometheus` crate → `/metrics` | worker metrics scraped by gateway, re-exported with `model` label |
| Tracing | `tracing` + `tracing-opentelemetry`, OTLP exporter (optional) | span per request: queue→prefill→decode→finish |
| Config | single TOML (`kiln.toml`), `figment` for env overrides | schema in Section 10 |
| Admin UI | SvelteKit (static adapter) served by gateway from embedded assets (`rust-embed`) | Phase 10; keep minimal |
| Testing | `cargo test`, `insta` snapshots, `criterion` benches; Python: `pytest` | golden-token parity harness (Section 11.2) |
| CI | GitHub Actions: macOS-14 arm64 runner for Metal tests; ubuntu for lint/compile-check (`--no-default-features`) | `cargo clippy -D warnings`, `cargo fmt --check`, `ruff` for Python |
| Packaging | `cargo dist` or plain release script → Homebrew formula; launchd plist | Phase 10 |

Rust crate policy for the agent: prefer std/tokio/well-known crates; any new dependency must be justified in the PR description; no crates with git dependencies except the vendored mlx-c submodule.

---

## 4. Repository Layout

```
kiln/
├── Cargo.toml                 # workspace
├── kiln.toml.example
├── CLAUDE.md                  # agent operating manual (Section 12.2)
├── PROGRESS.md                # agent-maintained task ledger (Section 12.3)
├── docs/
│   ├── SPEC.md                # this document
│   └── decisions/             # ADRs, one file per irreversible decision
├── proto/
│   └── kiln/v1/worker.proto   # THE contract (Section 5)
├── crates/
│   ├── kiln-proto/            # prost-generated types + shared enums
│   ├── kiln-mlx/              # unsafe FFI over mlx-c + safe wrappers (Array, Stream, Compile, FastOps)
│   │   └── vendor/mlx-c/      # git submodule, pinned
│   ├── kiln-models/           # model impls: llama.rs, qwen2.rs, qwen3.rs, gemma2.rs, gemma3.rs + config parsing
│   ├── kiln-engine/           # batching scheduler, block manager, radix prefix cache, sampler, spec-decode, ssd tier
│   ├── kiln-worker/           # binary: gRPC server wiring kiln-engine + kiln-models
│   ├── kiln-gateway/          # binary: axum app, adapters, router, supervisor, metrics
│   ├── kiln-jobs/             # binary: HF download + quantization job runner
│   └── kiln-tokenize/         # tokenizer + chat template + tool-call parsing (shared by gateway)
├── python/
│   └── kiln_worker_py/        # mlx-lm fallback worker (gRPC server, same proto)
├── admin/                     # SvelteKit app (Phase 10)
├── tests/
│   ├── golden/                # parity fixtures: prompts + expected token ids per model per config
│   └── e2e/                   # black-box HTTP tests against a running stack (Python, pytest)
└── scripts/
    ├── fetch-test-model.sh    # downloads pinned tiny models for CI (see 11.1)
    └── bench.sh               # throughput/latency benchmark harness
```

---

## 5. Worker Protocol (`proto/kiln/v1/worker.proto`)

The single contract between gateway and any worker. Request-level granularity (workers own batching internally). Sketch — the agent finalizes field numbering in Phase 0 and MUST NOT change wire semantics after Phase 2 without an ADR.

```proto
service Worker {
  rpc GetInfo(InfoRequest) returns (WorkerInfo);            // model id, arch, ctx len, tokenizer mode, capabilities
  rpc Health(HealthRequest) returns (HealthStatus);          // Ready|Loading|Draining|Unhealthy + memory report
  rpc Submit(SubmitRequest) returns (stream TokenEvent);     // the hot path
  rpc Cancel(CancelRequest) returns (CancelAck);
  rpc Drain(DrainRequest) returns (DrainAck);
  rpc Stats(StatsRequest) returns (WorkerStats);             // prometheus-friendly counters
  rpc Tokenize(TokenizeRequest) returns (TokenizeResponse);  // for workers that own tokenization (python worker)
}

message SubmitRequest {
  string request_id = 1;
  oneof input { TokenIds token_ids = 2; string raw_text = 3; }   // rust worker: gateway pre-tokenizes; py worker: raw text ok
  SamplingParams sampling = 4;       // temp, top_p, top_k, min_p, repetition penalties, seed
  StoppingParams stopping = 5;       // max_tokens, stop token ids, stop strings
  GrammarSpec grammar = 6;           // optional: json_schema | lark | regex (llguidance)
  Priority priority = 7;             // INTERACTIVE | BATCH
  uint64 prefix_hint = 8;            // optional prefix-cache hash from gateway (advisory)
}

message TokenEvent {
  oneof event {
    TokenChunk tokens = 1;           // token ids + optional detok text + logprobs
    RequestAdmitted admitted = 2;    // queue position, prefill estimate
    PrefixCacheHit cache = 3;        // tokens reused (observability)
    Finished finished = 4;           // finish_reason: stop|length|cancelled|error, usage counts, timings
  }
}
```

Capabilities in `WorkerInfo` (bitflags): `LOGPROBS`, `GRAMMAR`, `SPECULATIVE`, `PREFIX_CACHE`, `SSD_TIER`, `EMBEDDINGS`, `VISION`. Gateway feature-gates API behavior on these — the Python worker may lack `GRAMMAR` in v1; the gateway then returns 400 for structured-output requests routed to it.

Design rule: everything the gateway needs for the OpenAI/Anthropic responses (usage counts, finish reasons, timing) must arrive via `Finished`. The gateway never reaches into a worker.

---

## 6. `kiln-engine` — Batching, Paging, Scheduling (the core crate)

### 6.1 Request lifecycle inside a worker
`WAITING → PREFILLING → DECODING → FINISHED`, with `PREEMPTED` (blocks freed, request returns to WAITING with its generated tokens retained for re-prefill) under memory pressure. Preemption policy: lowest priority first, then most-recently-admitted.

### 6.2 Continuous batching loop
Single engine loop task owns the Metal stream (all MLX ops issued from one thread — mirrors MLX's stream semantics; enforced by making `kiln-mlx::Stream` `!Send` and confining it to the engine thread).

Each iteration:
1. Admit from waiting queue while token budget allows (`max_batch_tokens`, default 8192) using chunked prefill: long prompts prefill in chunks of `prefill_chunk` (default 2048) interleaved with decode steps so decode latency stays bounded.
2. Build the step: concatenated decode tokens for all DECODING requests + at most one prefill chunk.
3. Forward pass → logits; apply per-request logit processors (grammar masks, penalties); sample; append tokens; emit `TokenChunk`s (detokenization incremental, handling byte-fallback correctly).
4. Check stops; release blocks of finished requests (return to prefix cache, refcounted, not freed).
5. Every N iterations: run cache maintenance (SSD flush queue, eviction), update stats.

Target: steady-state decode step overhead outside the MLX forward call < 200µs at batch 16 (measure with criterion in Phase 4).

### 6.3 Paged KV + radix prefix cache
- Block size: 32 tokens (configurable, power of two). Per-layer K and V pools as preallocated MLX arrays; block tables map request → block indices; attention uses gather-based paged attention (see 7.4 for kernel strategy).
- Radix tree keyed on token-id sequences (block-aligned). On admit: longest-prefix match → reuse blocks (refcount++), skip those tokens in prefill. Copy-on-write when a shared block would be written.
- Eviction: leaf blocks with refcount 0, LRU. Evicted-to-SSD rather than dropped when the SSD tier is enabled.

### 6.4 SSD cold tier
Per-model directory `$KILN_CACHE_DIR/<model_hash>/blocks/`. Format: one file per block group (64 blocks) — a fixed-layout binary slab (header: magic, version, model fingerprint = hash of weights digest + arch config + dtype + block size; then raw K/V bytes per layer). No safetensors on the hot path (metadata parsing cost); write a `manifest.json` index rebuilt on startup by scanning headers. Async writes via a dedicated tokio blocking pool; reads are mmap + copy into pool blocks. Strict fingerprint check: mismatch → ignore file. LRU cap by bytes (`ssd_cache_max_gb`).
Persistence across restarts is a feature: on worker start, the radix tree is warm-loadable lazily (blocks pulled from SSD on prefix hit).

### 6.5 Speculative decoding (scheduler-native)
Two modes behind one abstraction, `Drafter`:
- `DraftModel(path)`: small model loaded in the same worker process (own weight set, shares nothing but the Metal device); proposes γ tokens (default 4).
- `SelfDraft` / MTP-style heads: deferred to Phase 8 stretch; interface reserved.
Verify: single target-model forward over draft tokens for each speculating request within the normal batch step (speculative requests contribute `γ+1` positions to the step budget). Accept longest agreeing prefix + bonus token; on rejection, roll back = release the speculative blocks and truncate the block table (paging makes rollback O(1) — this is the composition win over oMLX). Speculation auto-disables per-request when batch size > `spec_max_batch` (default 4) since batching already saturates the GPU.

> **BACKLOG:** attachment-time weights-byte-ratio guard — the worker knows draft and target weight byte counts at drafter attachment and should warn (or reject, behind config) when the ratio predicts a throughput loss: ~0.65 measured ratio loses at ANY acceptance rate, and the acceptance auto-disable structurally cannot catch cost-ratio losses; the profitable deployment shape is ratio ≈ 0.05–0.1 (ADR 0006; PROGRESS.md 2026-07-14).

### 6.6 Sampling
Implemented on-GPU with MLX ops where possible (temperature scale, top-k via `mlx_topk`, top-p via sorted cumsum mask, min-p), categorical draw via `mlx_random_categorical` with per-request keys derived from seed; repetition/frequency/presence penalties applied on gathered logits for the request's recent window. Deterministic given seed (document that determinism holds per-worker-version, not across releases).

---

## 7. `kiln-mlx` and `kiln-models`

### 7.1 FFI layer rules
- `kiln-mlx/src/sys.rs`: raw `extern "C"` bindings generated by `bindgen` against vendored mlx-c headers at a pinned commit (record commit hash in `docs/decisions/0001-mlx-c-pin.md`).
- Safe wrapper types: `Array` (owns `mlx_array`, `Drop` → `mlx_array_free`; `Clone` via `mlx_array_set` semantics — document aliasing), `Stream`, `Device`, `Closure`, `CompiledFn`. Every fallible mlx-c call (returns `int`) maps to `Result<_, MlxError>`; install a custom error handler via `mlx_set_error_handler` that records the message thread-locally and returns instead of exiting (this is mandatory — the default handler calls `exit()`).
- Invariant: **every `*_new` has exactly one `*_free`**; enforce with `Drop` impls and a debug-mode allocation counter; add a CI test that runs the engine loop 1k iterations under a leak counter assertion.
- Lazy-eval discipline: the engine calls `mlx_eval`/`mlx_async_eval` only at step boundaries on the outputs it needs (sampled token ids); never call `item()`-style reads mid-graph.
- Use `mlx_fast_scaled_dot_product_attention` for attention; `mlx_compile` the per-layer forward closures where profiling shows wins (Phase 7 optimization task, not before).

### 7.2 Model implementations (`kiln-models`)
Each architecture is a plain Rust module implementing:
```rust
trait Model {
    fn config(&self) -> &ModelConfig;                    // parsed from config.json
    fn forward(&mut self, batch: &StepBatch, kv: &mut PagedKv, stream: &Stream) -> Result<Array>; // logits [n_positions, vocab]
}
```
v1 architectures, in build order: **Llama** (covers Llama 2/3/3.x, Mistral, and most llamafied models), **Qwen2/2.5**, **Qwen3** (incl. GQA + qk-norm variants), **Gemma 2**, **Gemma 3 (text)**. Config parsing must follow mlx-lm's `config.json` conventions exactly (that's the ecosystem contract), including `rope_scaling` variants (linear, llama3, yarn).

### 7.3 Quantized weights
Support mlx-lm quantization format: affine group quantization (`quantization: {group_size, bits}` in config, weights stored as packed `uint32` + `scales` + `biases`). Matmuls via `mlx_quantized_matmul`. Support 4-bit and 8-bit, group sizes 32/64/128. FP16/BF16 unquantized also supported. Anything else → route model to Python worker.

### 7.4 Paged attention strategy
Phase 4 v0: gather KV blocks into contiguous per-request views (`mlx_gather`/`mlx_take`) then call `mlx_fast_scaled_dot_product_attention` — correct first, fast enough for moderate contexts. Phase 7: custom Metal kernel via `mlx_fast_metal_kernel_new` implementing block-table-aware attention (vLLM-style paged attention) to eliminate the gather copy. Keep both paths behind a config flag; parity-test the kernel against the gather path.

---

## 8. `kiln-gateway`

### 8.1 API surface
- OpenAI: `POST /v1/chat/completions` (stream + non-stream, tools, `response_format: json_schema`, logprobs), `POST /v1/completions`, `GET /v1/models`, `POST /v1/embeddings` (routes to a Python worker with an embeddings-capable model; v1 supports this only via `kiln-worker-py`).
- Anthropic: `POST /v1/messages` (stream + non-stream, tool use, system prompts, thinking passthrough as `thinking` content blocks when the model emits `<think>` — parsing in `kiln-tokenize`).
- Admin (bearer-token gated, separate token from API keys): `GET/POST /admin/models` (list/load/unload/pin), `GET /admin/stats`, `POST /admin/jobs/*` (proxied to kiln-jobs), `GET /metrics` (Prometheus, unauthenticated on localhost only by default).
- Health: `GET /healthz`, `GET /readyz`.

### 8.2 Request path
parse → validate → resolve model → apply chat template (minijinja, template from tokenizer_config/chat_template.jinja; templates for supported models vendored + tested) → tokenize → admission check → `Submit` to worker → translate `TokenEvent` stream to OpenAI/Anthropic SSE frames → final usage block. Tool-call extraction: streaming parsers per model family (Hermes-style `<tool_call>` JSON, Llama 3 python-tag, Qwen XML) in `kiln-tokenize`, selected by model metadata; emitted as proper `tool_calls` deltas.

### 8.3 Cross-cutting
API keys in config (hashed at rest, `argon2`); per-key rate limits (token bucket: requests/min and tokens/min) via `tower` middleware; request size limits; timeouts (TTFT timeout and total). Structured request logs with request_id propagated to worker spans.

> **BACKLOG:** per-key rate limits (rpm/tpm) and TTFT/total timeouts are parsed from kiln.toml since Phase 2 but UNENFORCED — implement in Phase 9 alongside priority/admission control (PROGRESS.md 2026-07-04).

---

## 9. `kiln-jobs` and Python worker

### 9.1 kiln-jobs
CLI + long-running job server: `kiln-jobs download <hf_repo>` (resumable, `hf_hub` REST, progress to stdout JSON lines), `kiln-jobs quantize <path> --bits 4 --group-size 64` (v1: shells out to `python -m mlx_lm convert` in the jobs venv — do not reimplement quantization in Rust in v1; wrap it). Job state in a SQLite file. Gateway admin API proxies job submission/status.

### 9.2 kiln-worker-py
~800 lines target. gRPC servicer implementing `worker.proto`; wraps `mlx_lm.load` + mlx-lm's batch generation (or a simple sequential loop for v1 — correctness first; batching in the Python worker is a Phase 9 improvement, since its whole purpose is compatibility, not peak throughput). Owns tokenization (`raw_text` input mode). Pinned `mlx-lm` version in `pyproject.toml`; a `make update-mlx-lm` task runs the golden suite before allowing a bump. Absolutely no monkey-patching: if a model needs a fix, it goes in an adapter module keyed by `model_type`.

---

## 10. Configuration (`kiln.toml`)

```toml
[server]
host = "127.0.0.1"; port = 8080
runtime_dir = "~/.kiln/run"; cache_dir = "~/.kiln/cache"; model_dir = "~/.kiln/models"

[memory]
budget_fraction = 0.80          # of unified memory

[defaults]
max_batch_tokens = 8192; prefill_chunk = 2048; block_size = 32
ssd_cache_max_gb = 64; ssd_tier = true

[[model]]
id = "qwen3-14b-4bit"
path = "mlx-community/Qwen3-14B-4bit"   # HF id or local path
worker = "rust"                          # rust | python | auto (auto = rust if arch supported)
pinned = true
ttl_seconds = 0                          # 0 = never auto-unload
[model.speculative]
draft = "mlx-community/Qwen3-0.6B-4bit"; gamma = 4

[auth]
admin_token_hash = "..."
[[auth.api_keys]]
name = "isaac"; key_hash = "..."; rpm = 600; tpm = 500000
```

---

## 11. Testing & Quality Strategy

### 11.1 Test models (pinned, tiny, CI-downloadable)
- `mlx-community/Llama-3.2-1B-Instruct-4bit` (rust worker primary)
- `mlx-community/Qwen3-0.6B-4bit` (also serves as draft model in spec-decode tests)
- `mlx-community/gemma-3-1b-it-4bit`
- One BF16 unquantized tiny model for dtype-path coverage.
`scripts/fetch-test-model.sh` pins exact revisions. CI caches them.

### 11.2 Golden-token parity harness (the keystone test)
`tests/golden/` fixtures: `{model, prompt, sampling: greedy, max_tokens: 64} → expected token ids`, generated once by a pinned mlx-lm reference script (`scripts/gen-golden.py`) and committed. The Rust worker must reproduce greedy token ids **exactly** for every fixture **on the fixture-generating device class** — bit-exactness is a same-device bar; foreign-device runs (e.g. CI) are permanently advisory (ADR 0004). (Same quantized weights on the same device ⇒ bitwise-identical logits are achievable; if a legitimate op-ordering divergence appears, the acceptance bar relaxes to: first divergence beyond token 48 AND logprob delta within 4 ULPs of the logit compute dtype at the divergence candidates' raw-logit magnitude — the dtype-aware calibration in ADR 0004, superseding the former fixed 1e-3, which is unsatisfiable for fp16/bf16 logits — requires an ADR naming the model and reason.) Every model-impl PR runs this. This is what makes loop-built model code trustworthy.

### 11.3 Layers
- Unit: block manager (alloc/free/COW/refcount invariants — property tests with `proptest`), radix tree, incremental detokenizer (fuzz against `tokenizers` full-decode), sampler distributions (chi-square at fixed seed), SSD slab round-trip + fingerprint rejection.
- Integration (macOS runner): worker gRPC black-box tests — submit/stream/cancel/drain; preemption under artificial memory cap; prefix-cache hit counters; spec-decode acceptance-rate sanity (>50% with same-family draft on English prose).
- E2E (pytest): full stack via HTTP — OpenAI + Anthropic conformance (validate against real client SDKs: `openai` and `anthropic` Python packages pointed at Kiln), streaming chunk shape, tool-call round trip, structured output validity (100/100 valid JSON against schema).
- Perf (`scripts/bench.sh`, criterion + a load generator): tracked per phase; regressions >10% fail the phase gate. Baseline targets on M4 Pro-class hardware, Qwen3-14B-4bit: single-stream decode ≥ mlx-lm `generate` −5%; batch-16 aggregate ≥ 3× single-stream on non-deterministic/mixed-majority load (deterministic-containing batch loads are governed by the amended bar in ADR 0003 — a recorded measured floor, gated on regression like any other bench number); TTFT p50 < 350ms for 500-token prompts cached-prefix-cold.
- Soak: 30-minute mixed-load run, assert RSS slope ~0 and MLX allocation counter stable (leak gate).

---

## 12. Build Plan — Phases for the Agent Loop

Each phase = one or more agent sessions. **Rule: a phase is done only when all its acceptance criteria pass via commands the agent runs and pastes output for.** Phases are ordered to keep the system runnable end-to-end from Phase 2 onward (tracer-bullet strategy: Python worker first, Rust worker replaces it under the same protocol).

### Phase 0 — Scaffold & Contract
Tasks: cargo workspace, crate stubs, CI (fmt/clippy/test on ubuntu compile-check + macos), `worker.proto` finalized, `kiln-proto` codegen (Rust + Python), `kiln.toml` parsing with figment, `CLAUDE.md` + `PROGRESS.md` seeded, mlx-c submodule pinned + `kiln-mlx` builds it via build.rs and links a hello-world (`mlx_array_new_float`, add, eval, read back).
Accept: `cargo build --workspace` clean on macOS; CI green; `cargo run -p kiln-mlx --example smoke` prints `3.0`; proto compiles in both languages.

### Phase 1 — Python worker end-to-end
Tasks: `kiln_worker_py` implementing GetInfo/Health/Submit(stream)/Cancel with mlx-lm sequential generation, raw-text input, sampling params (temp/top_p/top_k/seed), stop strings, usage in `Finished`.
Accept: pytest suite drives a real tiny model over UDS: streams tokens, cancel mid-stream stops within 2 steps, health reports memory numbers.

### Phase 2 — Gateway v0 (tracer bullet complete)
Tasks: axum app; model registry from config; worker supervisor (spawn python worker, health poll, restart-with-backoff); OpenAI chat completions (stream + non-stream) with chat templating via minijinja; `/v1/models`; API-key auth; Prometheus `/metrics`; structured logs.
Accept: `openai` Python SDK pointed at Kiln completes streaming + non-streaming chats against the tiny model; kill -9 the worker mid-request → client gets structured 502, worker auto-restarts, next request succeeds; metrics show request counters.

### Phase 3 — Rust worker v0: single-request Llama
Tasks: `kiln-mlx` safe wrappers (Array/Stream/error handler/leak counter); safetensors mmap loader; quantized matmul path; tokenizer-in-gateway mode (`token_ids` input); Llama forward pass, KV as simple contiguous cache (paging comes next); greedy + full sampler; incremental detok in gateway; golden harness + fixtures for Llama-3.2-1B.
Accept: golden parity exact on all Llama fixtures; single-stream decode tok/s within −10% of mlx-lm on same model (record number in PROGRESS.md); leak gate passes 1k-iteration loop.

### Phase 4 — Paged KV + continuous batching (kiln-engine core)
Tasks: block manager (+proptests), block tables, gather-based paged attention, chunked prefill, the batching loop per 6.2, preemption, Cancel/Drain semantics, per-request logit processing pipeline.
Accept: batch-16 aggregate ≥ 3× single-stream on tiny model; golden parity still exact (batching must not change greedy outputs — this catches position/masking bugs); preemption test passes under a 2-request memory cap; step-overhead criterion bench < 200µs recorded.

### Phase 5 — Radix prefix cache + SSD tier
Tasks: radix tree + refcount/COW integration; `PrefixCacheHit` events; SSD slab store, async flush, fingerprint checks, startup index scan, LRU cap; config flags.
Accept: resubmitting a 2k-token prompt shows ≥95% prefill skip and TTFT drops ≥5×; restart worker → prefix still hits from SSD; corrupt a slab header → cleanly ignored; proptest suite green.

### Phase 6 — Qwen + Gemma, quantization matrix, model routing
Tasks: Qwen2.5/Qwen3 and Gemma2/3 impls with golden fixtures; rope_scaling variants; 8-bit and BF16 paths; gateway `worker="auto"` routing (rust if supported+quant-format ok, else python); `/v1/completions`.
Accept: golden parity exact for all fixture models × dtypes; an unsupported-arch model transparently serves via python worker.

### Phase 7 — Structured output, tools, Anthropic API, paged-attention kernel
Tasks: llguidance integration (json_schema + regex) as a logit processor; tool-call streaming parsers (Hermes/Llama3/Qwen) in kiln-tokenize with unit fixtures; `/v1/messages` full adapter incl. thinking-block extraction; custom Metal paged-attention kernel behind flag + parity test vs gather path; `mlx_compile` experiments where profiled.
Accept: 100/100 schema-valid JSON generations; `anthropic` SDK conformance tests pass; tool-call e2e round trip; kernel path parity-exact and ≥15% decode throughput gain at 8k context (else flag stays off, documented).

### Phase 8 — Speculative decoding
Tasks: `Drafter` abstraction, draft-model loading, batched draft/verify per 6.5, O(1) rollback via block release, acceptance-rate metrics, auto-disable heuristics, config wiring.
Accept: greedy outputs identical with speculation on vs off (correctness invariant of spec decode); composes with prefix cache (test asserts both active). The ≥1.6×-at-acceptance->60% throughput clause is a deployment-shape precondition, not a CI bar (ADR 0006): it presumes substantial draft/target size asymmetry (e.g. Qwen3-0.6B drafting 8–14B), which the sub-1B pinned fleet cannot produce — measured 0.63–0.71× on every pinned pair — and it remains unverified in CI until such a pair enters the pinned fleet.

### Phase 9 — Multi-model supervision + memory governance
Tasks: LRU eviction with drain, pinning, TTL; machine budget accounting from worker heartbeats; per-worker admission (prefill projection); INTERACTIVE/BATCH priorities + preemption ordering; crash-loop backoff; python worker batching upgrade (mlx-lm batch API) if straightforward.
Accept: scripted scenario — load 3 models exceeding budget → correct LRU eviction order, pinned model survives; flood BATCH requests then send INTERACTIVE → interactive TTFT p95 unaffected >2×; 30-min soak leak gate on the full stack.

### Phase 10 — Jobs, admin UI, packaging, docs
Tasks: kiln-jobs (download + quantize wrapper, SQLite state); admin API; minimal SvelteKit admin (models table, load/unload/pin, live stats via SSE, job launcher) embedded in gateway; Homebrew formula + launchd plist; `kiln` CLI (`kiln serve`, `kiln models`, `kiln bench`); user docs (README, configuration reference, API compat notes).
Accept: `brew install --build-from-source ./Formula/kiln.rb` → `kiln serve` works; admin UI performs a full load/serve/unload cycle; e2e + soak + bench suites green; docs reviewed.

### Phase 11 (stretch, separate approval) — Embeddings-native, VLM via python worker, MTP self-draft, MLX distributed exploration.

---

## 13. Agent Loop Protocol (how to run Claude against this spec)

### 13.1 Session cadence
One task (or one small task cluster) per session. Prompt template:

> You are building Kiln per `docs/SPEC.md`. Read `CLAUDE.md` and `PROGRESS.md` first. Current phase: **{N}**. Task: **{task text from Section 12}**. Constraints: do not modify `worker.proto` semantics or any `docs/decisions/` content without writing a new ADR and stopping for my approval. Work until the phase's relevant acceptance criteria pass; run the commands and show me the output. Then update `PROGRESS.md` (what was done, decisions made, deviations from spec, next task) and stop. Do not start the next task.

Verification session every ~3 tasks: "Adversarially review the last 3 tasks against SPEC sections {X}. Run the full test suite. List any spec deviations, dead code, unhandled errors (`unwrap` outside tests), or missing tests. Fix or file in PROGRESS.md."

### 13.2 `CLAUDE.md` must contain
Build/test commands per crate; the mlx-c pin and how to rebuild it; "no monkey-patching, no new deps without justification, no `unwrap()` in library code, every mlx-c `new` has a `free`, all MLX ops on the engine thread"; golden-harness usage; how to fetch test models; PROGRESS.md format.

### 13.3 `PROGRESS.md` format
Append-only ledger: `## [date] Phase N / Task M — status`, bullet summary, acceptance-command outputs (trimmed), open questions flagged `DECISION NEEDED:` (these are your PM review queue).

### 13.4 Guardrails for you as PM
- Gate phase transitions yourself: run `scripts/bench.sh` and the e2e suite on your hardware before approving.
- Anything under `docs/decisions/` is agent-read-only after approval; changes go through you.
- Watch for the classic loop failure modes: silently weakened tests (diff test files every review session), acceptance criteria "reinterpreted," and dependency creep in `Cargo.toml`.
- Keep sessions scoped; when context degrades, start fresh — CLAUDE.md + PROGRESS.md + SPEC.md are the durable memory.

---

## 14. Risks & Open Decisions

| Risk | Mitigation |
|---|---|
| mlx-c API churn vs pinned commit | Pin hard; scheduled quarterly bump task with full golden re-run |
| Exact golden parity unattainable for some op orderings | Relaxed bar defined in 11.2 requires ADR per model |
| Paged-attention Metal kernel complexity | Gather path is always the correctness fallback; kernel is optional perf |
| Quantization format drift in mlx-community models | Router falls back to python worker on unknown `quantization` config |
| minijinja vs HF Jinja template edge cases | Vendored templates for supported models + template unit fixtures |
| Agent-introduced unsoundness in FFI | All `unsafe` confined to `kiln-mlx/src/sys.rs` + wrappers; review every diff touching that crate manually |
| Preemption resume cost: a resumed request replays its generated tokens as single-token forwards for bit-exactness — O(generated) extra steps per resume (PROGRESS 2026-07-04, Phase 4 part 3) | Acceptable while preemption is rare (Phase 4 pool sizes). Before Phase 9 makes preemption routine (memory governance, INTERACTIVE/BATCH floods), either batch the replay as one chunk — requires the batched-M bit-parity evidence from Phase 4 part 4 / the mlx#3120 note — or bound resumes via admission/eviction policy |

Open decisions to make before Phase 0: final project name; MSRV; whether gateway serves TLS natively or defers to a reverse proxy (recommend: localhost-only default, no TLS in v1).

— End of spec —
