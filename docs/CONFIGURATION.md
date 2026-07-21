# Kiln configuration reference (`kiln.toml`)

Schema authority: [SPEC.md §10](SPEC.md). Annotated starter:
[`kiln.toml.example`](../kiln.toml.example). Every `[server]`, `[memory]`,
and `[defaults]` value has a built-in default — omit any key, section, or
the whole file section to get it. Where a knob exists because of a recorded
design decision, the ADR is linked; read it before turning the knob.

## Environment overrides

Any scalar key in `[server]`, `[memory]`, `[defaults]`, or `[auth]` can be
overridden with a `KILN_`-prefixed variable, `__` separating nesting:

```bash
KILN_SERVER__PORT=9090
KILN_MEMORY__BUDGET_FRACTION=0.7
KILN_DEFAULTS__SSD_TIER=false
```

`[[model]]` entries come from the file only. Unrelated `KILN_*` variables
(e.g. `KILN_TEST_MODELS`) are ignored.

## `[server]`

| Key | Default | Meaning |
|---|---|---|
| `host` | `"127.0.0.1"` | Bind address. Localhost by default and no TLS in v1 (SPEC §14): terminate TLS/auth upstream if you expose it. |
| `port` | `8080` | Bind port (non-zero). |
| `runtime_dir` | `~/.kiln/run` | Worker/jobs Unix domain sockets. |
| `cache_dir` | `~/.kiln/cache` | SSD KV-cache tier, one subdirectory per model fingerprint. |
| `model_dir` | `~/.kiln/models` | Default destination root for `kiln-jobs download`/`quantize` output. |
| `python_worker_argv` | `["uv", "run", "--project", "python/kiln_worker_py", "python", "-m", "kiln_worker_py"]` | Command prefix launching the Python worker; the gateway appends `--model <path> --socket <path> --model-id <id>`. The default assumes a repo checkout; the Homebrew install overrides it to the packaged venv's python. |
| `rust_worker_argv` | sibling `kiln-worker` binary | Command prefix launching the Rust worker. Defaults to the `kiln-worker` next to the running gateway binary, falling back to `$PATH`. |
| `jobs_argv` | sibling `kiln-jobs` binary | Command prefix for the on-demand jobs server; the gateway appends `serve --socket <path> --db <path> --dest-root <model_dir>`. Flags may precede the subcommand, so `["kiln-jobs", "--venv", "<uv-project-dir>"]` points quantization at a packaged jobs venv. |
| `jobs_db` | `~/.kiln/jobs.sqlite` | SQLite job store used by the spawned `kiln-jobs` server (SPEC §9.1). |

## `[memory]`

| Key | Default | Meaning |
|---|---|---|
| `budget_fraction` | `0.80` | Machine budget = total unified memory × this fraction (SPEC §2.3). Every worker's heartbeat-measured footprint (max(MLX active, weights + KV pool) + MLX cache) is charged against it. |
| `budget_bytes` | unset | Absolute budget in bytes; overrides `budget_fraction` when set. |

Enforcement, as actually built (Phase 9):

- **Per load**: a `load()` that would exceed the budget first evicts the
  least-recently-used model that is unpinned and outside its TTL lease
  (Drain → SIGTERM → SIGKILL after grace); with nothing evictable the load
  is rejected and retried on the next request.
- **Per request**: projected KV-pool growth beyond live headroom returns a
  retriable `503 insufficient_memory` instead of drifting over budget.
  Admission runs against a reservation ledger, so concurrent requests
  cannot race past the budget (PROGRESS 2026-07-15, Phase 9 part 3).
- **Priorities**: requests carry an optional `priority` field
  (`"interactive"` default, `"batch"`); BATCH is preempted first under
  pressure and queues behind INTERACTIVE arrivals.

## `[defaults]` (engine, per worker)

| Key | Default | Meaning |
|---|---|---|
| `max_batch_tokens` | `8192` | Per-step token budget of the continuous-batching loop (SPEC §6.2). |
| `prefill_chunk` | `2048` | Long prompts prefill in chunks of this size, interleaved with decode steps to bound decode latency. |
| `block_size` | `32` | Tokens per paged-KV block; power of two. |
| `ssd_cache_max_gb` | `64` | LRU byte cap of the SSD cold tier. |
| `ssd_tier` | `true` | Evict cold KV blocks to `cache_dir` instead of dropping them; prefix cache then survives worker restarts. Slab files carry a model fingerprint; mismatches are silently skipped, never an error. |
| `paged_attention_kernel` | `false` | Block-table-aware Metal decode kernel (SPEC §7.4) instead of the gather-based path. On the Phase 7 dev machine it measured bit-identical to the gather path and 1.461× decode at 8k context (PROGRESS 2026-07-13), clearing both acceptance bars — but it ships **default-off by explicit ruling**: bit-exactness is a per-device measurement, not a promise (the offline-metallib-vs-JIT compiler risk is device/pin-specific), so enable it only after `cargo test -p kiln-engine --test paged_attn` and the `paged_attn_gate` release gate pass on *your* hardware. Kernel-dispatch determinism context: [ADR 0002](decisions/0002-parity-bars-under-mlx-kernel-dispatch.md). |

## `[[model]]` (one block per servable model)

You do not have to hand-edit these blocks: the admin dashboard's **Add
Model** action (or `POST /admin/models` with `{id, path, worker, pinned,
ttl_seconds}`) registers a model on the running gateway — downloading it
first through the job runner when needed — and appends the equivalent
`[[model]]` block to kiln.toml in place, preserving comments, formatting,
and any concurrent hand edits (the file is re-read at write time; a
duplicate id or unparseable file is refused loudly). No restart involved:
the model is immediately loadable via the existing load/unload/pin
machinery. `GET /admin/models/estimate?path=…` answers the size question
first (weights on disk or from the hub listing, against the live memory
budget). Removing or changing a model still means editing the file and
restarting.

| Key | Default | Meaning |
|---|---|---|
| `id` | required | The `model` string clients send. |
| `path` | required | Hugging Face repo id or local directory (mlx-lm layout: `config.json`, safetensors, `tokenizer.json`). |
| `worker` | `"auto"` | `rust` \| `python` \| `auto`. Auto = Rust worker when the architecture (llama, qwen2/2.5, qwen3, gemma2, gemma3-text) and quantization format (4/8-bit affine, group 32/64/128, or FP16/BF16) are supported, else the mlx-lm Python worker. |
| `pinned` | `false` | Never evicted under memory pressure. |
| `ttl_seconds` | `0` | Idle keep-alive lease; `0` = resident forever. An idle model past its TTL auto-unloads and reloads on the next request; within the lease it is also protected from LRU eviction. In-flight requests hold the lease open. |

### `[model.speculative]` (optional; Rust workers only)

| Key | Default | Meaning |
|---|---|---|
| `draft` | required | Draft model path/repo. Its tokenizer must match the target's; the worker refuses to start otherwise. |
| `gamma` | `4` | Tokens proposed per round. Clamped to the certified verify-kernel envelope — see [ADR 0005](decisions/0005-speculative-verify-kernel-envelope.md). |

Correctness is unconditional (greedy output is bit-identical with
speculation on or off), but **throughput is not**: speculation pays only
with a large draft/target size asymmetry (roughly a ≤1B draft proposing
for a ≥7–8B target). On similarly sized pairs it is a measured loss —
0.63–0.71× — at any acceptance rate. That deployment-shape precondition is
[ADR 0006](decisions/0006-speculative-throughput-bar-deployment-shape.md);
read it before enabling. At runtime speculation also stands down
per-request under wide batches or sustained low acceptance.

## `[auth]`

| Key | Default | Meaning |
|---|---|---|
| `admin_token_hash` | `""` | argon2 (PHC) hash of the admin bearer token — generate with `kiln-gateway hash-key`. Empty/unset ⇒ the whole `/admin/*` surface (and the `/ui` dashboard's data) is **disabled, fail-closed** (403). A non-empty value that does not parse as a PHC string is a startup error, not a warning. API keys never grant admin. |
| `[[auth.api_keys]]` | none | Per-key: `name`, `key_hash` (argon2), optional `rpm`, `tpm`. With **no** keys configured the `/v1/*` API is open (a warning is logged) — the admin surface is the opposite, closed until configured. |

> **Known gap — rate limits are parsed, not enforced.** `rpm`/`tpm` have
> been accepted by the config schema since Phase 2, but no token-bucket
> middleware was ever built (SPEC §8.3 BACKLOG; re-recorded at the Phase 9
> closeout, PROGRESS 2026-07-15). The same applies to the TTFT/total
> request timeouts SPEC §8.3 names. Keys authenticate; they do not
> currently limit.

## Unauthenticated endpoints

`/healthz`, `/readyz`, `/metrics` (Prometheus), and the static `/ui` shell
are unauthenticated by design — the gateway binds localhost by default
(SPEC §8.1); every piece of data behind `/ui` rides the bearer-gated
`/admin` API.
