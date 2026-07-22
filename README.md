# Kiln

**A production-grade LLM inference server for Apple Silicon.**

Kiln serves multiple LLMs concurrently on a single Apple Silicon machine —
a Rust control plane and data plane over [MLX](https://github.com/ml-explore/mlx)
(via a hand-rolled `mlx-c` FFI), with OpenAI- and Anthropic-compatible HTTP
APIs in front. Python is out of the token loop for the supported
architectures; a thin mlx-lm worker remains as a compatibility escape
hatch behind the identical worker protocol.

- **Process-per-model isolation** — one worker process per loaded model. A
  crash never takes down the gateway or other models; eviction returns
  memory to the OS by process exit, deterministically.
- **Continuous batching** with paged KV cache, radix-tree prefix sharing
  (copy-on-write), and an SSD cold tier that persists prefix cache across
  restarts.
- **Speculative decoding** as a scheduler-native capability that composes
  with batching and paging — greedy output is bit-identical with
  speculation on or off (see `docs/decisions/0006` before enabling it).
- **Deterministic greedy decoding** — bit-reproducible run to run on the
  same build and device; batching and caching do not change greedy output.
- **OpenAI (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) and
  Anthropic (`/v1/messages`) APIs** with SSE streaming, tool calling, and
  llguidance-backed structured output. Gaps are documented, not papered
  over: see [docs/API_COMPAT.md](docs/API_COMPAT.md).
- **Multi-model memory governance** — machine-level memory budget, LRU
  eviction with drain, pinning, idle TTL, INTERACTIVE/BATCH priorities,
  reservation-ledger admission control, and a live system-memory gate:
  loads the machine could only satisfy by swapping are refused up front,
  even when they fit the configured budget.
- **Operations built in** — Prometheus `/metrics`, structured JSON logs,
  a web admin dashboard (`/ui`), and a jobs runner for Hugging Face
  downloads and mlx-lm quantization.

Supported architectures on the Rust worker: Llama family (Llama 2/3/3.x,
Mistral, and most llamafied models), Qwen2/2.5, Qwen3, Gemma 2, Gemma 3
(text) — 4-bit/8-bit mlx-lm affine quantization (group sizes 32/64/128)
plus FP16/BF16. Anything else routes to the Python worker automatically
with `worker = "auto"`.

Requires macOS 14+ on Apple Silicon.

## Install

### Homebrew (build from source)

From a checkout (the formula is not in a published tap yet, so path
installs need Homebrew's developer mode):

```bash
HOMEBREW_DEVELOPER=1 brew install --build-from-source ./Formula/kiln.rb
```

or tap the repository directly:

```bash
brew tap ikematta/kiln https://github.com/ikematta/kiln.git
brew install --build-from-source ikematta/kiln/kiln
```

This builds the pinned Rust toolchain via rustup, embeds the admin UI,
installs `kiln`, `kiln-gateway`, `kiln-worker`, and `kiln-jobs`, sets up
the Python worker/jobs virtualenvs from the repo's own `uv.lock` pins,
and writes a starter config to `$(brew --prefix)/etc/kiln/kiln.toml`.
The build needs a Metal-capable compiler (Xcode, or Command Line Tools
with the Metal toolchain component).

### From source

```bash
git clone https://github.com/ikematta/kiln.git && cd kiln
git submodule update --init --recursive   # vendored mlx-c (pinned)
cargo build --release --workspace         # needs protobuf + cmake (brew install protobuf cmake)
uv sync --project python/kiln_worker_py   # optional: the Python fallback worker
```

## Quickstart

```bash
# 1. Download a model (resumable; JSON-lines progress on stdout)
kiln-jobs download mlx-community/Llama-3.2-1B-Instruct-4bit

# 2. Point a [[model]] block at it — Homebrew: $(brew --prefix)/etc/kiln/kiln.toml,
#    checkout: ./kiln.toml (start from kiln.toml.example)
#      [[model]]
#      id = "llama-3.2-1b"
#      path = "~/.kiln/models/mlx-community--Llama-3.2-1B-Instruct-4bit"
#      worker = "auto"

# 3. Serve (foreground; or `brew services start kiln` for launchd)
kiln serve

# 4. Complete
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "llama-3.2-1b", "messages": [{"role": "user", "content": "Say hello."}]}'
```

Any OpenAI or Anthropic SDK works pointed at the server:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused-when-no-keys-configured")
print(client.chat.completions.create(model="llama-3.2-1b",
      messages=[{"role": "user", "content": "Say hello."}]).choices[0].message.content)
```

## CLI

| Command | What it does |
|---|---|
| `kiln serve [--config <path>]` | Run the gateway (config resolution: `--config`, `$KILN_CONFIG`, `./kiln.toml`, `<prefix>/etc/kiln/kiln.toml`) |
| `kiln models [--json]` | The admin API's model table (id, worker, status, pinned, TTL, memory) plus the machine memory ledger. Needs `$KILN_ADMIN_TOKEN`. |
| `kiln bench --model <m> [...]` | Throughput/TTFT benchmark harness (`scripts/bench.sh`); `--url` benchmarks a running server, `--engine` adds the release engine perf gate |
| `kiln-gateway hash-key [key]` | argon2 hash for API keys / the admin token |
| `kiln-jobs download <hf_repo>` | Resumable Hugging Face download |
| `kiln-jobs quantize <path> --bits 4 --group-size 64` | Quantize via pinned `mlx_lm convert` |

## Admin API and dashboard

Set an admin token (the admin surface is disabled — 403 — until you do):

```bash
kiln-gateway hash-key my-secret-token   # → paste into auth.admin_token_hash
```

Then `http://127.0.0.1:8080/ui` is a live dashboard (models table,
load/unload/pin, live stats, download/quantize job launcher, and an Add
Model flow that takes an HF repo id from download to loaded — persisting
the `[[model]]` block to kiln.toml — with zero restarts), and the same
functionality is available over `GET/POST /admin/models*`, `/admin/jobs*`,
and the `GET /admin/stats` SSE stream with `Authorization: Bearer <token>`.

## Documentation

- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) — every `kiln.toml` field,
  with the design decisions (ADRs) behind the non-obvious knobs
- [docs/API_COMPAT.md](docs/API_COMPAT.md) — exactly which OpenAI/Anthropic
  features are supported, and the known gaps
- [docs/SPEC.md](docs/SPEC.md) — the build specification
- [docs/decisions/](docs/decisions/) — architecture decision records
- [PROGRESS.md](PROGRESS.md) — the append-only build ledger, including
  every phase's measured acceptance numbers

## Development

See [CLAUDE.md](CLAUDE.md) for the full build/test matrix. The short
version: `cargo test --workspace` (Metal tests auto-skip without a GPU),
`./scripts/fetch-test-model.sh` for the pinned test models,
`cargo test -p kiln-models --test golden` for the mlx-lm parity harness,
`./scripts/soak.sh` for the leak gate, `./scripts/bench.sh` for perf.
