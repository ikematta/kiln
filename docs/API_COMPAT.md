# API compatibility notes

Kiln exposes OpenAI-compatible and Anthropic-compatible surfaces (SPEC
§8.1). The adapters follow one policy throughout: **unknown top-level
fields are ignored; unsupported *features* are rejected with a clear 400**
— never silently dropped. This document is the honest inventory: what
works, what doesn't, and where the gaps were recorded during the build.

Real client SDKs are the conformance bar: the e2e suite drives the
`openai` and `anthropic` Python packages against a live stack on every CI
run.

## Authentication

- `/v1/chat/completions`, `/v1/completions`, `/v1/models`: `Authorization:
  Bearer <key>`. `/v1/messages` also accepts Anthropic's `x-api-key`
  header and returns the Anthropic error envelope.
- With no `[[auth.api_keys]]` configured the API is **open** (a warning is
  logged at startup) — the localhost-bind default is the guard rail.
- > **Known gap:** per-key `rpm`/`tpm` rate limits and the TTFT/total
  > timeouts of SPEC §8.3 are parsed from `kiln.toml` but **unenforced** —
  > no token-bucket middleware was ever built (SPEC §8.3 BACKLOG, Phase 2;
  > re-recorded open at the Phase 9 closeout). Keys authenticate; they do
  > not limit.

## OpenAI: `POST /v1/chat/completions`

Supported: `messages` (`system`/`user`/`assistant`/`tool` roles, string or
`text`-part array content), `stream` + `stream_options.include_usage`,
`temperature` (0–2), `top_p`, `presence_penalty`, `frequency_penalty`
(−2–2), `seed` (greedy/seeded decoding is deterministic per worker version
per device), `stop` (string or array), `max_tokens` /
`max_completion_tokens`, `tools` (type `function` only) with streamed
`tool_calls` deltas, multi-turn tool use (`role: "tool"` results),
`response_format` `text` / `json_object` / `json_schema`, `user`
(accepted, unused), `n: 1`.

Rejected with 400, by design or as a recorded gap:

| Field | Status |
|---|---|
| `n > 1` | not supported |
| `logprobs` / `top_logprobs` | not supported by either worker (the `LOGPROBS` capability flag exists in the worker protocol; no worker sets it in v1) |
| `tool_choice: "required"` or a named function | **not supported** — only `"auto"` and `"none"`. Forcing a call would need grammar-coupled decoding (constraining output to the tool-call format), which was never wired to the tool-call parsers. `"none"` drops the tools from the prompt entirely. |
| image/audio content parts | no vision or audio models in v1 (VLM input is a Phase 11 stretch item) |

Kiln extension: `priority: "interactive" | "batch"` (default
interactive) — BATCH requests are preempted first under memory pressure
and queue behind INTERACTIVE arrivals (SPEC §6.1).

## OpenAI: `POST /v1/completions`

String prompts only. Supported: `stream`, sampling/stop/seed as above,
`max_tokens`. Rejected: token-id prompts (arrays), `n > 1`, `best_of > 1`,
`logprobs`, `echo`, `suffix`.

## OpenAI: `GET /v1/models`

Lists the configured `[[model]]` ids (whether currently loaded or not).

## OpenAI: `POST /v1/embeddings` — not implemented

SPEC §8.1 scoped `/v1/embeddings` to Python workers with an
embeddings-capable model, but no build-plan phase ever delivered it (the
`EMBEDDINGS` capability flag is reserved in the worker protocol;
embeddings-native serving is the Phase 11 stretch item). The route does
not exist — clients get 404, not a degraded answer.

## Anthropic: `POST /v1/messages`

Supported: `max_tokens` (required, per the reference API), string or
block `system`, `user`/`assistant` messages with `text`, `tool_use`,
`tool_result` blocks, `temperature`, `top_p`, `top_k`, `stop_sequences`,
`stream` (Anthropic SSE event sequence), client `tools` + streamed
`tool_use` blocks, `thinking` passthrough (see below), `metadata`
(accepted, ignored), and the same `priority` extension as above.

Thinking: models that emit `<think>` reasoning have it extracted by the
streaming parser and returned as proper `thinking` content blocks.
`thinking.type: enabled | adaptive` is the models' native behavior;
`disabled` renders the chat template with `enable_thinking=false`
(thinking-trained templates honor it; others ignore it). Budgets
(`budget_tokens`) are unenforceable on open models and are
accepted-and-ignored.

Rejected with 400:

| Field | Status |
|---|---|
| `tool_choice: {type: "any"}` / `{type: "tool"}` | **not supported** — same forced-tool-call gap as the OpenAI adapter ("forced tool calls need grammar-coupled decoding") |
| `tool_choice.disable_parallel_tool_use` | not supported (the gateway cannot bound how many calls a model emits) |
| server tools (web search etc.) | client tools only |
| image content | no vision models in v1 |

## Tool calling (both APIs)

Tool-call extraction is a per-model-family streaming parser in
`kiln-tokenize` (Hermes-style `<tool_call>` JSON, Llama 3 python-tag,
Qwen XML), selected by model metadata and parity-tested against the HF
reference template rendering. Models without a known tool-call format
reject `tools` requests with a clear 400 rather than emitting text the
client would misparse.

## Structured output

`response_format: json_schema` (and `json_object`) compile to llguidance
grammars applied as decode-time logit masks — **Rust workers only**. The
Python worker does not set the `GRAMMAR` capability, and the gateway
returns 400 for structured-output requests routed to it. The e2e bar is
100/100 schema-valid generations (Phase 7).

## Streaming details

- SSE chunk shapes follow each API's reference framing; the incremental
  detokenizer never emits partial UTF-8 code points across chunks.
- OpenAI: `stream_options.include_usage` adds `"usage": null` on data
  chunks and a final usage-only chunk, per the reference behavior.
- A worker crash mid-stream yields a structured retriable 502; the model
  restarts with exponential backoff (max 3, then manual reset).

## Determinism

Greedy (and fixed-seed) decoding is bit-reproducible run to run on the
same build, same device — and batching, prefix caching, and speculative
decoding are required not to change greedy output (CI-gated). Determinism
is **not** promised across releases or device classes: MLX dispatches
shape-dependent Metal kernels whose reduction orders differ (ADRs
0002–0005 record the full story).
