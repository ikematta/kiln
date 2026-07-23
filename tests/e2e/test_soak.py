"""Phase 9 part 3: the 30-minute full-stack mixed-load soak (SPEC §11.3).

The phase-closing leak + correctness gate: everything Phases 4-9 built runs
TOGETHER against one gateway for 30 real minutes of concurrent multi-tenant
traffic — continuous batching, chunked prefill, preemption, the radix
prefix cache warm and cold, the SSD tier, speculative decoding, grammar
masking, INTERACTIVE/BATCH priorities, LRU eviction, TTL leases, pinning,
per-request admission, cancellation, and the python fallback worker — while
the harness tracks memory THROUGHOUT and spot-checks greedy determinism at
intervals.

Only runs when KILN_SOAK_MINUTES is set (scripts/soak.sh); the regular e2e
sweep skips it.

Fleet (all measured on the dev machine, PROGRESS 2026-07-15; the components
are device-stable — packed weights + the fixed 512-block KV pools dominate):

  llama-int    llama-3.2-1b-4bit, rust, PINNED.   warm ~1.24 GB
               (weights 695 MB + pool 537 MB). Primary tenant: interactive
               chat, batch floods, grammar, prefix traffic, cancellations,
               /v1/messages, and the determinism canary.
  spec-qwen25  qwen2.5-0.5b-4bit + same-checkpoint draft, gamma 3 — inside
               the ADR 0005 envelope (gqa_factor 7 ⇒ gamma+1 ≤ 4; quantized
               trunk, head_dim 64, fused SDPA), the standard self-draft
               gate shape. PLAIN → LRU-evictable. warm ~0.98 GB (2× weights
               278 MB + 2× pool 201 MB). The qwen3 e2e pair was measured at
               5.03 GB fully warm (two 1.88 GB pools) — it cannot coexist
               with this fleet inside a CI-sized budget.
  ttl-qwen25   qwen2.5-0.5b-4bit, rust, ttl_seconds=75. warm ~0.49 GB.
               Touched in ~110 s cycles so the lease expires between
               touches: idle_ttl unload/reload every cycle.
  burst-gemma  gemma-3-1b-it-4bit, rust, ttl_seconds=90. warm ~1.18 GB
               (pool 436 MB). Touched in ~5 min bursts: each burst's load
               forces LRU eviction and/or per-request 503s, then its own
               TTL frees the memory again.
  py-smollm    smollm2-135m-bf16, PYTHON worker, plain. warm ~0.28 GB.
               BF16 path + the second worker kind under the same roof.

Budget 3.9 GB (explicit budget_bytes: device-independent behavior, as in
test_lifecycle): all five idle ≈ 2.6 GB fit at startup; the warmup path
peaks near ~3.7 GB on the dev machine (idle sum + llama pool 537 MB +
spec pools 402 MB + ttl pool 201 MB + caches) and can legitimately be
refused on machines with fatter idle footprints (the CI runner, run
29398308457) — so warmup RETRIES each model for up to 240 s, riding the
system's own recovery (TTL expiry frees gemma's ~0.75 GB at 90 s). The
steady warm set (~3.0 GB + cache drift) leaves headroom well under a
gemma load+pool (~1.2 GB) — so every gemma burst must evict or be
refused, and spec-qwen25 re-warms only after a TTL expiry frees room.
Deliberate, recovering pressure — the machine is over-subscribed by
design (all-warm sum ≈ 4.2 GB) and must stay sane for the whole run.

Leak gates (SPEC §11.3), tracked throughout, not endpoint-compared:
  - RSS: absolute working-set caps (gateway 160 MB = 2x its measured
    54-82 MB plateau; pinned worker 1.2 GB = full weight mmap resident
    + heap slack; python worker 700 MB). Ten 30-min runs proved every
    derivative measure (fixed-window slopes, mid-run-referenced deltas)
    aliases macOS page-reclaim timing on identical workloads; slopes
    and deltas are still computed and reported. NOTE: MLX Metal buffers
    do not appear in RSS (measured: llama worker RSS ~40 MB with 1.2 GB
    mlx-active), so RSS caps bound the CPU/Rust-heap side coarsely; the
    fine-grained leak gates are the Metal-side counters below plus the
    committed/reservation ledgers.
  - mlx live objects: kiln_worker_mlx_live_objects (the CLAUDE.md
    debug-build wrapper counter, exported for this task) must RETURN TO
    its drained floor at each group's last quiesced checkpoint, with
    interior excursions bounded by LIVE_TRANSIENT_ALLOWANCE (engine-
    thread SSD-flush maintenance holds 2 handles per block mid-copy).
    Pool materialization is a one-time +2×layers step (measured: llama
    +32, gemma +52, qwen25 +48, qwen3 +2×56) absorbed by the warmup and
    the (generation, pool) grouping. Negative anywhere is a double-free.
    Sampling semantics (PROGRESS 2026-07-23 root-cause + ruling): the
    gated sample is taken FLUSH-IDLE — the checkpoint polls until every
    rust worker's kiln_worker_flush_pending_blocks reads 0 in the same
    scrape (bounded by FLUSH_IDLE_DEADLINE_S; on expiry it samples
    anyway, so a stuck queue is not a bypass). The quiesce itself
    triggers a cancel burst whose donations feed the SSD write-behind
    queue, and each in-flight block capture (read_block_bytes) holds
    exactly +2 handles for the duration of a sync'd 1 MiB read — on slow
    CI runners that tail outlives a fixed post-settle sleep, so an
    un-gated sample reads floor+2 with ~0.9 probability while flushes
    are in flight (measured; kiln-engine/tests/paged_attn_leak.rs R6).
    A real parked leak elevates EVERY sample, flush-idle ones included,
    so this is a measurement correction, not a bar change.
  - mlx_active equal (±2 MB transient band) and mlx_cache bounded at
    quiesced checkpoints within a generation.

Correctness gates, held THROUGHOUT:
  - Canaries: a fixed greedy prompt on llama-int every ~60 s and on
    spec-qwen25 (speculating) every ~75 s must yield bit-identical text
    every single time — across batching, floods, preemption, prefix
    warm/cold, SSD restores, eviction/reload.
  - Zero worker crash-restarts; /readyz never reports a crashed state.
  - Committed bytes (weights + materialized pools) ≤ budget at every 10 s
    sample — the invariant load/pool-growth admission actually enforces.
    The RAW ledger additionally counts mlx_cache and in-flight compute
    buffers, which have no admission lever on materialized pools (the
    recorded open Phase 9 gap: continuous-pressure eviction); its
    overshoot is measured and reported, not gated.
  - llama-int (pinned, warmed first): never evicted, never
    admission-rejected — its pool growth is 0 after warmup.
  - Every refusal is structured and expected: 503 insufficient_memory /
    model_loading / model_unloading / worker_draining (the worker-side
    drain race) are counted outcome classes; a 502
    worker_crashed is tolerated only as the bounded-drain severed tail
    of a deliberate unload (correlated ±60 s with that model's unload
    counter, never the pinned model, ≤ SEVERED_MAX per run — SPEC §2.2:
    Drain is 30 s bounded, then SIGTERM severs stragglers). Anything
    else is a hard failure. Grammar outputs 100% schema-valid.
  - Interactive requests complete < 90 s even mid-flood (priority
    admission), every gemma burst recovers to a 200 within its 180 s window.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field

import httpx
import pytest
from conftest import API_KEY, pinned_model_dir, running_stack

SOAK_MINUTES = float(os.environ.get("KILN_SOAK_MINUTES", "0") or "0")

pytestmark = pytest.mark.skipif(
    SOAK_MINUTES <= 0,
    reason="soak runs only via scripts/soak.sh (set KILN_SOAK_MINUTES)",
)

LLAMA = "llama-int"
SPEC = "spec-qwen25"
TTL = "ttl-qwen25"
GEMMA = "burst-gemma"
PYSMOL = "py-smollm"
RUST_MODELS = (LLAMA, SPEC, TTL, GEMMA)

BUDGET_BYTES = 3_900_000_000
TTL_SECONDS = 75
GEMMA_TTL_SECONDS = 90
SPEC_GAMMA = 3

# Gates. Count gates apply only to runs >= GATE_FULL_MINUTES (the CI shape);
# shorter smoke runs report but skip them. Slope thresholds sit well below
# any real per-request leak at this request rate (~0.5 rps x 30 min: a
# leaked KV block, SSE buffer, or request record shows up as MBs/min) and
# well above measured idle noise.
GATE_FULL_MINUTES = 20
# RSS gates are absolute working-set caps, and ONLY that. Ten 30-min
# runs (7 local + 3 CI) established the envelope: the gateway converges
# to a 54-82 MB working set every run, but macOS page-reclaim dips and
# re-climbs land anywhere in the timeline, whipsawing every derivative
# measure tried — fixed-window slopes read -2027..+3370 KiB/min and a
# mid-run-referenced late delta read -46..+23 MB across runs with
# IDENTICAL workloads and flat mlx-side counters. The pinned worker is
# worse: its RSS breathes with the page cache over the 695 MB weight
# mmap (observed 36..846 MB resident, slopes -47308..+1816). A real
# CPU-heap leak of consequence blows an absolute cap within the run
# (~1900 requests x 40 KB = 76 MB over the gateway plateau); finer-
# grained leak detection is owned by the bit-exact mlx live-object
# gate, the flat mlx_active gate, the committed/reservation ledgers,
# and the 1k-iteration leak suites. Slopes and deltas are still
# computed and REPORTED for the record.
GW_RSS_FINAL_CAP = 160 * 1024 * 1024
LLAMA_RSS_CAP = 1200 * 1024 * 1024  # full weight mmap resident + heap slack
PY_RSS_CAP = 700 * 1024 * 1024  # evictable python worker: working-set cap
# Bounded-drain severed tail (SPEC §2.2): a request in flight on a worker
# being deliberately unloaded gets 30 s of drain (supervisor
# DRAIN_DEADLINE), then SIGTERM severs the stream and the gateway maps it
# to a retriable 502 worker_crashed. Tolerated ONLY when correlated with
# an unload of that model (±60 s), never on the pinned model, and at most
# this many per run (observed: 1 in ~20k requests across four runs, on
# the slowest runner).
SEVERED_MAX = 3
ACTIVE_BAND_BYTES = 2 * 1024 * 1024
CACHE_CAP_BYTES = 768 * 1024 * 1024
# Live-object transient allowance at quiesced checkpoints: SSD-flush
# maintenance runs on the engine thread after traffic stops (engine.rs
# flush_entries -> read_block_bytes) and holds exactly 2 handles (K+V
# gather) per block mid-copy; the heartbeat thread can sample mid-drain
# (observed: one +2 excursion that returned to baseline at every later
# checkpoint). 8 allows four in-flight block copies; a real leak fails
# the return-to-baseline check regardless of this allowance.
LIVE_TRANSIENT_ALLOWANCE = 8
# Flush-idle sampling (see the live-objects gate note in the module
# docstring): how long a quiesced checkpoint will wait for every rust
# worker's SSD write-behind queue to drain before sampling anyway.
# Sized from the instrumented CI evidence run (29980519514): gated
# samples observed with 133-397 blocks pending and a drain pace of a
# few blocks/s on the runner (debug-build Sha256 + 1 MiB copy per block
# dominates each flush cycle), so a large post-burst backlog can take
# minutes. 240 s bounds the wait; on expiry the sample is taken anyway
# and the stuck depths printed — a deadline, not a bypass.
FLUSH_IDLE_DEADLINE_S = 240.0
INTERACTIVE_P100_S = 90.0
REQ_TIMEOUT = httpx.Timeout(30.0, read=420.0, write=30.0, pool=60.0)

METRIC_LINE = re.compile(r"^(\w+)(?:\{([^}]*)\})?\s+(-?[0-9eE+.]+)$")

# ~64 repetitions ≈ 1030 llama tokens: long enough that a warm resubmission
# is a big measurable prefix hit, short enough to prefill fast under load.
PREFIX_FILLER = (
    "Paged attention splits the key-value cache into fixed-size blocks so "
    "that requests can grow without contiguous reservations. "
) * 64

CANARY_LLAMA_PROMPT = (
    "Kiln soak canary: list the first eight prime numbers in ascending "
    "order, separated by commas."
)
CANARY_SPEC_PROMPT = (
    "The invention of the printing press changed European society because"
)

INTERACTIVE_PROMPTS = [
    "Summarize why unit tests matter in two sentences.",
    "What is the capital of Japan?",
    "Explain what a hash map is to a beginner.",
    "Write a haiku about mountains.",
    "Name three uses for a magnet.",
    "What does HTTP stand for?",
    "Give one tip for writing clear emails.",
    "Why is the sky blue, briefly?",
]

BATCH_PROMPTS = [
    "Write a short story about a lighthouse keeper who finds a map.",
    "Describe the water cycle in detail for a science pamphlet.",
    "Draft a product description for a mechanical keyboard.",
    "Explain how continuous batching improves GPU utilization.",
]

SPEC_PROMPTS = [
    "The industrial revolution began in England because",
    "Photosynthesis is the process by which plants",
    "The primary difference between weather and climate is",
    "In distributed systems, consensus protocols are used to",
]

GRAMMAR_SCHEMA = {
    "x-guidance": {"whitespace_flexible": False},
    "type": "object",
    "properties": {
        "name": {"type": "string", "maxLength": 12},
        "kind": {"type": "string", "enum": ["cat", "dog", "bird"]},
        "age": {"type": "integer", "minimum": 0, "maximum": 30},
    },
    "required": ["name", "kind", "age"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "pet", "schema": GRAMMAR_SCHEMA, "strict": True},
}


def require(model_id: str) -> str:
    path = pinned_model_dir(model_id)
    if path is None:
        pytest.skip(
            f"pinned test model '{model_id}' not found; run "
            "./scripts/fetch-test-model.sh"
        )
    return str(path)


# ---------------------------------------------------------------------------
# Metrics scraping
# ---------------------------------------------------------------------------


def scrape(base_url: str) -> list[tuple[str, dict[str, str], float]]:
    text = httpx.get(f"{base_url}/metrics", timeout=15).text
    out = []
    for line in text.splitlines():
        match = METRIC_LINE.match(line)
        if match:
            labels = dict(re.findall(r'(\w+)="([^"]*)"', match.group(2) or ""))
            out.append((match.group(1), labels, float(match.group(3))))
    return out


def mval(
    samples: list[tuple[str, dict[str, str], float]], name: str, **labels: str
) -> float | None:
    """First sample of `name` whose labels include `labels`, else None."""
    for got_name, got_labels, value in samples:
        if got_name == name and all(got_labels.get(k) == v for k, v in labels.items()):
            return value
    return None


def msum(
    samples: list[tuple[str, dict[str, str], float]], name: str, **labels: str
) -> float:
    """Sum over all samples of `name` whose labels include `labels`."""
    return sum(
        value
        for got_name, got_labels, value in samples
        if got_name == name and all(got_labels.get(k) == v for k, v in labels.items())
    )


def slope_kb_per_min(points: list[tuple[float, float]]) -> float:
    """Least-squares slope of (seconds, bytes) points, in KiB/minute."""
    n = len(points)
    if n < 3:
        return 0.0
    mean_t = sum(p[0] for p in points) / n
    mean_v = sum(p[1] for p in points) / n
    num = sum((t - mean_t) * (v - mean_v) for t, v in points)
    den = sum((t - mean_t) ** 2 for t, _ in points)
    if den == 0:
        return 0.0
    return (num / den) * 60.0 / 1024.0


# ---------------------------------------------------------------------------
# Load-generator framework
# ---------------------------------------------------------------------------


@dataclass
class Ctx:
    stack: object
    stop: threading.Event
    gate: threading.Event  # set = generators may run; cleared = quiesce
    flood_active: threading.Event
    hard_errors: list[str] = field(default_factory=list)
    # 502 worker_crashed responses, held for post-run correlation: an
    # in-flight request severed by the BOUNDED drain of a deliberate
    # unload (SPEC §2.2: Drain 30 s -> SIGTERM; the severed stream maps
    # to a retriable 502) is within contract; anything uncorrelated with
    # an unload of that model is a hard failure. (model, t, detail).
    severed: list[tuple[str, float, str]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    started: float = 0.0

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def should_abort(self) -> bool:
        return self.stop.is_set() or not self.gate.is_set()

    def record_error(self, source: str, detail: str) -> None:
        with self.lock:
            self.hard_errors.append(f"[t+{self.elapsed():7.1f}s] {source}: {detail}")


class Runner(threading.Thread):
    """One traffic class: runs `fn` on a jittered period, pausable via
    ctx.gate for quiesced checkpoints. `fn` must never raise for expected
    outcomes — anything raised or recorded via error() is a hard failure."""

    def __init__(self, ctx, name, fn, period, initial_delay=0.0):
        super().__init__(name=f"soak-{name}", daemon=True)
        self.ctx = ctx
        self.label = name
        self.fn = fn
        self.period = period
        self.initial_delay = initial_delay
        self.rng = random.Random(name)
        self.busy = False
        self.oks = 0
        self.rejects = 0
        self.cancelled = 0
        self.last_503 = ""
        # One-shot extension of the next inter-request delay: a class can
        # stand down (e.g. after an insufficient_memory refusal) without
        # holding `busy` through a sleep.
        self.extra_delay = 0.0
        self.extra: dict[str, float] = {}
        self.client = httpx.Client(
            base_url=ctx.stack.base_url,
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=REQ_TIMEOUT,
        )

    def error(self, detail: str) -> None:
        self.ctx.record_error(self.label, detail)

    def bump(self, key: str, delta: float = 1.0) -> None:
        with self.ctx.lock:
            self.extra[key] = self.extra.get(key, 0) + delta

    def run(self) -> None:
        if self.ctx.stop.wait(timeout=self.initial_delay):
            return
        while not self.ctx.stop.is_set():
            self.ctx.gate.wait(timeout=1.0)
            if not self.ctx.gate.is_set():
                continue
            if self.ctx.stop.is_set():
                break
            self.busy = True
            try:
                self.fn(self)
            except Exception as exc:  # noqa: BLE001 - soak must keep going
                self.error(f"uncaught {type(exc).__name__}: {exc}")
            finally:
                self.busy = False
            delay = self.rng.uniform(*self.period) + self.extra_delay
            self.extra_delay = 0.0
            if self.ctx.stop.wait(timeout=delay):
                break
        self.client.close()


def classify(runner: Runner, response: httpx.Response, what: str, model: str) -> bool:
    """True when 200. Structured refusals that this scenario's deliberate
    pressure makes EXPECTED are counted, not failed: 503
    `insufficient_memory` (the part 2 admission gate), 503 `model_loading`
    (on-demand reload in progress — part 1's "rejected and retried"
    path), 503 `model_unloading` (routed during a drain window), and 503
    `worker_draining` (routed while Ready, refused worker-side after its
    Drain RPC — the race's structured leg). A
    502 `worker_crashed` is held for post-run correlation (see
    Ctx.severed): tolerated only as the bounded-drain severed tail of a
    deliberate unload of this exact model. Anything else is a hard
    error."""
    if response.status_code == 200:
        return True
    code = None
    with contextlib.suppress(Exception):
        code = response.json()["error"]["code"]
    if response.status_code == 503:
        if code == "insufficient_memory":
            runner.rejects += 1
            runner.last_503 = response.text[:300]
            return False
        if code == "model_loading":
            runner.bump("loading_retry")
            runner.last_503 = response.text[:300]
            return False
        if code == "model_unloading":
            runner.bump("unloading_retry")
            runner.last_503 = response.text[:300]
            return False
        if code == "worker_draining":
            # The worker-side leg of the drain race: routed while Ready,
            # arrived after the Drain RPC; the worker refuses with a
            # structured retriable (gateway error.rs worker_draining).
            # First observed on CI run 29436961038.
            runner.bump("draining_retry")
            runner.last_503 = response.text[:300]
            return False
        runner.error(f"{what}: unexpected 503 ({code}): {response.text[:200]}")
        return False
    if response.status_code == 502 and code == "worker_crashed":
        with runner.ctx.lock:
            runner.ctx.severed.append(
                (model, runner.ctx.elapsed(), f"{what}: {response.text[:200]}")
            )
        return False
    runner.error(f"{what}: HTTP {response.status_code}: {response.text[:200]}")
    return False


def completion(
    runner: Runner, model: str, prompt: str, max_tokens: int, **extra
) -> str | None:
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens}
    body.update(extra)
    try:
        response = runner.client.post("/v1/completions", json=body)
    except httpx.HTTPError as exc:
        runner.error(f"completion({model}): {type(exc).__name__}: {exc}")
        return None
    if not classify(runner, response, f"completion({model})", model):
        return None
    choice = response.json()["choices"][0]
    if not choice["text"]:
        # EOS as the very first sampled token is legal model behavior
        # (finish "stop"); an empty text with any other finish reason
        # would be a serving bug.
        if choice.get("finish_reason") == "stop":
            runner.bump("empty_stop")
            runner.oks += 1
            return None
        runner.error(
            f"completion({model}): empty text with finish_reason="
            f"{choice.get('finish_reason')!r}"
        )
        return None
    runner.oks += 1
    return choice["text"]


def sse_data_lines(response):
    for line in response.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            yield json.loads(line[len("data: ") :])


def stream_completion(
    runner: Runner,
    model: str,
    prompt: str,
    max_tokens: int,
    abort_after_chunks: int | None = None,
    **extra,
) -> int:
    """Streams a completion; returns chunks read. Aborting early (or on
    quiesce/stop) closes the SSE stream — the gateway must Cancel the
    worker request."""
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
    }
    body.update(extra)
    chunks = 0
    try:
        with runner.client.stream("POST", "/v1/completions", json=body) as r:
            if r.status_code != 200:
                r.read()
                classify(runner, r, f"stream({model})", model)
                return 0
            for _event in sse_data_lines(r):
                chunks += 1
                if abort_after_chunks and chunks >= abort_after_chunks:
                    runner.cancelled += 1
                    return chunks
                if runner.ctx.should_abort():
                    runner.cancelled += 1
                    return chunks
    except httpx.HTTPError as exc:
        runner.error(f"stream({model}): {type(exc).__name__}: {exc}")
        return chunks
    runner.oks += 1
    return chunks


# ---------------------------------------------------------------------------
# Traffic classes
# ---------------------------------------------------------------------------

interactive_latencies: list[tuple[float, bool]] = []


def interactive_fn(runner: Runner) -> None:
    prompt = runner.rng.choice(INTERACTIVE_PROMPTS)
    during_flood = runner.ctx.flood_active.is_set()
    started = time.monotonic()
    if runner.rng.random() < 0.5:
        try:
            response = runner.client.post(
                "/v1/chat/completions",
                json={
                    "model": LLAMA,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 32,
                },
            )
        except httpx.HTTPError as exc:
            runner.error(f"chat: {type(exc).__name__}: {exc}")
            return
        if classify(runner, response, "chat", LLAMA):
            runner.oks += 1
    else:
        body = {
            "model": LLAMA,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "stream": True,
        }
        chunks = 0
        try:
            with runner.client.stream("POST", "/v1/chat/completions", json=body) as r:
                if r.status_code != 200:
                    r.read()
                    classify(runner, r, "chat-stream", LLAMA)
                    return
                for _event in sse_data_lines(r):
                    chunks += 1
        except httpx.HTTPError as exc:
            runner.error(f"chat-stream: {type(exc).__name__}: {exc}")
            return
        runner.oks += 1
    with runner.ctx.lock:
        interactive_latencies.append((time.monotonic() - started, during_flood))


def batch_fn(runner: Runner) -> None:
    completion(
        runner,
        LLAMA,
        runner.rng.choice(BATCH_PROMPTS),
        96,
        priority="batch",
        temperature=0.8,
        top_p=0.95,
    )


def flood_fn(runner: Runner) -> None:
    """12 concurrent unique-prefix BATCH streams: saturates the 512-block
    pool (12 x ~42 blocks, the test_priority sizing) so interactive
    arrivals exercise priority preemption. Streams abort promptly on
    quiesce/stop (counted as cancellations)."""
    runner.ctx.flood_active.set()
    runner.bump("floods")
    threads = []
    for index in range(12):
        tag = f"[flood {uuid.uuid4().hex[:8]}-{index}] "

        def one(tag=tag):
            stream_completion(
                runner,
                LLAMA,
                tag + PREFIX_FILLER,
                192,
                priority="batch",
                temperature=0,
            )

        thread = threading.Thread(target=one, daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join(timeout=REQ_TIMEOUT.read)
    runner.ctx.flood_active.clear()


def grammar_fn(runner: Runner) -> None:
    try:
        response = runner.client.post(
            "/v1/chat/completions",
            json={
                "model": LLAMA,
                "messages": [{"role": "user", "content": "Describe a pet."}],
                "response_format": RESPONSE_FORMAT,
                "max_tokens": 96,
            },
        )
    except httpx.HTTPError as exc:
        runner.error(f"grammar: {type(exc).__name__}: {exc}")
        return
    if not classify(runner, response, "grammar", LLAMA):
        return
    text = response.json()["choices"][0]["message"]["content"]
    try:
        value = json.loads(text)
        assert set(value) == {"name", "kind", "age"}
        assert isinstance(value["name"], str) and len(value["name"]) <= 12
        assert value["kind"] in ("cat", "dog", "bird")
        assert isinstance(value["age"], int) and 0 <= value["age"] <= 30
    except Exception:  # noqa: BLE001
        runner.error(f"grammar: schema-invalid output: {text[:200]!r}")
        return
    runner.oks += 1


def prefix_fn(runner: Runner) -> None:
    if runner.rng.random() < 0.6:  # warm: shared long prefix, varied tail
        tail = runner.rng.choice(
            ["Summarize.", "Why?", "Continue.", "Restate briefly."]
        )
        completion(runner, LLAMA, PREFIX_FILLER + tail, 24)
        runner.bump("warm")
    else:  # cold: unique first tokens defeat radix sharing
        completion(
            runner,
            LLAMA,
            f"[cold {uuid.uuid4().hex[:10]}] {PREFIX_FILLER[:600]} Continue.",
            24,
        )
        runner.bump("cold")


def anthropic_fn(runner: Runner) -> None:
    body = {
        "model": LLAMA,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Name two colors and stop."}],
    }
    headers = {"x-api-key": API_KEY}
    if runner.rng.random() < 0.3:
        body["stream"] = True
        try:
            with runner.client.stream(
                "POST", "/v1/messages", json=body, headers=headers
            ) as r:
                if r.status_code != 200:
                    r.read()
                    classify(runner, r, "messages-stream", LLAMA)
                    return
                saw_stop = False
                for line in r.iter_lines():
                    if line.startswith("event: message_stop"):
                        saw_stop = True
        except httpx.HTTPError as exc:
            runner.error(f"messages-stream: {type(exc).__name__}: {exc}")
            return
        if not saw_stop:
            runner.error("messages-stream: no message_stop event")
            return
        runner.oks += 1
    else:
        try:
            response = runner.client.post("/v1/messages", json=body, headers=headers)
        except httpx.HTTPError as exc:
            runner.error(f"messages: {type(exc).__name__}: {exc}")
            return
        if not classify(runner, response, "messages", LLAMA):
            return
        if not response.json()["content"]:
            runner.error("messages: empty content")
            return
        runner.oks += 1


def cancel_fn(runner: Runner) -> None:
    stream_completion(
        runner,
        LLAMA,
        runner.rng.choice(BATCH_PROMPTS),
        256,
        abort_after_chunks=runner.rng.randint(2, 4),
        priority="batch",
    )


def spec_fn(runner: Runner) -> None:
    rejects_before = runner.rejects
    prompt = runner.rng.choice(SPEC_PROMPTS)
    if runner.rng.random() < 0.5:
        completion(runner, SPEC, prompt, 48, temperature=0)
    else:
        completion(
            runner,
            SPEC,
            prompt,
            48,
            temperature=0.7,
            top_p=0.9,
            seed=runner.rng.randint(1, 10_000),
        )
    if runner.rejects > rejects_before:
        # Refused for memory: stand down instead of racing the gemma
        # burst for the next headroom slot. Under the reservation ledger
        # a spec re-warm honestly needs its whole 402 MB double pool, and
        # a 4-8 s retry cadence beats gemma's burst retries to every slot
        # the TTL valve opens — a harness livelock, seen as a starved
        # burst on the slow CI runner (run 29449124438).
        runner.extra_delay = runner.rng.uniform(20, 30)


def ttl_fn(runner: Runner) -> None:
    # Two touches, then the runner period (> ttl 75 s) lets the lease
    # expire: one idle_ttl unload + on-demand reload per cycle. Each
    # touch retries through 503 model_loading like a real client — on a
    # slow runner the reload can eat several attempts (CI run
    # 29408682271 landed 26 loading retries and only 3 successes when
    # touches gave up after one attempt). But it BACKS OFF on
    # insufficient_memory: retrying through pressure keeps touching the
    # model, which keeps its TTL lease alive and removes the lease-expiry
    # release valve the gemma bursts need to recover (run D starved a
    # burst for its full 120 s window exactly this way, ttl rejects
    # 2 -> 50).
    for _ in range(2):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not runner.ctx.should_abort():
            rejects_before = runner.rejects
            if completion(runner, TTL, "The capital of France is", 24):
                break
            if runner.rejects > rejects_before:
                return  # pressure: stand down for the whole cycle
            time.sleep(5)
        if runner.ctx.should_abort():
            return
        time.sleep(runner.rng.uniform(2, 5))


def gemma_burst_fn(runner: Runner) -> None:
    """One pressure burst: keep asking until the stack makes room (LRU
    eviction at load, TTL expiry for headroom). 503s along the way are the
    admission gate doing its job; never recovering within the window is a
    governance failure."""
    runner.bump("bursts")
    # 180 s window: on the slow CI runner a recovery legitimately chains
    # gemma load (10-30 s) after a TTL lease expiry (up to 75 s away)
    # after an eviction round-trip (up to 45 s) — 120 s left no slack
    # once the reservation ledger made spec re-warms honest competitors
    # for the same headroom.
    deadline = time.monotonic() + 180
    successes = 0
    while time.monotonic() < deadline and not runner.ctx.should_abort():
        if completion(runner, GEMMA, "A fun fact about volcanoes:", 32):
            successes += 1
            if successes >= 3:
                break
            time.sleep(runner.rng.uniform(4, 8))
        else:
            time.sleep(8)
    if successes == 0 and not runner.ctx.should_abort():
        # A burst cut short by a quiesce checkpoint is neither a success
        # nor a governance failure — only a full window without a 200 is.
        runner.bump("failed_bursts")
        runner.error("gemma burst never recovered to a 200 within 180s")


def python_fn(runner: Runner) -> None:
    completion(runner, PYSMOL, "The capital of France is", 24)


canary_texts: dict[str, list[tuple[float, str]]] = {LLAMA: [], SPEC: []}


def canary_fn_for(model: str, prompt: str):
    def canary(runner: Runner) -> None:
        text = completion(runner, model, prompt, 48, temperature=0)
        if text is None:
            # spec canary may be admission-rejected mid-burst (rejects
            # counted); identity is asserted over the successful samples.
            return
        with runner.ctx.lock:
            canary_texts[model].append((runner.ctx.elapsed(), text))

    return canary


# ---------------------------------------------------------------------------
# Checkpoints (quiesced) + sampling
# ---------------------------------------------------------------------------


@dataclass
class Checkpoint:
    label: str
    t: float
    busy_left: list[str]
    gateway_rss: int
    per_model: dict[str, dict[str, float]]


def take_sample(stack, gateway_pid: int):
    metrics = scrape(stack.base_url)
    rss_kb = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(gateway_pid)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    gateway_rss = int(rss_kb) * 1024 if rss_kb else 0
    ready = httpx.get(f"{stack.base_url}/readyz", timeout=15)
    return metrics, gateway_rss, ready.json()["models"]


def model_snapshot(metrics, model: str) -> dict[str, float]:
    def get(name: str) -> float:
        value = mval(metrics, name, model=model)
        return 0.0 if value is None else value

    return {
        "footprint": get("kiln_worker_memory_bytes"),
        "rss": get("kiln_worker_process_rss_bytes"),
        "active": get("kiln_worker_mlx_active_bytes"),
        "cache": get("kiln_worker_mlx_cache_bytes"),
        "live": get("kiln_worker_mlx_live_objects"),
        "kv_alloc": get("kiln_worker_kv_pool_allocated_bytes"),
        "up": get("kiln_worker_up"),
        # Instrumentation only (PROGRESS 2026-07-23 root-cause): the SSD
        # flush-queue depth paired with the live-object sample above —
        # both fields ride the same heartbeat, so a live excursion can be
        # attributed to (or cleared of) an in-flight block capture. Logged
        # in the checkpoint table; no gate consumes these.
        "flush": get("kiln_worker_flush_pending_blocks"),
        "ssd_w": get("kiln_worker_ssd_writes_total"),
        # generation marker: any unload or restart recycles the process
        "generation": msum(metrics, "kiln_worker_unloads_total", model=model)
        + msum(metrics, "kiln_worker_restarts_total", model=model),
    }


def sample_flush_idle(stack, gateway_pid: int, label: str):
    """Takes the checkpoint sample only once every rust worker's SSD
    write-behind queue reads empty (`kiln_worker_flush_pending_blocks`
    == 0) in the SAME scrape that supplies the gated live-object value —
    the flush-idle sampling semantics from the module docstring. `live`
    and `flush_pending` ride one heartbeat, so a scrape with fp == 0 has
    its live value read at an instant with no block capture in flight.
    Bounded: on deadline the last scrape is used unchanged — a stuck
    queue or a real leak is still sampled and still gated (a parked
    handle elevates every sample, flush-idle ones included)."""
    deadline = time.monotonic() + FLUSH_IDLE_DEADLINE_S
    while True:
        sample = take_sample(stack, gateway_pid)
        metrics = sample[0]
        pending = {
            m: mval(metrics, "kiln_worker_flush_pending_blocks", model=m) or 0
            for m in RUST_MODELS
        }
        if all(v == 0 for v in pending.values()):
            return sample
        if time.monotonic() >= deadline:
            stuck = {m: int(v) for m, v in pending.items() if v}
            print(
                f"checkpoint {label}: flush queues still pending after "
                f"{FLUSH_IDLE_DEADLINE_S:.0f}s: {stuck} — sampling anyway"
            )
            return sample
        time.sleep(2.0)


def quiesce(ctx: Ctx, runners: list[Runner], label: str, gateway_pid: int):
    ctx.gate.clear()
    deadline = time.monotonic() + 180
    while any(r.busy for r in runners) and time.monotonic() < deadline:
        time.sleep(0.25)
    busy_left = [r.label for r in runners if r.busy]
    time.sleep(6.0)  # let heartbeat/stats polling catch up (1 s cadence)
    metrics, gateway_rss, _ready = sample_flush_idle(ctx.stack, gateway_pid, label)
    checkpoint = Checkpoint(
        label=label,
        t=ctx.elapsed(),
        busy_left=busy_left,
        gateway_rss=gateway_rss,
        per_model={m: model_snapshot(metrics, m) for m in RUST_MODELS},
    )
    ctx.gate.set()
    return checkpoint, metrics


# ---------------------------------------------------------------------------
# The soak
# ---------------------------------------------------------------------------


def test_full_stack_soak():
    llama_path = require("llama-3.2-1b-4bit")
    qwen25_path = require("qwen2.5-0.5b-4bit")
    gemma_path = require("gemma-3-1b-it-4bit")
    smol_path = require("smollm2-135m-bf16")

    duration_s = SOAK_MINUTES * 60.0
    full_run = SOAK_MINUTES >= GATE_FULL_MINUTES
    failures: list[str] = []

    models = [
        (LLAMA, "rust", llama_path, "pinned = true"),
        (
            SPEC,
            "rust",
            qwen25_path,
            f'[model.speculative]\ndraft = "{qwen25_path}"\ngamma = {SPEC_GAMMA}',
        ),
        (TTL, "rust", qwen25_path, f"ttl_seconds = {TTL_SECONDS}"),
        (GEMMA, "rust", gemma_path, f"ttl_seconds = {GEMMA_TTL_SECONDS}"),
        (PYSMOL, "python", smol_path, ""),
    ]
    extra = f"[memory]\nbudget_bytes = {BUDGET_BYTES}\nmin_available_bytes = 0\n"

    with running_stack(models, extra_toml=extra) as stack:
        stack.wait_ready()
        gateway_pid = stack.gateway.pid
        ctx = Ctx(
            stack=stack,
            stop=threading.Event(),
            gate=threading.Event(),
            flood_active=threading.Event(),
        )
        ctx.gate.set()
        ctx.started = time.monotonic()

        warm = Runner(ctx, "warmup", lambda r: None, (1, 1))

        # -- Warmup: materialize pools in budget-safe order (llama first —
        # it must never be admission-rejected afterwards), populate the
        # shared prefix, engage speculation. gemma stays cold: its pools
        # materialize inside the pressure bursts.
        def must_warm(model: str, prompt: str, max_tokens: int, **extra):
            """Warms with retries: on constrained machines (the CI runner's
            idle footprints + caches run fatter than the dev box, run
            29398308457) the later warms can be legitimately refused with
            insufficient_memory until a TTL lease expires and frees room
            (gemma's 90 s lease is the guaranteed release), or answered
            model_loading if the model itself TTL-cycled while earlier
            warms ran. Both are the system's own recovery paths — wait
            them out instead of assuming dev-machine margins."""
            deadline = time.monotonic() + 240
            before = warm.oks
            while time.monotonic() < deadline:
                completion(warm, model, prompt, max_tokens, **extra)
                if warm.oks > before:
                    return
                time.sleep(6)
            raise AssertionError(
                f"{model} warmup failed for 240s (rejects={warm.rejects}, "
                f"loading_retries={warm.extra.get('loading_retry', 0)}, "
                f"last_503={warm.last_503!r}, errors={ctx.hard_errors[-3:]})"
            )

        must_warm(LLAMA, "Warmup. " + PREFIX_FILLER, 32)
        # 64 greedy tokens: speculation must engage so the draft-side pools
        # materialize now, not mid-run.
        must_warm(SPEC, SPEC_PROMPTS[0], 64, temperature=0)
        must_warm(TTL, "The capital of France is", 16)
        must_warm(PYSMOL, "The capital of France is", 16)
        warm.client.close()

        runners = [
            Runner(ctx, "interactive", interactive_fn, (2.5, 5.0)),
            Runner(ctx, "batch", batch_fn, (6, 10)),
            Runner(ctx, "flood", flood_fn, (300, 360), initial_delay=200),
            Runner(ctx, "grammar", grammar_fn, (15, 25)),
            Runner(ctx, "prefix", prefix_fn, (12, 18)),
            Runner(ctx, "anthropic", anthropic_fn, (20, 30)),
            Runner(ctx, "cancel", cancel_fn, (15, 25)),
            Runner(ctx, "spec", spec_fn, (4, 8)),
            Runner(ctx, "ttl", ttl_fn, (95, 125)),
            Runner(ctx, "gemma-burst", gemma_burst_fn, (240, 300), initial_delay=150),
            Runner(ctx, "python", python_fn, (14, 22)),
            Runner(
                ctx,
                "canary-llama",
                canary_fn_for(LLAMA, CANARY_LLAMA_PROMPT),
                (58, 62),
            ),
            Runner(
                ctx,
                "canary-spec",
                canary_fn_for(SPEC, CANARY_SPEC_PROMPT),
                (72, 78),
                initial_delay=5,
            ),
        ]

        # Baseline checkpoint before load starts: post-warmup quiesced state.
        baseline, _baseline_metrics = quiesce(ctx, [], "baseline", gateway_pid)
        checkpoints = [baseline]

        for runner in runners:
            runner.start()

        # -- Main loop: 10 s samples, quiesced checkpoint every ~6 min.
        samples: list[dict] = []
        ledger_violations: list[str] = []
        crashed_states: list[str] = []
        # Worker Stats counters reset with each eviction/reload generation
        # (spec-qwen25 cycled 13-21 times in the local runs), so the
        # speculation totals are accumulated across resets here: bank the
        # running value whenever it drops.
        spec_totals = {"proposed": [0.0, 0.0], "accepted": [0.0, 0.0]}
        checkpoint_interval = max(300.0, duration_s / 5.0)
        next_checkpoint = checkpoint_interval
        next_status = 60.0
        while ctx.elapsed() < duration_s:
            time.sleep(10.0)
            metrics, gateway_rss, ready = take_sample(stack, gateway_pid)
            now = ctx.elapsed()
            for key, name in (
                ("proposed", "kiln_worker_spec_tokens_proposed_total"),
                ("accepted", "kiln_worker_spec_tokens_accepted_total"),
            ):
                banked_prev = spec_totals[key]
                current = mval(metrics, name, model=SPEC) or 0.0
                if current < banked_prev[1]:
                    banked_prev[0] += banked_prev[1]
                banked_prev[1] = current
            # The ENFORCED invariant is committed bytes (weights +
            # materialized pools) <= budget: every load/pool-growth
            # admission bounds it conservatively. The raw ledger (which
            # adds mlx_cache and in-flight compute buffers) has NO
            # enforcement lever for cache drift on materialized pools —
            # the recorded open Phase 9 gap (continuous-pressure
            # eviction) — so its overshoot is measured and reported, not
            # gated.
            committed = sum(
                (mval(metrics, "kiln_worker_weights_bytes", model=m) or 0)
                + (mval(metrics, "kiln_worker_kv_pool_allocated_bytes", model=m) or 0)
                for m in (*RUST_MODELS, PYSMOL)
            )
            row = {
                "t": now,
                "gateway_rss": gateway_rss,
                "used": mval(metrics, "kiln_memory_used_bytes") or 0,
                "budget": mval(metrics, "kiln_memory_budget_bytes") or 0,
                "committed": committed,
                "reserved": mval(metrics, "kiln_memory_reserved_bytes") or 0,
                "models": {m: model_snapshot(metrics, m) for m in RUST_MODELS},
                "py_rss": mval(metrics, "kiln_worker_process_rss_bytes", model=PYSMOL)
                or 0,
                # Per-model unload+restart totals, for correlating any
                # severed-stream 502 with the deliberate unload that cut it.
                "gen": {
                    m: msum(metrics, "kiln_worker_unloads_total", model=m)
                    + msum(metrics, "kiln_worker_restarts_total", model=m)
                    for m in (*RUST_MODELS, PYSMOL)
                },
            }
            samples.append(row)
            if row["committed"] > row["budget"]:
                ledger_violations.append(
                    f"t+{now:.0f}s committed={row['committed']:.0f} > "
                    f"budget={row['budget']:.0f}"
                )
            for model, state in ready.items():
                if "crash" in state.lower() or "unhealthy" in state.lower():
                    crashed_states.append(f"t+{now:.0f}s {model}={state}")
            if now >= next_status:
                next_status += 60.0
                up = {m: int(row["models"][m]["up"]) for m in RUST_MODELS}
                oks = sum(r.oks for r in runners)
                rejects = sum(r.rejects for r in runners)
                print(
                    f"[soak t+{now:6.0f}s] committed="
                    f"{row['committed'] / 1e9:.2f} used="
                    f"{row['used'] / 1e9:.2f}/{row['budget'] / 1e9:.2f}GB "
                    f"gw_rss={gateway_rss / 1e6:.0f}MB up={up} ok={oks} "
                    f"rej={rejects} err={len(ctx.hard_errors)}",
                    flush=True,
                )
            if now >= next_checkpoint and duration_s - now > 60:
                next_checkpoint += checkpoint_interval
                checkpoint, _ = quiesce(ctx, runners, f"t+{now:.0f}s", gateway_pid)
                checkpoints.append(checkpoint)

        # -- Drain and final quiesced checkpoint.
        ctx.stop.set()
        for runner in runners:
            runner.join(timeout=REQ_TIMEOUT.read + 30)
        still_alive = [r.label for r in runners if r.is_alive()]

        final_canary = Runner(ctx, "final-canary", lambda r: None, (1, 1))
        llama_final = completion(
            final_canary, LLAMA, CANARY_LLAMA_PROMPT, 48, temperature=0
        )
        spec_final = completion(
            final_canary, SPEC, CANARY_SPEC_PROMPT, 48, temperature=0
        )
        if llama_final:
            canary_texts[LLAMA].append((ctx.elapsed(), llama_final))
        if spec_final:
            canary_texts[SPEC].append((ctx.elapsed(), spec_final))
        final_canary.client.close()

        time.sleep(6.0)
        final_metrics, final_gateway_rss, final_ready = sample_flush_idle(
            stack, gateway_pid, "final"
        )
        checkpoints.append(
            Checkpoint(
                label="final",
                t=ctx.elapsed(),
                busy_left=[],
                gateway_rss=final_gateway_rss,
                per_model={m: model_snapshot(final_metrics, m) for m in RUST_MODELS},
            )
        )
        # Close the correlation window for any 502 severed during the
        # drain tail (after the sampler loop stopped): without a sample
        # past the event, its ±60 s unload-counter check cannot see the
        # unload that cut it.
        samples.append(
            {
                "t": ctx.elapsed(),
                "gateway_rss": final_gateway_rss,
                "used": mval(final_metrics, "kiln_memory_used_bytes") or 0,
                "budget": mval(final_metrics, "kiln_memory_budget_bytes") or 0,
                "committed": sum(
                    (mval(final_metrics, "kiln_worker_weights_bytes", model=m) or 0)
                    + (
                        mval(
                            final_metrics,
                            "kiln_worker_kv_pool_allocated_bytes",
                            model=m,
                        )
                        or 0
                    )
                    for m in (*RUST_MODELS, PYSMOL)
                ),
                "models": {m: model_snapshot(final_metrics, m) for m in RUST_MODELS},
                "py_rss": mval(
                    final_metrics, "kiln_worker_process_rss_bytes", model=PYSMOL
                )
                or 0,
                "gen": {
                    m: msum(final_metrics, "kiln_worker_unloads_total", model=m)
                    + msum(final_metrics, "kiln_worker_restarts_total", model=m)
                    for m in (*RUST_MODELS, PYSMOL)
                },
            }
        )

        # ------------------------------------------------------------------
        # Report
        # ------------------------------------------------------------------
        print("\n================ SOAK REPORT ================")
        print(f"duration: {ctx.elapsed():.0f}s (requested {duration_s:.0f}s)")

        print("\n-- quiesced checkpoints (live = mlx live objects) --")
        header = f"{'label':>12} {'t':>7} {'gw_rss':>8}  " + "  ".join(
            f"{m:>28}" for m in RUST_MODELS
        )
        print(header)
        print(
            f"{'':>29}"
            + "  ".join(f"{'live/act_MB/cache_MB/gen/fp/sw':>28}" for _ in RUST_MODELS)
        )
        for cp in checkpoints:
            cells = []
            for m in RUST_MODELS:
                s = cp.per_model[m]
                # fp = flush_pending_blocks and sw = ssd_writes_total at
                # the SAME heartbeat as `live` — instrumentation for the
                # PR #35 +2 attribution (PROGRESS 2026-07-23); print-only.
                cells.append(
                    f"{int(s['live'])}/{s['active'] / 1e6:.1f}/"
                    f"{s['cache'] / 1e6:.1f}/g{int(s['generation'])}"
                    f"/fp{int(s['flush'])}/sw{int(s['ssd_w'])}"
                )
            print(
                f"{cp.label:>12} {cp.t:7.0f} {cp.gateway_rss / 1e6:7.1f}M  "
                + "  ".join(f"{c:>28}" for c in cells)
            )
            if cp.busy_left:
                failures.append(
                    f"checkpoint {cp.label}: runners still busy after 180s: "
                    f"{cp.busy_left}"
                )

        # RSS trends. Two windows per process: the middle-out view (last
        # 2/3, reported) and the GATED view (final 1/3). The first local
        # 30-min run showed plateau-shaped growth — checkpoint deltas of
        # +15.0/+9.8/+3.4/-0.2/+2.7 MB per ~6 min for the gateway —
        # i.e. allocator/arena growth flattening out, not a linear leak;
        # a genuine per-request leak stays linear through the final
        # third, which is where the gate looks.
        def rss_series(extract):
            report_w = [
                (s["t"], extract(s))
                for s in samples
                if s["t"] >= duration_s / 3.0 and extract(s) > 0
            ]
            gate_w = [(t, v) for t, v in report_w if t >= duration_s * 2.0 / 3.0]
            return report_w, slope_kb_per_min(report_w), slope_kb_per_min(gate_w)

        gw_points, gw_slope_23, gw_slope = rss_series(lambda s: s["gateway_rss"])
        llama_points, llama_slope_23, llama_slope = rss_series(
            lambda s: s["models"][LLAMA]["rss"]
        )
        py_points, py_slope_23, py_slope = rss_series(lambda s: s["py_rss"])
        print("\n-- RSS trends (report window: last 2/3; gate window: last 1/3) --")
        for name, points, slope_23, slope_3 in (
            ("gateway", gw_points, gw_slope_23, gw_slope),
            (LLAMA, llama_points, llama_slope_23, llama_slope),
            (PYSMOL, py_points, py_slope_23, py_slope),
        ):
            if points:
                print(
                    f"{name}: {points[0][1] / 1e6:.1f} -> "
                    f"{points[-1][1] / 1e6:.1f} MB, slope "
                    f"{slope_23:+.1f} KiB/min (last 2/3), "
                    f"{slope_3:+.1f} KiB/min (last 1/3, gated)"
                )

        # The known Phase 9 residual gap, quantified: raw ledger (adds
        # mlx_cache + in-flight compute buffers) vs budget over the run.
        if samples:
            worst = max(samples, key=lambda s: s["used"] - s["budget"])
            over = [s for s in samples if s["used"] > s["budget"]]
            peak_committed = max(s["committed"] for s in samples)
            peak_reserved = max(s.get("reserved", 0) for s in samples)
            uncovered = sum(
                v
                for name, labels, v in final_metrics
                if name == "kiln_admission_uncovered_bytes_total"
            )
            print(
                f"\n-- ledger vs budget --\n"
                f"committed (weights+pools, the enforced bound): peak "
                f"{peak_committed / 1e9:.2f} GB of {BUDGET_BYTES / 1e9:.2f} "
                f"GB budget\nreservation ledger: peak in-flight "
                f"{peak_reserved / 1e6:.0f} MB, uncovered growth "
                f"{uncovered / 1e6:.1f} MB (must be 0)\n"
                f"raw used (adds caches/compute buffers): above "
                f"budget in {len(over)}/{len(samples)} samples, worst "
                f"t+{worst['t']:.0f}s used={worst['used'] / 1e9:.2f} GB "
                f"(+{(worst['used'] - worst['budget']) / 1e6:.0f} MB) — the "
                f"recorded cache-drift gap (no continuous-pressure "
                f"eviction yet)"
            )

        print("\n-- traffic --")
        for runner in runners:
            extras = " ".join(f"{k}={int(v)}" for k, v in sorted(runner.extra.items()))
            print(
                f"{runner.label:>14}: ok={runner.oks} "
                f"rejected={runner.rejects} cancelled={runner.cancelled} "
                f"{extras}"
            )
        for model in (LLAMA, SPEC, TTL, GEMMA, PYSMOL):
            unloads = {
                reason: msum(
                    final_metrics,
                    "kiln_worker_unloads_total",
                    model=model,
                    reason=reason,
                )
                for reason in ("evicted", "idle_ttl", "over_budget")
            }
            restarts = msum(final_metrics, "kiln_worker_restarts_total", model=model)
            rejects = msum(final_metrics, "kiln_admission_rejects_total", model=model)
            print(
                f"{model:>14}: unloads={unloads} restarts={int(restarts)} "
                f"admission_rejects={int(rejects)}"
            )
        preempted = (
            mval(final_metrics, "kiln_worker_requests_preempted_total", model=LLAMA)
            or 0
        )
        cancelled = (
            mval(final_metrics, "kiln_worker_requests_cancelled_total", model=LLAMA)
            or 0
        )
        prefix_reused = (
            mval(
                final_metrics,
                "kiln_worker_prefix_tokens_reused_total",
                model=LLAMA,
            )
            or 0
        )
        ssd_writes = (
            mval(final_metrics, "kiln_worker_ssd_writes_total", model=LLAMA) or 0
        )
        # Fold the final scrape into the cross-generation accumulators.
        for key, name in (
            ("proposed", "kiln_worker_spec_tokens_proposed_total"),
            ("accepted", "kiln_worker_spec_tokens_accepted_total"),
        ):
            banked_prev = spec_totals[key]
            current = mval(final_metrics, name, model=SPEC) or 0.0
            if current < banked_prev[1]:
                banked_prev[0] += banked_prev[1]
            banked_prev[1] = current
        proposed = sum(spec_totals["proposed"])
        accepted = sum(spec_totals["accepted"])
        print(
            f"\nllama: preempted={int(preempted)} "
            f"worker_cancelled={int(cancelled)} "
            f"prefix_reused={int(prefix_reused)} ssd_writes={int(ssd_writes)}"
        )
        acceptance = accepted / proposed if proposed else 0.0
        print(
            f"spec (run total across generations): proposed={int(proposed)} "
            f"accepted={int(accepted)} rate={acceptance:.2f} "
            f"(final generation: "
            f"{int(spec_totals['proposed'][1])}/{int(spec_totals['accepted'][1])})"
        )

        lat_normal = sorted(d for d, f in interactive_latencies if not f)
        lat_flood = sorted(d for d, f in interactive_latencies if f)

        def pct(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            return values[min(len(values) - 1, int(p * len(values)))]

        print(
            f"interactive latency: normal n={len(lat_normal)} "
            f"p50={pct(lat_normal, 0.5):.2f}s p95={pct(lat_normal, 0.95):.2f}s "
            f"max={max(lat_normal, default=0):.2f}s | during-flood "
            f"n={len(lat_flood)} p50={pct(lat_flood, 0.5):.2f}s "
            f"p95={pct(lat_flood, 0.95):.2f}s "
            f"max={max(lat_flood, default=0):.2f}s"
        )
        print(
            f"canaries: llama n={len(canary_texts[LLAMA])} "
            f"spec n={len(canary_texts[SPEC])}"
        )
        if canary_texts[LLAMA]:
            print(f"llama canary text: {canary_texts[LLAMA][0][1][:80]!r}")
        if canary_texts[SPEC]:
            print(f"spec canary text:  {canary_texts[SPEC][0][1][:80]!r}")

        # ------------------------------------------------------------------
        # Gates
        # ------------------------------------------------------------------
        # Severed-stream 502s: each must be the bounded-drain tail of a
        # deliberate unload of that model — the per-model unload counter
        # must move within ±60 s of the error. Uncorrelated ones are hard
        # errors (with restarts==0 asserted separately, a correlated one
        # cannot be a hidden crash: crashes increment restarts, unloads
        # don't).
        severed_ok = 0
        for model, when, detail in ctx.severed:
            if model == LLAMA:
                failures.append(
                    f"pinned {LLAMA} had a severed request (it is never "
                    f"unloaded, so this cannot be a drain tail): {detail}"
                )
                continue
            window = [s for s in samples if abs(s["t"] - when) <= 60]
            moved = len(window) >= 2 and window[-1]["gen"].get(model, 0) > window[0][
                "gen"
            ].get(model, 0)
            if moved:
                severed_ok += 1
                print(
                    f"severed-by-drain 502 accepted: {model} at t+{when:.1f}s "
                    f"(unload counter moved in ±60s window)"
                )
            else:
                failures.append(
                    f"502 worker_crashed NOT correlated with an unload of "
                    f"{model} at t+{when:.1f}s: {detail}"
                )
        if severed_ok > SEVERED_MAX:
            failures.append(
                f"{severed_ok} drain-severed requests exceeds the "
                f"SEVERED_MAX={SEVERED_MAX} bound — eviction is cutting "
                "in-flight work too often to be the rare bounded tail"
            )
        if ctx.hard_errors:
            preview = "\n  ".join(ctx.hard_errors[:20])
            failures.append(
                f"{len(ctx.hard_errors)} hard errors (first 20):\n  {preview}"
            )
        if still_alive:
            failures.append(f"runners failed to stop: {still_alive}")
        if ledger_violations:
            failures.append(
                f"committed bytes (weights+pools) exceeded budget "
                f"{len(ledger_violations)}x: {ledger_violations[:5]}"
            )
        if uncovered > 0:
            failures.append(
                f"{uncovered:.0f} bytes of pool growth were never covered "
                "by an admission reservation (kiln_admission_uncovered_"
                "bytes_total — unpriced memory materialized)"
            )
        if crashed_states:
            failures.append(f"crashed/unhealthy states: {crashed_states[:5]}")

        # Determinism canaries: every sample identical, per model.
        for model, minimum in ((LLAMA, 15), (SPEC, 8)):
            texts = canary_texts[model]
            distinct = {text for _, text in texts}
            if len(distinct) > 1:
                failures.append(
                    f"{model} canary NON-DETERMINISTIC: "
                    f"{len(distinct)} distinct outputs across "
                    f"{len(texts)} samples: "
                    + " | ".join(repr(t[:60]) for t in sorted(distinct))
                )
            if full_run and len(texts) < minimum:
                failures.append(
                    f"{model} canary: only {len(texts)} samples (need >= {minimum})"
                )

        # Live-object gate: equal at every quiesced checkpoint within a
        # (generation, materialized-pool) group; strictly for llama-int,
        # which must stay in ONE group for the whole run.
        groups: dict[tuple, list[tuple[str, float]]] = {}
        for cp in checkpoints:
            for model in RUST_MODELS:
                s = cp.per_model[model]
                if s["up"] != 1:
                    continue  # unloaded at this checkpoint
                key = (model, s["generation"], s["kv_alloc"])
                groups.setdefault(key, []).append((cp.label, s["live"]))
        for (model, gen, kv), entries in sorted(groups.items()):
            values = [live for _, live in entries]
            floor = min(values)
            # Return-to-baseline is the leak signal: the group's LAST
            # quiesced sample must sit at its drained floor. Bounded
            # upward excursions at interior checkpoints are engine-thread
            # maintenance caught mid-flight (2 handles per KV block being
            # copied to SSD — see LIVE_TRANSIENT_ALLOWANCE), and they must
            # drain, not accumulate.
            if values[-1] != floor:
                failures.append(
                    f"mlx live objects did not return to baseline for "
                    f"{model} (gen {gen:.0f}, pool {kv / 1e6:.0f}MB): "
                    f"{entries} (floor {floor:.0f})"
                )
            if max(values) > floor + LIVE_TRANSIENT_ALLOWANCE:
                failures.append(
                    f"mlx live objects excursion beyond the maintenance "
                    f"allowance for {model} (gen {gen:.0f}): {entries} "
                    f"(floor {floor:.0f}, allowance {LIVE_TRANSIENT_ALLOWANCE})"
                )
            if any(live < 0 for _, live in entries):
                failures.append(
                    f"NEGATIVE mlx live objects for {model}: {entries} (double-free)"
                )
        llama_groups = [k for k in groups if k[0] == LLAMA]
        if len(llama_groups) != 1:
            failures.append(
                f"{LLAMA} must hold one (generation, pool) group across the "
                f"whole run (pinned, warmed in warmup); saw {llama_groups}"
            )

        # mlx_active band + cache cap within each group at checkpoints.
        for (model, gen, kv), entries in sorted(groups.items()):
            actives = [
                cp.per_model[model]["active"]
                for cp in checkpoints
                if cp.per_model[model]["up"] == 1
                and cp.per_model[model]["generation"] == gen
                and cp.per_model[model]["kv_alloc"] == kv
            ]
            if actives and max(actives) - min(actives) > ACTIVE_BAND_BYTES:
                failures.append(
                    f"mlx_active drift for {model} gen {gen:.0f}: "
                    f"{min(actives):.0f}..{max(actives):.0f} "
                    f"(> {ACTIVE_BAND_BYTES} band)"
                )
        for cp in checkpoints:
            for model in RUST_MODELS:
                if cp.per_model[model]["cache"] > CACHE_CAP_BYTES:
                    failures.append(
                        f"mlx_cache above cap for {model} at {cp.label}: "
                        f"{cp.per_model[model]['cache']:.0f}"
                    )

        # RSS: absolute working-set caps only (see the GW_RSS_FINAL_CAP
        # comment for the ten-run evidence that every derivative measure
        # aliases page-reclaim timing). Slopes/deltas are printed above
        # for the record; the mlx-side gates carry fine-grained leak
        # detection.
        gw_final = gw_points[-1][1] if gw_points else 0
        if gw_final > GW_RSS_FINAL_CAP:
            failures.append(
                f"gateway RSS working set {gw_final / 1e6:.1f} MB exceeds "
                f"the {GW_RSS_FINAL_CAP / 1e6:.0f} MB cap (2x measured plateau)"
            )
        mid_band = [
            v for t, v in gw_points if duration_s * 0.5 <= t <= duration_s * 0.6
        ]
        late_band = [v for t, v in gw_points if t >= duration_s - 180]
        if mid_band and late_band:
            late_delta = sum(late_band) / len(late_band) - sum(mid_band) / len(mid_band)
            print(
                f"gateway RSS late delta (mean last 3 min vs mean of "
                f"50-60% window): {late_delta / 1e6:+.1f} MB (reported, "
                "not gated: aliases reclaim-dip timing)"
            )
        llama_peak = max((v for _, v in llama_points), default=0)
        if llama_peak > LLAMA_RSS_CAP:
            failures.append(
                f"{LLAMA} RSS peaked at {llama_peak / 1e6:.1f} MB "
                f"(> {LLAMA_RSS_CAP / 1e6:.0f} MB cap: full weight mmap "
                "resident + heap slack)"
            )
        # py-smollm is EVICTABLE (9-24 evictions per run): its RSS series
        # crosses process generations, each ramping ~50 -> ~470 MB as
        # python mlx materializes weights, so a cross-generation slope
        # measures churn phase, not leaks (observed -17,490 and +4,716
        # KiB/min on back-to-back clean runs). Gate the working set
        # instead: every sample under an absolute cap.
        py_peak = max((v for _, v in py_points), default=0)
        if py_peak > PY_RSS_CAP:
            failures.append(
                f"{PYSMOL} RSS peaked at {py_peak / 1e6:.1f} MB "
                f"(> {PY_RSS_CAP / 1e6:.0f} MB cap; warm working set is "
                "~450-470 MB)"
            )

        # Governance sanity.
        llama_unloads = sum(
            msum(final_metrics, "kiln_worker_unloads_total", model=LLAMA, reason=reason)
            for reason in ("evicted", "idle_ttl", "over_budget")
        )
        if llama_unloads:
            failures.append(f"pinned {LLAMA} was unloaded {llama_unloads}x")
        llama_rejects = msum(final_metrics, "kiln_admission_rejects_total", model=LLAMA)
        if llama_rejects:
            failures.append(
                f"warm pinned {LLAMA} was admission-rejected "
                f"{llama_rejects:.0f}x (growth should be 0)"
            )
        total_restarts = sum(
            msum(final_metrics, "kiln_worker_restarts_total", model=m)
            for m in (LLAMA, SPEC, TTL, GEMMA, PYSMOL)
        )
        if total_restarts:
            failures.append(f"crash-restarts observed: {total_restarts:.0f}")
        if mval(final_metrics, "kiln_worker_up", model=LLAMA) != 1:
            failures.append(f"{LLAMA} not up at the end")
        for model, state in final_ready.items():
            if "crash" in state.lower():
                failures.append(f"final readyz: {model}={state}")

        if lat_flood and max(lat_flood) > INTERACTIVE_P100_S:
            failures.append(
                f"interactive request took {max(lat_flood):.1f}s during a "
                f"flood (> {INTERACTIVE_P100_S}s: priority admission failed)"
            )
        if lat_normal and max(lat_normal) > INTERACTIVE_P100_S:
            failures.append(
                f"interactive request took {max(lat_normal):.1f}s outside "
                f"floods (> {INTERACTIVE_P100_S}s)"
            )

        if full_run:
            by_label = {r.label: r for r in runners}
            minimums = {
                "interactive": 100,
                "batch": 60,
                "grammar": 30,
                "prefix": 40,
                "anthropic": 25,
                "spec": 80,
                "ttl": 8,
                "python": 30,
            }
            for label, minimum in minimums.items():
                if by_label[label].oks < minimum:
                    failures.append(
                        f"class '{label}' only {by_label[label].oks} "
                        f"successes (need >= {minimum})"
                    )
            burst = by_label["gemma-burst"]
            if burst.extra.get("bursts", 0) < 3:
                failures.append(f"only {burst.extra.get('bursts', 0)} gemma bursts ran")
            if burst.extra.get("failed_bursts", 0):
                failures.append(
                    f"{burst.extra['failed_bursts']:.0f} gemma bursts never recovered"
                )
            idle_ttl_unloads = msum(
                final_metrics, "kiln_worker_unloads_total", reason="idle_ttl"
            )
            evictions = msum(
                final_metrics, "kiln_worker_unloads_total", reason="evicted"
            )
            total_rejects = sum(r.rejects for r in runners)
            if idle_ttl_unloads < 2:
                failures.append(
                    f"only {idle_ttl_unloads:.0f} idle_ttl unloads (need>=2)"
                )
            if evictions < 2:
                failures.append(f"only {evictions:.0f} evictions (need >= 2)")
            if total_rejects < 1:
                failures.append(
                    "no admission rejections at all — the pressure scenario "
                    "did not exercise the gate"
                )
            if preempted < 1:
                failures.append("no preemptions despite 12-stream floods")
            if cancelled < 5:
                failures.append(
                    f"only {cancelled:.0f} worker-side cancellations "
                    "(client aborts must reach the worker)"
                )
            if prefix_reused < 10_000:
                failures.append(
                    f"prefix reuse only {prefix_reused:.0f} tokens "
                    "(warm-prefix traffic should exceed 10k)"
                )
            if ssd_writes < 1:
                failures.append("no SSD tier writes despite pool churn")
            if proposed < 500:
                failures.append(
                    f"speculation barely ran: proposed={proposed:.0f} "
                    "(run total across generations)"
                )
            if proposed and acceptance < 0.5:
                failures.append(
                    f"spec acceptance {acceptance:.2f} < 0.5 (SPEC §11.3 "
                    "same-family sanity)"
                )

        print("\n-- verdict --")
        if failures:
            print(f"FAIL: {len(failures)} gate(s) violated")
        else:
            print("PASS: all gates held")
        assert not failures, "\n".join(failures)
