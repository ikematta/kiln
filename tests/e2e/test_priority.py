"""Phase 9 part 2: INTERACTIVE/BATCH priority classes (SPEC §6.1 / §12).

The Phase 9 acceptance scenario, against a real stack: flood the worker
with BATCH-priority requests until its KV pool is saturated, then send
INTERACTIVE requests and confirm their TTFT is not meaningfully degraded —
because the scheduler admits INTERACTIVE arrivals by preempting BATCH
work, instead of queueing them behind it.

Threshold (stated, per SPEC §12 Phase 9 "interactive TTFT p95 unaffected
>2x"): the worst flooded interactive TTFT must stay within 2x the worst
unloaded baseline TTFT, plus a 500 ms absolute floor that absorbs timer
noise and the one-decode-step admission latency on sub-second baselines.
The failure mode this discriminates against is not subtle: without
priority admission the probe waits for a BATCH request to finish naturally
(512 tokens x tens of ms = >10 s).

Numbers: llama-3.2-1b pool = 512 blocks x 32 = 16384 token slots. The
flood (12 x ~1150-token prompts growing by 512) demands ~20k slots, so
the pool saturates while every flood stream is still running. Each
interactive probe (~1150-token prompt) needs ~37 free blocks, forcing the
admission path to reclaim them from BATCH victims.
"""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest
from conftest import API_KEY, model_dir, running_stack
from test_lifecycle import metric_value

FLOOD_STREAMS = 12
FLOOD_MAX_TOKENS = 512
PROBE_MAX_TOKENS = 32
BASELINE_PROBES = 4
FLOODED_PROBES = 4
TTFT_RATIO = 2.0
TTFT_FLOOR_S = 0.5

# ~96 words -> ~1150 llama tokens per repetition block. Each request gets
# a unique numbered prefix so the radix prefix cache cannot share blocks
# between streams (sharing would deflate the pool pressure this test needs).
FILLER = (
    "Paged attention splits the key-value cache into fixed-size blocks so "
    "that requests can grow without contiguous reservations. "
) * 72


def prompt_for(tag: str) -> str:
    return f"[request {tag}] {FILLER}"


def stream_completion(
    stack, tag: str, priority: str, max_tokens: int
) -> tuple[float | None, str | None, int]:
    """Runs one streaming completion; returns (ttft_seconds, finish_reason,
    status_code). TTFT is the wall time from request start to the first SSE
    data chunk that carries completion text."""
    started = time.monotonic()
    ttft = None
    finish = None
    with httpx.stream(
        "POST",
        f"{stack.base_url}/v1/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": stack.model_id,
            "prompt": prompt_for(tag),
            "max_tokens": max_tokens,
            "stream": True,
            "priority": priority,
            # Greedy: the repetition-loop continuation never samples EOS,
            # so every flood stream holds its blocks for all 512 tokens,
            # and preempted streams resume bit-exact (finish "length").
            "temperature": 0,
        },
        timeout=httpx.Timeout(10.0, read=600.0),
    ) as response:
        if response.status_code != 200:
            response.read()
            return None, None, response.status_code
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            if ttft is None and choices[0].get("text"):
                ttft = time.monotonic() - started
            finish = choices[0].get("finish_reason") or finish
    return ttft, finish, 200


def wait_for_saturation(stack, timeout_s: float = 120.0) -> None:
    """Waits until every flood stream is running and the pool is too full
    to admit a probe without reclaiming blocks (probe needs ~37)."""
    deadline = time.monotonic() + timeout_s
    last = "<no sample>"
    while time.monotonic() < deadline:
        text = stack.metrics_text()
        free = metric_value(text, "kiln_worker_kv_blocks_free", model=stack.model_id)
        allocated = metric_value(
            text, "kiln_worker_kv_blocks_allocated", model=stack.model_id
        )
        last = f"allocated={allocated} free={free}"
        if free is not None and allocated is not None and free < 37 and allocated > 400:
            return
        time.sleep(0.25)
    pytest.fail(f"flood never saturated the KV pool ({last})")


def test_interactive_ttft_survives_batch_flood():
    path = model_dir()
    if path is None:
        pytest.skip(
            "pinned llama test model not found; run ./scripts/fetch-test-model.sh"
        )
    with running_stack([("llama", "rust", str(path))]) as stack:
        stack.wait_ready()

        # Warm-up: materialize the pool and template/tokenizer caches so the
        # baseline measures steady-state TTFT, not first-touch costs.
        ttft, _, status = stream_completion(stack, "warmup", "interactive", 8)
        assert status == 200 and ttft is not None

        baseline = []
        for i in range(BASELINE_PROBES):
            ttft, _, status = stream_completion(
                stack, f"baseline-{i}", "interactive", PROBE_MAX_TOKENS
            )
            assert status == 200, "baseline probe failed"
            assert ttft is not None, "baseline probe streamed no text"
            baseline.append(ttft)
        baseline_worst = max(baseline)

        # BATCH flood: a dozen long streams, launched together.
        results: list[tuple[float | None, str | None, int]] = [
            (None, None, 0)
        ] * FLOOD_STREAMS

        def flood(i: int) -> None:
            results[i] = stream_completion(
                stack, f"flood-{i}", "batch", FLOOD_MAX_TOKENS
            )

        threads = [
            threading.Thread(target=flood, args=(i,), daemon=True)
            for i in range(FLOOD_STREAMS)
        ]
        for thread in threads:
            thread.start()
        wait_for_saturation(stack)

        # INTERACTIVE probes into the saturated pool.
        flooded = []
        for i in range(FLOODED_PROBES):
            ttft, _, status = stream_completion(
                stack, f"probe-{i}", "interactive", PROBE_MAX_TOKENS
            )
            assert status == 200, "interactive probe failed under flood"
            assert ttft is not None, "interactive probe streamed no text"
            flooded.append(ttft)
        flooded_worst = max(flooded)

        for thread in threads:
            thread.join(timeout=600)
            assert not thread.is_alive(), "flood stream never finished"

        # The stated bar: worst flooded TTFT within 2x worst baseline plus
        # the 500 ms floor. Report the numbers either way.
        limit = TTFT_RATIO * baseline_worst + TTFT_FLOOR_S
        print(
            f"baseline TTFTs: {[f'{t:.3f}' for t in baseline]} "
            f"flooded TTFTs: {[f'{t:.3f}' for t in flooded]} limit={limit:.3f}s"
        )
        assert flooded_worst <= limit, (
            f"interactive TTFT degraded under BATCH flood: worst flooded "
            f"{flooded_worst:.3f}s vs baseline {baseline_worst:.3f}s "
            f"(limit {limit:.3f}s) — BATCH work was not preempted"
        )

        # The room was made by preempting BATCH work, and the preempted
        # streams still completed: preemption is scheduling, not failure.
        text = stack.metrics_text()
        preempted = metric_value(
            text, "kiln_worker_requests_preempted_total", model=stack.model_id
        )
        assert preempted and preempted >= 1, (
            f"no preemptions recorded ({preempted}); the flood scenario "
            "did not exercise priority preemption"
        )
        for i, (ttft, finish, status) in enumerate(results):
            assert status == 200, f"flood stream {i} failed with {status}"
            assert ttft is not None, f"flood stream {i} streamed no text"
            assert finish == "length", f"flood stream {i} finished '{finish}'"
