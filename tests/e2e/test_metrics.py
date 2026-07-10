"""Phase 2 acceptance: /metrics shows request counters incrementing."""

from __future__ import annotations

import re
import time


def _counter(text: str, name: str, **labels) -> float:
    """Sum of a counter's samples whose labels include all of `labels`."""
    total = 0.0
    for line in text.splitlines():
        if not line.startswith(name):
            continue
        if all(f'{key}="{value}"' in line for key, value in labels.items()):
            total += float(line.rsplit(" ", 1)[1])
    return total


def test_request_counters_increment(stack, client):
    before = stack.metrics_text()

    client.chat.completions.create(
        model=stack.model_id,
        messages=[{"role": "user", "content": "Say hello."}],
        temperature=0,
        max_tokens=8,
    )

    after = stack.metrics_text()

    chat_ok = _counter(
        after, "kiln_chat_completions_total", model=stack.model_id, outcome="ok"
    )
    assert (
        chat_ok
        >= _counter(
            before, "kiln_chat_completions_total", model=stack.model_id, outcome="ok"
        )
        + 1
    )

    http_chat = _counter(
        after, "kiln_http_requests_total", path="/v1/chat/completions", status="200"
    )
    assert http_chat >= 1

    assert _counter(after, "kiln_prompt_tokens_total", model=stack.model_id) > 0
    assert _counter(after, "kiln_completion_tokens_total", model=stack.model_id) > 0

    # Gauge: the worker is up right now.
    assert re.search(
        rf'^kiln_worker_up{{model="{stack.model_id}"}} 1$', after, flags=re.MULTILINE
    ), "kiln_worker_up gauge should be 1"

    # Latency histogram exists for the chat route.
    assert (
        'kiln_http_request_duration_seconds_bucket{path="/v1/chat/completions"' in after
    )


def test_worker_stats_reexported_with_model_label(stack, client):
    """Phase 5: the gateway polls the worker's Stats RPC alongside Health
    and re-exports it with a `model` label (SPEC §5/§2.3). The python
    worker has no Stats yet — the gateway must simply skip it."""
    client.chat.completions.create(
        model=stack.model_id,
        messages=[{"role": "user", "content": "Say hello."}],
        temperature=0,
        max_tokens=8,
    )

    if stack.worker_kind != "rust":
        return  # nothing re-exported; the call above must still succeed

    # Stats is polled on the 1s health cadence; allow a few ticks. A poll
    # can land mid-request — engine steps already counted, generated
    # tokens not yet — and /metrics serves that snapshot until the next
    # tick (observed: CI run 28754975315), so wait until EVERY counter
    # this test asserts on has caught up, not just the first.
    def stats_caught_up(text: str) -> bool:
        return (
            _counter(text, "kiln_worker_engine_steps_total", model=stack.model_id) > 0
            and _counter(text, "kiln_worker_requests_total", model=stack.model_id) >= 1
            and _counter(
                text, "kiln_worker_tokens_generated_total", model=stack.model_id
            )
            > 0
        )

    deadline = time.time() + 10
    while time.time() < deadline:
        text = stack.metrics_text()
        if stats_caught_up(text):
            break
        time.sleep(0.5)
    else:
        raise AssertionError(
            "worker stats never fully re-exported within the deadline: "
            f"steps={_counter(text, 'kiln_worker_engine_steps_total', model=stack.model_id)} "
            f"requests={_counter(text, 'kiln_worker_requests_total', model=stack.model_id)} "
            f"tokens={_counter(text, 'kiln_worker_tokens_generated_total', model=stack.model_id)}"
        )
    blocks = _counter(
        text, "kiln_worker_kv_blocks_allocated", model=stack.model_id
    ) + _counter(text, "kiln_worker_kv_blocks_free", model=stack.model_id)
    assert blocks == 512, f"block gauges should cover the pool, got {blocks}"
