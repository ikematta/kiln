"""Phase 2 acceptance: /metrics shows request counters incrementing."""

from __future__ import annotations

import re


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
