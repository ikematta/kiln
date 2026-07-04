"""Phase 2 acceptance: kill -9 the worker mid-request → the client gets a
structured 502, the supervisor auto-restarts the worker, and the next
request succeeds (SPEC §2.2, §12)."""

from __future__ import annotations

import os
import signal
import threading
import time

import openai
import pytest

RECOVERY_TIMEOUT_S = 180


def test_kill9_mid_request_yields_502_then_recovers(stack, client):
    restarts_before = _restart_count(stack)

    # Long generation so the kill lands mid-decode.
    result: dict = {}

    def request():
        try:
            result["completion"] = client.chat.completions.create(
                model=stack.model_id,
                messages=[
                    {"role": "user", "content": "Write a long story about a potter."}
                ],
                max_tokens=512,
                temperature=0.8,
            )
        except openai.APIStatusError as exc:
            result["error"] = exc

    thread = threading.Thread(target=request)
    thread.start()
    time.sleep(
        1.5
    )  # tokenize + prefill are sub-second; decode of 512 tokens is several seconds

    pids = stack.worker_pids()
    assert pids, "no worker process found to kill"
    for pid in pids:
        os.kill(pid, signal.SIGKILL)

    thread.join(timeout=60)
    assert not thread.is_alive(), "request did not fail after the worker was killed"

    error = result.get("error")
    assert error is not None, (
        f"request unexpectedly succeeded: {result.get('completion')}"
    )
    assert error.status_code == 502
    body = error.response.json()
    assert body["error"]["code"] == "worker_crashed"
    assert body["error"]["type"] == "server_error"
    assert body["error"]["message"]

    # Auto-restart: keep retrying until the request path works again.
    deadline = time.monotonic() + RECOVERY_TIMEOUT_S
    while True:
        try:
            completion = client.chat.completions.create(
                model=stack.model_id,
                messages=[{"role": "user", "content": "Say the word ready."}],
                temperature=0,
                max_tokens=8,
            )
            break
        except openai.APIError:
            if time.monotonic() > deadline:
                pytest.fail("worker never recovered after kill -9")
            time.sleep(1)
    assert completion.choices[0].message.content

    assert _restart_count(stack) > restarts_before, "restart counter did not increment"
    assert stack.worker_pids(), "no worker process running after recovery"


def _restart_count(stack) -> float:
    needle = f'kiln_worker_restarts_total{{model="{stack.model_id}"}}'
    for line in stack.metrics_text().splitlines():
        if line.startswith(needle):
            return float(line.rsplit(" ", 1)[1])
    return 0.0
