"""TTFT/total timeout enforcement + request size limits (SPEC §8.3).

Enforcement model under test (crates/kiln-gateway/src/timeout.rs):
- Both budgets anchor at request ARRIVAL at the gateway, so queue wait
  counts against TTFT — the scenario here creates REAL queue pressure by
  occupying the python worker (a sequential engine: one request runs, the
  rest wait in its queue) with a long "holder" generation, then submitting
  victims that sit queued past their TTFT budget. No artificial sleeps in
  server code, and no short-timeout-races-a-fast-model shortcuts.
- On expiry the request is cancelled through the worker's Cancel RPC (the
  same ≤2-step path as a user stop). Verified two ways: the gateway logs
  the worker's CancelAck (`cancel_found`), and behaviorally — after the
  holder is released, a fresh probe request completes immediately, which
  is impossible if the timed-out victim were still queued or running
  (the victims carry multi-hundred-token max_tokens on purpose).
- Timeouts are 504 `timeout_error` (codes ttft_timeout|total_timeout),
  deliberately NOT the 429 rate-limit family, and never carry Retry-After.
- A timed-out request's tpm reservation forfeits exactly like a client
  disconnect: not refunded (anti-abuse), and never double-charged.

Size limits: bodies over the 2 MiB cap are 413 `request_too_large` on the
OpenAI envelope and Anthropic's `request_too_large` type on /v1/messages.
"""

from __future__ import annotations

import json
import subprocess
import time

import httpx
import pytest
from conftest import MODEL_ID, build_binaries, model_dir, running_stack

TTFT_S = 4
TOTAL_S = 12
TPM_KEY = "timeout-tpm-key"  # tpm = 600 → refills 10 tokens/s
# Raw continuation on /v1/completions (no chat template): an instruct model
# continuing a bare number sequence has no turn structure to end, so unlike
# a chat-templated "count for me" it does not EOS after a polite answer —
# the generation reliably runs until max_tokens or a Cancel. This is what
# makes the worker-occupying "holder" requests deterministic.
NUMBER_PROMPT = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,"


def hash_key(binary, key: str) -> str:
    return subprocess.run(
        [binary, "hash-key", key], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture(scope="module")
def to_stack():
    if model_dir() is None:
        pytest.skip(
            f"pinned test model '{MODEL_ID}' not found; run "
            "./scripts/fetch-test-model.sh"
        )
    binary = build_binaries()
    extra = (
        f"ttft_timeout_secs = {TTFT_S}\n"
        f"total_timeout_secs = {TOTAL_S}\n"
        f'[[auth.api_keys]]\nname = "timeout-tpm"\n'
        f'key_hash = "{hash_key(binary, TPM_KEY)}"\ntpm = 600\n'
    )
    with running_stack([(MODEL_ID, "python")], extra_toml=extra) as stack:
        stack.wait_ready()
        # Warm the worker (first-generation kernel compile must not eat a
        # later test's TTFT budget). A cold first request may itself time
        # out — that is correct behavior, so retry until warm.
        for _ in range(3):
            if chat(stack, stack.api_key, max_tokens=2).status_code == 200:
                break
        else:
            pytest.fail("warmup request never succeeded under timeout config")
        yield stack


def chat(stack, key: str, max_tokens: int, prompt: str = "Hi", **params):
    return httpx.post(
        f"{stack.base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": stack.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            **params,
        },
        timeout=120,
    )


class Holder:
    """A streaming raw-completions request that occupies the sequential
    python worker, confirmed running (first text delta received) before
    victims are submitted behind it. Closing the stream disconnects the
    client, which cancels the holder through the same worker Cancel path."""

    def __init__(self, stack):
        self._cm = httpx.stream(
            "POST",
            f"{stack.base_url}/v1/completions",
            headers={"Authorization": f"Bearer {stack.api_key}"},
            json={
                "model": stack.model_id,
                "prompt": NUMBER_PROMPT,
                "max_tokens": 4000,
                "temperature": 0,
                "stream": True,
            },
            timeout=60,
        )
        self._response = self._cm.__enter__()
        assert self._response.status_code == 200, self._response.read()
        # Keep a strong reference: abandoning the generator chain would get
        # it GC'd, which closes the response — a client disconnect that
        # cancels the holder and silently releases the worker.
        self._frames = sse_frames(self._response.iter_lines())
        for frame in self._frames:
            if "error" in frame:
                pytest.fail(f"holder errored before producing text: {frame}")
            if any(choice.get("text") for choice in frame.get("choices", [])):
                return  # generation confirmed underway
        pytest.fail("holder stream ended before producing text")

    def close(self):
        self._cm.__exit__(None, None, None)


def sse_frames(lines):
    for line in lines:
        if line.startswith("data: ") and line != "data: [DONE]":
            yield json.loads(line[6:])


def timeout_count(stack, scope: str) -> float:
    needle = f'kiln_timeout_total{{model="{stack.model_id}",scope="{scope}"}}'
    for line in stack.metrics_text().splitlines():
        if line.startswith(needle):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def test_ttft_timeout_cancels_queued_request_cleanly(to_stack):
    """A request stuck in the worker queue behind a running generation is
    504'd at the TTFT budget — not left hanging — and its worker-side entry
    is cancelled, proven by the CancelAck log and by the queue being empty
    once the holder is released."""
    before = timeout_count(to_stack, "ttft")
    holder = Holder(to_stack)
    try:
        started = time.monotonic()
        # Big max_tokens on purpose: if this victim were NOT cancelled it
        # would occupy the worker for tens of seconds once dequeued, and
        # the freed-worker probe below would fail.
        victim = chat(to_stack, to_stack.api_key, max_tokens=3000)
        elapsed = time.monotonic() - started

        assert victim.status_code == 504, victim.text
        error = victim.json()["error"]
        assert error["type"] == "timeout_error"
        assert error["code"] == "ttft_timeout"
        assert "retry-after" not in victim.headers, "a timeout is not a 429"
        assert TTFT_S - 1.5 <= elapsed <= TOTAL_S - 2, (
            f"TTFT abort at {elapsed:.1f}s; budget {TTFT_S}s (holder still "
            f"held the worker until {TOTAL_S}s, so this must be the TTFT timer)"
        )

        # Anthropic surface, same queue pressure: its taxonomy has no
        # timeout type, so the envelope reports the 5xx catch-all.
        anthropic_victim = httpx.post(
            f"{to_stack.base_url}/v1/messages",
            headers={"x-api-key": to_stack.api_key},
            json={
                "model": to_stack.model_id,
                "max_tokens": 3000,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            timeout=120,
        )
        assert anthropic_victim.status_code == 504, anthropic_victim.text
        body = anthropic_victim.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "api_error"
        assert "ttft_timeout_secs" in body["error"]["message"]
    finally:
        holder.close()

    # Cancel delivery: the gateway logged the worker's CancelAck for the
    # queued victims (found=true — the worker knew the request).
    log = to_stack.log_path.read_text()
    assert '"cancel_found":true' in log and '"scope":"ttft"' in log, (
        "expected a timeout-abort log line with CancelAck found=true"
    )
    assert timeout_count(to_stack, "ttft") >= before + 2

    # Freed-worker probe: holder disconnected, victims cancelled → a fresh
    # request must complete within ITS OWN TTFT budget. If either victim
    # were still queued (not cancelled), it would run first for ~10s+ and
    # this probe would 504.
    probe = chat(to_stack, to_stack.api_key, max_tokens=2)
    assert probe.status_code == 200, probe.text


def test_total_timeout_cancels_mid_stream_after_partial_output(to_stack):
    """A streaming generation that exceeds the total budget ends with the
    terminal timeout error event (no [DONE]); deltas already sent stand as
    delivered. The worker-side request is cancelled at the timer, proven by
    the worker being free immediately afterwards."""
    before = timeout_count(to_stack, "total")
    started = time.monotonic()
    content = ""
    terminal_error = None
    with httpx.stream(
        "POST",
        f"{to_stack.base_url}/v1/completions",
        headers={"Authorization": f"Bearer {to_stack.api_key}"},
        json={
            "model": to_stack.model_id,
            "prompt": NUMBER_PROMPT,
            "max_tokens": 4000,
            "temperature": 0,
            "stream": True,
        },
        timeout=60,
    ) as response:
        assert response.status_code == 200
        saw_done = False
        for frame in sse_frames(response.iter_lines()):
            if "error" in frame:
                terminal_error = frame["error"]
                break
            for choice in frame.get("choices", []):
                content += choice.get("text") or ""
        else:
            saw_done = True
    elapsed = time.monotonic() - started

    assert terminal_error is not None and not saw_done, (
        f"stream must end in the timeout error event, got content={content[:80]!r}"
    )
    assert terminal_error["type"] == "timeout_error"
    assert terminal_error["code"] == "total_timeout"
    assert content, "partial output before the timer must have been delivered"
    assert TOTAL_S - 2 <= elapsed <= TOTAL_S + 15, f"total abort at {elapsed:.1f}s"
    assert timeout_count(to_stack, "total") >= before + 1

    # The cancel landed: the worker is free right away, not still grinding
    # toward max_tokens=4000 (which would take far longer than one TTFT
    # budget to finish).
    probe = chat(to_stack, to_stack.api_key, max_tokens=2)
    assert probe.status_code == 200, probe.text


def test_ttft_timeout_forfeits_tpm_reservation_like_a_disconnect(to_stack):
    """B1 interaction: a timed-out request's tpm hold is released the same
    way a client disconnect is — forfeited until refill (never refunded,
    per the anti-abuse rule) and charged exactly once (never leaked or
    double-counted)."""
    holder = Holder(to_stack)
    try:
        # Reserves ~(50 prompt + 265) ≈ 315 of the 600 budget, then times
        # out in queue. The hold is dropped unsettled.
        victim = chat(to_stack, TPM_KEY, max_tokens=265)
        assert victim.status_code == 504, victim.text
        assert victim.json()["error"]["code"] == "ttft_timeout"

        # NOT refunded: ~565 needed cannot fit the ~285+refill remaining.
        # (Rejected before Submit, so the busy worker is irrelevant.)
        refund_probe = chat(to_stack, TPM_KEY, max_tokens=515)
        assert refund_probe.status_code == 429, refund_probe.text
        assert refund_probe.json()["error"]["type"] == "tokens"
    finally:
        holder.close()

    # Charged exactly ONCE: ~265 needed fits the remaining budget. A
    # double-count would leave ~0 and reject this too.
    once_probe = chat(to_stack, TPM_KEY, max_tokens=215)
    assert once_probe.status_code == 200, once_probe.text


def test_oversized_body_is_413_on_both_surfaces(to_stack):
    """SPEC §8.3 request size limits: a body over the 2 MiB cap is a
    proper 413 (`request_too_large`), not a generic 400."""
    oversized = {
        "model": to_stack.model_id,
        "max_tokens": 2,
        "messages": [{"role": "user", "content": "x" * (3 * 1024 * 1024)}],
    }
    openai_resp = httpx.post(
        f"{to_stack.base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {to_stack.api_key}"},
        json=oversized,
        timeout=60,
    )
    assert openai_resp.status_code == 413, openai_resp.text
    assert openai_resp.json()["error"]["code"] == "request_too_large"

    anthropic_resp = httpx.post(
        f"{to_stack.base_url}/v1/messages",
        headers={"x-api-key": to_stack.api_key},
        json=oversized,
        timeout=60,
    )
    assert anthropic_resp.status_code == 413, anthropic_resp.text
    assert anthropic_resp.json()["error"]["type"] == "request_too_large"
