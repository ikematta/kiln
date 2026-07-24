"""Per-key rate limiting (SPEC §8.3): rpm and tpm token buckets.

Enforcement model under test (crates/kiln-gateway/src/ratelimit.rs):
- rpm is taken before the request is processed (middleware after auth).
- tpm reserves the worst case (prompt + max_tokens) before Submit and
  refunds the unused remainder when the response settles; a request whose
  worst case exceeds the whole per-minute budget is rejected up front.

Every limited key here is dedicated to one test: buckets are per-key
state, and sharing one would couple the tests' timing. The stack's default
"e2e" key has no limits — the other suites must stay unlimited.

Timing note: buckets refill continuously at limit/60 tokens per second, so
an rpm=2 key regains a request 30s after exhaustion — the reset test
sleeps for real and is the slowest test in this file by design ("timed and
verified, not assumed").
"""

from __future__ import annotations

import concurrent.futures
import pathlib
import subprocess
import time

import httpx
import pytest
from conftest import MODEL_ID, build_binaries, model_dir, running_stack

# name -> (raw key, limits toml lines)
LIMITED_KEYS = {
    "rpm": ("rl-rpm-key", "rpm = 2"),
    "burst": ("rl-burst-key", "rpm = 5"),
    "anthropic": ("rl-anthropic-key", "rpm = 1"),
    "tpm-cap": ("rl-tpm-cap-key", "tpm = 300"),
    "tpm-reconcile": ("rl-tpm-reconcile-key", "tpm = 600"),
    "tpm-race": ("rl-tpm-race-key", "tpm = 400"),
}


def hash_key(binary: pathlib.Path, key: str) -> str:
    return subprocess.run(
        [binary, "hash-key", key], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture(scope="module")
def rl_stack():
    if model_dir() is None:
        pytest.skip(
            f"pinned test model '{MODEL_ID}' not found; run "
            "./scripts/fetch-test-model.sh"
        )
    binary = build_binaries()
    extra = "\n".join(
        f'[[auth.api_keys]]\nname = "{name}"\nkey_hash = "{hash_key(binary, key)}"\n{limits}\n'
        for name, (key, limits) in LIMITED_KEYS.items()
    )
    with running_stack([(MODEL_ID, "rust")], extra_toml=extra) as stack:
        stack.wait_ready()
        yield stack


def chat(stack, key_name: str, max_tokens: int, prompt: str = "Hi", **params):
    raw_key = LIMITED_KEYS[key_name][0]
    return httpx.post(
        f"{stack.base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {raw_key}"},
        json={
            "model": stack.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            **params,
        },
        timeout=120,
    )


def assert_openai_429(response: httpx.Response, scope: str) -> None:
    assert response.status_code == 429, response.text
    error = response.json()["error"]
    assert error["type"] == scope
    assert error["code"] == "rate_limit_exceeded"


def test_rpm_limit_429_then_recovers_after_the_window(rl_stack):
    """Exceeding rpm yields structured 429s with Retry-After; the same key
    is admitted again once the advertised wait has actually elapsed."""
    started = time.monotonic()
    for _ in range(2):
        response = chat(rl_stack, "rpm", max_tokens=2)
        assert response.status_code == 200, response.text
    if time.monotonic() - started > 20:
        pytest.skip("host too slow: bucket refilled during the setup requests")

    denied = chat(rl_stack, "rpm", max_tokens=2)
    assert_openai_429(denied, "requests")
    retry_after = int(denied.headers["retry-after"])
    assert 1 <= retry_after <= 30

    # Still inside the window: denied again (proves the 200 below is the
    # refill, not a lucky second bucket).
    still_denied = chat(rl_stack, "rpm", max_tokens=2)
    assert still_denied.status_code == 429

    time.sleep(retry_after + 1.5)
    recovered = chat(rl_stack, "rpm", max_tokens=2)
    assert recovered.status_code == 200, recovered.text


def test_concurrent_burst_admits_exactly_the_limit(rl_stack):
    """12 simultaneous requests against an rpm=5 key: exactly 5 in, 7 out.
    The racy over-admission failure mode would show up as >5 successes."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        responses = list(
            pool.map(lambda _: chat(rl_stack, "burst", max_tokens=2), range(12))
        )
    statuses = sorted(response.status_code for response in responses)
    assert statuses.count(200) == 5, f"expected exactly 5 admitted, got {statuses}"
    assert statuses.count(429) == 7, f"expected exactly 7 rejected, got {statuses}"
    for response in responses:
        if response.status_code == 429:
            assert_openai_429(response, "requests")
            assert int(response.headers["retry-after"]) >= 1


def test_anthropic_surface_gets_rate_limit_error_envelope(rl_stack):
    """/v1/messages: same buckets, Anthropic error shape on 429."""
    raw_key = LIMITED_KEYS["anthropic"][0]

    def messages():
        return httpx.post(
            f"{rl_stack.base_url}/v1/messages",
            headers={"x-api-key": raw_key},
            json={
                "model": rl_stack.model_id,
                "max_tokens": 2,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            timeout=120,
        )

    assert messages().status_code == 200
    denied = messages()  # rpm = 1: the second request in the minute
    assert denied.status_code == 429, denied.text
    body = denied.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "rate_limit_error"
    assert int(denied.headers["retry-after"]) >= 1


def test_tpm_request_that_can_never_fit_is_rejected_up_front(rl_stack):
    """max_tokens alone above the tpm budget: immediate 429 telling the
    client what to change — and the failed attempt consumes nothing."""
    denied = chat(rl_stack, "tpm-cap", max_tokens=1000)
    assert_openai_429(denied, "tokens")
    message = denied.json()["error"]["message"]
    assert "max_tokens" in message, message
    # Waiting can never help, so no Retry-After is advertised.
    assert "retry-after" not in denied.headers

    # The whole budget is still there: a request that fits succeeds.
    ok = chat(rl_stack, "tpm-cap", max_tokens=8)
    assert ok.status_code == 200, ok.text


def test_tpm_reservation_is_reconciled_to_actual_usage(rl_stack):
    """Reserve-then-reconcile: request A reserves prompt + 450 of the 600
    budget but a stop string ends it after a handful of tokens. Only its
    ACTUAL usage may stay charged — otherwise request B (another ~470
    reservation) could not fit in the same minute."""
    first = chat(
        rl_stack,
        "tpm-reconcile",
        max_tokens=450,
        prompt="Count from 1 to 100, separated by commas:",
        stop=[" "],
    )
    assert first.status_code == 200, first.text
    usage = first.json()["usage"]
    # The stop string must have fired early for the refund to be visible.
    assert usage["completion_tokens"] <= 20, usage
    assert usage["total_tokens"] <= 100, usage

    second = chat(rl_stack, "tpm-reconcile", max_tokens=450)
    assert second.status_code == 200, (
        f"reservation was not refunded down to actual usage "
        f"(A used {usage['total_tokens']} of its ~490 reservation): {second.text}"
    )


def test_tpm_concurrent_reservations_cannot_overcommit(rl_stack):
    """Two simultaneous requests whose reservations individually fit the
    tpm budget but jointly exceed it: exactly one wins the atomic
    check-and-take, the other gets a structured 429 with Retry-After."""
    # Prewarm: pays the one-time argon2 verification so both racers hit
    # the bucket near-simultaneously (costs ~30 of the 400 budget).
    prewarm = chat(rl_stack, "tpm-race", max_tokens=2)
    assert prewarm.status_code == 200, prewarm.text

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(
            pool.map(lambda _: chat(rl_stack, "tpm-race", max_tokens=280), range(2))
        )
    statuses = sorted(response.status_code for response in responses)
    assert statuses == [200, 429], f"expected exactly one winner, got {statuses}"
    denied = next(r for r in responses if r.status_code == 429)
    assert_openai_429(denied, "tokens")
    assert int(denied.headers["retry-after"]) >= 1
