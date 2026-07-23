"""Phase 9 part 1: machine memory budget + model lifecycle (SPEC §2.2/§2.3).

Real stacks, real workers, real measured memory numbers — nothing mocked:
every budget decision the gateway makes here is arithmetic over worker
heartbeat MemoryReports, asserted through /metrics and /readyz.

The budgets are placed between measured bounds (dev machine, PROGRESS
2026-07-14; the components are device-stable — packed weight bytes and the
fixed 512-block KV pool dominate, and the pool only materializes on first
traffic):

  gemma-3-1b-it-4bit rust worker: ~766 MB idle after load, ~1176 MB once
  the KV pool is touched (pool commitment 436 MB); weights on disk 733 MB.
  LRU_BUDGET (2.08 GB) sits between "one traffic-warmed resident + one
  load" (~1.91 GB) and "two idle residents + one load" (~2.26 GB), so a
  third load always evicts exactly one model.

  qwen2.5-0.5b-4bit rust worker: ~300 MB idle, ~486 MB after traffic
  (pool commitment 201 MB); weights 278 MB. PINNED_BUDGET (730 MB) fits
  two idle residents but not two residents + a third load (~880 MB).

Phase 9 part 2 note: these budgets deliberately over-pack the machine —
part 1 let requests materialize KV pools past the budget (the recorded
continuous-drift gap). Part 2's per-request admission now REFUSES the
request that would start that drift with a structured 503
(`insufficient_memory`), so the flows below assert the 503 exactly where
part 1 silently went over, and the ledger stays <= budget throughout.
The routing touch still lands before the gate, so LRU recency semantics
are unchanged. test_admission.py holds the isolated gate scenarios.
"""

from __future__ import annotations

import re
import time

import httpx
import pytest
from conftest import API_KEY, pinned_model_dir, running_stack

GEMMA = "gemma-3-1b-it-4bit"
QWEN25 = "qwen2.5-0.5b-4bit"

LRU_BUDGET = 2_080_000_000
PINNED_BUDGET = 730_000_000
TRAP_BUDGET = 900_000_000

METRIC_LINE = re.compile(r"^(\w+)(?:\{([^}]*)\})?\s+([0-9eE+.-]+)$")


def require(model_id: str) -> str:
    path = pinned_model_dir(model_id)
    if path is None:
        pytest.skip(
            f"pinned test model '{model_id}' not found; run "
            "./scripts/fetch-test-model.sh"
        )
    return str(path)


def metric_value(text: str, name: str, **labels) -> float | None:
    """Value of the first sample of `name` whose labels include `labels`;
    None when no such series exists (e.g. a counter that never fired)."""
    for line in text.splitlines():
        match = METRIC_LINE.match(line)
        if not match or match.group(1) != name:
            continue
        got = dict(re.findall(r'(\w+)="([^"]*)"', match.group(2) or ""))
        if all(got.get(key) == value for key, value in labels.items()):
            return float(match.group(3))
    return None


def readyz(stack) -> tuple[int, dict[str, str]]:
    response = httpx.get(f"{stack.base_url}/readyz", timeout=10)
    return response.status_code, response.json()["models"]


def wait_for_status(stack, model: str, want: str, timeout_s: float = 180) -> None:
    deadline = time.monotonic() + timeout_s
    status = "<unknown>"
    while time.monotonic() < deadline:
        _, models = readyz(stack)
        status = models.get(model, "<missing>")
        if status == want:
            return
        time.sleep(0.5)
    pytest.fail(f"model '{model}' never reached '{want}' (last: '{status}')")


def complete(stack, model: str, max_tokens: int = 16) -> httpx.Response:
    """One real completion (the request-path 'touch'); returns the raw
    response so callers can assert 200s and 503s alike."""
    return httpx.post(
        f"{stack.base_url}/v1/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": model,
            "prompt": "The capital of France is",
            "max_tokens": max_tokens,
        },
        timeout=120,
    )


def assert_completes(stack, model: str) -> None:
    response = complete(stack, model)
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["text"], response.text


def test_lru_eviction_order_and_on_demand_reload():
    """Three models over a budget sized for two: startup evicts the LRU;
    request recency (not load order) picks the next victim; an evicted
    model reloads on demand and serves."""
    path = require(GEMMA)
    models = [
        ("alpha", "rust", path),
        ("bravo", "rust", path),
        ("charlie", "rust", path),
    ]
    memory = f"[memory]\nbudget_bytes = {LRU_BUDGET}\nmin_available_bytes = 0\n"
    with running_stack(models, extra_toml=memory) as stack:
        stack.wait_ready()

        # Startup (loads sequenced in config order): alpha and bravo fit;
        # charlie's load exceeded the budget and evicted the LRU — alpha,
        # READY the longest with no traffic yet.
        _, statuses = readyz(stack)
        assert statuses == {
            "alpha": "unloaded (evicted)",
            "bravo": "ready",
            "charlie": "ready",
        }, statuses
        assert stack.worker_command("alpha") == "", "evicted worker still alive"
        assert stack.worker_command("bravo") and stack.worker_command("charlie")

        # The numbers driving that decision are real heartbeat bytes.
        text = stack.metrics_text()
        assert metric_value(text, "kiln_memory_budget_bytes") == LRU_BUDGET
        used = metric_value(text, "kiln_memory_used_bytes")
        f_bravo = metric_value(text, "kiln_worker_memory_bytes", model="bravo")
        f_charlie = metric_value(text, "kiln_worker_memory_bytes", model="charlie")
        assert f_bravo > 500_000_000, f"implausible measured footprint: {f_bravo}"
        assert f_charlie > 500_000_000, f"implausible measured footprint: {f_charlie}"
        assert metric_value(text, "kiln_worker_memory_bytes", model="alpha") == 0
        assert used <= LRU_BUDGET, f"over budget after eviction: {used}"
        assert (
            metric_value(
                text, "kiln_worker_unloads_total", model="alpha", reason="evicted"
            )
            == 1
        )

        # Request recency: touch charlie, then bravo. Charlie becomes the
        # LRU even though bravo is the older worker — a load-order policy
        # would evict bravo here.
        assert_completes(stack, "charlie")
        time.sleep(3)  # charlie's busy-heartbeat touches settle first, and
        # its pool-inflated footprint (~1.18 GB) reaches the ledger.

        # Part 2's per-request admission: warming bravo's pool too (436 MB
        # growth) no longer fits the headroom charlie's warm left
        # (~140 MB). Part 1 let exactly this request drift the machine to
        # ~2.35 GB against the 2.08 GB budget; the structured 503 is that
        # gap closing. The routing touch landed before the gate, so bravo
        # is still the most recently used model for the eviction below.
        response = complete(stack, "bravo")
        assert response.status_code == 503, response.text
        assert response.json()["error"]["code"] == "insufficient_memory", response.text
        text = stack.metrics_text()
        assert metric_value(text, "kiln_admission_rejects_total", model="bravo") == 1, (
            text
        )
        used = metric_value(text, "kiln_memory_used_bytes")
        assert used <= LRU_BUDGET, f"admission let the ledger drift over budget: {used}"

        # On-demand reload: a request for the evicted model 503s (retriable)
        # and starts the load, which must evict charlie and spare bravo.
        response = complete(stack, "alpha")
        assert response.status_code == 503, response.text
        assert response.json()["error"]["code"] == "model_loading", response.text
        wait_for_status(stack, "alpha", "ready")
        _, statuses = readyz(stack)
        assert statuses["charlie"] == "unloaded (evicted)", statuses
        assert statuses["bravo"] == "ready", statuses
        assert stack.worker_command("charlie") == "", "evicted worker still alive"
        assert stack.worker_command("bravo"), "LRU eviction took the wrong victim"

        # The reloaded model actually serves.
        assert_completes(stack, "alpha")
        text = stack.metrics_text()
        assert (
            metric_value(
                text, "kiln_worker_unloads_total", model="charlie", reason="evicted"
            )
            == 1
        )
        assert (
            metric_value(
                text, "kiln_worker_unloads_total", model="bravo", reason="evicted"
            )
            is None
        )


def test_pinned_model_survives_eviction_pressure():
    """The pinned model is the machine-wide LRU when eviction pressure
    arrives (loaded first, never touched) — pressure that would otherwise
    evict it must take the unpinned model instead."""
    path = require(QWEN25)
    models = [
        ("pin", "rust", path, "pinned = true"),
        ("alpha", "rust", path),
        ("charlie", "rust", path),
    ]
    memory = f"[memory]\nbudget_bytes = {PINNED_BUDGET}\nmin_available_bytes = 0\n"
    with running_stack(models, extra_toml=memory) as stack:
        stack.wait_ready()

        _, statuses = readyz(stack)
        assert statuses == {
            "pin": "ready",
            "alpha": "unloaded (evicted)",
            "charlie": "ready",
        }, statuses
        assert stack.worker_command("pin"), "pinned worker was evicted"
        assert stack.worker_command("alpha") == ""

        text = stack.metrics_text()
        assert (
            metric_value(
                text, "kiln_worker_unloads_total", model="alpha", reason="evicted"
            )
            == 1
        )
        assert (
            metric_value(
                text, "kiln_worker_unloads_total", model="pin", reason="evicted"
            )
            is None
        )
        f_pin = metric_value(text, "kiln_worker_memory_bytes", model="pin")
        assert f_pin > 250_000_000, f"implausible measured footprint: {f_pin}"

        # This budget deliberately over-packs the machine: warming any
        # pool (201 MB) exceeds the ~130 MB of headroom two idle residents
        # leave. Part 1 served here by silently drifting over budget;
        # part 2's per-request admission refuses with a structured 503 and
        # the worker stays READY — the load-time gate passed, the request
        # gate did not. (Serving within a budget that has pool headroom is
        # test_admission.py's positive control.)
        response = complete(stack, "pin")
        assert response.status_code == 503, response.text
        assert response.json()["error"]["code"] == "insufficient_memory", response.text
        _, statuses = readyz(stack)
        assert statuses["pin"] == "ready", statuses
        text = stack.metrics_text()
        assert metric_value(text, "kiln_admission_rejects_total", model="pin") == 1, (
            text
        )


def test_reload_prices_pool_and_evicts_instead_of_stranding():
    """The 2026-07-23 soak burst-starvation trap, reproduced deterministically
    (PROGRESS root cause, runs 30018908142 / 29930224449 / 29967425675 /
    29540167586): a demand-driven reload whose WEIGHTS fit the drifted
    budget but whose pool does not. Weights-only load pricing admitted the
    worker without evicting, stranding it READY-with-cold-pool behind
    per-request `insufficient_memory` 503s — a denial path with no
    eviction lever — while an evictable idle model held the room the
    whole time. Commitment-carried pricing must instead evict at the
    reload and serve the triggering request off the READY-seeded
    reservation, with zero admission rejects.

    Sizing (qwen2.5: ~300 MB idle, ~486 MB warm, weights-on-disk 278 MB
    + 64 MB margin = 342 MB load base, pool commitment 201 MB), budget
    900 MB: boot fits both idle (~642 MB peak); victim warm (~486 MB)
    leaves the reload's weights (342 <= ~414 headroom) but not
    weights + pool (543) — the trap window. Post-eviction the full
    reload fits alone (543 <= 900)."""
    path = require(QWEN25)
    models = [
        ("victim", "rust", path),
        ("trap", "rust", path, "ttl_seconds = 10"),
    ]
    memory = f"[memory]\nbudget_bytes = {TRAP_BUDGET}\nmin_available_bytes = 0\n"
    with running_stack(models, extra_toml=memory) as stack:
        stack.wait_ready()
        _, statuses = readyz(stack)
        assert statuses == {"victim": "ready", "trap": "ready"}, statuses

        # trap's first READY records its pool commitment (GetInfo geometry;
        # the pool itself never materializes — it stays cold, exactly like
        # the soak's gemma between bursts). Warm the victim so its drift
        # occupies the pool-sized share of the headroom.
        assert_completes(stack, "victim")
        time.sleep(3)  # warmed footprint reaches the ledger

        # trap idles past its TTL and unloads.
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            _, statuses = readyz(stack)
            if statuses["trap"] == "unloaded (idle ttl)":
                break
            time.sleep(1)
        assert statuses["trap"] == "unloaded (idle ttl)", statuses

        # The demand reload. Weights-only pricing loads WITHOUT evicting
        # (642 + 342 fits 900) and the follow-up request starves on
        # insufficient_memory; commitment-carried pricing must evict the
        # victim at the load and serve off the seeded reservation.
        response = complete(stack, "trap")
        assert response.status_code == 503, response.text
        assert response.json()["error"]["code"] == "model_loading", response.text
        wait_for_status(stack, "trap", "ready", timeout_s=120)

        _, statuses = readyz(stack)
        assert statuses["victim"] == "unloaded (evicted)", (
            f"reload was admitted without evicting — the starvation trap: {statuses}"
        )

        # The triggering request's retry serves promptly: its pool room is
        # reserved, so it must never bounce off insufficient_memory.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            response = complete(stack, "trap")
            if response.status_code == 200:
                break
            assert response.json()["error"]["code"] != "insufficient_memory", (
                f"reloaded model starved on its own pool growth: {response.text}"
            )
            time.sleep(2)
        assert response.status_code == 200, response.text

        text = stack.metrics_text()
        assert metric_value(text, "kiln_admission_rejects_total", model="trap") is None
        assert (
            metric_value(
                text, "kiln_worker_unloads_total", model="victim", reason="evicted"
            )
            == 1
        )
        used = metric_value(text, "kiln_memory_used_bytes")
        assert used <= TRAP_BUDGET, f"ledger over budget: {used}"


def test_ttl_idle_model_auto_unloads_and_reloads_on_demand():
    """A ttl_seconds model auto-unloads on schedule once idle (process
    exits, memory released, /readyz stays 200 — it is a settled state)
    and reloads on the next request."""
    path = require(QWEN25)
    with running_stack([("solo", "rust", path, "ttl_seconds = 10")]) as stack:
        stack.wait_ready()
        assert_completes(stack, "solo")

        # The TTL clock starts at the completion; the 1s health poll fires
        # the unload shortly after 10s idle.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            _, statuses = readyz(stack)
            if statuses["solo"] == "unloaded (idle ttl)":
                break
            time.sleep(1)
        code, statuses = readyz(stack)
        assert statuses["solo"] == "unloaded (idle ttl)", statuses
        assert code == 200, "an idle-unloaded model must not fail readiness"
        assert stack.worker_command("solo") == "", "worker process still alive"

        text = stack.metrics_text()
        assert (
            metric_value(
                text, "kiln_worker_unloads_total", model="solo", reason="idle_ttl"
            )
            == 1
        )
        assert metric_value(text, "kiln_worker_memory_bytes", model="solo") == 0
        assert metric_value(text, "kiln_worker_up", model="solo") == 0

        # On-demand reload: first request 503s and triggers the load.
        response = complete(stack, "solo")
        assert response.status_code == 503, response.text
        assert response.json()["error"]["code"] == "model_loading", response.text
        wait_for_status(stack, "solo", "ready", timeout_s=120)
        assert_completes(stack, "solo")
