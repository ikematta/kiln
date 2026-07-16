"""Phase 9 part 2: per-request memory admission (SPEC §2.3 / §8.2).

The gate under test is the REQUEST-level check, distinct from part 1's
load-time `load()` budget check: a model can load fine (weights fit) and
still be refused traffic, because serving a request would materialize its
lazily-allocated KV pool — real machine bytes the load-time projection
never saw. The gateway projects that growth from the worker-reported pool
geometry (`WorkerInfo.kv_bytes_per_block × kv_pool_blocks`) against the
LIVE ledger (budget minus heartbeat footprints), so drift since load is
what the check prices in.

Measured bounds (dev machine, PROGRESS 2026-07-14; device-stable — packed
weight bytes and the fixed 512-block pool dominate):

  qwen2.5-0.5b-4bit rust worker: ~300 MB idle, ~486 MB after traffic;
  weights on disk 278 MB; pool commitment 24 layers x 2 kv-heads x 64
  head-dim x 32 x 512 blocks x 2 bytes x 2 (K+V) = 201 MB.

Budgets: REJECT_BUDGET (450 MB) admits the load (278 MB projection, then
~300 MB measured) but not the pool growth (201 MB > ~150 MB headroom).
SERVE_BUDGET (850 MB) leaves ~550 MB of headroom, so the same request
passes and the pool actually materializes. DRIFT_BUDGET (900 MB) holds two
idle residents (~600 MB) and one warm (201 MB growth <= ~300 MB headroom),
but the second model's first request must then be refused: the first
model's drift (~486 MB warm) left only ~114 MB.
"""

from __future__ import annotations

import threading
import time

import pytest
from conftest import pinned_model_dir, running_stack
from test_lifecycle import complete, metric_value, readyz

QWEN25 = "qwen2.5-0.5b-4bit"

REJECT_BUDGET = 450_000_000
SERVE_BUDGET = 850_000_000
DRIFT_BUDGET = 900_000_000


def require(model_id: str) -> str:
    path = pinned_model_dir(model_id)
    if path is None:
        pytest.skip(
            f"pinned test model '{model_id}' not found; run "
            "./scripts/fetch-test-model.sh"
        )
    return str(path)


def test_request_rejected_when_pool_growth_exceeds_headroom():
    """The load-time check passes; the per-request check refuses. The 503
    is structured and retriable, the worker stays READY, and — the point —
    the pool never materializes, so the machine never drifts over budget."""
    path = require(QWEN25)
    memory = f"[memory]\nbudget_bytes = {REJECT_BUDGET}\n"
    with running_stack([("solo", "rust", path)], extra_toml=memory) as stack:
        stack.wait_ready()
        _, statuses = readyz(stack)
        assert statuses["solo"] == "ready", statuses

        response = complete(stack, "solo")
        assert response.status_code == 503, response.text
        body = response.json()["error"]
        assert body["code"] == "insufficient_memory", response.text
        assert "bytes" in body["message"], response.text

        # Rejected, not wounded: the worker is still READY and nothing
        # materialized — usage stays where the load left it, under budget.
        _, statuses = readyz(stack)
        assert statuses["solo"] == "ready", statuses
        text = stack.metrics_text()
        assert metric_value(text, "kiln_admission_rejects_total", model="solo") == 1
        assert (
            metric_value(text, "kiln_worker_kv_pool_allocated_bytes", model="solo") == 0
        ), "a rejected request must not have materialized the pool"
        used = metric_value(text, "kiln_memory_used_bytes")
        assert used <= REJECT_BUDGET, f"ledger over budget: {used}"


def test_request_admitted_when_headroom_allows():
    """Positive control: the identical request under a budget with pool
    headroom serves, the pool materializes, and the ledger stays under
    budget — the gate refuses over-commitment, not traffic."""
    path = require(QWEN25)
    memory = f"[memory]\nbudget_bytes = {SERVE_BUDGET}\n"
    with running_stack([("solo", "rust", path)], extra_toml=memory) as stack:
        stack.wait_ready()
        response = complete(stack, "solo")
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["text"], response.text

        time.sleep(2)  # pool-inflated footprint reaches the ledger
        text = stack.metrics_text()
        assert metric_value(text, "kiln_admission_rejects_total", model="solo") is None
        assert (
            metric_value(text, "kiln_worker_kv_pool_allocated_bytes", model="solo")
            > 100_000_000
        ), "the served request should have materialized the pool"
        used = metric_value(text, "kiln_memory_used_bytes")
        assert used <= SERVE_BUDGET, f"ledger over budget: {used}"


def test_drift_from_one_model_gates_anothers_requests():
    """The continuous-drift closure (part 1's recorded gap): both models
    loaded within budget, but one model's traffic grows its footprint, and
    the OTHER model's first request is then refused — the admission check
    runs per-request against the live ledger, not per-load against a
    snapshot."""
    path = require(QWEN25)
    models = [("hot", "rust", path), ("cold", "rust", path)]
    memory = f"[memory]\nbudget_bytes = {DRIFT_BUDGET}\n"
    with running_stack(models, extra_toml=memory) as stack:
        stack.wait_ready()
        _, statuses = readyz(stack)
        assert statuses == {"hot": "ready", "cold": "ready"}, statuses

        # Warm the hot model: passes (201 MB growth vs ~300 MB headroom).
        response = complete(stack, "hot")
        assert response.status_code == 200, response.text
        time.sleep(2)  # the drifted footprint reaches the ledger

        # The cold model's first request now projects growth the drifted
        # machine cannot hold. Same request, same model config — only the
        # OTHER model's usage changed since both loads passed.
        response = complete(stack, "cold")
        assert response.status_code == 503, response.text
        assert response.json()["error"]["code"] == "insufficient_memory", response.text

        text = stack.metrics_text()
        assert metric_value(text, "kiln_admission_rejects_total", model="cold") == 1
        assert metric_value(text, "kiln_admission_rejects_total", model="hot") is None
        used = metric_value(text, "kiln_memory_used_bytes")
        assert used <= DRIFT_BUDGET, f"ledger over budget: {used}"
        # Both workers remain READY: the refusal is per-request policy,
        # never worker damage.
        _, statuses = readyz(stack)
        assert statuses == {"hot": "ready", "cold": "ready"}, statuses


def test_concurrent_admissions_cannot_jointly_overshoot():
    """Phase 9 part 3 addendum, Option A ruling: the admission TOCTOU.

    Two cold models under a budget with headroom for exactly ONE pool
    growth (the drift scenario's budget, fired CONCURRENTLY instead of
    sequentially). Before the reservation ledger, both admissions priced
    against the same heartbeat-lagged footprints and both passed — the
    machine materialized both pools and overshot the budget by ~100 MB
    (the CI run 29436961038 shape, +6.8 MB, made deterministic here by
    sub-millisecond simultaneous requests vs the 1 s heartbeat cadence).
    With the ledger, the winning admission's reservation is immediately
    visible to the loser's check: exactly one 200, one structured 503,
    and committed bytes (weights + materialized pools) NEVER exceed the
    budget — including transiently, sampled at 200 ms throughout."""
    path = require(QWEN25)
    models = [("left", "rust", path), ("right", "rust", path)]
    memory = f"[memory]\nbudget_bytes = {DRIFT_BUDGET}\n"
    with running_stack(models, extra_toml=memory) as stack:
        stack.wait_ready()
        time.sleep(2)  # measured heartbeats replace the load reservations

        results: dict[str, int] = {}
        barrier = threading.Barrier(2)

        def fire(model: str) -> None:
            barrier.wait()
            results[model] = complete(stack, model).status_code

        committed_samples: list[float] = []
        stop_polling = threading.Event()

        def poll_committed() -> None:
            while not stop_polling.is_set():
                text = stack.metrics_text()
                committed = sum(
                    (metric_value(text, "kiln_worker_weights_bytes", model=m) or 0)
                    + (
                        metric_value(
                            text, "kiln_worker_kv_pool_allocated_bytes", model=m
                        )
                        or 0
                    )
                    for m in ("left", "right")
                )
                committed_samples.append(committed)
                time.sleep(0.2)

        poller = threading.Thread(target=poll_committed, daemon=True)
        poller.start()
        threads = [
            threading.Thread(target=fire, args=(m,), daemon=True)
            for m in ("left", "right")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        time.sleep(6)  # heartbeats confirm materialization; reservations drain
        stop_polling.set()
        poller.join(timeout=10)

        statuses = sorted(results.values())
        assert statuses == [200, 503], (
            f"exactly one concurrent admission must win: {results} "
            "(both 200 = the pre-reservation TOCTOU; both 503 = no headroom "
            "for even one growth, budget miscalibrated)"
        )

        # The committed ledger never exceeded the budget, transiently
        # included — the part 2 "<= budget throughout" claim under real
        # concurrency.
        worst = max(committed_samples)
        assert worst <= DRIFT_BUDGET, (
            f"committed bytes overshot the budget: {worst:.0f} > "
            f"{DRIFT_BUDGET} across {len(committed_samples)} samples"
        )

        # Same pair again: the winner's pool is fully materialized (zero
        # growth -> 200); the loser still projects growth the machine
        # cannot hold (503). Deterministic now that decisions are
        # reservation-serialized.
        rerun = {m: complete(stack, m).status_code for m in results}
        for model, first_status in results.items():
            assert rerun[model] == first_status, (
                f"outcome flapped on retry: first={results} rerun={rerun}"
            )

        text = stack.metrics_text()
        loser = next(m for m, s in results.items() if s == 503)
        assert metric_value(text, "kiln_admission_rejects_total", model=loser) == 2
        # Reservations are transient bookkeeping: fully reconciled against
        # heartbeats once the dust settles.
        assert metric_value(text, "kiln_memory_reserved_bytes") == 0, (
            "reservations must drain to zero after heartbeats confirm usage"
        )
        # No uncovered growth: every materialized byte was priced by an
        # admission first.
        for model in ("left", "right"):
            assert (
                metric_value(text, "kiln_admission_uncovered_bytes_total", model=model)
                is None
            ), f"pool growth on {model} was never covered by a reservation"
        _, ready = readyz(stack)
        assert ready == {"left": "ready", "right": "ready"}, ready
