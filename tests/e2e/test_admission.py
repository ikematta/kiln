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
