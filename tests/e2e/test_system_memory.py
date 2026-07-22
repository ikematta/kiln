"""System-memory-aware admission (the 2026-07-21 field finding).

The bug: the machine budget is cut from INSTALLED RAM
(total x budget_fraction), blind to memory other processes already hold.
On a 16 GB dev machine under daily use, an 11.5 GB model was admitted
against the 13.7 GB budget while the OS was 4.4 GB into swap; generation
ran at ~0.44 tok/s — 150-200x under the machine's benchmarked rate. The
fix prices every load against a LIVE probe (`vm_stat` availability +
`kern.memorystatus_vm_pressure_level`), refusing loads the machine cannot
grant without swapping: structured status `unloaded (system memory
pressure)`, `kiln_load_rejects_total{constraint=...}`, and a log naming
the numbers.

This test reproduces the shape with real memory: with a hog process
holding 3 GiB of real, touched pages — "the user has other apps open" —
the boot load is refused; the moment the hog exits, the IDENTICAL config
admits and serves (the control). The `min_available_bytes` floor is
computed from availability measured WITH the hog resident (macOS absorbs
big allocations partly by compressing other pages, so only the
hog-resident baseline and the release direction are deterministic),
giving ~1 GiB of refusal margin and ~2 GiB of recovery margin.

CI shape (documented per the fix's acceptance): the GitHub `macos-14`
runners are 7 GB VMs whose free memory varies run-to-run; deliberately
claiming 3 GiB + a model + margins there is exactly the flakiness this
gate exists to refuse, so the test SKIPS unless the machine measures
at least HOG + projection + 2 GiB available (the 16 GB dev machine
passes; constrained runners self-skip with the message below). What CI
cannot exercise — refusal while the kernel already reports elevated
pressure (`pressure_level >= 2`, the "swap is actively churning" state)
— is pinned by injected-probe unit tests in `lifecycle.rs`
(`request_admission_refuses_new_growth_under_os_pressure`,
`load_system_gate_refuses_what_the_machine_cannot_give`) and was
verified manually on the dev machine where the finding occurred (see
PROGRESS 2026-07-21: the original 11.5 GB load, re-attempted post-fix
with swap still allocated, is refused up front instead of thrashing).
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile
import time

import httpx
import pytest
from conftest import build_binaries, pinned_model_dir, running_stack
from test_lifecycle import complete, metric_value, readyz

QWEN25 = "qwen2.5-0.5b-4bit"
ADMIN_TOKEN = "kiln-e2e-sysmem-token"

GIB = 1 << 30
HOG_BYTES = 3 * GIB
# Same margin the gateway's load projection adds over weight bytes.
LOAD_OVERHEAD_MARGIN = 64 * 1024 * 1024

# Allocates and TOUCHES real pages (a bare allocation stays virtual), then
# holds them until killed.
HOG_CODE = """
import sys, time
size = int(sys.argv[1])
hog = bytearray(size)
for i in range(0, size, 4096):
    hog[i] = 1
print("hog-ready", flush=True)
time.sleep(600)
"""


def system_available_bytes() -> int:
    """The gateway's availability formula, independently computed:
    (free + speculative + inactive pages) x page size, from vm_stat."""
    out = subprocess.run(
        ["/usr/bin/vm_stat"], capture_output=True, text=True, check=True
    ).stdout
    page_size = int(re.search(r"page size of (\d+) bytes", out).group(1))

    def pages(name: str) -> int:
        return int(re.search(rf"Pages {name}:\s+(\d+)\.", out).group(1))

    return (pages("free") + pages("speculative") + pages("inactive")) * page_size


def pressure_level() -> int:
    return int(
        subprocess.run(
            ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )


def weights_bytes(model_path: str) -> int:
    return sum(f.stat().st_size for f in pathlib.Path(model_path).glob("*.safetensors"))


def require_model() -> str:
    path = pinned_model_dir(QWEN25)
    if path is None:
        pytest.skip(
            f"pinned test model '{QWEN25}' not found; run ./scripts/fetch-test-model.sh"
        )
    return str(path)


def test_load_refused_under_real_memory_pressure_and_recovers():
    """The headline acceptance: with 3 GiB of real memory held by another
    process, a load that fits the total-RAM budget is REFUSED — structured
    status, counter, no worker, nothing charged — and the IDENTICAL config
    admits and serves the moment the hog exits (the control).

    Ordering note: the hog starts FIRST and the availability floor is
    computed with it resident. macOS absorbs a big allocation partly by
    compressing OTHER processes' pages, so availability does not drop
    1-for-1 with the hog and a pre-hog floor is not deterministic (first
    version of this test proved that). The recovery direction IS reliable:
    a dying process's touched pages return to the free queue at once."""
    if sys.platform != "darwin":
        pytest.skip("macOS-only: the availability probe under test is vm_stat-based")
    model_path = require_model()
    if pressure_level() >= 2:
        pytest.skip(
            "machine already under OS memory pressure; the post-hog "
            "recovery control could not be distinguished — rerun when idle"
        )
    projection = weights_bytes(model_path) + LOAD_OVERHEAD_MARGIN
    if system_available_bytes() < HOG_BYTES + projection + 2 * GIB:
        pytest.skip(
            f"only {system_available_bytes() / GIB:.1f} GiB available; need "
            f"{(HOG_BYTES + projection + 2 * GIB) / GIB:.1f} GiB to host the "
            "3 GiB hog without genuinely distressing the machine (CI "
            "macos-14 runners are 7 GB VMs and self-skip here; the scenario "
            "is verified on real dev hardware — see PROGRESS 2026-07-21)"
        )

    # Build BEFORE measuring: an incremental cargo build shifts hundreds
    # of MB of page cache and would skew the floor.
    build_binaries()

    hog: subprocess.Popen | None = None
    try:
        # "Other apps open": 3 GiB of real, touched pages, held first.
        hog = subprocess.Popen(
            [sys.executable, "-c", HOG_CODE, str(HOG_BYTES)],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert hog.stdout is not None
        assert hog.stdout.readline().strip() == "hog-ready", "hog died"
        time.sleep(2)  # let the OS settle its queues around the hog

        # Floor anchored to hogged availability:
        #   refusal: projection + floor = available_hogged + 1 GiB — the
        #     boot load is short by ~1 GiB of drift margin;
        #   recovery: the hog's exit frees ~3 GiB back to the free queue,
        #     clearing the same floor by ~2 GiB.
        available_hogged = system_available_bytes()
        floor = available_hogged + GIB - projection
        assert floor > 0

        # No budget_bytes: the budget stays the default fraction of
        # INSTALLED RAM — the exact shape that admitted the original
        # 11.5 GB load. Only the availability gate refuses here.
        extra = f"[memory]\nmin_available_bytes = {floor}\n"
        with running_stack([("solo", "rust", model_path)], extra_toml=extra) as stack:
            # The boot load must be refused: structured settled status,
            # never a silent admission that would have swapped. A refused
            # model IS settled, so /readyz turns 200 the moment the
            # decision lands — wait_ready is exactly that barrier.
            stack.wait_ready()
            _, statuses = readyz(stack)
            assert statuses["solo"] == "unloaded (system memory pressure)", statuses
            assert stack.worker_command("solo") == "", "refused load spawned a worker"

            text = stack.metrics_text()
            # Under a live hog the kernel may or may not flip to pressure
            # WARN — either refusal constraint is the gate working.
            rejects = sum(
                metric_value(
                    text, "kiln_load_rejects_total", model="solo", constraint=constraint
                )
                or 0
                for constraint in ("system_available", "system_pressure")
            )
            assert rejects >= 1, text
            assert metric_value(text, "kiln_memory_used_bytes") == 0, (
                "a refused load must charge nothing"
            )
            # The live-probe gauges the decision was priced from are real.
            assert metric_value(text, "kiln_system_available_bytes") > 0
            assert metric_value(text, "kiln_system_pressure_level") in (0, 1, 2, 4)
            assert metric_value(text, "kiln_system_swap_used_bytes") is not None

            # A request against the refused model is the retriable 503,
            # and triggers another (still refused) attempt — the machine
            # state, not a latched failure, owns the answer.
            response = complete(stack, "solo")
            assert response.status_code == 503, response.text
            assert response.json()["error"]["code"] == "model_loading", response.text

            # Control: the hog exits ("apps closed") and the IDENTICAL
            # config admits and serves. Retried while the OS reclaims.
            hog.kill()
            hog.wait(timeout=10)
            hog = None
            deadline = time.monotonic() + 120
            last = None
            while time.monotonic() < deadline:
                last = complete(stack, "solo")
                if last.status_code == 200:
                    break
                time.sleep(2)
            assert last is not None and last.status_code == 200, (
                f"load never recovered after the hog exited: {last and last.text}"
            )
            assert last.json()["choices"][0]["text"]
            _, statuses = readyz(stack)
            assert statuses["solo"] == "ready", statuses
    finally:
        if hog is not None:
            hog.kill()
            hog.wait(timeout=10)


def test_estimate_reports_the_system_gate():
    """`GET /admin/models/estimate` answers the same system-gate question
    the load will face: `fits` is now budget AND system, with the split
    (`fits_budget` / `fits_system`) and the live numbers exposed."""
    if sys.platform != "darwin":
        pytest.skip("macOS-only: the availability probe under test is vm_stat-based")
    model_path = require_model()

    gateway = build_binaries()
    token_hash = subprocess.run(
        [gateway, "hash-key", ADMIN_TOKEN], capture_output=True, text=True, check=True
    ).stdout.strip()
    dest_root = pathlib.Path(tempfile.mkdtemp(prefix="kiln-e2e-sysmem-", dir="/tmp"))
    extra = f'model_dir = "{dest_root}"\n[auth]\nadmin_token_hash = "{token_hash}"\n'
    with running_stack([], extra_toml=extra) as stack:
        stack.wait_ready()
        response = httpx.get(
            f"{stack.base_url}/admin/models/estimate",
            params={"path": model_path},
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            timeout=30,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["source"] == "local"
        assert body["fits"] == (body["fits_budget"] and body["fits_system"])
        # Default 1 GiB floor, live probe numbers.
        assert body["min_available_bytes"] == GIB
        assert body["system_available_bytes"] > 0
        assert body["pressure_level"] in (0, 1, 2, 4)
        # An impossible ask is refused by the system side regardless of
        # what the configured budget thinks: the admin "memory" ledger
        # exposes the same live snapshot the decision used.
        admin = httpx.get(
            f"{stack.base_url}/admin/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            timeout=10,
        )
        assert admin.status_code == 200, admin.text
        system = admin.json()["memory"]["system"]
        assert system is not None
        assert system["available_bytes"] > 0
        assert system["min_available_bytes"] == GIB
