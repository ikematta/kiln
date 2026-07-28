"""Guards the soak's gateway-log forensics (test_soak.py, failure path).

The soak itself only runs under KILN_SOAK_MINUTES, so nothing in the normal
sweep would notice if its log forensics stopped working — and the forensics
only ever execute when a 30-minute run has ALREADY failed, which is the
worst moment to discover a typo. Two failure modes are covered here, both
cheap and stack-free:

  - String coupling. The recycle-cause literals are the supervisor's
    `tracing` format strings, copied. Reword one in supervisor.rs and the
    forensics silently report "no supervisor recycle line" for a real
    crash — indistinguishable from the log genuinely lacking one. So every
    literal must still be present in supervisor.rs.
  - Parsing and attribution. tracing-subscriber's JSON shape (fmt().json():
    fields.message plus flattened fields, RFC3339 UTC timestamps) against
    the parser; the asymmetric worker-output window, which must catch a
    line landing just after the exit line without reaching into the
    restarted worker's output 500 ms later; and the degradation paths for
    a missing or garbled log.
"""

from __future__ import annotations

import json
import pathlib
import time

import test_soak as soak
from conftest import REPO

SUPERVISOR = REPO / "crates" / "kiln-gateway" / "src" / "supervisor.rs"


def test_recycle_cause_literals_exist_in_supervisor():
    """Every cause string the forensics match on is still logged by the
    supervisor. A rename here is a silent loss of root-cause evidence."""
    source = SUPERVISOR.read_text()
    missing = [msg for msg in soak.LIFECYCLE_MESSAGES if msg not in source]
    assert not missing, (
        "these test_soak forensics literals no longer appear in "
        f"{SUPERVISOR.relative_to(REPO)}: {missing}. Re-copy them from the "
        "tracing calls; the gate they annotate (crash-restarts observed) "
        "cannot be root-caused without them."
    )


def test_every_crashed_return_has_a_matched_cause_line():
    """Each `return RunExit::Crashed` in the supervisor is preceded by a log
    line the forensics recognise — so no recycle path can go unlabelled."""
    lines = SUPERVISOR.read_text().splitlines()
    unlabelled = []
    for index, line in enumerate(lines):
        if "RunExit::Crashed" not in line or "return" not in line:
            continue
        # The cause is logged within the same short arm; 12 lines back
        # covers the widest (spawn failure: log, release, return).
        window = "\n".join(lines[max(0, index - 12) : index])
        if not any(msg in window for msg in soak.RECYCLE_CAUSES):
            unlabelled.append(index + 1)
    assert not unlabelled, (
        f"{SUPERVISOR.relative_to(REPO)} returns RunExit::Crashed at line(s) "
        f"{unlabelled} without a preceding log line in test_soak."
        "RECYCLE_CAUSES — a restart down that path would leave "
        "kiln_worker_restarts_total unexplained."
    )


def json_line(when: float, level: str, target: str, message: str, **fields) -> str:
    """One line in tracing-subscriber's fmt().json() shape."""
    base = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(when))
    return json.dumps(
        {
            "timestamp": f"{base}.{int((when % 1) * 1e9):09d}Z",
            "level": level,
            "fields": {"message": message, **fields},
            "target": target,
        }
    )


def write_log(tmp_path: pathlib.Path, t0: float) -> pathlib.Path:
    path = tmp_path / "gateway.log"
    path.write_text(
        "\n".join(
            [
                json_line(t0 + 1, "INFO", "kiln_gateway", "kiln-gateway listening"),
                json_line(
                    t0 + 2,
                    "INFO",
                    "kiln_gateway::supervisor",
                    "worker spawned",
                    model="py-smollm",
                    pid=4242,
                ),
                json_line(
                    t0 + 30,
                    "INFO",
                    "kiln::worker",
                    "MemoryError: unable to allocate",
                    model="py-smollm",
                    source="stderr",
                ),
                json_line(
                    t0 + 31,
                    "ERROR",
                    "kiln_gateway::supervisor",
                    "worker process exited",
                    model="py-smollm",
                    status="Some(ExitStatus(unix_wait_status(9)))",
                ),
                # forward_output drains on its own task: the dying worker's
                # last line can land AFTER the supervisor's exit line.
                json_line(
                    t0 + 31.4,
                    "INFO",
                    "kiln::worker",
                    "Traceback (most recent call last):",
                    model="py-smollm",
                    source="stderr",
                ),
                # The replacement generation, 500 ms later (backoff(1)), and
                # its own first stderr — neither may be attributed to the
                # death above.
                json_line(
                    t0 + 31.5,
                    "INFO",
                    "kiln_gateway::supervisor",
                    "worker spawned",
                    model="py-smollm",
                    pid=4343,
                ),
                json_line(
                    t0 + 33,
                    "INFO",
                    "kiln::worker",
                    "NotImplementedError: Method not implemented!",
                    model="py-smollm",
                    source="stderr",
                ),
                json_line(
                    t0 + 60,
                    "ERROR",
                    "kiln_gateway::supervisor",
                    "worker missed health deadline; recycling",
                    model="burst-gemma",
                    silent_ms=3211,
                ),
                json_line(
                    t0 + 70,
                    "INFO",
                    "kiln_gateway::supervisor",
                    "unloading worker",
                    model="ttl-qwen25",
                    reason="idle_ttl",
                ),
                "{ this line is not valid json",
            ]
        )
        + "\n"
    )
    return path


def test_parses_tracing_json_and_attributes_per_model(tmp_path):
    t0 = time.time()
    records = soak.read_gateway_log(write_log(tmp_path, t0))

    assert len(records) == 9, "the one non-JSON line must be skipped, not fatal"
    assert all(r.when is not None for r in records), "sub-second timestamps parse"
    assert records[0].when == pytest_approx(t0 + 1)

    causes = soak.restart_attribution(records, t0)
    assert sorted(causes) == ["burst-gemma", "py-smollm"]

    smol = "\n".join(causes["py-smollm"])
    assert "worker process exited" in smol
    # The discriminating field, not just the message: this is what says the
    # process was signalled rather than exiting on its own.
    assert "unix_wait_status(9)" in smol
    # Worker output on BOTH sides of the exit line: forward_output drains on
    # its own task, so the last stderr can arrive after the exit is logged.
    assert "MemoryError" in smol and "Traceback" in smol
    # ...but NOT across the generation boundary. The replacement spawns
    # 500 ms later; its first stderr belongs to it, not to the death.
    assert "NotImplementedError" not in smol, (
        "output from the RESTARTED worker was attributed to its "
        "predecessor's death — the forward window must stop at the next "
        "`worker spawned` for that model"
    )
    # ...rebased onto the soak's t+ clock, so the log lines up with the
    # `[soak t+...]` status lines and the checkpoint table.
    assert "t   +31.0s" in smol

    gemma = "\n".join(causes["burst-gemma"])
    assert "missed health deadline" in gemma and "silent_ms=3211" in gemma
    # No forwarded output for gemma: the silent-death note must say so
    # rather than leaving an empty gap.
    assert "no worker output" in gemma


def test_report_includes_causes_timeline_and_raw_tail(tmp_path):
    t0 = time.time()
    path = write_log(tmp_path, t0)
    report = soak.gateway_log_report(soak.read_gateway_log(path), t0, path)

    assert "recycle causes" in report
    assert "worker process exited" in report
    # Deliberate lifecycle traffic is context, not a cause.
    assert "unloading worker" in report
    assert "py-smollm" not in report.split("recycle causes")[0]
    assert "raw tail" in report


def test_report_degrades_without_a_log(tmp_path):
    missing = tmp_path / "gone.log"
    report = soak.gateway_log_report([], time.time(), missing)
    assert "<no log>" in report

    garbled = tmp_path / "garbled.log"
    garbled.write_text("not json\nstill not json\n")
    report = soak.gateway_log_report(
        soak.read_gateway_log(garbled), time.time(), garbled
    )
    assert "no parseable JSON lines" in report
    assert "still not json" in report


def test_preserve_copies_the_log_out_of_the_runtime_dir(tmp_path, monkeypatch):
    """running_stack deletes the runtime dir at teardown; the copy is what
    ci.yml uploads."""
    source = tmp_path / "gateway.log"
    source.write_text("line one\nline two\n")
    dest_dir = tmp_path / "preserved"
    monkeypatch.setenv("KILN_SOAK_LOG_DIR", str(dest_dir))

    saved = soak.preserve_gateway_log(source)
    assert saved is not None and saved.read_text() == "line one\nline two\n"
    assert saved.parent == dest_dir

    monkeypatch.setenv("KILN_SOAK_LOG_DIR", str(tmp_path / "nope"))
    assert soak.preserve_gateway_log(tmp_path / "missing.log") is None


def test_failure_handler_dumps_and_preserves_then_reraises(
    tmp_path, monkeypatch, capsys
):
    """The soak body's escape hatch: whatever raised — a violated gate, a
    warmup that never got admitted — the log must be dumped and copied out
    before running_stack deletes the runtime dir, and the original failure
    must still propagate."""
    t0 = time.time()
    log = write_log(tmp_path, t0)
    dest_dir = tmp_path / "preserved"
    monkeypatch.setenv("KILN_SOAK_LOG_DIR", str(dest_dir))

    class FakeStack:
        log_path = log

    import pytest

    with pytest.raises(AssertionError, match="crash-restarts observed"):
        with soak.gateway_log_on_failure(FakeStack()) as forensics:
            forensics.origin = t0
            raise AssertionError("crash-restarts observed: 1 (py-smollm=1)")

    printed = capsys.readouterr().out
    assert "recycle causes" in printed
    assert "worker process exited" in printed
    assert "preserved at" in printed
    assert list(dest_dir.iterdir()), "the log copy ci.yml uploads must exist"


def test_failure_handler_is_silent_on_success(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KILN_SOAK_LOG_DIR", str(tmp_path / "preserved"))
    with soak.gateway_log_on_failure(object()):
        pass
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "preserved").exists()


def pytest_approx(value):
    import pytest

    return pytest.approx(value, abs=0.001)
