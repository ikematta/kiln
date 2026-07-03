"""Fixtures: an in-process worker serving the pinned tiny model over UDS.

Integration tests need the pinned Llama test model; they skip (with an
actionable message) when KILN_TEST_MODELS is unset or the model has not been
fetched. Heavy imports (mlx via kiln_worker_py.server) stay inside the
fixture so collection works on machines without MLX.
"""

from __future__ import annotations

import os
import pathlib
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import grpc
import pytest

from kiln_worker_py._gen import worker_pb2 as pb
from kiln_worker_py._gen import worker_pb2_grpc

MODEL_NAME = "llama-3.2-1b-4bit"
READY_TIMEOUT_S = 300


def _model_dir() -> pathlib.Path | None:
    root = os.environ.get("KILN_TEST_MODELS")
    if not root:
        return None
    model_dir = pathlib.Path(root).expanduser() / MODEL_NAME
    return model_dir if (model_dir / "config.json").is_file() else None


@dataclass
class Worker:
    stub: worker_pb2_grpc.WorkerStub
    engine: Any  # kiln_worker_py.engine.Engine (typed loosely: mlx-free import)
    model_id: str


@pytest.fixture(scope="session")
def worker():
    model_dir = _model_dir()
    if model_dir is None:
        pytest.skip(
            f"set KILN_TEST_MODELS and fetch {MODEL_NAME} "
            "(./scripts/fetch-test-model.sh) to run worker integration tests"
        )
    pytest.importorskip("mlx_lm")
    from kiln_worker_py.server import build_server

    # mkdtemp over tmp_path: UDS paths must stay under the 104-char macOS cap.
    tmp = tempfile.mkdtemp(prefix="kiln-wt-")
    socket_path = os.path.join(tmp, "w.sock")
    server, engine = build_server(
        str(model_dir), socket_path, model_id="llama-test", max_workers=8
    )
    server.start()
    channel = grpc.insecure_channel(f"unix:{socket_path}")
    stub = worker_pb2_grpc.WorkerStub(channel)

    deadline = time.monotonic() + READY_TIMEOUT_S
    while True:
        status = stub.Health(pb.HealthRequest())
        if status.state == pb.WORKER_STATE_READY:
            break
        if status.state == pb.WORKER_STATE_UNHEALTHY:
            pytest.fail(f"worker unhealthy during load: {status.detail}")
        if time.monotonic() > deadline:
            pytest.fail("worker never reached READY")
        time.sleep(0.25)

    yield Worker(stub=stub, engine=engine, model_id="llama-test")

    channel.close()
    server.stop(grace=None).wait()
    engine.stop()
    pathlib.Path(socket_path).unlink(missing_ok=True)
