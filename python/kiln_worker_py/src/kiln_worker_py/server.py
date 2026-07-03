"""Worker process assembly: engine + gRPC server over a Unix domain socket."""

from __future__ import annotations

import logging
import pathlib
import resource
import signal
import threading
from concurrent import futures

import grpc

from ._gen import worker_pb2_grpc
from .engine import Engine
from .servicer import WorkerServicer

_LOG = logging.getLogger(__name__)

# mmap'd slabs + sockets exhaust the default macOS soft limit (CLAUDE.md).
_NOFILE_TARGET = 8192


def raise_nofile_limit(target: int = _NOFILE_TARGET) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = target if hard == resource.RLIM_INFINITY else min(target, hard)
    if soft < want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))


def build_server(
    model_path: str,
    socket_path: str,
    model_id: str | None = None,
    max_workers: int = 16,
) -> tuple[grpc.Server, Engine]:
    """Create the engine (model load starts immediately) and a bound server.

    The caller owns both: ``server.start()`` to serve, then ``server.stop()``
    and ``engine.stop()`` to shut down.
    """
    raise_nofile_limit()
    resolved_id = model_id or pathlib.Path(model_path).expanduser().name
    engine = Engine(model_path=model_path, model_id=resolved_id)
    engine.start()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    worker_pb2_grpc.add_WorkerServicer_to_server(WorkerServicer(engine), server)

    sock = pathlib.Path(socket_path)
    sock.parent.mkdir(parents=True, exist_ok=True)
    sock.unlink(missing_ok=True)
    server.add_insecure_port(f"unix:{socket_path}")
    return server, engine


def serve(model_path: str, socket_path: str, model_id: str | None = None) -> int:
    server, engine = build_server(model_path, socket_path, model_id=model_id)
    server.start()
    _LOG.info("worker listening on unix:%s", socket_path)

    shutdown = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: shutdown.set())
    shutdown.wait()

    _LOG.info("shutting down")
    server.stop(grace=10).wait()
    engine.stop()
    pathlib.Path(socket_path).unlink(missing_ok=True)
    return 0
