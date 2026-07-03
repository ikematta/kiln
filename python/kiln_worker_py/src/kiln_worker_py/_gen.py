"""Import shim for the generated protobuf/gRPC modules.

The generated code under ``gen/`` imports itself as ``kiln.v1.worker_pb2``
(absolute import emitted by grpc_tools.protoc), so the ``gen`` directory must
be on ``sys.path``. This module is the single place that arranges that; all
worker code imports the stubs from here.
"""

import pathlib
import sys

_GEN_DIR = str(pathlib.Path(__file__).parent / "gen")
if _GEN_DIR not in sys.path:
    sys.path.insert(0, _GEN_DIR)

from kiln.v1 import worker_pb2, worker_pb2_grpc  # noqa: E402

__all__ = ["worker_pb2", "worker_pb2_grpc"]
