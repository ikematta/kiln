"""Kiln Python worker: mlx-lm fallback behind the Kiln worker protocol.

The gRPC servicer is implemented in Phase 1 (SPEC §9.2). This package
currently ships only the generated protocol stubs under
``kiln_worker_py/gen`` (regenerate with the grpc_tools.protoc command in
CLAUDE.md).
"""
