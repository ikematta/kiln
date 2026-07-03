"""Kiln Python worker: mlx-lm fallback behind the Kiln worker protocol.

Phase 1 surface (SPEC §9.2): GetInfo, Health, Submit (streaming), Cancel and
Tokenize over gRPC/UDS, sequential mlx-lm generation, worker-owned
tokenization (``raw_text`` input). Run with
``python -m kiln_worker_py --model <dir> --socket <path>``.

Generated protocol stubs live under ``kiln_worker_py/gen`` (regenerate with
the grpc_tools.protoc command in CLAUDE.md).
"""
