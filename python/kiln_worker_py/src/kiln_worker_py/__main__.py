"""CLI entry point: ``python -m kiln_worker_py --model <dir> --socket <path>``."""

from __future__ import annotations

import argparse
import logging
import sys

from .server import serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kiln-worker-py",
        description="Kiln mlx-lm fallback worker (gRPC over a Unix domain socket)",
    )
    parser.add_argument(
        "--model", required=True, help="local model directory (mlx-lm layout)"
    )
    parser.add_argument("--socket", required=True, help="Unix domain socket path")
    parser.add_argument(
        "--model-id", default=None, help="model id to report (default: dir name)"
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    return serve(args.model, args.socket, model_id=args.model_id)


if __name__ == "__main__":
    sys.exit(main())
