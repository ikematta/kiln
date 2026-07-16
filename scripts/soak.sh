#!/usr/bin/env bash
# soak.sh — full-stack mixed-load soak (SPEC §11.3 leak gate; CLAUDE.md).
#
# Runs tests/e2e/test_soak.py for the requested wall-clock duration against
# a real 5-model gateway stack (see the module docstring for the scenario
# and gates). Debug binaries on purpose: the mlx live-object counter that
# backs the leak gate is compiled out of release builds.
#
# Usage: ./scripts/soak.sh [--minutes N]     (default 30, per SPEC §11.3)
set -euo pipefail

MINUTES=30
while [[ $# -gt 0 ]]; do
  case "$1" in
    --minutes)
      [[ $# -ge 2 ]] || { echo "--minutes needs a value" >&2; exit 2; }
      MINUTES="$2"
      shift 2
      ;;
    *)
      echo "usage: $0 [--minutes N]" >&2
      exit 2
      ;;
  esac
done

cd "$(dirname "$0")/.."
export KILN_TEST_MODELS="${KILN_TEST_MODELS:-$HOME/.kiln/test-models}"

exec env KILN_SOAK_MINUTES="$MINUTES" \
  uv run --project tests/e2e pytest tests/e2e/test_soak.py -v -s
