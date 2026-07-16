#!/usr/bin/env bash
# bench.sh — throughput/TTFT benchmark harness (SPEC §11.3; CLAUDE.md).
#
# Two modes:
#   stack mode (default): builds the release gateway+worker from this
#     checkout, serves the given model on a scratch config, runs the HTTP
#     load lanes (scripts/bench.py), and writes JSON results to
#     bench/results/<utc-timestamp>-<model>.json.
#   --url mode: runs the load lanes against an already-running gateway
#     (works from an installed kiln without a checkout; `kiln bench`).
#
# --engine additionally runs the release engine perf gate
# (crates/kiln-models/tests/throughput.rs — the ADR 0003 bars), saving its
# output next to the JSON. Checkout + KILN_TEST_MODELS required.
#
# Usage:
#   ./scripts/bench.sh --model <name-or-path> [options]
#   ./scripts/bench.sh --url http://127.0.0.1:8080 --model <served-id> [options]
# Options:
#   --model <x>        model directory name (resolved under $KILN_TEST_MODELS,
#                      ~/.kiln/test-models, ~/.kiln/models) or a path; in
#                      --url mode, the served model id
#   --url <base>       benchmark a running gateway instead of spawning one
#   --api-key <key>    API key for --url mode (or $KILN_API_KEY)
#   --requests <n>     single-stream lane requests        (default 8)
#   --concurrency <n>  batch lane width                   (default 16)
#   --max-tokens <n>   decode length per request          (default 128)
#   --out <dir>        results directory                  (default bench/results)
#   --engine           also run the release engine perf gate
set -euo pipefail

MODEL="" URL="" API_KEY="${KILN_API_KEY:-}" REQUESTS=8 CONCURRENCY=16
MAX_TOKENS=128 OUT_DIR="" ENGINE=0

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL="${2:?--model needs a value}"; shift 2 ;;
    --url)         URL="${2:?--url needs a value}"; shift 2 ;;
    --api-key)     API_KEY="${2:?--api-key needs a value}"; shift 2 ;;
    --requests)    REQUESTS="${2:?--requests needs a value}"; shift 2 ;;
    --concurrency) CONCURRENCY="${2:?--concurrency needs a value}"; shift 2 ;;
    --max-tokens)  MAX_TOKENS="${2:?--max-tokens needs a value}"; shift 2 ;;
    --out)         OUT_DIR="${2:?--out needs a value}"; shift 2 ;;
    --engine)      ENGINE=1; shift ;;
    --help|-h)     usage; exit 0 ;;
    *) echo "bench.sh: unknown argument '$1' (--help for usage)" >&2; exit 2 ;;
  esac
done
[[ -n "$MODEL" ]] || { echo "bench.sh: --model is required (--help for usage)" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_PY="$SCRIPT_DIR/bench.py"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GATEWAY_PID=""
TMP_DIR=""
cleanup() {
  if [[ -n "$GATEWAY_PID" ]] && kill -0 "$GATEWAY_PID" 2>/dev/null; then
    kill -TERM "$GATEWAY_PID" 2>/dev/null || true
    wait "$GATEWAY_PID" 2>/dev/null || true
  fi
  [[ -n "$TMP_DIR" ]] && rm -rf "$TMP_DIR"
}
trap cleanup EXIT

MODEL_ID="$MODEL"
if [[ -z "$URL" ]]; then
  # Stack mode needs the checkout: release binaries + a scratch config.
  if [[ ! -f "$REPO_ROOT/Cargo.toml" ]]; then
    echo "bench.sh: stack mode needs a Kiln checkout; use --url against a running gateway" >&2
    exit 2
  fi
  # Resolve --model to a local directory.
  MODEL_DIR=""
  if [[ -d "$MODEL" ]]; then
    MODEL_DIR="$(cd "$MODEL" && pwd)"
  else
    for root in "${KILN_TEST_MODELS:-}" "$HOME/.kiln/test-models" "$HOME/.kiln/models"; do
      [[ -n "$root" && -d "$root/$MODEL" ]] && { MODEL_DIR="$root/$MODEL"; break; }
    done
  fi
  [[ -n "$MODEL_DIR" ]] || {
    echo "bench.sh: model '$MODEL' not found (searched \$KILN_TEST_MODELS, ~/.kiln/test-models, ~/.kiln/models)" >&2
    exit 2
  }
  MODEL_ID="$(basename "$MODEL_DIR")"

  echo "bench.sh: building release gateway + worker..." >&2
  (cd "$REPO_ROOT" && cargo build --release -p kiln-gateway -p kiln-worker >&2)

  PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
  TMP_DIR="$(mktemp -d /tmp/kiln-bench.XXXXXX)"
  mkdir -p "$TMP_DIR/run" "$TMP_DIR/cache"
  cat > "$TMP_DIR/kiln.toml" <<EOF
[server]
host = "127.0.0.1"
port = $PORT
runtime_dir = "$TMP_DIR/run"
cache_dir = "$TMP_DIR/cache"

[[model]]
id = "$MODEL_ID"
path = "$MODEL_DIR"
worker = "auto"
EOF

  echo "bench.sh: starting gateway on port $PORT..." >&2
  "$REPO_ROOT/target/release/kiln-gateway" --config "$TMP_DIR/kiln.toml" \
    > "$TMP_DIR/gateway.log" 2>&1 &
  GATEWAY_PID=$!
  URL="http://127.0.0.1:$PORT"

  for _ in $(seq 1 180); do
    kill -0 "$GATEWAY_PID" 2>/dev/null || {
      echo "bench.sh: gateway exited during startup; log:" >&2
      cat "$TMP_DIR/gateway.log" >&2
      exit 1
    }
    curl -fsS -o /dev/null "$URL/readyz" 2>/dev/null && break
    sleep 1
  done
  curl -fsS -o /dev/null "$URL/readyz" || {
    echo "bench.sh: gateway never became ready; log:" >&2
    cat "$TMP_DIR/gateway.log" >&2
    exit 1
  }
fi

OUT_DIR="${OUT_DIR:-$REPO_ROOT/bench/results}"
mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_FILE="$OUT_DIR/$STAMP-$MODEL_ID.json"

echo "bench.sh: load lanes against $URL (model $MODEL_ID)..." >&2
python3 "$BENCH_PY" --base "$URL" --model "$MODEL_ID" --api-key "$API_KEY" \
  --requests "$REQUESTS" --concurrency "$CONCURRENCY" --max-tokens "$MAX_TOKENS" \
  | tee "$RESULT_FILE"
echo "bench.sh: wrote $RESULT_FILE" >&2

if [[ "$ENGINE" == 1 ]]; then
  [[ -f "$REPO_ROOT/Cargo.toml" ]] || {
    echo "bench.sh: --engine needs a Kiln checkout" >&2
    exit 2
  }
  ENGINE_LOG="$OUT_DIR/$STAMP-$MODEL_ID.engine.log"
  echo "bench.sh: release engine perf gate (ADR 0003 bars)..." >&2
  (cd "$REPO_ROOT" && \
    KILN_TEST_MODELS="${KILN_TEST_MODELS:-$HOME/.kiln/test-models}" \
    cargo test -p kiln-models --release --test throughput -- --ignored --nocapture) \
    | tee "$ENGINE_LOG"
  echo "bench.sh: wrote $ENGINE_LOG" >&2
fi
