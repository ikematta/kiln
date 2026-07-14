#!/usr/bin/env bash
# fetch-test-model.sh — download the pinned tiny test models (SPEC §11.1).
#
# Destination: $KILN_TEST_MODELS if set, else ~/.kiln/test-models.
# Layout: <dest>/<local-name>/<repo files>, plus a .kiln-revision marker.
#
# The revisions below are FROZEN (CLAUDE.md). Bumping one invalidates every
# golden fixture generated against it; do not change without instruction.
#
# Usage:
#   ./scripts/fetch-test-model.sh              # fetch all pinned models
#   ./scripts/fetch-test-model.sh --only llama-3.2-1b-4bit [--only ...]
#   ./scripts/fetch-test-model.sh --list

set -euo pipefail

# local-name  hf-repo  pinned-revision
PINS=(
  "llama-3.2-1b-4bit   mlx-community/Llama-3.2-1B-Instruct-4bit  08231374eeacb049a0eade7922910865b8fce912"
  "qwen3-0.6b-4bit     mlx-community/Qwen3-0.6B-4bit             73e3e38d981303bc594367cd910ea6eb48349da8"
  "gemma-3-1b-it-4bit  mlx-community/gemma-3-1b-it-4bit          2d44e83dc9e80843d22fb941d3d699a0b1351aa6"
  "smollm2-135m-bf16   mlx-community/SmolLM2-135M-Instruct       422de227b90002f443a21a58b1087f6ee7632731"
  # Phase 6 additions (new pins, existing pins untouched): qwen2 arch
  # coverage and the 8-bit quantization path (SPEC §12 Phase 6 matrix).
  "qwen2.5-0.5b-4bit   mlx-community/Qwen2.5-0.5B-Instruct-4bit  a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
  # Phase 6 Task 2: gemma2 arch coverage (smallest gemma-2 is 2B; no
  # tinier checkpoint exists). gemma3 is covered by the pinned
  # gemma-3-1b-it-4bit above.
  "gemma-2-2b-it-4bit  mlx-community/gemma-2-2b-it-4bit          2c715097ff9c081a6ac1e5cd239e2ac756b5bd99"
  # Phase 6 Task 3: the 8-bit quantization path (SPEC §7.3 matrix). Same
  # base model as the qwen3-0.6b-4bit pin, so the 4-bit/8-bit cells of
  # the dtype matrix differ only in quantization. Uniform affine
  # {group_size: 64, bits: 8} — no per-module overrides (config.rs
  # fail-loud check). BF16 is covered by smollm2-135m-bf16 above.
  "qwen3-0.6b-8bit     mlx-community/Qwen3-0.6B-8bit             11de96878523501bcaa86104e3c186de07ff9068"
)

DEST="${KILN_TEST_MODELS:-$HOME/.kiln/test-models}"
ONLY=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      ONLY+=("$2")
      shift 2
      ;;
    --list)
      for pin in "${PINS[@]}"; do echo "$pin" | awk '{printf "%-20s %-45s %s\n", $1, $2, $3}'; done
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

wanted() {
  [[ ${#ONLY[@]} -eq 0 ]] && return 0
  local name="$1" o
  for o in "${ONLY[@]}"; do [[ "$o" == "$name" ]] && return 0; done
  return 1
}

mkdir -p "$DEST"

for pin in "${PINS[@]}"; do
  read -r name repo rev <<<"$pin"
  wanted "$name" || continue
  echo "==> $name ($repo @ ${rev:0:12})"
  python3 - "$repo" "$rev" "$DEST/$name" <<'PYEOF'
"""Stdlib-only pinned-revision HF downloader with sha256 verification.

Every request carries a socket timeout and a bounded retry budget so a dead
connection fails fast instead of blocking read() forever (CI run 29305594680:
1h42m hang in this step). Large files stream to a .part file and resume via
HTTP Range. The stall detector only sees a fully silent socket; a connection
trickling a few bytes per timeout window is bounded by the CI step
timeout-minutes, not here.
"""

import hashlib
import http.client
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

repo, rev, dest = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
marker = dest / ".kiln-revision"

SKIP_PREFIXES = (".", "README")
# Standard hub override knob (mirrors; the stall tests point it at a local
# misbehaving server). Default is the real hub — pins stay frozen either way.
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
UA = {"User-Agent": "kiln-fetch-test-model"}
STALL_TIMEOUT_S = 30  # no bytes on the socket for this long = dead connection
MAX_ATTEMPTS = 4
RETRYABLE_HTTP = {408, 425, 429, 500, 502, 503, 504}


def backoff(attempt: int) -> None:
    if attempt > 1:
        time.sleep(5 * (attempt - 1))


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch_json(url: str):
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        backoff(attempt)
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=STALL_TIMEOUT_S) as resp:  # noqa: S310
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code not in RETRYABLE_HTTP:
                sys.exit(f"HTTP {e.code} for {url}")
            last = e
        except (OSError, http.client.HTTPException) as e:
            last = e
        print(f"    attempt {attempt}/{MAX_ATTEMPTS} failed: {last}", flush=True)
    sys.exit(f"giving up on {url} after {MAX_ATTEMPTS} attempts: {last}")


def download(url: str, out: pathlib.Path, size: int, lfs_sha) -> None:
    part = out.parent / (out.name + ".part")
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        backoff(attempt)
        try:
            part.touch()  # size-0 files skip the request loop but still rename
            offset = part.stat().st_size
            if offset > size:
                part.unlink()
                offset = 0
            if offset < size:
                headers = dict(UA)
                if offset:
                    print(f"    resuming {out.name} at {offset / 1e6:.1f} MB", flush=True)
                    headers["Range"] = f"bytes={offset}-"
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=STALL_TIMEOUT_S) as resp:  # noqa: S310
                    if offset and resp.status != 206:
                        offset = 0  # server ignored the Range; restart
                    with part.open("ab" if offset else "wb") as f:
                        # read1, not read: read(n) blocks until n bytes are
                        # buffered, so a stall mid-chunk would discard the
                        # partial chunk; read1 banks bytes as they arrive and
                        # the next attempt resumes from the true offset.
                        while chunk := resp.read1(1 << 20):
                            f.write(chunk)
            got = part.stat().st_size
            if got != size:
                raise OSError(f"short body: got {got}, want {size}")
            if lfs_sha is not None and sha256_file(part) != lfs_sha:
                part.unlink()  # never retry on top of a corrupt prefix
                raise OSError("sha256 mismatch, partial discarded")
            part.replace(out)
            return
        except urllib.error.HTTPError as e:
            if e.code == 416:  # resume window refused; start over
                part.unlink(missing_ok=True)
            elif e.code not in RETRYABLE_HTTP:
                sys.exit(f"HTTP {e.code} for {url}")
            last = e
        except (OSError, http.client.HTTPException) as e:
            last = e
        print(f"    attempt {attempt}/{MAX_ATTEMPTS} failed for {out.name}: {last}", flush=True)
    sys.exit(f"giving up on {out.name} after {MAX_ATTEMPTS} attempts: {last}")


tree = fetch_json(f"{ENDPOINT}/api/models/{repo}/tree/{rev}?recursive=true")
files = [
    e
    for e in tree
    if e["type"] == "file" and not e["path"].startswith(SKIP_PREFIXES)
]
if not files:
    sys.exit(f"no files listed for {repo}@{rev}")

dest.mkdir(parents=True, exist_ok=True)
for entry in files:
    path, size = entry["path"], entry["size"]
    lfs_sha = (entry.get("lfs") or {}).get("oid")
    out = dest / path
    if out.exists() and out.stat().st_size == size:
        if lfs_sha is None or sha256_file(out) == lfs_sha:
            print(f"    ok       {path}")
            continue
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"    fetching {path} ({size / 1e6:.1f} MB)", flush=True)
    download(f"{ENDPOINT}/{repo}/resolve/{rev}/{path}", out, size, lfs_sha)

marker.write_text(f"{repo}@{rev}\n")
print(f"    done -> {dest}")
PYEOF
done
