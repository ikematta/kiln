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
"""Stdlib-only pinned-revision HF downloader with sha256 verification."""

import hashlib
import json
import pathlib
import sys
import urllib.request

repo, rev, dest = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
marker = dest / ".kiln-revision"

SKIP_PREFIXES = (".", "README")


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "kiln-fetch-test-model"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (https only)
        return resp.read()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


tree = json.loads(
    fetch(f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true")
)
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
    print(f"    fetching {path} ({size / 1e6:.1f} MB)")
    data = fetch(f"https://huggingface.co/{repo}/resolve/{rev}/{path}")
    if len(data) != size:
        sys.exit(f"size mismatch for {path}: got {len(data)}, want {size}")
    if lfs_sha is not None:
        got = hashlib.sha256(data).hexdigest()
        if got != lfs_sha:
            sys.exit(f"sha256 mismatch for {path}: got {got}, want {lfs_sha}")
    out.write_bytes(data)

marker.write_text(f"{repo}@{rev}\n")
print(f"    done -> {dest}")
PYEOF
done
