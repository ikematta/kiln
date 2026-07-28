#!/usr/bin/env bash
# sweep-build-cache.sh — prune stale Cargo build artifacts from the Kiln workspace.
#
# Runs `cargo sweep --time 14` against the repo root: removes only artifacts
# whose timestamps show no use in the last 14 days. This is INTENTIONALLY
# conservative — it is not `cargo clean` on a schedule. Anything touched by a
# recent build is left alone, so the next build after a sweep stays warm; the
# job only reclaims space from artifacts no build has looked at in two weeks
# (dep versions we've moved past, stale incremental state, old feature combos).
#
# Safety: exits without sweeping if any cargo/rustc process is running for this
# user — a sweep mid-compile could corrupt incremental state. A skipped week is
# free; the next scheduled run picks it up.
#
# Scheduling: installed as a weekly launchd LaunchAgent (macOS):
#   ~/Library/LaunchAgents/com.kiln.build-cache-sweep.plist
#   label: com.kiln.build-cache-sweep — fires Mondays 09:30 local
# New machine setup:
#   cargo install cargo-sweep
#   create the plist pointing at this script, then:
#   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.kiln.build-cache-sweep.plist
#
# Logging: appends to ~/.kiln/logs/cargo-sweep.log; the log is size-capped
# (rotated once to cargo-sweep.log.1 past ~512 KB) so it can't grow unbounded
# either. macOS-only script (stat -f, launchd) — matches the repo's platform.

set -euo pipefail

export PATH="$HOME/.cargo/bin:/usr/bin:/bin:/usr/sbin:/sbin"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$HOME/.kiln/logs"
LOG_FILE="$LOG_DIR/cargo-sweep.log"
MAX_LOG_BYTES=$((512 * 1024))
SWEEP_DAYS=14

mkdir -p "$LOG_DIR"

if [[ -f "$LOG_FILE" ]] && (( $(stat -f%z "$LOG_FILE") > MAX_LOG_BYTES )); then
    mv -f "$LOG_FILE" "$LOG_FILE.1"
fi

exec >>"$LOG_FILE" 2>&1

echo "==== $(date '+%Y-%m-%d %H:%M:%S %z') cargo-sweep (${SWEEP_DAYS}d) ===="

if pgrep -U "$(id -u)" -x cargo >/dev/null || pgrep -U "$(id -u)" -x rustc >/dev/null; then
    echo "SKIP: cargo/rustc running — not sweeping under an active build."
    exit 0
fi

if ! command -v cargo-sweep >/dev/null; then
    echo "ERROR: cargo-sweep not installed. Run: cargo install cargo-sweep"
    exit 1
fi

before_kb=$(du -sk "$REPO_ROOT/target" 2>/dev/null | cut -f1 || echo 0)
cargo sweep --time "$SWEEP_DAYS" "$REPO_ROOT"
after_kb=$(du -sk "$REPO_ROOT/target" 2>/dev/null | cut -f1 || echo 0)
echo "target/: $((before_kb / 1024)) MB -> $((after_kb / 1024)) MB (freed $(( (before_kb - after_kb) / 1024 )) MB)"
