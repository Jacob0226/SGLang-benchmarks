#!/usr/bin/env bash
# Copy the small, irreplaceable part of an AgentX sweep into the tracked tree.
#
# results/ is gitignored because the aiperf artifacts run to ~300 MB a point,
# but the numbers that matter fit in a few KB: the result JSON the summariser
# reads, the workload distribution, the exact server and client command lines,
# and sweep.log with the pool sizes. Archiving those means a lost machine
# reservation costs the raw traces, not the curve.
#
#   ./archive_agentx.sh <sweep-dir>     # -> Agentic/archive/<model>/<image>/<leaf>/
set -euo pipefail

SRC="$(cd "${1:?usage: $0 <sweep-dir>}" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEAF="$(basename "$SRC")"
IMAGE="$(basename "$(dirname "$SRC")")"
MODEL="$(basename "$(dirname "$(dirname "$SRC")")")"
DST="$HERE/archive/$MODEL/$IMAGE/$LEAF"
mkdir -p "$DST"

cp "$SRC/sweep.log" "$DST/" 2>/dev/null || true
for point in "$SRC"/*conc*/; do
    name="$(basename "$point")"
    # Skip a point that is still running: no result JSON yet.
    ls "$point"/*_local-*.json >/dev/null 2>&1 || continue
    mkdir -p "$DST/$name"
    cp "$point"/*_local-*.json "$DST/$name/"
    for f in workload_distribution_summary.txt sglang_command.txt benchmark_command.txt; do
        [ -f "$point/$f" ] && cp "$point/$f" "$DST/$name/"
    done
done
echo "archived to $DST"
