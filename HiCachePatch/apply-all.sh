#!/usr/bin/env bash
# Apply all HiCachePatch patches to a sglang checkout. Idempotent.
#
# Default target is /sgl-workspace/sglang (the container's checkout).
# Do NOT run on the host edit repo at ~/PR/sglang (commits originate
# from there and the patch would pollute pushes).
#
# Usage:
#   bash /home/jacchang/local-patches/HiCachePatch/apply-all.sh
#   bash /home/jacchang/local-patches/HiCachePatch/apply-all.sh /path/to/sglang
set -euo pipefail

REPO="${1:-/sgl-workspace/sglang}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== HiCachePatch: applying to $REPO =="
bash "$HERE/no-aiter-mem-fraction.sh" "$REPO"
bash "$HERE/pr25556-aiter-mla-page-size.sh" "$REPO"
echo "== done =="
