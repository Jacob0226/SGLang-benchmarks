#!/usr/bin/env bash
# Apply PR #25154 (ROCm fix for the HiCache JIT kernel) inside a container.
#
# The JIT kernel is compiled at runtime from hicache.cuh, so we only need to
# swap that one source file and clear the JIT cache -- no rebuild of sgl_kernel.
#
# Run this INSIDE the container (or via:  docker exec <name> bash tools/apply_pr25154_jit.sh)
# A copy of the PR's hicache.cuh is kept at out/hicache_pr25154.cuh (on the
# /home mount, so it is visible from any fresh container).
#
#   apply:   bash tools/apply_pr25154_jit.sh
#   restore: bash tools/apply_pr25154_jit.sh --restore
set -euo pipefail

PR_SRC="$(cd "$(dirname "$0")/.." && pwd)/out/hicache_pr25154.cuh"
# locate the installed sglang jit_kernel hicache.cuh
DST="$(python3 - <<'PY'
import os, sglang
base=os.path.dirname(sglang.__file__)
print(os.path.join(base, "jit_kernel", "csrc", "hicache.cuh"))
PY
)"
echo "target: $DST"

if [[ "${1:-}" == "--restore" ]]; then
  if [[ -f "$DST.orig.bak" ]]; then
    cp "$DST.orig.bak" "$DST"; echo "restored original hicache.cuh"
  else
    echo "no backup found ($DST.orig.bak)"; exit 1
  fi
else
  [[ -f "$PR_SRC" ]] || { echo "ERROR: PR source not found: $PR_SRC"; exit 1; }
  cp -n "$DST" "$DST.orig.bak" 2>/dev/null || true   # backup once
  cp "$PR_SRC" "$DST"
  echo "applied PR #25154 hicache.cuh (backup at $DST.orig.bak)"
fi

rm -rf "$HOME/.cache/tvm-ffi" 2>/dev/null || true
echo "cleared JIT cache (~/.cache/tvm-ffi)"
python3 - <<'PY'
from sglang.jit_kernel.hicache import can_use_hicache_jit_kernel as c
print("can_use(1152) =", c(element_size=1152))
PY
