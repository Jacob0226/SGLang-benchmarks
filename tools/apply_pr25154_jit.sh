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

# hicache.py gates JIT on element_size % N. The kernel now supports any
# element that tiles across the threads (multiple of 64 for our unrolls), so
# relax 128 -> 64 to enable DeepSeek MLA fp8 (element_size=576).
PY_SRC="$(python3 -c 'import sglang.jit_kernel.hicache as h; print(h.__file__)')"
# memory_pool_host gates JIT use on `_is_cuda`, which is False on ROCm, so the
# server never picks the JIT path on MI35x even when the kernel compiles.
# Relax to `(_is_cuda or _is_hip)` so ROCm uses the JIT transfer kernels too.
MP_SRC="$(python3 -c 'import sglang.srt.mem_cache.memory_pool_host as m; print(m.__file__)')"

if [[ "${1:-}" == "--restore" ]]; then
  if [[ -f "$DST.orig.bak" ]]; then
    cp "$DST.orig.bak" "$DST"; echo "restored original hicache.cuh"
  else
    echo "no backup found ($DST.orig.bak)"; exit 1
  fi
  sed -i 's/element_size % 64 != 0/element_size % 128 != 0/g' "$PY_SRC" || true
  sed -i 's/self.can_use_jit = (_is_cuda or _is_hip) and/self.can_use_jit = _is_cuda and/g' "$MP_SRC" || true
  echo "reverted hicache.py gate to % 128 + memory_pool_host JIT gate to _is_cuda"
else
  [[ -f "$PR_SRC" ]] || { echo "ERROR: PR source not found: $PR_SRC"; exit 1; }
  cp -n "$DST" "$DST.orig.bak" 2>/dev/null || true   # backup once
  cp "$PR_SRC" "$DST"
  echo "applied PR #25154 hicache.cuh (backup at $DST.orig.bak)"
  sed -i 's/element_size % 128 != 0/element_size % 64 != 0/g' "$PY_SRC" || true
  sed -i 's/self.can_use_jit = _is_cuda and/self.can_use_jit = (_is_cuda or _is_hip) and/g' "$MP_SRC" || true
  echo "relaxed hicache.py gate 128->64 + memory_pool_host JIT gate to (_is_cuda or _is_hip)"
fi

rm -rf "$HOME/.cache/tvm-ffi" 2>/dev/null || true
echo "cleared JIT cache (~/.cache/tvm-ffi)"
python3 - <<'PY'
from sglang.jit_kernel.hicache import can_use_hicache_jit_kernel as c
print("can_use(576)  =", c(element_size=576), "(fp8 MLA)")
print("can_use(1152) =", c(element_size=1152), "(bf16 MLA)")
PY
