#!/bin/bash
# Install (or remove) a candidate GEMM-A16W16-N=288-K=4096.json into aiter.
#   glm53_install_candidate.sh best_single | best_overall | --revert
set -uo pipefail
CAND=${1:-best_single}
SRC=/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm_0928/GEMM-A16W16-N=288-K=4096.$CAND.json
DST_DIR=/sgl-workspace/aiter/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16
DST=$DST_DIR/GEMM-A16W16-N=288-K=4096.json

if [ "$CAND" = "--revert" ]; then
    rm -f "$DST" && echo "removed $DST" || echo "nothing to revert"
    exit 0
fi

[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }
cp "$SRC" "$DST"
chmod 644 "$DST"
echo "installed $CAND -> $DST"

python - <<'PY'
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16 import (
    _get_config as g,
)
from aiter.ops.triton.utils.gemm_config_utils import _get_gemm_config_cached

_get_gemm_config_cached.cache_clear()
for M in (1, 4, 8, 16, 32, 64, 128, 8192):
    c, tuned = g(M, 288, 4096)
    print(f"  M={M:5d} tuned={tuned!s:5s} BM={c['BLOCK_SIZE_M']:>3} "
          f"BN={c['BLOCK_SIZE_N']:>3} BK={c['BLOCK_SIZE_K']:>3} "
          f"KS={c['NUM_KSPLIT']} w={c['num_warps']} "
          f"wpe={c['waves_per_eu']} cm={c['cache_modifier']}")
print("  --- unrelated shape (must stay on DEFAULT) ---")
c, tuned = g(4, 512, 4096)
print(f"  N=512 tuned={tuned} BM={c['BLOCK_SIZE_M']} wpe={c['waves_per_eu']} "
      f"cm={c['cache_modifier']}")
PY
