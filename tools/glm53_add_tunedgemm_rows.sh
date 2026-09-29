#!/bin/bash
# Route the router GEMM (N=288, K=4096, bf16) to triton for decode batches 24..64.
#
# aiter.tuned_gemm matches on (gfx, cu_num, M, N, K, ...) trying M, then
# get_padded_m(M,N,K,0), then get_padded_m(M,N,K,1). model_configs/
# glm53_bf16_tuned_gemm.csv ships rows at M=2/4/16 only, so decode batch sizes
# 24..64 find nothing and fall back to "torch solution:0" (hipblaslt), which is
# why the tuned triton JSON does nothing for conc24..64.
#
# Padding means two rows cover the whole range:
#   bs=24,32       -> padded 32
#   bs=40,48,56,64 -> padded 64 (via gl=1)
#
# libtype=triton makes tuned_gemm call gemm_a16w16() with no explicit config,
# so the tile still comes from GEMM-A16W16-N=288-K=4096.json.
#
#   glm53_add_tunedgemm_rows.sh          add the rows
#   glm53_add_tunedgemm_rows.sh --revert restore the original file
set -uo pipefail
C=/sgl-workspace/aiter/aiter/configs/model_configs/glm53_bf16_tuned_gemm.csv
[ -f "$C.orig" ] || cp "$C" "$C.orig"

if [ "${1:-}" = "--revert" ]; then
    cp "$C.orig" "$C"
    rm -f /tmp/aiter_configs/bf16_tuned_gemm.csv
    echo "reverted $C and dropped the merged /tmp copy"
    exit 0
fi

cp "$C.orig" "$C"
# us/tflops/bw are informational; the values are the measured medians from
# tools/glm53_sweep_router_buckets.py for the tuned single-kernel config.
cat >> "$C" <<'ROWS'
gfx950,256,32,288,4096,False,torch.bfloat16,torch.bfloat16,False,False,triton,0,0,5.2700,auto,0.0,14.3,481.2
gfx950,256,64,288,4096,False,torch.bfloat16,torch.bfloat16,False,False,triton,0,0,5.5600,auto,0.0,27.2,456.1
ROWS

# The runtime table is a merge of model_configs/*bf16_tuned_gemm*.csv written to
# /tmp/aiter_configs/; drop it so the next process rebuilds it.
rm -f /tmp/aiter_configs/bf16_tuned_gemm.csv
echo "appended 2 rows to $C and dropped the merged /tmp copy"
echo
echo "--- N=288 rows now in the source file ---"
awk -F, 'NR==1 || $4==288' "$C"

cd /sgl-workspace
python - <<'PY'
import aiter
from aiter.tuned_gemm import get_GEMM_A16W16_config
from aiter.ops.gemm_op_common import get_padded_m

print("\n--- lookup per decode batch size (N=288, K=4096, bf16) ---")
for M in (1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 128):
    cfg = get_GEMM_A16W16_config(
        M=M, N=288, K=4096, bias=False,
        dtype="torch.bfloat16", otype="torch.bfloat16",
    )
    lib = cfg.get("libtype", "?")
    mark = "  <-- triton" if lib == "triton" else ""
    print(f"  M={M:>4}  padded(gl0)={get_padded_m(M,288,4096,0):>4} "
          f"padded(gl1)={get_padded_m(M,288,4096,1):>4}  libtype={lib}{mark}")
PY
