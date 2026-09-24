#!/usr/bin/env bash
# Are PR36851 and PR37889 in this image, and can GLM-5.3-Flash reach the topk_v2
# kernel at all?
#
# GLM.sh forces SGLANG_OPT_USE_TOPK_V2=0 on ROCm. The reason recorded there is a
# build failure: the topk_v2 kernel was JIT-compiled from
# deepseek_v4/topk_impl.cuh, which #includes <cooperative_groups.h>, a CUDA
# header ROCm 7.2 does not ship, so hipcc died at cuda-graph capture. This image
# is ROCm 10 and ships the flag defaulting to True, so the question is whether
# the workaround is now stale AND whether this model's k-pool indexer takes that
# path in the first place -- the two named PRs say "GLM", which may mean GLM-5.2,
# which is pure DSA with index_kpool=1 while GLM-5.3-Flash has index_kpool=4.
#
# Output: SGLang-benchmarks/tmp/logs/topk_v2_probe.txt
OUT=/home/jacchang/SGLang-benchmarks/tmp/logs
mkdir -p "$OUT"
exec > "$OUT/topk_v2_probe.txt" 2>&1

SRC=${SRC:-/sgl-workspace/sglang}
git config --global --add safe.directory "$SRC" 2>/dev/null

echo "=== are the two PRs in this checkout? ==="
LOG=$(git -C "$SRC" log --since=2026-06-01 --format='%h %cs %s' 2>/dev/null)
for n in 36851 37889 30715; do
    hit=$(printf '%s\n' "$LOG" | grep -E "\(#$n\)" | head -1)
    if [ -n "$hit" ]; then printf 'PR%-6s IN   %s\n' "$n" "$(echo "$hit" | cut -c1-100)"
    else printf 'PR%-6s ---  not found\n' "$n"; fi
done

echo
echo "=== where SGLANG_OPT_USE_TOPK_V2 is read, and what else gates it ==="
grep -rn "OPT_USE_TOPK_V2" "$SRC/python" | head -20

echo
echo "=== the fast_topk / topk_v2 dispatch ==="
grep -rn -B10 -A22 "use_topk_v2\|fast_topk_v2" "$SRC/python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py" 2>/dev/null | head -70

echo
echo "=== does the k-pool indexer (GLM-5.3-Flash) reach it? ==="
grep -rn "topk_v2\|fast_topk" "$SRC/python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py" | head -12
echo "--- and the non-kpool indexer, for contrast:"
grep -rn "topk_v2\|fast_topk" "$SRC/python/sglang/srt/layers/attention/dsa/dsa_indexer.py" | head -12

echo
echo "=== is the old build blocker still there? (cooperative_groups in the jit header) ==="
h=$(find "$SRC/python/sglang" -name 'topk_impl.cuh' 2>/dev/null | head -1)
if [ -n "$h" ]; then
    echo "  header: $h"
    grep -n "cooperative_groups\|hipcub\|rocprim" "$h" | head -5
else
    echo "  topk_impl.cuh not found under python/sglang -- the JIT path may be gone"
    find "$SRC/python/sglang" -path '*deepseek_v4*' -name '*.cuh' 2>/dev/null | head -5
fi

echo
echo "=== can the kernel be imported / built right now? ==="
python3 - <<'PY'
try:
    from sgl_kernel import fast_topk_v2
    print("  sgl_kernel.fast_topk_v2 imports:", fast_topk_v2)
except Exception as e:
    print("  import fast_topk_v2 failed:", type(e).__name__, e)
try:
    from sglang.srt.environ import envs
    print("  envs.SGLANG_OPT_USE_TOPK_V2 default =", envs.SGLANG_OPT_USE_TOPK_V2.get())
except Exception as e:
    print("  envs read failed:", e)
PY
