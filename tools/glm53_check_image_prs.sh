#!/usr/bin/env bash
# Are the ten GLM-5.3-Flash Day-0 PRs in this image's own sglang checkout?
#
# Two independent checks, because neither alone is conclusive:
#   1. Commit archaeology: is a commit whose subject ends in (#NNNNN) reachable
#      from HEAD. Cheap, but an image can carry a cherry-pick with a rewritten
#      subject, or a squash, and then this misses it.
#   2. Functional probes: does the code each PR adds actually exist and fire.
#      This is what decides whether the stack needs to be applied on top.
#
# Output: SGLang-benchmarks/tmp/logs/image_pr_check.txt
OUT=/home/jacchang/SGLang-benchmarks/tmp/logs
mkdir -p "$OUT"
exec > "$OUT/image_pr_check.txt" 2>&1

SRC=${SRC:-/sgl-workspace/sglang}
git config --global --add safe.directory "$SRC" 2>/dev/null

echo "=== image sglang checkout ==="
git -C "$SRC" log --oneline -1
git -C "$SRC" rev-parse HEAD
python3 -c 'import sglang, os; print("package:", sglang.__version__, os.path.dirname(sglang.__file__))'
echo "rocm: $(cat /opt/rocm/.info/version 2>/dev/null)  triton: $(python3 -c 'import triton;print(triton.__version__)' 2>/dev/null)"
echo "aiter: $(git -C /sgl-workspace/aiter rev-parse --short HEAD 2>/dev/null || echo '?')"

echo
echo "=== 1. commits mentioning each PR number, one pass over recent history ==="
LOG=$(git -C "$SRC" log --since=2026-08-01 --format='%h %cs %s' 2>/dev/null)
for n in 39338 39339 39340 39341 38545 38546 38547 39317 39778 39779; do
    hit=$(printf '%s\n' "$LOG" | grep -E "\(#$n\)" | head -1)
    if [ -n "$hit" ]; then
        printf 'PR%-6s IN   %s\n' "$n" "$(echo "$hit" | cut -c1-96)"
    else
        printf 'PR%-6s ---  no commit subject with (#%s)\n' "$n" "$n"
    fi
done

echo
echo "=== 2. functional probes ==="

probe() {   # probe <label> <file> <pattern>
    local label=$1 file=$2 pat=$3
    if [ ! -f "$SRC/$file" ]; then
        printf '  %-58s FILE MISSING (%s)\n' "$label" "$file"
        return
    fi
    if grep -q "$pat" "$SRC/$file"; then
        printf '  %-58s yes\n' "$label"
    else
        printf '  %-58s NO\n' "$label"
    fi
}

# #38545 routes mHC through aiter; the cheapest proof is the gate itself.
probe "#38545 mhc.py has the aiter gfx95 path" \
      python/sglang/kernels/ops/layernorm/mhc.py "Using AITER gfx950 mHC pre/post kernels"
# #38546 + #39317 are the mixed Quark MXFP4 / block-FP8 loading.
probe "#38546 quark.py resolves layer_quant_config per layer" \
      python/sglang/srt/layers/quantization/quark/quark.py "_find_matched_config"
probe "#39317 quark utils honors fused / per-expert exclude names" \
      python/sglang/srt/layers/quantization/quark/utils.py "should_ignore_layer"
# #39341 / #39339 / #39340 are the k-pool DSA indexer and its transform.
probe "#39341 k-pool DSA indexer exists" \
      python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py "index_kpool"
probe "#39340 page-table transform takes non-2048 top-k" \
      python/sglang/kernels/ops/attention/dsa/transform_index.py "topk"
probe "#39338 zero-RoPE MHA prefill on ROCm" \
      python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha_rocm.py "norope"
probe "#38547 TileLang DSA zero-RoPE" \
      python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py "norope"
# #39779 loads the MXFP4 MTP draft layer.
probe "#39779 nextn draft layer module present" \
      python/sglang/srt/models/glm5_next_nextn.py "class"
# The gate the whole gfx950 fast path hangs on.
probe "shared-experts fusion gate still CUDA-only (NOT patched)" \
      python/sglang/srt/models/glm5_next.py "requires CUDA devices"
probe "#39200 fused mHC attn->MLP boundary (upstream, not one of the ten)" \
      python/sglang/srt/models/glm5_next.py "hc_ffn_post_pre"

echo
echo "=== 3. does the model build take the gfx950 AITER path? ==="
python3 -c 'from sglang.srt.models.glm5_next import _use_aiter_gfx95; print("  _use_aiter_gfx95 =", _use_aiter_gfx95)' 2>&1 | tail -2

echo
echo "=== 4. per-layer quark quant resolution (the #38546 + #39317 acid test) ==="
python3 /home/jacchang/SGLang-benchmarks/tools/glm53_mxfp4_quant_resolution.py \
    /data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4/config.json 2>&1 | tail -18
