#!/usr/bin/env bash
# Why does the trace show fast_hadamard_transform_kernel and act_quant_kernel as
# separate launches when SGLANG_DSA_FUSE_HADAMARD_QUANT=1 is set?
#
# Three things to establish before spending GPU time on an A/B:
#   1. where the flag is read, and whether that is at import (so a toggle needs a
#      fresh server) or per call
#   2. what the fused path is gated on besides the flag -- PR #30715 is
#      shape-guarded on gfx950 and head_dim == block_size == 128
#   3. whether this model's shapes satisfy that guard: GLM-5.3-Flash has
#      index_head_dim 128, so it should, but index_kpool=4 changes the layout
#
# Output: SGLang-benchmarks/tmp/logs/hadamard_fuse_probe.txt
OUT=/home/jacchang/SGLang-benchmarks/tmp/logs
mkdir -p "$OUT"
exec > "$OUT/hadamard_fuse_probe.txt" 2>&1

SRC=${SRC:-/sgl-workspace/sglang}

echo "=== where the flag is read ==="
grep -rn "SGLANG_DSA_FUSE_HADAMARD_QUANT\|DSA_FUSE_HADAMARD" "$SRC/python" 2>/dev/null | head -10

echo
echo "=== the env declaration (default, type) ==="
grep -rn -A3 "SGLANG_DSA_FUSE_HADAMARD_QUANT" "$SRC/python/sglang/srt/environ.py" 2>/dev/null | head -12

echo
echo "=== every gate on the fused path ==="
for f in $(grep -rln "fuse_hadamard\|FUSE_HADAMARD" "$SRC/python" 2>/dev/null | head -5); do
    echo "--- $f"
    grep -n -B6 -A16 "fuse_hadamard\|FUSE_HADAMARD" "$f" | head -60
done

echo
echo "=== the unfused kernels the trace shows, and who calls them ==="
grep -rn "fast_hadamard_transform\|act_quant_kernel\|act_quant(" "$SRC/python/sglang/srt/layers/attention/dsa/" 2>/dev/null | head -12

echo
echo "=== is the flag on in this process, and what does the code think? ==="
python3 - <<'PY'
import os
print("  env SGLANG_DSA_FUSE_HADAMARD_QUANT =", os.environ.get("SGLANG_DSA_FUSE_HADAMARD_QUANT", "<unset>"))
try:
    from sglang.srt.environ import envs
    for name in dir(envs):
        if "HADAMARD" in name or "FUSE" in name.upper():
            try:
                print(f"  envs.{name} =", getattr(envs, name).get())
            except Exception as e:
                print(f"  envs.{name} -> {e}")
except Exception as e:
    print("  envs import failed:", e)
PY
