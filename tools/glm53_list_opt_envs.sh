#!/usr/bin/env bash
# Every SGLANG_* knob this build exposes, with its default, plus which ones the
# GLM-5.3-Flash decode path can actually reach.
#
# Motivation: GLM.sh exports SGLANG_DSA_FUSE_HADAMARD_QUANT, and this build has
# no such variable -- the knob from PR #30715 was renamed or removed, so the
# export has been inert. Rather than guess again, list what exists and grep the
# DSA / MoE / mHC paths for the ones they read.
#
# Output: SGLang-benchmarks/tmp/logs/opt_envs.txt
OUT=/home/jacchang/SGLang-benchmarks/tmp/logs
mkdir -p "$OUT"
exec > "$OUT/opt_envs.txt" 2>&1

SRC=${SRC:-/sgl-workspace/sglang}

echo "=== hadamard / act_quant fusion: is there a knob at all? ==="
grep -rn "hadamard" "$SRC/python/sglang" --include=*.py -il | head
echo "--- any env mentioning hadamard:"
grep -rn "HADAMARD" "$SRC/python" | head
echo "--- the indexer's hadamard call site and what guards it:"
grep -rn -B8 -A4 "hadamard_transform" "$SRC/python/sglang/srt/layers/attention/dsa/dsa_indexer.py" | head -40

echo
echo "=== full env inventory with defaults ==="
python3 - <<'PY'
from sglang.srt.environ import envs
rows = []
for name in sorted(dir(envs)):
    if not name.startswith("SGLANG"):
        continue
    try:
        rows.append((name, repr(envs.__getattribute__(name).get())))
    except Exception as e:
        rows.append((name, f"<{type(e).__name__}>"))
print(f"  {len(rows)} SGLANG_* knobs")
for n, v in rows:
    print(f"  {n:<62} {v}")
PY

echo
echo "=== which of them the DSA / MoE / mHC decode paths read ==="
for d in srt/layers/attention/dsa srt/layers/moe kernels/ops/layernorm srt/models/glm5_next.py; do
    echo "--- $d"
    grep -rhoE "envs\.SGLANG_[A-Z0-9_]+" "$SRC/python/sglang/$d" 2>/dev/null | sort -u | sed 's/^/    /'
done
