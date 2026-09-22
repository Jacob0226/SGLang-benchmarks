#!/usr/bin/env bash
# Pre-flight for the GLM-5.3-Flash Day-0 PR stack, before spending 6 min on a
# model load. Runs inside the container against the tree in SGLANG_SRC.
# Output: ~/glm53_stack_check.txt on the shared home.
SGLANG_SRC="${SGLANG_SRC:-/home/jacchang/PR/glm53-day0-stack}"
exec > /home/jacchang/glm53_stack_check.txt 2>&1

export PYTHONPATH="$SGLANG_SRC/python:/sgl-workspace/aiter"
export PYTHONDONTWRITEBYTECODE=1

echo "=== tree ==="
git -C "$SGLANG_SRC" rev-parse --short HEAD
git -C "$SGLANG_SRC" status --short | head

echo "=== which sglang is imported (must be under \$SGLANG_SRC) ==="
python3 -c 'import inspect, sglang; print(inspect.getfile(sglang))'

echo "=== #38545 took effect (must print True on gfx950) ==="
python3 -c 'from sglang.srt.models.glm5_next import _use_aiter_gfx95; print(_use_aiter_gfx95)'

echo "=== aiter revision ==="
git -C /sgl-workspace/aiter rev-parse HEAD

echo "=== #38546 + #39317: per-layer quant resolution ==="
python3 /home/jacchang/SGLang-benchmarks/tools/glm53_mxfp4_quant_resolution.py \
    /data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4/config.json
