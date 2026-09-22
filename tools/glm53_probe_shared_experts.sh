#!/usr/bin/env bash
# Is shared-experts fusion safe on this checkpoint? It appends the shared expert
# as one more slot in the routed-expert tensors, so the two must carry the same
# quantization scheme and the same per-expert shapes. A shared expert left at a
# different precision is exactly what the modelopt_fp4 guard in
# shared_experts_fusion_disable_reason() refuses.
# Output: ~/glm53_shared_experts.txt
SGLANG_SRC="${SGLANG_SRC:-/home/jacchang/PR/glm53-day0-stack}"
exec > /home/jacchang/glm53_shared_experts.txt 2>&1

export PYTHONPATH="$SGLANG_SRC/python:/sgl-workspace/aiter"
export PYTHONDONTWRITEBYTECODE=1

python3 - <<'PY'
import json, re
from types import SimpleNamespace
from safetensors import safe_open

from sglang.srt.layers.quantization.quark.quark import QuarkConfig
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration

ROOT = "/data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4"
raw = json.load(open(f"{ROOT}/config.json"))
qc = dict(raw["quantization_config"])
text = raw.get("text_config", raw)
hf = SimpleNamespace(
    model_type=raw.get("model_type", "glm5_next"),
    num_hidden_layers=text.get("num_hidden_layers"),
    num_nextn_predict_layers=text.get("num_nextn_predict_layers", 0),
)
qc["packed_modules_mapping"] = Glm5NextForConditionalGeneration.packed_modules_mapping
cfg = QuarkConfig.from_config(qc)


def spec(name):
    c = cfg._find_matched_config(name, None) if hasattr(cfg, "_find_matched_config") else None
    if c is None:
        c = getattr(cfg, "quant_config", {})
    w = (c or {}).get("weight") or {}
    return f"{w.get('dtype')}/{w.get('qscheme')} block={w.get('block_size')}"


print("=== quant scheme: routed vs shared, a few layers ===")
for lid in (3, 20, 44, 45):
    for mod in ("experts.0", "shared_experts"):
        n = f"model.layers.{lid}.mlp.{mod}.up_proj"
        print(f"{n:55s} {spec(n)}")
    print()

print("=== checkpoint tensors for one sparse layer ===")
idx = json.load(open(f"{ROOT}/model.safetensors.index.json"))["weight_map"]
want = [n for n in idx if re.search(r"layers\.20\.mlp\.(shared_experts|experts\.0)\.", n)]
for n in sorted(want):
    with safe_open(f"{ROOT}/{idx[n]}", framework="pt") as f:
        sl = f.get_slice(n)
        print(f"{n:70s} {tuple(sl.get_shape())}  {sl.get_dtype()}")

print()
print("=== is the shared expert in quark 'exclude'? ===")
ex = qc.get("exclude", [])
hits = [e for e in ex if "shared_expert" in e]
print(f"exclude entries mentioning shared_expert: {len(hits)}")
for h in hits[:5]:
    print("  ", h)
PY
