#!/usr/bin/env bash
# Dump the Quark-MXFP4 card + the parts of config.json that decide sharding.
exec > /home/jacchang/SGLang-benchmarks/tmp/logs/glm53_readme.txt 2>&1
root=/data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4
echo "=== README.md ==="
head -120 "$root/README.md"
echo
echo "=== config.json (no quantization_config) ==="
python3 -c 'import json;c=json.load(open("'"$root"'/config.json"));c.pop("quantization_config",None);print(json.dumps(c,indent=1)[:4000])'
echo
echo "=== quantization_config.exclude (unique suffixes) ==="
python3 - <<'PY'
import json, re, collections
c = json.load(open("/data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4/config.json"))
q = c.get("quantization_config", {})
ex = q.get("exclude", [])
print("count:", len(ex))
sfx = collections.Counter(re.sub(r"layers\.\d+\.", "layers.N.", e) for e in ex)
for k, v in sorted(sfx.items()):
    print(f"{v:5d}  {k}")
print("--- layer_quant_config keys ---")
lq = q.get("layer_quant_config", {})
print(list(lq)[:10], "..." if len(lq) > 10 else "", "total", len(lq))
PY
