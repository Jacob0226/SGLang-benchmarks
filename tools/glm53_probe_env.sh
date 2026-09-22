#!/usr/bin/env bash
# One-off probe: which sglang the container runs, and what the Quark-MXFP4
# checkpoint declares. Output goes to ~/glm53_env.txt on the shared home.
exec > /home/jacchang/glm53_env.txt 2>&1

echo "=== /sgl-workspace/sglang ==="
git -C /sgl-workspace/sglang log --oneline -3 2>&1
git -C /sgl-workspace/sglang status -sb 2>&1 | head -3

echo "=== python sglang package ==="
python3 -c 'import sglang, os; print(sglang.__version__, os.path.dirname(sglang.__file__))'

echo "=== checkpoint config (Quark-MXFP4) ==="
python3 - <<'PY'
import json
p = "/data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4/config.json"
c = json.load(open(p))
q = c.pop("quantization_config", {})
for k in ("architectures", "num_hidden_layers", "hidden_size", "intermediate_size",
          "moe_intermediate_size", "n_routed_experts", "num_experts",
          "n_shared_experts", "num_attention_heads", "num_key_value_heads",
          "transformers_version", "torch_dtype"):
    if k in c:
        print(f"{k}: {c[k]}")
print("--- quantization_config ---")
print(json.dumps(q)[:1500])
PY

echo "=== files ==="
ls /data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4 | head -20
