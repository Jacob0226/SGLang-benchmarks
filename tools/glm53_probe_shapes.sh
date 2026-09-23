#!/usr/bin/env bash
# Find which Quark-MXFP4 tensors cannot be row-parallel sharded at TP=4.
# A row-parallel MXFP4 weight is [out, in/2] (two fp4 per byte) with a scale
# [out, in/32], so the per-rank input shard needs in % (TP*32) == 0.
# Output: ~/glm53_shapes.txt on the shared home.
exec > /home/jacchang/SGLang-benchmarks/tmp/logs/glm53_shapes.txt 2>&1

python3 - <<'PY'
import json, collections, re

root = "/data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4"
idx = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]

# shapes are not in the index; read them from the headers of the shards we need
from safetensors import safe_open

by_shard = collections.defaultdict(list)
for name, shard in idx.items():
    by_shard[shard].append(name)

# Only inspect layers 0-2 plus anything outside the layer stack: the pattern
# repeats per layer and reading all 62 shards is pointless.
def interesting(n):
    m = re.search(r"layers\.(\d+)\.", n)
    return m is None or int(m.group(1)) <= 2

shapes = {}
for shard, names in sorted(by_shard.items()):
    want = [n for n in names if interesting(n)]
    if not want:
        continue
    with safe_open(f"{root}/{shard}", framework="pt") as f:
        for n in want:
            shapes[n] = tuple(f.get_slice(n).get_shape())

TP = 4
print(f"=== tensors for layers 0-2 + non-layer, TP={TP} ===")
for n in sorted(shapes):
    print(f"{n}  {shapes[n]}")

print()
print("=== row-parallel candidates whose input dim is not divisible by TP*32 ===")
ROW_PARALLEL = ("o_proj", "down_proj", "b_proj", "f_b_proj", "g_b_proj")
for n in sorted(shapes):
    if not any(k in n for k in ROW_PARALLEL):
        continue
    s = shapes[n]
    if len(s) < 2:
        continue
    in_dim = s[-1]
    flags = []
    if in_dim % TP:
        flags.append(f"in%{TP}={in_dim % TP}")
    if in_dim % (TP * 32):
        flags.append(f"in%{TP*32}={in_dim % (TP*32)}")
    if flags:
        print(f"{n}  {s}  -> {' '.join(flags)}")
PY
