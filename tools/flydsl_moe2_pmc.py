#!/usr/bin/env python3
"""Two-phase harness so rocprofv3 can profile flydsl_moe_stage2 without aborting.

Phase 1 (gen, run normally): build inputs via aiter's test data-gen, torch.save them.
Phase 2 (prof, run under rocprofv3): torch.load inputs, run flydsl_moe_stage2 in a
loop. No torch.randn / heavy init under the profiler -> avoids the signal-6 abort.

  python3 flydsl_moe2_pmc.py gen    /tmp/moe2_inputs.pt
  rocprofv3 --pmc ... -- python3 flydsl_moe2_pmc.py prof /tmp/moe2_inputs.pt
"""
import sys
import torch

sys.path.insert(0, "/sgl-workspace/aiter")

SHAPE = dict(token=16304, model_dim=6144, inter_dim=512, E=256, topk=8, block_m=64)
TILE = dict(tile_m=64, tile_n=256, tile_k=512, mode="reduce")
KEYS = ["a2_qt", "w2_qt_shuf", "sorted_ids", "sorted_expert_ids",
        "num_valid_ids", "w2_scale_shuf", "a2_scale_sort", "sorted_weights_s2"]


def gen(path):
    from aiter.ops.flydsl.test_flydsl_moe_a4w4 import _generate_a4w4_data
    data = _generate_a4w4_data(**SHAPE)
    torch.save({k: data[k] for k in KEYS}, path)
    print(f"saved inputs -> {path}")


def prof(path, iters=30, warmup=8):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2
    d = torch.load(path, map_location="cuda")

    def call():
        return flydsl_moe_stage2(
            inter_states=d["a2_qt"], w2=d["w2_qt_shuf"],
            sorted_token_ids=d["sorted_ids"], sorted_expert_ids=d["sorted_expert_ids"],
            num_valid_ids=d["num_valid_ids"], topk=SHAPE["topk"],
            tile_m=TILE["tile_m"], tile_n=TILE["tile_n"], tile_k=TILE["tile_k"],
            a_dtype="fp4", b_dtype="fp4", out_dtype="bf16", mode=TILE["mode"],
            w2_scale=d["w2_scale_shuf"], a2_scale=d["a2_scale_sort"],
            sorted_weights=d["sorted_weights_s2"],
        )

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    for _ in range(iters):
        call()
    torch.cuda.synchronize()
    print(f"ran {iters} iters of flydsl_moe_stage2 {TILE}")


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    (gen if mode == "gen" else prof)(path)
