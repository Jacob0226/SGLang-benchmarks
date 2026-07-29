#!/usr/bin/env python3
"""Timing microbench for FlyDSL MoE stage2 (down GEMM) at GLM-5.2 shape.

Reuses aiter's flydsl a4w4 test data generation, then times flydsl_moe_stage2
across (mode, tile_m, tile_n, tile_k) so we can push HBM BW utilization without
standing up a server. Compares against the mfma_moe2 baseline (~1391us big step)
and B200 bmm_Bfloat16 (~435us).

Run inside the MI355X aiter container:
  python3 flydsl_moe2_bench.py --tokens 16304 --inter-dim 512
"""
import argparse
import sys
import torch

sys.path.insert(0, "/sgl-workspace/aiter")
from aiter.ops.flydsl.test_flydsl_moe_a4w4 import _generate_a4w4_data  # noqa: E402
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2  # noqa: E402


def time_call(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16304)
    ap.add_argument("--model-dim", type=int, default=6144)
    ap.add_argument("--inter-dim", type=int, default=512)
    ap.add_argument("-E", "--experts", type=int, default=256)
    ap.add_argument("-k", "--topk", type=int, default=8)
    ap.add_argument("--tile-m", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--tile-n", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--tile-k", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--modes", nargs="+", default=["atomic", "reduce"])
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    T, D, I, E, topk = args.tokens, args.model_dim, args.inter_dim, args.experts, args.topk

    # bytes (reduce-mode useful traffic): w2 fp4 + reduced output bf16 + input act fp4
    w2_bytes = E * D * I * 0.5
    out_bytes = T * D * 2
    act_bytes = T * topk * I * 0.5
    mem_reduce = w2_bytes + out_bytes + act_bytes
    flops = 2 * (T * topk) * I * D

    print(f"shape: token={T} model_dim={D} inter_dim={I} E={E} topk={topk}")
    print(f"FLOPs={flops/1e9:.1f} GFLOP  mem(reduce)~{mem_reduce/1e6:.0f}MB")
    print(f"{'mode':7} {'tm':>4} {'tn':>4} {'tk':>4} {'us':>9} {'TFLOP/s':>9} {'BW(TB/s)':>9}")

    # generate data once per block_m (data layout depends on block_m)
    best = None
    for tm in args.tile_m:
        data = _generate_a4w4_data(token=T, model_dim=D, inter_dim=I, E=E, topk=topk, block_m=tm)
        out_dtype_str = "bf16"
        for tn in args.tile_n:
            for tk in args.tile_k:
                for mode in args.modes:
                    def fn():
                        return flydsl_moe_stage2(
                            inter_states=data["a2_qt"],
                            w2=data["w2_qt_shuf"],
                            sorted_token_ids=data["sorted_ids"],
                            sorted_expert_ids=data["sorted_expert_ids"],
                            num_valid_ids=data["num_valid_ids"],
                            topk=topk,
                            tile_m=tm, tile_n=tn, tile_k=tk,
                            a_dtype="fp4", b_dtype="fp4", out_dtype=out_dtype_str,
                            mode=mode,
                            w2_scale=data["w2_scale_shuf"],
                            a2_scale=data["a2_scale_sort"],
                            sorted_weights=data["sorted_weights_s2"],
                        )
                    try:
                        us = time_call(fn, iters=args.iters)
                    except Exception as e:
                        print(f"{mode:7} {tm:>4} {tn:>4} {tk:>4}  FAIL: {str(e)[:40]}")
                        continue
                    tfs = flops / (us * 1e-6) / 1e12
                    bw = mem_reduce / (us * 1e-6) / 1e12
                    print(f"{mode:7} {tm:>4} {tn:>4} {tk:>4} {us:9.1f} {tfs:9.1f} {bw:9.2f}")
                    if best is None or us < best[0]:
                        best = (us, mode, tm, tn, tk)
    if best:
        print(f"\nBEST: {best[0]:.1f}us  mode={best[1]} tile=({best[2]},{best[3]},{best[4]})")
        print(f"vs mfma_moe2 baseline ~1391us (big step)  |  B200 bmm ~435us")


if __name__ == "__main__":
    main()
