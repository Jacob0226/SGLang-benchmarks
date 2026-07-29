#!/usr/bin/env python3
"""Validate + benchmark the TileLang decode early-exit (topk_length) change.

Baseline  = tilelang_sparse_fwd(..., topk_length=None)   # full 2048 groups
Early-exit = tilelang_sparse_fwd(..., topk_length=valid)  # skip padded groups

Correctness: early-exit must match baseline (padded groups contribute 0 either
way). Perf: i1k (context<2048) should get faster; i8k (all 2048 valid) unchanged.

Run against the branch under test:
    HIP_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=~/PR/sglang/python \
        python3 glm5_decode_earlyexit_test.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm5_decode_attn_microbench import (  # noqa: E402
    D_V, DIM, NUM_HEADS_TOTAL, TOPK, build_inputs, cos_sim, cuda_time_graph,
)


def main():
    from sglang.srt.layers.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

    H = NUM_HEADS_TOTAL // 4
    sm = DIM ** -0.5
    dev = "cuda"
    seqlens = [1024, 8192]
    concs = [4, 8, 16, 32, 64]

    print(f"GPU: {torch.cuda.get_device_name(0)}  |  TileLang decode early-exit test")
    print("=" * 92)
    print(f"{'in':>6} {'conc':>5} {'valid':>6} | {'base(us)':>10} {'earlyexit(us)':>13} "
          f"{'speedup':>8} | {'cos(base,ee)':>13}")
    print("-" * 92)

    for seq_len in seqlens:
        in_tag = f"i{seq_len // 1024}k"
        for bs in concs:
            q, kv, indices = build_inputs(bs, seq_len, H, TOPK, dev)
            valid = min(seq_len, TOPK)
            tl_len = torch.full((bs,), valid, dtype=torch.int32, device=dev)

            def run_base():
                return tilelang_sparse_fwd(q=q, kv=kv, indices=indices,
                                           sm_scale=sm, d_v=D_V, topk_length=None)

            def run_ee():
                return tilelang_sparse_fwd(q=q, kv=kv, indices=indices,
                                           sm_scale=sm, d_v=D_V, topk_length=tl_len)

            o_base = run_base()
            o_ee = run_ee()
            cs = cos_sim(o_base, o_ee)
            t_base = cuda_time_graph(run_base, 200, 40)
            t_ee = cuda_time_graph(run_ee, 200, 40)
            print(f"{in_tag:>6} {bs:>5} {valid:>6} | {t_base:>10.2f} {t_ee:>13.2f} "
                  f"{t_base / t_ee:>7.2f}x | {cs:>13.6f}", flush=True)


if __name__ == "__main__":
    main()
