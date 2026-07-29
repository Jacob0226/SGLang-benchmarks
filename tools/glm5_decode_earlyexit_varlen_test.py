#!/usr/bin/env python3
"""Correctness of early-exit with *varying per-token* topk_length (production case).

The uniform-length microbench can't catch a per-token indexing bug. Here each row
of the batch gets a different valid length (front-packed indices, -1 tail), and we
require the early-exit output to match the full-compute baseline exactly.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm5_decode_attn_microbench import D_V, DIM, FP8, NUM_HEADS_TOTAL, TOPK, cos_sim


def main():
    from sglang.srt.layers.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

    H = NUM_HEADS_TOTAL // 4
    sm = DIM ** -0.5
    dev = "cuda"
    g = torch.Generator("cpu").manual_seed(1)

    # A batch mixing short and long contexts (the exact production heterogeneity).
    lengths = [37, 128, 512, 1, 2048, 1999, 1024, 65, 1536, 800, 2000, 333]
    bs = len(lengths)
    seq = 2048  # max context; per-token valid varies below
    q = (torch.randn(bs, H, DIM, generator=g) * 0.5).to(dev).to(FP8)
    kv = (torch.randn(bs * seq, 1, DIM, generator=g) * 0.5).to(dev).to(FP8)

    idx = torch.full((bs, TOPK), -1, dtype=torch.int32)
    for t, L in enumerate(lengths):
        sel = torch.randperm(seq, generator=g)[:L].sort().values
        idx[t, :L] = (t * seq + sel).to(torch.int32)
    indices = idx.unsqueeze(1).to(dev)
    tk_len = torch.tensor(lengths, dtype=torch.int32, device=dev)

    o_full = tilelang_sparse_fwd(q=q, kv=kv, indices=indices, sm_scale=sm, d_v=D_V,
                                 topk_length=None)
    o_ee = tilelang_sparse_fwd(q=q, kv=kv, indices=indices, sm_scale=sm, d_v=D_V,
                               topk_length=tk_len)

    overall = cos_sim(o_full, o_ee)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"per-token lengths: {lengths}")
    print(f"overall cos(full, earlyexit) = {overall:.7f}")
    # per-row max abs diff to prove every request (incl. short ones) matches
    a = o_full.reshape(bs, H, D_V).float()
    b = o_ee.reshape(bs, H, D_V).float()
    worst = 0.0
    for t, L in enumerate(lengths):
        d = (a[t] - b[t]).abs().max().item()
        worst = max(worst, d)
        print(f"  row {t:2d} len={L:4d}  max|Δ|={d:.3e}")
    print(f"worst per-row max|Δ| = {worst:.3e}")
    print("RESULT:", "PASS" if overall > 0.99999 and worst < 1e-2 else "FAIL")


if __name__ == "__main__":
    main()
