#!/usr/bin/env python3
"""Does the decode kernel waste work on padded (idx=-1) topk slots?

GLM-5.2 uses index_topk=2048. When the live context is shorter (e.g. i1k -> only
~1024 real tokens), the topk row is padded with -1. This isolates whether the
kernel skips those or still processes all 2048 slots.

Method: fix seq_len=2048 so the KV footprint / cache behavior is CONSTANT, and
sweep only the number of VALID topk entries (rest = -1). Flat time => the kernel
processes all 2048 slots regardless (compute wasted on padding).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm5_decode_attn_microbench import (  # noqa: E402
    DIM, D_V, FP8, NUM_HEADS_TOTAL, TOPK, cuda_time_graph,
)


def main():
    from sglang.srt.layers.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
    from sglang.srt.layers.attention.dsa.triton_sparse_mla_decode import (
        triton_sparse_mla_decode,
    )

    H = NUM_HEADS_TOTAL // 4
    sm = DIM ** -0.5
    dev = "cuda"
    bs = int(os.environ.get("CONC", "8"))
    seq = 2048  # fixed footprint; all pages real
    g = torch.Generator("cpu").manual_seed(0)

    q = (torch.randn(bs, H, DIM, generator=g) * 0.5).to(dev).to(FP8)
    kv = (torch.randn(bs * seq, 1, DIM, generator=g) * 0.5).to(dev).to(FP8)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"conc={bs}, seq_len={seq} (KV footprint FIXED, all pages real), "
          f"sweep #valid of topk={TOPK}:")
    print(f"{'valid':>6} {'TileLang(us)':>13} {'Triton(us)':>11}")
    for valid in (128, 256, 512, 1024, 1536, 2048):
        idx = torch.full((bs, TOPK), -1, dtype=torch.int32)
        for t in range(bs):
            sel = torch.randperm(seq, generator=g)[:valid].sort().values
            idx[t, :valid] = (t * seq + sel).to(torch.int32)
        indices = idx.unsqueeze(1).to(dev)
        t_tl = cuda_time_graph(
            lambda: tilelang_sparse_fwd(q=q, kv=kv, indices=indices, sm_scale=sm, d_v=D_V),
            200, 40)
        t_tr = cuda_time_graph(
            lambda: triton_sparse_mla_decode(q, kv, indices, sm, d_v=D_V), 200, 40)
        print(f"{valid:>6} {t_tl:>13.2f} {t_tr:>11.2f}", flush=True)


if __name__ == "__main__":
    main()
