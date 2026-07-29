#!/usr/bin/env python3
"""3-way GLM-5.2 DSA decode kernel comparison (pure kernel, no SGLang E2E).

Compares, on identical inputs, the three decode sparse-MLA kernels:
  1. TileLang            -- tilelang_sparse_fwd (baseline, full topk=2048)
  2. Triton (PR #30575)  -- triton_sparse_mla_decode (split-K, no early-stop)
  3. TileLang + early-stop -- tilelang_sparse_fwd(topk_length=valid) (AttnOpt)

Timing is HIP-graph replay (GPU-only, matches serving). Correctness is cosine
similarity vs the TileLang baseline. Requires the merged branch that has BOTH
the tilelang early-exit AND PR #30575's triton decode kernel, e.g.:

  HIP_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=~/PR/sglang/python \
      python3 glm5_decode_3way_compare.py --iters 300 --csv out.csv
"""
import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm5_decode_attn_microbench import (  # noqa: E402
    D_V, DIM, NUM_HEADS_TOTAL, TOPK, build_inputs, cos_sim, cuda_time_graph,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--seqlens", type=int, nargs="+", default=[1024, 8192])
    ap.add_argument("--concs", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    from sglang.srt.layers.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
    from sglang.srt.layers.attention.dsa.triton_sparse_mla_decode import (
        triton_sparse_mla_decode,
    )

    H = NUM_HEADS_TOTAL // args.tp
    sm = DIM ** -0.5
    dev = "cuda"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GLM-5.2-MXFP4 decode 3-way | TP={args.tp} -> H={H}, DIM={DIM}, "
          f"topk={TOPK}, KV fp8_e4m3 | graph timing")
    print("=" * 108)
    print(f"{'in':>6} {'conc':>5} {'valid':>6} | "
          f"{'TileLang':>9} {'Triton':>9} {'EarlyStop':>9} | "
          f"{'Tri/TL':>7} {'ES/TL':>7} | {'cos(Tri)':>9} {'cos(ES)':>9}")
    print("-" * 108)

    rows = []
    for seq_len in args.seqlens:
        in_tag = f"i{seq_len // 1024}k" if seq_len % 1024 == 0 else f"i{seq_len}"
        for bs in args.concs:
            q, kv, indices = build_inputs(bs, seq_len, H, TOPK, dev)
            valid = min(seq_len, TOPK)
            tk_len = torch.full((bs,), valid, dtype=torch.int32, device=dev)

            # PR #30575 triton kernel computes page*KV_DIM in int32 -> overflow
            # (GPU memory fault) when num_pages*DIM exceeds 2^31. Skip triton there.
            triton_safe = (kv.shape[0] * DIM) < 2**31

            def run_tl():
                return tilelang_sparse_fwd(q=q, kv=kv, indices=indices,
                                           sm_scale=sm, d_v=D_V, topk_length=None)

            def run_es():
                return tilelang_sparse_fwd(q=q, kv=kv, indices=indices,
                                           sm_scale=sm, d_v=D_V, topk_length=tk_len)

            def run_tri():
                return triton_sparse_mla_decode(q, kv, indices, sm, d_v=D_V)

            o_tl = run_tl()
            o_es = run_es()
            cos_es = cos_sim(o_tl, o_es)
            t_tl = cuda_time_graph(run_tl, args.iters, args.warmup)
            t_es = cuda_time_graph(run_es, args.iters, args.warmup)

            if triton_safe:
                o_tri = run_tri()
                cos_tri = cos_sim(o_tl, o_tri)
                t_tri = cuda_time_graph(run_tri, args.iters, args.warmup)
                tri_s = f"{t_tri:9.2f}"
                tri_ratio = f"{t_tri / t_tl:6.2f}x"
                cos_tri_s = f"{cos_tri:9.5f}"
            else:
                t_tri = float("nan")
                cos_tri = float("nan")
                tri_s = f"{'int32OVF':>9}"
                tri_ratio = f"{'—':>7}"
                cos_tri_s = f"{'—':>9}"

            print(f"{in_tag:>6} {bs:>5} {valid:>6} | "
                  f"{t_tl:9.2f} {tri_s} {t_es:9.2f} | "
                  f"{tri_ratio} {t_es / t_tl:6.2f}x | {cos_tri_s} {cos_es:9.5f}")
            rows.append(dict(input=in_tag, seq_len=seq_len, conc=bs, valid_topk=valid,
                             tilelang_us=t_tl, triton_us=t_tri, earlystop_us=t_es,
                             triton_vs_tl=t_tri / t_tl if triton_safe else float("nan"),
                             earlystop_vs_tl=t_es / t_tl,
                             cos_triton=cos_tri, cos_earlystop=cos_es))

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
