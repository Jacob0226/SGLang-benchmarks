#!/usr/bin/env python3
"""Per-kernel breakdown for GLM-5.2 DSA decode sparse-MLA.

Both backends run in TWO passes; this splits the total decode-attn time into
the two component kernels so we can see how much is the online-softmax "reduce":

  TileLang : partial (main_kernel, split-K online softmax) + combine (main_kernel)
  Triton   : _sparse_mla_decode_split_kernel + _sparse_mla_decode_reduce_kernel

Each sub-kernel is captured into its own HIP graph and timed independently
(GPU-only). Reuses the exact dispatch logic of `tilelang_sparse_fwd` and
`triton_sparse_mla_decode_splitk` so the shapes/heuristics match production.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from glm5_decode_attn_microbench import (  # noqa: E402
    D_TAIL, D_V, DIM, FP8, NUM_HEADS_TOTAL, TOPK, build_inputs, cuda_time_graph,
)


def time_tilelang_breakdown(q, kv, indices, sm_scale, iters, warmup):
    from sglang.srt.layers.attention.dsa.tilelang_kernel import (
        _is_gfx95_supported, _pick_inner_iter, sparse_mla_fwd_decode_combine,
        sparse_mla_fwd_decode_partial_fp8,
    )

    num_heads = q.shape[1]
    tail_dim = DIM - D_V
    topk = indices.shape[-1]
    if _is_gfx95_supported:
        block_I, threads, block_per_cu, cu = 64, 256, 2, 256
    else:
        block_I, threads, block_per_cu, cu = 64, 256, 1, 304
    ni = topk // block_I
    inner_iter = _pick_inner_iter(q.shape[0], ni, cu, block_per_cu)

    kernel_partial = sparse_mla_fwd_decode_partial_fp8(
        num_heads, D_V, tail_dim, topk, sm_scale=sm_scale,
        block_I=block_I, inner_iter=inner_iter, threads=threads,
    )
    qb, kvb, idxb = q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0)
    partial_o, partial_lse = kernel_partial(qb, kvb, idxb)

    n_groups = ni // inner_iter
    kernel_combine = sparse_mla_fwd_decode_combine(
        num_heads, D_V, n_groups * block_I, head_per_block=4,
        block_I=block_I, threads=threads,
    )

    t_partial = cuda_time_graph(lambda: kernel_partial(qb, kvb, idxb), iters, warmup)
    t_combine = cuda_time_graph(
        lambda: kernel_combine(partial_o, partial_lse), iters, warmup)
    return t_partial, t_combine


def time_triton_breakdown(q, kv, indices, sm_scale, iters, warmup):
    from sglang.srt.layers.attention.dsa.triton_sparse_mla_decode import (
        LOG2E, _kv_splits_heuristic, _sparse_mla_decode_fused_kernel,
        _sparse_mla_decode_reduce_kernel, _sparse_mla_decode_split_kernel,
    )

    bs, H, q_dim = q.shape
    kv_dim = kv.shape[-1]
    d_tail = q_dim - D_V
    topk = indices.shape[-1]
    idx_flat = indices.squeeze(1).contiguous()

    BLOCK_H = BLOCK_K = 16
    n_head_blocks = (H + BLOCK_H - 1) // BLOCK_H
    h_padded = n_head_blocks * BLOCK_H
    kv_splits = _kv_splits_heuristic(bs, H, BLOCK_H)
    qk_scale = float(sm_scale) * LOG2E

    out = torch.empty(bs, H, D_V, device=q.device, dtype=torch.bfloat16)

    # bs large enough -> heuristic returns kv_splits==1 -> the real code takes the
    # FUSED single-pass kernel (no separate reduce). Time that; reduce = 0.
    if kv_splits == 1:
        def run_fused():
            _sparse_mla_decode_fused_kernel[(bs, n_head_blocks)](
                q, kv, idx_flat, out, qk_scale,
                topk=topk, H=H, Q_DIM=q_dim, KV_DIM=kv_dim, D_V=D_V, D_TAIL=d_tail,
                BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K, num_warps=4, num_stages=2,
            )
        return cuda_time_graph(run_fused, iters, warmup), 0.0, 1
    m_partial = torch.empty(bs, kv_splits, h_padded, dtype=torch.float32, device=q.device)
    l_partial = torch.empty_like(m_partial)
    acc_partial = torch.empty(bs, kv_splits, h_padded, D_V, dtype=torch.float32, device=q.device)

    def run_split():
        _sparse_mla_decode_split_kernel[(bs, n_head_blocks, kv_splits)](
            q, kv, idx_flat, m_partial, l_partial, acc_partial, qk_scale,
            topk=topk, H=H, Q_DIM=q_dim, KV_DIM=kv_dim, D_V=D_V, D_TAIL=d_tail,
            KV_SPLITS=kv_splits, BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

    D_CHUNK = 64
    grid_reduce = (bs, H, (D_V + D_CHUNK - 1) // D_CHUNK)

    def run_reduce():
        _sparse_mla_decode_reduce_kernel[grid_reduce](
            m_partial, l_partial, acc_partial, out,
            H=H, D_V=D_V, KV_SPLITS=kv_splits, D_CHUNK=D_CHUNK, BLOCK_K=BLOCK_K,
            num_warps=4,
        )

    t_split = cuda_time_graph(run_split, iters, warmup)
    t_reduce = cuda_time_graph(run_reduce, iters, warmup)
    return t_split, t_reduce, kv_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--seqlens", type=int, nargs="+", default=[1024, 8192])
    ap.add_argument("--concs", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    dev = "cuda"
    H = NUM_HEADS_TOTAL // args.tp
    sm_scale = DIM ** -0.5

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GLM-5.2-MXFP4 decode sparse-MLA per-kernel breakdown | TP={args.tp} -> H={H}")
    print("=" * 116)
    print(f"{'in':>6} {'conc':>5} | "
          f"{'TL_partial':>10} {'TL_combine':>10} {'TL_tot':>8} {'comb%':>6} | "
          f"{'kvsplit':>7} {'Tri_split':>10} {'Tri_reduce':>10} {'Tri_tot':>8} {'red%':>6}")
    print("-" * 116)

    for seq_len in args.seqlens:
        in_tag = f"i{seq_len // 1024}k" if seq_len % 1024 == 0 else f"i{seq_len}"
        for bs in args.concs:
            q, kv, indices = build_inputs(bs, seq_len, H, TOPK, dev)
            tl_p, tl_c = time_tilelang_breakdown(q, kv, indices, sm_scale, args.iters, args.warmup)
            if (kv.shape[0] * DIM) >= 2**31:  # PR Triton int32 page*KV_DIM overflow
                tl_tot = tl_p + tl_c
                print(f"{in_tag:>6} {bs:>5} | "
                      f"{tl_p:>10.2f} {tl_c:>10.2f} {tl_tot:>8.2f} {100*tl_c/tl_tot:>5.1f}% | "
                      f"{'—':>7} {'int32-OVF':>10} {'—':>10} {'—':>8} {'—':>6}")
                continue
            tr_s, tr_r, ks = time_triton_breakdown(q, kv, indices, sm_scale, args.iters, args.warmup)
            tl_tot = tl_p + tl_c
            tr_tot = tr_s + tr_r
            print(f"{in_tag:>6} {bs:>5} | "
                  f"{tl_p:>10.2f} {tl_c:>10.2f} {tl_tot:>8.2f} {100*tl_c/tl_tot:>5.1f}% | "
                  f"{ks:>7} {tr_s:>10.2f} {tr_r:>10.2f} {tr_tot:>8.2f} {100*tr_r/tr_tot:>5.1f}%")


if __name__ == "__main__":
    main()
