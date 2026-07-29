#!/usr/bin/env python3
"""GLM-5.2 DSA sparse-MLA *decode* attention unit-kernel microbenchmark.

Isolates ONLY the decode-stage sparse-MLA attention kernel (the piece that
`dsa_backend._forward_decode` dispatches to) and benchmarks two backends on the
exact same inputs:

  * TileLang  -> tilelang_sparse_fwd  (production "main_kernel", --nsa-decode-backend tilelang)
  * Triton    -> triton_sparse_mla_decode (PR #30575, split-K + fused fast path)

Instead of running the whole server (`GLM.sh --prof`) we translate the GLM.sh
sweep (model=amd/GLM-5.2-MXFP4, TP=4, in_out=1024:1024 / 8192:1024,
concurrency 4..64) into the kernel's tensor dimensions and drive the kernel
directly. This is *seconds* per config instead of minutes, and is the right
harness for iterating on the kernel during development.

Kernel-shape mapping (from GLM-5.2-MXFP4 config.json + GLM.sh):
    num_attention_heads = 64, TP=4          -> H   = 16   (heads per rank)
    kv_lora_rank                            -> D_V = 512  (nope)
    qk_rope_head_dim                        -> D_TAIL = 64 (rope)
    DIM = D_V + D_TAIL                       -> 576
    index_topk                              -> topk = 2048  (TileLang asserts ==2048)
    concurrency (max running decode reqs)   -> bs (one decode token per request)
    input length "iNk"                      -> KV context length (seq_len) per request
                                               valid topk = min(seq_len, topk)

Both kernels take identical inputs:
    q:       [bs, H, DIM]         fp8_e4m3
    kv:      [num_pages, 1, DIM]  fp8_e4m3   (raw MLA layout: 512 nope + 64 rope)
    indices: [bs, 1, topk]        int32      (-1 = padding slot, skipped)
    -> out:  [1, bs, H, D_V]      bf16

The Triton decode kernel only exists on PR #30575, so the sglang package must be
the PR checkout. Reproduce with the wrapper (handles worktree + PYTHONPATH):
    ./run_decode_micro.sh                       # microbench (this file)
    ./run_decode_micro.sh --breakdown           # per-kernel breakdown
Or manually:
    git -C <sglang> fetch upstream pull/30575/head
    git -C <sglang> worktree add ../sglang-pr30575 FETCH_HEAD
    HIP_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=../sglang-pr30575/python \
        python3 glm5_decode_attn_microbench.py --iters 300 --csv out.csv
"""
import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# GLM-5.2-MXFP4 fixed kernel dims
# --------------------------------------------------------------------------- #
NUM_HEADS_TOTAL = 64
D_V = 512  # kv_lora_rank (nope)
D_TAIL = 64  # qk_rope_head_dim (rope)
DIM = D_V + D_TAIL  # 576
TOPK = 2048  # index_topk (TileLang requires exactly 2048)
FP8 = torch.float8_e4m3fn


def build_inputs(bs, seq_len, H, topk, device, seed=0):
    """Construct one decode step's kernel inputs.

    KV pool layout: request t owns the contiguous slot range
    [t*seq_len, (t+1)*seq_len). Each request selects `valid = min(seq_len, topk)`
    slots from its own range (scattered, unique, sorted) and pads the rest of the
    2048-wide topk row with -1 (both kernels skip idx < 0).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    num_pages = bs * seq_len
    valid = min(seq_len, topk)

    # fp8 q / kv (values ~N(0,1) then cast to fp8, mirroring absorbed MLA latents)
    q = (torch.randn(bs, H, DIM, generator=g) * 0.5).to(device).to(FP8)
    kv = (torch.randn(num_pages, 1, DIM, generator=g) * 0.5).to(device).to(FP8)

    idx = torch.full((bs, topk), -1, dtype=torch.int32)
    for t in range(bs):
        base = t * seq_len
        if valid >= seq_len:
            sel = torch.arange(seq_len, dtype=torch.int64)
        else:
            sel = torch.randperm(seq_len, generator=g)[:valid].sort().values
        idx[t, :valid] = (base + sel).to(torch.int32)
    indices = idx.unsqueeze(1).to(device)  # [bs, 1, topk]
    return q, kv, indices


def cuda_time_eager(fn, iters, warmup=10):
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
    return start.elapsed_time(end) / iters * 1000.0  # us / call


def cuda_time_graph(fn, iters, warmup=10):
    """Capture fn into a HIP/CUDA graph and time replays. This is how decode
    actually runs in serving (captured graph), so it isolates GPU kernel time
    from Python/host launch overhead."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # us / call


def cos_sim(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--seqlens", type=int, nargs="+", default=[1024, 8192],
                    help="KV context lengths, i.e. i1k / i8k")
    ap.add_argument("--concs", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--mode", choices=["graph", "eager"], default="graph",
                    help="graph = HIP-graph replay (GPU-only, matches serving); "
                         "eager = per-call launch (includes host overhead)")
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()
    cuda_time = cuda_time_graph if args.mode == "graph" else cuda_time_eager

    assert torch.cuda.is_available(), "needs a GPU"
    dev = "cuda"
    H = NUM_HEADS_TOTAL // args.tp
    sm_scale = DIM ** -0.5

    from sglang.srt.layers.attention.dsa.tilelang_kernel import tilelang_sparse_fwd
    try:
        from sglang.srt.layers.attention.dsa.triton_sparse_mla_decode import (
            triton_sparse_mla_decode,
        )
    except ImportError as e:
        raise SystemExit(
            "Could not import the Triton decode kernel from sglang. It only exists "
            "on PR #30575. Check out that branch and put it on PYTHONPATH, e.g.:\n"
            "  git -C <sglang> fetch upstream pull/30575/head\n"
            "  git -C <sglang> worktree add ../sglang-pr30575 FETCH_HEAD\n"
            "  PYTHONPATH=../sglang-pr30575/python python3 " + os.path.basename(__file__)
            + f"\n(original error: {e})"
        )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GLM-5.2-MXFP4 decode sparse-MLA | TP={args.tp} -> H={H}, "
          f"D_V={D_V}, D_TAIL={D_TAIL}, DIM={DIM}, topk={TOPK}, KV fp8_e4m3 | timing={args.mode}")
    print("=" * 86)
    hdr = (f"{'in':>6} {'conc':>5} {'valid_topk':>10} | "
           f"{'TileLang(us)':>13} {'Tri-splitK(us)':>15} | "
           f"{'splitK vs TL':>12} | {'cos(sK)':>8}")
    print(hdr)
    print("-" * 86)

    rows = []
    for seq_len in args.seqlens:
        in_tag = f"i{seq_len // 1024}k" if seq_len % 1024 == 0 else f"i{seq_len}"
        for bs in args.concs:
            q, kv, indices = build_inputs(bs, seq_len, H, TOPK, dev)
            valid = min(seq_len, TOPK)

            # The PR #30575 Triton kernels compute `page * KV_DIM` in int32; when
            # num_pages*DIM exceeds 2^31 the address overflows -> GPU memory-access
            # fault (uncatchable, aborts the process). TileLang indexes safely.
            # Predict + skip the Triton kernels for those configs instead of crashing.
            triton_safe = (kv.shape[0] * DIM) < 2**31
            if not triton_safe:
                t_tl = cuda_time(lambda: tilelang_sparse_fwd(
                    q=q, kv=kv, indices=indices, sm_scale=sm_scale, d_v=D_V),
                    args.iters, args.warmup)
                print(f"{in_tag:>6} {bs:>5} {valid:>10} | "
                      f"{t_tl:>13.2f} {'int32-OVF':>15} | "
                      f"{'—':>12} | {'—':>8}")
                rows.append(dict(input=in_tag, seq_len=seq_len, conc=bs, valid_topk=valid,
                                 tilelang_us=t_tl, triton_splitk_us=float("nan"),
                                 speedup_splitk=float("nan"), cos_splitk=float("nan")))
                continue

            def run_tl():
                return tilelang_sparse_fwd(q=q, kv=kv, indices=indices,
                                           sm_scale=sm_scale, d_v=D_V)

            def run_sk():
                return triton_sparse_mla_decode(q, kv, indices, sm_scale, d_v=D_V)

            try:
                o_tl = run_tl()
                o_sk = run_sk()
            except Exception as e:
                print(f"{in_tag:>6} {bs:>5} {valid:>10} | ERROR: {e}")
                continue

            cs_sk = cos_sim(o_tl, o_sk)

            t_tl = cuda_time(run_tl, args.iters, args.warmup)
            t_sk = cuda_time(run_sk, args.iters, args.warmup)

            sp_sk = t_tl / t_sk
            print(f"{in_tag:>6} {bs:>5} {valid:>10} | "
                  f"{t_tl:>13.2f} {t_sk:>15.2f} | "
                  f"{sp_sk:>11.2f}x | "
                  f"{cs_sk:>8.5f}")
            rows.append(dict(input=in_tag, seq_len=seq_len, conc=bs, valid_topk=valid,
                             tilelang_us=t_tl, triton_splitk_us=t_sk,
                             speedup_splitk=sp_sk, cos_splitk=cs_sk))

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
