"""Validate aiter mla_decode_fwd(return_lse=True) for DSA decode under DCP.

Mirrors sglang DSA backend `_forward_aiter` (bf16 Q, fp8 KV, page_size=1,
persistent metadata from get_mla_metadata_v1) at GLM-5.2 shapes
(kv_lora_rank=512, rope=64, top-k=2048) and checks:

  1. single rank: out / lse vs an fp32 reference, and which log base lse uses
  2. DCP emulation: split one global top-k by owner (v % dcp), run each rank on
     its local pool with the gathered head count, LSE-merge, compare against
     attention over the full top-k
  3. rows that own zero KV on a rank (seq_len < dcp)
  4. fallbacks: non-persistent path, fp8 Q with q_scale

Usage (one GPU):
  HIP_VISIBLE_DEVICES=0 python3 dsa_dcp_aiter_lse_test.py [--json out.json]
"""

import argparse
import json
import math

import torch
from aiter import dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd

KV_LORA = 512
ROPE = 64
D = KV_LORA + ROPE
TOPK = 2048
MAX_SPLIT = 64
SM_SCALE = 1.0 / math.sqrt(256)
FP8 = dtypes.fp8
DEV = "cuda"


def build_csr(rows):
    lens = torch.tensor([len(r) for r in rows], dtype=torch.int32)
    indptr = torch.zeros(len(rows) + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(lens, 0)
    flat = [i for r in rows for i in r]
    indices = torch.tensor(flat if flat else [0], dtype=torch.int32)
    return indptr.to(DEV), indices.to(DEV)


def persistent_kwargs(qo_indptr, kv_indptr, bs, meta_heads, q_dtype):
    info = get_mla_metadata_info_v1(
        bs,
        1,
        meta_heads,
        q_dtype,
        FP8,
        is_sparse=True,
        fast_mode=False,
        num_kv_splits=MAX_SPLIT,
        intra_batch_mode=True,
    )
    work_meta, work_indptr, work_info, red_indptr, red_final, red_partial = [
        torch.empty(s, dtype=t, device=DEV) for s, t in info
    ]
    kv_last = torch.ones(bs, dtype=torch.int32, device=DEV)
    get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last,
        meta_heads,
        1,
        False,
        work_meta,
        work_info,
        work_indptr,
        red_indptr,
        red_final,
        red_partial,
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=False,
        topk=TOPK,
        max_split_per_batch=MAX_SPLIT,
        intra_batch_mode=True,
        dtype_q=q_dtype,
        dtype_kv=FP8,
    )
    return kv_last, dict(
        work_meta_data=work_meta,
        work_indptr=work_indptr,
        work_info_set=work_info,
        reduce_indptr=red_indptr,
        reduce_final_map=red_final,
        reduce_partial_map=red_partial,
        intra_batch_mode=True,
        num_kv_splits=MAX_SPLIT,
    )


def run_kernel(q, kv_fp8, kv_indptr, kv_indices, mode, meta_heads=None):
    """mode: persistent | nonpersistent | fp8q (persistent with fp8 Q)."""
    bs, nhead, _ = q.shape
    qo_indptr = torch.arange(bs + 1, dtype=torch.int32, device=DEV)
    kv_scale = torch.ones((), dtype=torch.float32, device=DEV)
    q_scale = None
    q_in = q
    if mode == "fp8q":
        q_scale = (q.abs().amax().float() / 448.0).clamp(min=1e-8).reshape(())
        q_in = (q.float() / q_scale).to(FP8)
    o = torch.empty(bs, nhead, KV_LORA, dtype=torch.bfloat16, device=DEV)
    extra = {}
    if mode in ("persistent", "fp8q"):
        kv_last, extra = persistent_kwargs(
            qo_indptr, kv_indptr, bs, meta_heads or nhead, q_in.dtype
        )
    else:
        kv_last = torch.ones(bs, dtype=torch.int32, device=DEV)
    _, lse = mla_decode_fwd(
        q_in,
        kv_fp8.view(-1, 1, 1, D),
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last,
        1,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        q_scale=q_scale,
        kv_scale=kv_scale,
        return_lse=True,
        **extra,
    )
    torch.cuda.synchronize()
    return o.float(), lse.float().view(bs, nhead)


def reference(q, kv_f32, rows):
    """fp32 attention over each row's KV list. lse is natural log."""
    bs, nhead, _ = q.shape
    out = torch.zeros(bs, nhead, KV_LORA, device=DEV)
    lse = torch.full((bs, nhead), float("-inf"), device=DEV)
    for b, r in enumerate(rows):
        if not r:
            continue
        k = kv_f32[torch.tensor(r, device=DEV, dtype=torch.long)]
        s = (q[b].float() @ k.T) * SM_SCALE
        lse[b] = torch.logsumexp(s, -1)
        out[b] = torch.softmax(s, -1) @ k[:, :KV_LORA]
    return out, lse


def lse_merge(outs, lses):
    lse = torch.stack(lses)
    m = lse.amax(0)
    m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    w = torch.exp(lse - m_safe)
    w = torch.where(torch.isfinite(lse), w, torch.zeros_like(w))
    denom = w.sum(0)
    out = (w.unsqueeze(-1) * torch.stack(outs)).sum(0) / denom.clamp(min=1e-30).unsqueeze(-1)
    return out, m_safe + torch.log(denom)


def err(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    if a.numel() == 0:
        return 0.0
    finite = torch.isfinite(a) & torch.isfinite(b)
    if not bool(finite.all()):
        return float("nan")
    return float((a - b).abs().max())


def make_rows(bs, lens, pool, gen):
    rows = []
    for b in range(bs):
        n = lens[b % len(lens)]
        rows.append(torch.randperm(pool, generator=gen)[:n].tolist())
    return rows


def single_rank_case(name, nhead, bs, lens, mode, meta_heads=None, pool=200_000, seed=0):
    gen = torch.Generator().manual_seed(seed)
    kv = torch.randn(pool, D, generator=gen).to(DEV)
    kv_fp8 = kv.to(FP8)
    kv_f32 = kv_fp8.float()
    q = torch.randn(bs, nhead, D, generator=gen).to(DEV, torch.bfloat16)
    rows = make_rows(bs, lens, pool, gen)
    indptr, indices = build_csr(rows)
    try:
        o, lse = run_kernel(q, kv_fp8, indptr, indices, mode, meta_heads)
    except Exception as e:  # noqa: BLE001
        return dict(case=name, nhead=nhead, bs=bs, mode=mode, meta_heads=meta_heads, error=repr(e)[:300])
    o_ref, lse_ref = reference(q, kv_f32, rows)
    nonempty = torch.tensor([len(r) > 0 for r in rows], device=DEV)
    res = dict(
        case=name,
        nhead=nhead,
        bs=bs,
        mode=mode,
        meta_heads=meta_heads or nhead,
        out_maxerr=err(o, o_ref, nonempty),
        lse_maxerr_base_e=err(lse, lse_ref, nonempty),
        lse_maxerr_base_2=err(lse, lse_ref / math.log(2), nonempty),
    )
    if not bool(nonempty.all()):
        empty = ~nonempty
        res["empty_rows_out_absmax"] = float(o[empty].abs().max()) if bool(torch.isfinite(o[empty]).all()) else float("nan")
        res["empty_rows_lse_sample"] = [float(x) for x in lse[empty][:, 0][:4]]
    return res


def dcp_case(name, dcp, local_heads, bs, lens, mode, pool_global=400_000, seed=1):
    nhead = local_heads * dcp
    gen = torch.Generator().manual_seed(seed)
    kv = torch.randn(pool_global, D, generator=gen).to(DEV)
    kv_fp8 = kv.to(FP8)
    kv_f32 = kv_fp8.float()
    q = torch.randn(bs, nhead, D, generator=gen).to(DEV, torch.bfloat16)
    rows = make_rows(bs, lens, pool_global, gen)
    outs, lses, owned = [], [], []
    try:
        for r in range(dcp):
            local_rows = [[v // dcp for v in row if v % dcp == r] for row in rows]
            owned.append([len(x) for x in local_rows])
            local_pool = kv_fp8[r::dcp].contiguous()
            indptr, indices = build_csr(local_rows)
            o_r, lse_r = run_kernel(q, local_pool, indptr, indices, mode)
            empty = torch.tensor([len(x) == 0 for x in local_rows], device=DEV)
            lse_r = torch.where(empty.unsqueeze(-1), torch.full_like(lse_r, float("-inf")), lse_r)
            o_r = torch.where(empty.view(-1, 1, 1), torch.zeros_like(o_r), o_r)
            outs.append(o_r)
            lses.append(lse_r)
    except Exception as e:  # noqa: BLE001
        return dict(case=name, dcp=dcp, nhead=nhead, bs=bs, mode=mode, error=repr(e)[:300])
    o, lse = lse_merge(outs, lses)
    o_ref, lse_ref = reference(q, kv_f32, rows)
    nonempty = torch.tensor([len(r) > 0 for r in rows], device=DEV)
    min_owned = min(min(x) for x in owned)
    return dict(
        case=name,
        dcp=dcp,
        nhead=nhead,
        bs=bs,
        mode=mode,
        min_owned_per_rank=min_owned,
        merged_out_maxerr=err(o, o_ref, nonempty),
        merged_lse_maxerr=err(lse, lse_ref, nonempty),
    )


def raw_empty_rank_case(nhead, mode, seed=2):
    """A rank that owns zero KV for some rows: report what aiter writes there
    without any masking, so we know whether the caller must mask."""
    gen = torch.Generator().manual_seed(seed)
    pool = 10_000
    kv_fp8 = torch.randn(pool, D, generator=gen).to(DEV).to(FP8)
    bs = 8
    q = torch.randn(bs, nhead, D, generator=gen).to(DEV, torch.bfloat16)
    rows = [[] if b % 2 else torch.randperm(pool, generator=gen)[:300].tolist() for b in range(bs)]
    indptr, indices = build_csr(rows)
    try:
        o, lse = run_kernel(q, kv_fp8, indptr, indices, mode)
    except Exception as e:  # noqa: BLE001
        return dict(case="raw_empty_rows", nhead=nhead, mode=mode, error=repr(e)[:300])
    empty = torch.tensor([len(r) == 0 for r in rows], device=DEV)
    oe, le = o[empty], lse[empty]
    return dict(
        case="raw_empty_rows",
        nhead=nhead,
        mode=mode,
        out_has_nan=bool(torch.isnan(oe).any()),
        out_absmax=float(oe.nan_to_num().abs().max()),
        lse_values=sorted({round(float(x), 4) for x in le.flatten()[:64].tolist()})[:6],
    )


def bench(nhead, bs, nvalid, mode, pool=2_000_000, iters=50, seed=3):
    gen = torch.Generator().manual_seed(seed)
    kv_fp8 = torch.randn(pool, D, generator=gen, dtype=torch.bfloat16).to(DEV).to(FP8)
    q = torch.randn(bs, nhead, D, generator=gen).to(DEV, torch.bfloat16)
    rows = [torch.randint(0, pool, (nvalid,), generator=gen).tolist() for _ in range(bs)]
    indptr, indices = build_csr(rows)
    qo_indptr = torch.arange(bs + 1, dtype=torch.int32, device=DEV)
    kv_scale = torch.ones((), dtype=torch.float32, device=DEV)
    o = torch.empty(bs, nhead, KV_LORA, dtype=torch.bfloat16, device=DEV)
    kv_last, extra = persistent_kwargs(qo_indptr, indptr, bs, nhead, q.dtype)

    def call():
        mla_decode_fwd(
            q, kv_fp8.view(-1, 1, 1, D), o, qo_indptr, indptr, indices, kv_last, 1,
            sm_scale=SM_SCALE, logit_cap=0.0, kv_scale=kv_scale,
            return_lse=(mode == "lse"), **extra,
        )

    for _ in range(5):
        call()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        call()
    end.record()
    torch.cuda.synchronize()
    return dict(case="bench", nhead=nhead, bs=bs, nvalid=nvalid, mode=mode, us=start.elapsed_time(end) * 1000 / iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    print("device:", torch.cuda.get_device_name(0), "fp8:", FP8)
    results = []
    mixed = [2048, 512, 513, 300, 64, 7, 1]

    for nhead in (16, 64, 128):
        for bs in (1, 16, 64):
            results.append(single_rank_case(f"single_h{nhead}_bs{bs}", nhead, bs, mixed, "persistent"))
    results.append(single_rank_case("single_h64_bs16_meta16", 64, 16, mixed, "persistent", meta_heads=16))
    for mode in ("nonpersistent", "fp8q"):
        for nhead in (16, 64):
            results.append(single_rank_case(f"single_h{nhead}_bs16_{mode}", nhead, 16, mixed, mode))
    results.append(single_rank_case("single_h64_extend_n512", 64, 512, mixed, "persistent"))

    for mode in ("persistent", "nonpersistent", "fp8q"):
        results.append(raw_empty_rank_case(64, mode))

    short = [2048, 1500, 600, 64, 5, 3, 2, 1]
    for mode in ("persistent", "nonpersistent", "fp8q"):
        results.append(dcp_case(f"dcp4_tp4_{mode}", 4, 16, 16, short, mode))
    results.append(dcp_case("dcp8_tp8_persistent", 8, 8, 16, short, "persistent"))
    results.append(dcp_case("dcp2_tp4_persistent", 2, 16, 16, short, "persistent"))

    if not args.skip_bench:
        for bs in (4, 16, 64):
            results.append(bench(16, bs, 2048, "nolse"))
            results.append(bench(64, bs, 512, "nolse"))
            results.append(bench(64, bs, 512, "lse"))

    for r in results:
        print(json.dumps(r))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
