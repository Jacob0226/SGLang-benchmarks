#!/usr/bin/env python3
"""Sweep the GLM-5.3-Flash MoE router GEMM (N=288, K=4096, bf16) for every
M bucket that conc4..conc64 decode hits, and emit two candidate config files.

Buckets (sglang pads decode to captured bs 1,2,4,8,12,16,24,32,40,48,56,64):
  conc4        -> M=4   -> M_LEQ_4
  conc8..16    -> M<=16 -> M_LEQ_8  / M_LEQ_16
  conc24..32   -> M<=32 -> M_LEQ_32
  conc40..64   -> M<=64 -> M_LEQ_64

Two candidates are emitted because the trace shows a ~4 us per-kernel floor in
the decode HIP graph: NUM_KSPLIT>1 pays it twice (GEMM 4.36 us + split-K reduce
4.32 us for a shape that only reduces 18 KB), and a slower single-kernel config
can still win end to end. This script's own combined timing under-reports that
second kernel -- back-to-back graph nodes pipeline the overhead in a
microbenchmark in a way they do not in the model -- so the GEMM and the reduce
are timed separately here and the pick must be confirmed against a trace.

  best_overall.json  -- lowest measured total, split-K allowed
  best_single.json   -- lowest measured total among NUM_KSPLIT==1 only
"""

import copy
import itertools
import json
import os

import torch
import triton

OUT_DIR = "/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm_0928"
os.makedirs(OUT_DIR, exist_ok=True)

N, K = 288, 4096
NBUF = 42
REPS = 7
LAUNCHES = 42
DTYPE = torch.bfloat16
DEV = "cuda"

from aiter.ops.triton._triton_kernels.common.splitk_reduce import (  # noqa: E402
    _gemm_splitk_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16 import (  # noqa: E402
    _gemm_a16_w16_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16 import (  # noqa: E402
    _get_config as _get_triton_config,
)
from aiter.ops.triton.utils.config_utils import resolve_config_dir  # noqa: E402
from aiter.ops.triton.utils.gemm_config_utils import (  # noqa: E402
    compute_splitk_params,
)


def log(*a):
    print(*a, flush=True)


CFG_DIR = resolve_config_dir("gemm", "GEMM-A16W16", backend="triton")
with open(f"{CFG_DIR}/DEFAULT.json") as f:
    DEFAULT = json.load(f)
log(f"config dir = {CFG_DIR}")
log(f"DEFAULT buckets: {list(DEFAULT.keys())}")
log(f"device={torch.cuda.get_device_name(0)} "
    f"CU={torch.cuda.get_device_properties(0).multi_processor_count}")

torch.manual_seed(0)
ws = [torch.randn((N, K), dtype=DTYPE, device=DEV) * 0.05 for _ in range(NBUF)]
wsT = [w.T for w in ws]


def graph_time(build, n, reps=REPS):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            build()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        build()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0 / n)
    ts.sort()
    del g
    return ts[len(ts) // 2]


def make_cfg(bm, bn, bk, ks, w, st, wpe, cm, gsm=1):
    return compute_splitk_params(
        {
            "BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
            "GROUP_SIZE_M": gsm, "NUM_KSPLIT": ks, "num_warps": w,
            "num_stages": st, "waves_per_eu": wpe,
            "matrix_instr_nonkdim": 16, "cache_modifier": cm,
        },
        K,
    )


def eval_cfg(cfg, M, x, ref):
    """Return (gemm_us, reduce_us, total_us, nkernels)."""
    ks = cfg["NUM_KSPLIT"]
    y = torch.empty((M, N), dtype=DTYPE, device=DEV)
    y_pp = (
        torch.empty((ks, M, N), dtype=torch.float32, device=DEV) if ks > 1 else None
    )

    def gemm(i):
        grid = lambda META: (  # noqa: E731
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        _gemm_a16_w16_kernel[grid](
            x, wsT[i % NBUF], None, y if ks == 1 else y_pp, M, N, K,
            x.stride(0), x.stride(1), wsT[0].stride(0), wsT[0].stride(1),
            0 if ks == 1 else y_pp.stride(0),
            y.stride(0) if ks == 1 else y_pp.stride(1),
            y.stride(1) if ks == 1 else y_pp.stride(2),
            activation="", use_activation=False, ADD_BIAS=False,
            SKIP_REDUCE=False, **cfg,
        )

    actual_ksplit = triton.cdiv(K, cfg["SPLITK_BLOCK_SIZE"])

    def reduce():
        _gemm_splitk_reduce_kernel[(triton.cdiv(M, 32), triton.cdiv(N, 32))](
            y_pp, y, None, M, N,
            y_pp.stride(0), y_pp.stride(1), y_pp.stride(2),
            y.stride(0), y.stride(1), 32, 32, actual_ksplit,
            triton.next_power_of_2(ks), ADD_BIAS=False, activation="",
            use_activation=False, KERNEL_NAME="_gemm_a16w16_reduce_kernel",
        )

    gemm(0)
    if ks > 1:
        reduce()
    torch.cuda.synchronize()
    err = ((y.float() - ref.float()).abs().max()
           / ref.abs().max().clamp_min(1e-6)).item()
    if not (err < 2e-2):
        raise RuntimeError(f"mismatch {err:.3e}")

    g_us = graph_time(lambda: [gemm(i) for i in range(LAUNCHES)], LAUNCHES)
    if ks > 1:
        r_us = graph_time(lambda: [reduce() for _ in range(LAUNCHES)], LAUNCHES)
        return g_us, r_us, g_us + r_us, 2
    return g_us, 0.0, g_us, 1


KEEP = (
    "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
    "NUM_KSPLIT", "num_warps", "num_stages", "waves_per_eu",
    "matrix_instr_nonkdim", "cache_modifier",
)

# (M to measure, bucket key to write). M is the largest value the bucket serves,
# since that is the worst case for a tile sized to the bucket.
TARGETS = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]

csv = open(os.path.join(OUT_DIR, "bucket_sweep.csv"), "w")
csv.write("M,bucket,label,BM,BN,BK,KSPLIT,warps,stages,wpe,grid,"
          "gemm_us,reduce_us,total_us,nkernels\n")
csv.flush()

best_overall, best_single, summary = {}, {}, []

for M, bucket in TARGETS:
    x = torch.randn((M, K), dtype=DTYPE, device=DEV) * 0.1
    ref = torch.nn.functional.linear(x, ws[0])

    prod_cfg, _ = _get_triton_config(M, N, K)
    try:
        _, _, prod_us, prod_nk = eval_cfg(dict(prod_cfg), M, x, ref)
    except Exception as exc:
        log(f"M={M}: DEFAULT failed: {exc}")
        prod_us, prod_nk = float("nan"), 1

    bms = sorted({1, M // 4 or 1, M // 2 or 1, M, M * 2, 16})
    bms = [b for b in bms if 1 <= b <= 256]
    space = list(itertools.product(bms, [16, 32, 64], [256, 512],
                                   [1, 2, 4, 8, 16], [1, 2, 4, 8]))
    log(f"\n===== M={M} (bucket M_LEQ_{bucket}) DEFAULT={prod_us:.2f} us "
        f"({prod_nk} kernel) | {len(space)} coarse configs =====")

    seen, results = set(), []
    for bm, bn, bk, ks, w in space:
        cfg = make_cfg(bm, bn, bk, ks, w, 2, 0, None)
        key = tuple(cfg[k] for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N",
                                     "BLOCK_SIZE_K", "NUM_KSPLIT", "num_warps"))
        if key in seen:
            continue
        seen.add(key)
        try:
            g, r, t, nk = eval_cfg(cfg, M, x, ref)
        except Exception:
            continue
        grid = (cfg["NUM_KSPLIT"] * triton.cdiv(M, cfg["BLOCK_SIZE_M"])
                * triton.cdiv(N, cfg["BLOCK_SIZE_N"]))
        lab = f"bm{bm}_bn{bn}_bk{bk}_ks{ks}_w{w}"
        csv.write(f"{M},{bucket},{lab},{bm},{bn},{bk},{cfg['NUM_KSPLIT']},{w},2,0,"
                  f"{grid},{g:.3f},{r:.3f},{t:.3f},{nk}\n")
        csv.flush()
        results.append((t, cfg, nk))

    # refine the top few of each family on stages / waves_per_eu / cache_modifier
    results.sort(key=lambda r: r[0])
    tops = results[:4]
    singles = [r for r in results if r[2] == 1][:4]
    for _t, base, _nk in tops + singles:
        for st, wpe, cm in itertools.product([1, 2, 3], [0, 2, 4, 8], [None, ".cg"]):
            cfg = make_cfg(base["BLOCK_SIZE_M"], base["BLOCK_SIZE_N"],
                           base["BLOCK_SIZE_K"], base["NUM_KSPLIT"],
                           base["num_warps"], st, wpe, cm)
            try:
                g, r, t, nk = eval_cfg(cfg, M, x, ref)
            except Exception:
                continue
            grid = (cfg["NUM_KSPLIT"] * triton.cdiv(M, cfg["BLOCK_SIZE_M"])
                    * triton.cdiv(N, cfg["BLOCK_SIZE_N"]))
            lab = (f"r_bm{cfg['BLOCK_SIZE_M']}_bn{cfg['BLOCK_SIZE_N']}"
                   f"_bk{cfg['BLOCK_SIZE_K']}_ks{cfg['NUM_KSPLIT']}"
                   f"_w{cfg['num_warps']}_s{st}_wpe{wpe}_cm{'cg' if cm else 'none'}")
            csv.write(f"{M},{bucket},{lab},{cfg['BLOCK_SIZE_M']},{cfg['BLOCK_SIZE_N']},"
                      f"{cfg['BLOCK_SIZE_K']},{cfg['NUM_KSPLIT']},{cfg['num_warps']},"
                      f"{st},{wpe},{grid},{g:.3f},{r:.3f},{t:.3f},{nk}\n")
            csv.flush()
            results.append((t, cfg, nk))

    results.sort(key=lambda r: r[0])
    bo_t, bo_cfg, bo_nk = results[0]
    sing = [r for r in results if r[2] == 1]
    bs_t, bs_cfg, _ = sing[0] if sing else results[0]

    best_overall[f"M_LEQ_{bucket}"] = {k: bo_cfg[k] for k in KEEP}
    best_single[f"M_LEQ_{bucket}"] = {k: bs_cfg[k] for k in KEEP}
    summary.append((M, bucket, prod_us, prod_nk, bo_t, bo_nk, bs_t))

    log(f"  DEFAULT      {prod_us:7.2f} us ({prod_nk} kernel)")
    log(f"  best overall {bo_t:7.2f} us ({bo_nk} kernel)  "
        f"BM={bo_cfg['BLOCK_SIZE_M']} BN={bo_cfg['BLOCK_SIZE_N']} "
        f"BK={bo_cfg['BLOCK_SIZE_K']} KSPLIT={bo_cfg['NUM_KSPLIT']} "
        f"warps={bo_cfg['num_warps']}")
    log(f"  best single  {bs_t:7.2f} us (1 kernel)  "
        f"BM={bs_cfg['BLOCK_SIZE_M']} BN={bs_cfg['BLOCK_SIZE_N']} "
        f"BK={bs_cfg['BLOCK_SIZE_K']} warps={bs_cfg['num_warps']}")

csv.close()

log("\n===================== summary =====================")
log(f"{'M':>5} {'bucket':>10} {'DEFAULT':>9} {'best_all':>9} {'best_1k':>9}")
for M, b, p, pnk, bo, bonk, bs in summary:
    log(f"{M:>5} {'M_LEQ_'+str(b):>10} {p:>8.2f} {bo:>8.2f}({bonk}) {bs:>8.2f}")

# M_LEQ_1 mirrors M_LEQ_4: the smallest bucket DEFAULT defines is M_LEQ_8, so
# M=1..4 currently lands on a tile built for M=8.
for name, buckets in (("best_overall", best_overall), ("best_single", best_single)):
    out = copy.deepcopy(DEFAULT)
    out.update(buckets)
    out["M_LEQ_1"] = dict(buckets["M_LEQ_4"])
    # keys must stay in ascending bucket order for readability, not correctness
    ordered = {}
    for k in sorted(out, key=lambda s: (s == "any", int(s.split("_")[-1]) if s != "any" else 0)):
        ordered[k] = out[k]
    p = os.path.join(OUT_DIR, f"GEMM-A16W16-N=288-K=4096.{name}.json")
    with open(p, "w") as f:
        json.dump(ordered, f, indent=4)
        f.write("\n")
    log(f"wrote {p}")

log("SWEEP_DONE")
