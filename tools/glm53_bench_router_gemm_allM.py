#!/usr/bin/env python3
"""Phase 2: tune the router GEMM (N=288, K=4096, bf16) across every M bucket
that GLM-5.3-Flash actually hits, then emit GEMM-A16W16-N=288-K=4096.json.

A specialized config file *replaces* DEFAULT.json wholesale in aiter's lookup
(see _get_gemm_config_cached), so the emitted file must carry every bucket
DEFAULT has, not just the tuned ones.
"""

import copy
import itertools
import json
import os

import torch
import triton

OUT_DIR = "/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm"
os.makedirs(OUT_DIR, exist_ok=True)

N, K = 288, 4096
NBUF = 42
REPS = 7
DTYPE = torch.bfloat16
DEV = "cuda"

from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16 import (  # noqa: E402
    _gemm_a16_w16_kernel,
)
from aiter.ops.triton._triton_kernels.common.splitk_reduce import (  # noqa: E402
    _gemm_splitk_reduce_kernel,
)
from aiter.ops.triton.utils.config_utils import resolve_config_dir  # noqa: E402
from aiter.ops.triton.utils.gemm_config_utils import (  # noqa: E402
    compute_splitk_params,
)


def log(*a):
    print(*a, flush=True)


CFG_DIR = resolve_config_dir("gemm", "GEMM-A16W16", backend="triton")
log(f"config dir = {CFG_DIR}")
with open(f"{CFG_DIR}/DEFAULT.json") as f:
    DEFAULT = json.load(f)
log(f"DEFAULT buckets: {list(DEFAULT.keys())}")

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
    return ts[0], ts[len(ts) // 2]


def make_cfg(bm, bn, bk, ks, w, st, wpe, cm):
    return compute_splitk_params(
        {
            "BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
            "GROUP_SIZE_M": 1, "NUM_KSPLIT": ks, "num_warps": w,
            "num_stages": st, "waves_per_eu": wpe,
            "matrix_instr_nonkdim": 16, "cache_modifier": cm,
        },
        K,
    )


def eval_cfg(cfg, M, x, ref, launches):
    ks = cfg["NUM_KSPLIT"]
    y = torch.empty((M, N), dtype=DTYPE, device=DEV)
    y_pp = (
        torch.empty((ks, M, N), dtype=torch.float32, device=DEV) if ks > 1 else None
    )
    actual_ksplit = triton.cdiv(K, cfg["SPLITK_BLOCK_SIZE"])

    def one(i):
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
        if ks > 1:
            _gemm_splitk_reduce_kernel[(triton.cdiv(M, 32), triton.cdiv(N, 32))](
                y_pp, y, None, M, N,
                y_pp.stride(0), y_pp.stride(1), y_pp.stride(2),
                y.stride(0), y.stride(1), 32, 32, actual_ksplit,
                triton.next_power_of_2(ks), ADD_BIAS=False, activation="",
                use_activation=False, KERNEL_NAME="_gemm_a16w16_reduce_kernel",
            )

    one(0)
    torch.cuda.synchronize()
    err = ((y.float() - ref.float()).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
    if not (err < 2e-2):
        raise RuntimeError(f"mismatch {err:.3e}")
    return graph_time(lambda: [one(i) for i in range(launches)], launches)


# Space around the phase-1 winner plus the small-M / large-M extremes.
SPACE = list(
    itertools.product(
        [1, 4, 8, 16, 32, 64, 128, 256],  # BLOCK_M
        [16, 32, 64, 128],  # BLOCK_N
        [128, 256, 512],  # BLOCK_K
        [1, 2, 4, 8],  # NUM_KSPLIT
        [1, 2, 4, 8],  # num_warps
    )
)

# M values the model actually runs: decode conc4/conc64 (x4 TP ranks share the
# token batch) and the i8k / i70k prefill chunks.
M_TARGETS = [
    (4, 1, 42),
    (8, 4, 42),
    (16, 8, 42),
    (64, 16, 42),
    (256, 64, 42),
    (2048, 256, 16),
    (8192, 1024, 8),
]

winners = {}
summary = []
for M, bucket, launches in M_TARGETS:
    x = torch.randn((M, K), dtype=DTYPE, device=DEV) * 0.1
    ref = torch.nn.functional.linear(x, ws[0])

    # what production does today
    from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16 import (
        _get_config as _get_triton_config,
    )

    prod_cfg, _ = _get_triton_config(M, N, K)
    try:
        _, prod_us = eval_cfg(dict(prod_cfg), M, x, ref, launches)
    except Exception as exc:
        log(f"M={M}: production config failed: {exc}")
        prod_us = float("nan")

    best = None
    tried = set()
    for bm, bn, bk, ks, w in SPACE:
        if bm > max(16, 2 * M):  # a tile taller than the problem is pure waste
            continue
        cfg = make_cfg(bm, bn, bk, ks, w, 2, 0, None)
        key = tuple(
            cfg[k] for k in
            ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "NUM_KSPLIT", "num_warps")
        )
        if key in tried:
            continue
        tried.add(key)
        try:
            _, us = eval_cfg(cfg, M, x, ref, launches)
        except Exception:
            continue
        if best is None or us < best[0]:
            best = (us, cfg)

    # refine num_stages / waves_per_eu / cache_modifier on the winner
    if best is not None:
        base = best[1]
        for st, wpe, cm in itertools.product([1, 2, 3], [0, 2, 4, 8], [None, ".cg"]):
            cfg = make_cfg(
                base["BLOCK_SIZE_M"], base["BLOCK_SIZE_N"], base["BLOCK_SIZE_K"],
                base["NUM_KSPLIT"], base["num_warps"], st, wpe, cm,
            )
            try:
                _, us = eval_cfg(cfg, M, x, ref, launches)
            except Exception:
                continue
            if us < best[0]:
                best = (us, cfg)

    us, cfg = best
    keep = {
        k: cfg[k] for k in (
            "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
            "NUM_KSPLIT", "num_warps", "num_stages", "waves_per_eu",
            "matrix_instr_nonkdim", "cache_modifier",
        )
    }
    winners[bucket] = keep
    speedup = prod_us / us if us else float("nan")
    summary.append((M, bucket, prod_us, us, speedup))
    log(f"M={M:5d} bucket=M_LEQ_{bucket:<5d} default={prod_us:7.2f} us  "
        f"tuned={us:6.2f} us  speedup={speedup:.2f}x  {keep}")

log("\n===== summary =====")
for M, bucket, p, t, s in summary:
    log(f"  M={M:5d}  default {p:7.2f} us -> tuned {t:6.2f} us  ({s:.2f}x)")

# ------------------------------------------------------------ emit the config
out = copy.deepcopy(DEFAULT)
for bucket, cfg in winners.items():
    out[f"M_LEQ_{bucket}"] = cfg
# Keep every DEFAULT bucket we did not tune so the specialized file is a strict
# superset of the fallback it replaces.
path = os.path.join(OUT_DIR, "GEMM-A16W16-N=288-K=4096.json")
with open(path, "w") as f:
    json.dump(out, f, indent=4)
    f.write("\n")
log(f"\nwrote {path}")
log(json.dumps(out, indent=2))
log("ALLM_DONE")
