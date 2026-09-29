#!/usr/bin/env python3
"""Which knob makes DEFAULT's M_LEQ_8 entry cost 27 us on the router GEMM?

The tile itself is not the problem: BM=16/BN=16/BK=256/KSPLIT=1 at M=4 measures
8.2-9.2 us across num_warps 1/2/4/8 when swept with stages=2, waves_per_eu=0
and no cache modifier. DEFAULT's M_LEQ_8 uses the same tile but adds
stages=3, waves_per_eu=8 and cache_modifier=".cg", and measures 27.1 us.

This walks the 3 x 4 x 2 grid over (num_stages, waves_per_eu, cache_modifier)
on that fixed tile so the responsible knob is named rather than guessed. Also
reports the compiled kernel's register usage and spill counts, since
waves_per_eu is an occupancy hint that buys waves with registers.
"""

import itertools
import json
import os

import torch
import triton

OUT_DIR = "/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm_0928"
os.makedirs(OUT_DIR, exist_ok=True)

M, N, K = 4, 288, 4096
NBUF, LAUNCHES, REPS = 42, 42, 7
DTYPE, DEV = torch.bfloat16, "cuda"

from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16 import (  # noqa: E402
    _gemm_a16_w16_kernel,
)
from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params  # noqa: E402


def log(*a):
    print(*a, flush=True)


torch.manual_seed(0)
x = torch.randn((M, K), dtype=DTYPE, device=DEV) * 0.1
ws = [torch.randn((N, K), dtype=DTYPE, device=DEV) * 0.05 for _ in range(NBUF)]
wsT = [w.T for w in ws]
ref = torch.nn.functional.linear(x, ws[0])
y = torch.empty((M, N), dtype=DTYPE, device=DEV)


def run(cfg, i=0):
    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    return _gemm_a16_w16_kernel[grid](
        x, wsT[i % NBUF], None, y, M, N, K,
        x.stride(0), x.stride(1), wsT[0].stride(0), wsT[0].stride(1),
        0, y.stride(0), y.stride(1),
        activation="", use_activation=False, ADD_BIAS=False,
        SKIP_REDUCE=False, **cfg,
    )


def graph_time(cfg):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            run(cfg, i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(LAUNCHES):
            run(cfg, i)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0 / LAUNCHES)
    ts.sort()
    del g
    return ts[len(ts) // 2]


def compiled_stats(cfg):
    """n_regs / n_spills of the compiled kernel, if triton exposes them."""
    k = run(cfg)
    torch.cuda.synchronize()
    out = {}
    for attr in ("n_regs", "n_spills", "shared"):
        v = getattr(k, attr, None)
        if v is not None:
            out[attr] = v
    md = getattr(k, "metadata", None)
    if md is not None and "shared" not in out:
        v = getattr(md, "shared", None)
        if v is not None:
            out["shared"] = v
    return out


BASE = dict(
    BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=256,
    GROUP_SIZE_M=1, NUM_KSPLIT=1, matrix_instr_nonkdim=16,
)

log(f"M={M} N={N} K={K}  tile BM=16 BN=16 BK=256 KSPLIT=1  "
    f"grid={triton.cdiv(M,16)*triton.cdiv(N,16)}")
log(f"device={torch.cuda.get_device_name(0)} triton={triton.__version__}\n")

rows = []
log(f"{'warps':>5} {'stages':>6} {'wpe':>4} {'cache':>6} {'us':>8}  "
    f"{'regs':>5} {'spills':>6}")
for warps, stages, wpe, cm in itertools.product(
    [4, 8], [1, 2, 3], [0, 2, 4, 6, 8], [None, ".cg"]
):
    cfg = compute_splitk_params(
        dict(BASE, num_warps=warps, num_stages=stages,
             waves_per_eu=wpe, cache_modifier=cm),
        K,
    )
    try:
        run(cfg)
        torch.cuda.synchronize()
        err = ((y.float() - ref.float()).abs().max()
               / ref.abs().max().clamp_min(1e-6)).item()
        if not (err < 2e-2):
            raise RuntimeError(f"mismatch {err:.2e}")
        us = graph_time(cfg)
        st = compiled_stats(cfg)
    except Exception as exc:
        log(f"{warps:>5} {stages:>6} {wpe:>4} {str(cm):>6}  FAILED {type(exc).__name__}")
        continue
    tag = ""
    if (warps, stages, wpe, cm) == (8, 3, 8, ".cg"):
        tag = "   <-- DEFAULT M_LEQ_8"
    if (warps, stages, wpe, cm) == (4, 3, 6, ".cg"):
        tag = "   <-- DEFAULT M_LEQ_16"
    log(f"{warps:>5} {stages:>6} {wpe:>4} {str(cm):>6} {us:>8.2f}  "
        f"{st.get('n_regs', '?'):>5} {st.get('n_spills', '?'):>6}{tag}")
    rows.append(dict(num_warps=warps, num_stages=stages, waves_per_eu=wpe,
                     cache_modifier=cm, us=us, **st))

rows.sort(key=lambda r: r["us"])
log("\n--- fastest 5 ---")
for r in rows[:5]:
    log(f"  {r['us']:6.2f} us  warps={r['num_warps']} stages={r['num_stages']} "
        f"wpe={r['waves_per_eu']} cm={r['cache_modifier']} "
        f"regs={r.get('n_regs','?')} spills={r.get('n_spills','?')}")
log("--- slowest 5 ---")
for r in rows[-5:]:
    log(f"  {r['us']:6.2f} us  warps={r['num_warps']} stages={r['num_stages']} "
        f"wpe={r['waves_per_eu']} cm={r['cache_modifier']} "
        f"regs={r.get('n_regs','?')} spills={r.get('n_spills','?')}")

with open(os.path.join(OUT_DIR, "default_knob_ablation.json"), "w") as f:
    json.dump(rows, f, indent=2, default=str)
log("\nABLATION_DONE")
