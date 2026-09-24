#!/usr/bin/env python3
"""Microbenchmark the GLM-5.3-Flash MoE router GEMM on gfx950.

Shape: M=4, N=288, K=4096, bf16 (x @ w.T), 42 launches per decode forward.
On the 0923 image this falls back to DEFAULT.json M_LEQ_8 and costs 28.1 us
per launch vs 8.4 us on the 0914 image with the *same* config, so the goal is
to find a config that triton 3.8 compiles well.

Timing uses HIP graph replay: the aiter wrapper JSON-serializes the config on
every call, which costs ~27 us of CPU per launch and completely masks the GPU
kernel if you time a plain python loop.

Compares: current default config / swept triton configs / aiter skinny-GEMM /
torch (hipblaslt) reference.
"""

import itertools
import json
import os
import traceback

import torch

OUT_DIR = os.environ.get(
    "OUT_DIR", "/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm"
)
os.makedirs(OUT_DIR, exist_ok=True)

M = int(os.environ.get("BENCH_M", 4))
N = int(os.environ.get("BENCH_N", 288))
K = int(os.environ.get("BENCH_K", 4096))
NBUF = int(os.environ.get("BENCH_NBUF", 42))  # rotate 42 weights, as in the model
GRAPH_ITERS = int(os.environ.get("BENCH_GRAPH_ITERS", 42))  # launches per graph
REPS = int(os.environ.get("BENCH_REPS", 7))
DTYPE = torch.bfloat16
DEV = "cuda"


def log(*a):
    print(*a, flush=True)


import triton  # noqa: E402

import aiter  # noqa: E402
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16 import (  # noqa: E402
    _gemm_a16_w16_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16 import (  # noqa: E402
    _get_config as _get_triton_config,
)
from aiter.ops.triton.utils._triton import arch_info  # noqa: E402
from aiter.ops.triton.utils.gemm_config_utils import (  # noqa: E402
    compute_splitk_params,
)

ARCH = arch_info.get_arch()
CU = torch.cuda.get_device_properties(0).multi_processor_count
log(f"arch={ARCH} triton={triton.__version__} torch={torch.__version__}")
log(f"shape M={M} N={N} K={K} dtype={DTYPE} nbuf={NBUF} CU={CU}")
log(f"device={torch.cuda.get_device_name(0)}")

# ------------------------------------------------------------------- operands
torch.manual_seed(0)
x = torch.randn((M, K), dtype=DTYPE, device=DEV) * 0.1
# 42 distinct weight buffers (288*4096*2B = 2.36 MB each, ~95 MiB total) so the
# weight stream is not served from a single warm line the way one buffer would be.
ws = [torch.randn((N, K), dtype=DTYPE, device=DEV) * 0.05 for _ in range(NBUF)]
wsT = [w.T for w in ws]
log(f"weight working set = {NBUF * N * K * 2 / 2**20:.1f} MiB")

refs = [torch.nn.functional.linear(x, w) for w in ws]


def max_rel_err(got, ref):
    denom = ref.abs().max().clamp_min(1e-6)
    return ((got.float() - ref.float()).abs().max() / denom).item()


# --------------------------------------------------------------------- timing
def graph_time(build_launches, n=GRAPH_ITERS, reps=REPS):
    """Capture n launches into a HIP graph, replay, return (best_us, med_us) per launch.

    build_launches() must issue exactly n kernel launches on the current stream.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            build_launches()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        build_launches()

    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()

    ts = []
    for _ in range(reps):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        g.replay()
        ev1.record()
        torch.cuda.synchronize()
        ts.append(ev0.elapsed_time(ev1) * 1000.0 / n)
    ts.sort()
    del g
    return ts[0], ts[len(ts) // 2]


# ------------------------------------------------- direct triton kernel launch
def launch_triton(cfg, y, y_pp, i):
    """One raw _gemm_a16_w16_kernel launch, bypassing the aiter custom-op wrapper."""
    ksplit = cfg["NUM_KSPLIT"]
    out = y if ksplit == 1 else y_pp
    grid = lambda META: (  # noqa: E731
        META["NUM_KSPLIT"]
        * triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _gemm_a16_w16_kernel[grid](
        x,
        wsT[i % NBUF],
        None,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        wsT[0].stride(0),
        wsT[0].stride(1),
        0 if ksplit == 1 else y_pp.stride(0),
        y.stride(0) if ksplit == 1 else y_pp.stride(1),
        y.stride(1) if ksplit == 1 else y_pp.stride(2),
        activation="",
        use_activation=False,
        ADD_BIAS=False,
        SKIP_REDUCE=False,
        **cfg,
    )


def make_cfg(bm, bn, bk, ksplit, warps, stages, wpe, gsm, cm, nonkdim=16):
    cfg = {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "GROUP_SIZE_M": gsm,
        "NUM_KSPLIT": ksplit,
        "num_warps": warps,
        "num_stages": stages,
        "waves_per_eu": wpe,
        "matrix_instr_nonkdim": nonkdim,
        "cache_modifier": cm,
    }
    return compute_splitk_params(cfg, K)


def run_cfg(cfg):
    """Validate + graph-time one triton config. Returns (best_us, med_us, rel_err).

    NUM_KSPLIT>1 writes fp32 partials that the caller reduces in a second
    kernel; both are counted so the comparison against NUM_KSPLIT==1 is fair.
    """
    ksplit = cfg["NUM_KSPLIT"]
    y = torch.empty((M, N), dtype=DTYPE, device=DEV)
    y_pp = (
        torch.empty((ksplit, M, N), dtype=torch.float32, device=DEV)
        if ksplit > 1
        else None
    )

    if ksplit > 1:
        from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
            _gemm_splitk_reduce_kernel,
        )

        actual_ksplit = triton.cdiv(K, cfg["SPLITK_BLOCK_SIZE"])

        def reduce():
            _gemm_splitk_reduce_kernel[(triton.cdiv(M, 32), triton.cdiv(N, 32))](
                y_pp, y, None, M, N,
                y_pp.stride(0), y_pp.stride(1), y_pp.stride(2),
                y.stride(0), y.stride(1),
                32, 32, actual_ksplit,
                triton.next_power_of_2(ksplit),
                ADD_BIAS=False, activation="", use_activation=False,
                KERNEL_NAME="_gemm_a16w16_reduce_kernel",
            )
    else:
        reduce = lambda: None  # noqa: E731

    launch_triton(cfg, y, y_pp, 0)
    reduce()
    torch.cuda.synchronize()
    err = max_rel_err(y, refs[0])
    if not (err < 2e-2):
        raise RuntimeError(f"numerical mismatch rel_err={err:.3e}")

    def build():
        for i in range(GRAPH_ITERS):
            launch_triton(cfg, y, y_pp, i)
            reduce()

    b, m = graph_time(build)
    return b, m, err


results = []
csv_path = os.path.join(OUT_DIR, "sweep.csv")
csv = open(csv_path, "w")
csv.write(
    "label,BLOCK_M,BLOCK_N,BLOCK_K,NUM_KSPLIT,num_warps,num_stages,"
    "waves_per_eu,GROUP_SIZE_M,cache_modifier,grid,best_us,med_us,rel_err,status\n"
)
csv.flush()


def record(label, cfg, b, m, err, status="ok"):
    grid = (
        cfg["NUM_KSPLIT"]
        * triton.cdiv(M, cfg["BLOCK_SIZE_M"])
        * triton.cdiv(N, cfg["BLOCK_SIZE_N"])
    )
    csv.write(
        f"{label},{cfg['BLOCK_SIZE_M']},{cfg['BLOCK_SIZE_N']},{cfg['BLOCK_SIZE_K']},"
        f"{cfg['NUM_KSPLIT']},{cfg['num_warps']},{cfg['num_stages']},"
        f"{cfg['waves_per_eu']},{cfg['GROUP_SIZE_M']},{cfg['cache_modifier']},"
        f"{grid},{b if b else ''},{m if m else ''},{err if err is not None else ''},{status}\n"
    )
    csv.flush()
    if status == "ok":
        results.append((m, b, label, dict(cfg), grid))


def record_plain(label, b, m, err):
    csv.write(f"{label},,,,,,,,,,,{b},{m},{err},ok\n")
    csv.flush()


# ------------------------------------------- 0. what production actually uses
log("\n===== 0. production default config =====")
prod_cfg, prod_tuned = _get_triton_config(M, N, K)
log(f"is_tuned={prod_tuned}  {json.dumps(prod_cfg, default=str)}")
try:
    b, m, err = run_cfg(prod_cfg)
    log(f"PRODUCTION DEFAULT: best={b:.2f} us  median={m:.2f} us  rel_err={err:.2e}")
    record("production_default", prod_cfg, b, m, err)
    PROD = m
except Exception as exc:
    log(f"production default FAILED: {exc}")
    traceback.print_exc()
    PROD = float("nan")

# ------------------------------------------- 1. torch / hipblaslt reference
log("\n===== 1. torch reference (hipblaslt) =====")
y_t = torch.empty((M, N), dtype=DTYPE, device=DEV)
try:
    b, m = graph_time(lambda: [torch.mm(x, wsT[i % NBUF], out=y_t) for i in range(GRAPH_ITERS)])
    log(f"torch.mm: best={b:.2f} us  median={m:.2f} us")
    record_plain("torch_mm", b, m, 0)
except Exception as exc:
    log(f"torch.mm failed: {exc}")

# ------------------------------------------- 2. aiter skinny GEMM
log("\n===== 2. aiter skinny GEMM =====")


def try_skinny(name, f):
    y_s = torch.zeros((M, N), dtype=DTYPE, device=DEV)
    try:
        f(ws[0], x, y_s, M, CU)
        torch.cuda.synchronize()
        err = max_rel_err(y_s, refs[0])
        if not (err < 2e-2):
            log(f"  {name}: WRONG rel_err={err:.3e}")
            return
        b, m = graph_time(
            lambda: [f(ws[i % NBUF], x, y_s, M, CU) for i in range(GRAPH_ITERS)]
        )
        log(f"  {name}: best={b:.2f} us  median={m:.2f} us  rel_err={err:.2e}")
        record_plain(name, b, m, err)
    except Exception as exc:
        log(f"  {name}: FAILED {type(exc).__name__}: {exc}")


for nm, fname in [
    ("skinny_wv_splitk_small", "wv_splitk_small_fp16_bf16"),
    ("skinny_wvSpltK", "wvSpltK"),
]:
    fn = getattr(aiter, fname, None)
    if fn is None:
        from aiter.ops import custom as _custom

        fn = getattr(_custom, fname, None)
    if fn is None:
        log(f"  {nm}: not exported")
        continue
    try_skinny(nm, fn)

# ------------------------------------------- 3. coarse triton sweep
log("\n===== 3. coarse triton config sweep =====")
space = list(
    itertools.product(
        [1, 4, 8, 16],  # BLOCK_M
        [16, 32, 64, 128],  # BLOCK_N
        [128, 256, 512],  # BLOCK_K
        [1, 2, 4, 8, 16],  # NUM_KSPLIT
        [1, 2, 4, 8],  # num_warps
    )
)
log(f"coarse space = {len(space)} configs")
seen = set()
for idx, (bm, bn, bk, ks, w) in enumerate(space):
    cfg = make_cfg(bm, bn, bk, ks, w, 2, 0, 1, None)
    key = tuple(
        cfg[k]
        for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "NUM_KSPLIT", "num_warps")
    )
    if key in seen:
        continue
    seen.add(key)
    label = f"c_bm{bm}_bn{bn}_bk{bk}_ks{ks}_w{w}"
    try:
        b, m, err = run_cfg(cfg)
        record(label, cfg, b, m, err)
        if m < PROD * 0.6:
            log(f"[{idx}/{len(space)}] {label}: {m:.2f} us  <-- beats default")
    except Exception as exc:
        record(label, cfg, None, None, None, status=f"fail:{type(exc).__name__}")
    if idx % 50 == 0:
        log(f"  ...{idx}/{len(space)} done, best so far "
            f"{min(results)[0]:.2f} us" if results else f"  ...{idx}/{len(space)}")

results.sort()
log("\n--- coarse top 15 ---")
for m, b, label, cfg, grid in results[:15]:
    log(f"  {m:7.2f} us (best {b:6.2f})  grid={grid:5d}  {label}")

# ------------------------------------------- 4. refine top candidates
log("\n===== 4. refine top candidates =====")
top = [r for r in results if r[2].startswith("c_")][:10]
for _m, _b, label, cfg, _g in top:
    for stages, wpe, cm in itertools.product([1, 2, 3], [0, 2, 4, 8], [None, ".cg"]):
        c2 = make_cfg(
            cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"],
            cfg["NUM_KSPLIT"], cfg["num_warps"], stages, wpe, 1, cm,
        )
        lab = f"r_{label}_s{stages}_wpe{wpe}_cm{'cg' if cm else 'none'}"
        try:
            b, m, err = run_cfg(c2)
            record(lab, c2, b, m, err)
        except Exception as exc:
            record(lab, c2, None, None, None, status=f"fail:{type(exc).__name__}")

results.sort()
log("\n===== FINAL TOP 20 =====")
log(f"(production default = {PROD:.2f} us)")
for m, b, label, cfg, grid in results[:20]:
    log(f"  {m:7.2f} us (best {b:6.2f})  grid={grid:5d}  {label}")

if results:
    best_m, best_b, best_label, best_cfg, best_grid = results[0]
    log(f"\nBEST: {best_label}  median={best_m:.2f} us  ({PROD / best_m:.2f}x vs default)")
    keep = {
        k: best_cfg[k]
        for k in (
            "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
            "NUM_KSPLIT", "num_warps", "num_stages", "waves_per_eu",
            "matrix_instr_nonkdim", "cache_modifier",
        )
    }
    log(json.dumps(keep, indent=2))
    with open(os.path.join(OUT_DIR, "best_config.json"), "w") as f:
        json.dump(
            {"M_LEQ_8": keep, "_label": best_label, "_median_us": best_m,
             "_default_us": PROD},
            f, indent=2,
        )

csv.close()
log(f"\nCSV written to {csv_path}")
log("BENCH_DONE")
