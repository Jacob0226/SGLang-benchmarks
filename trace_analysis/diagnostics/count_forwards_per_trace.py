"""How many forwards did the profiler actually capture, and what is in each one?

Anchors on the CPU-side annotation, which is one record per forward, and then
counts the GPU work that lands between that forward's start and the next one's.
Clustering GPU annotations by a time gap (which why_sigma_exceeds_wall.py does)
cannot answer this: with cuda graphs and many streams the GPU windows can run
back to back, so two forwards merge into one cluster and the per-forward figures
come out halved.

    python3 count_forwards_per_trace.py <trace.json.gz> --match "DECODE bs=" \
        [--kernel all_reduce]
"""

import argparse
import gzip
import json
from collections import defaultdict


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--match", required=True)
    ap.add_argument("--kernel", default=None)
    args = ap.parse_args()

    ev = load(args.trace).get("traceEvents", [])
    cpu, gpu, ker = [], [], []
    for e in ev:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            ker.append(e)
        elif args.match in e.get("name", ""):
            if cat == "gpu_user_annotation":
                gpu.append(e)
            elif cat in ("user_annotation", "cpu_op"):
                cpu.append(e)

    cpu.sort(key=lambda e: e["ts"])
    print(f"=== {args.trace.split('/')[-1]}")
    print(f"match {args.match!r}: {len(cpu)} CPU-side forward annotations, "
          f"{len(gpu)} GPU-side copies, {len(ker)} kernels in the trace")
    if not cpu:
        return

    # Forward i owns everything from its CPU start until the next CPU start.
    # The GPU trails the CPU launch loop, so the window has to extend past the
    # annotation's own dur -- that is exactly why a dur-sized window undercounts.
    bounds = []
    for i, e in enumerate(cpu):
        s = e["ts"]
        t = cpu[i + 1]["ts"] if i + 1 < len(cpu) else float("inf")
        bounds.append((s, t, e.get("dur", 0)))

    ker.sort(key=lambda e: e["ts"])
    print(f"\n{'#':>2} {'cpu_dur_ms':>11} {'kernels':>8} {'sigma_ms':>9} "
          f"{'gpu_span_ms':>12}" + (f" {args.kernel[:14]+'_n':>18}" if args.kernel else ""))
    for i, (s, t, dur) in enumerate(bounds):
        inside = [k for k in ker if s <= k["ts"] < t]
        sigma = sum(k.get("dur", 0) for k in inside) / 1000
        span = ((max(k["ts"] + k.get("dur", 0) for k in inside) - min(k["ts"] for k in inside))
                / 1000) if inside else 0.0
        line = f"{i:>2} {dur/1000:>11.3f} {len(inside):>8} {sigma:>9.3f} {span:>12.3f}"
        if args.kernel:
            sel = [k for k in inside if args.kernel in k.get("name", "")]
            tot = sum(k.get("dur", 0) for k in sel) / 1000
            line += f" {len(sel):>8} / {tot:>7.3f}ms"
        print(line)

    if args.kernel:
        print(f"\n(last column: launches of {args.kernel!r} in that forward, and their sigma)")

    # Which streams carry the GPU-side copies, to show why gap clustering fails.
    per_tid = defaultdict(int)
    for e in gpu:
        per_tid[e.get("tid")] += 1
    print(f"\nGPU-side annotation copies live on {len(per_tid)} stream(s); "
          f"{sum(1 for v in per_tid.values() if v > 1)} of them carry more than one copy")


if __name__ == "__main__":
    main()
