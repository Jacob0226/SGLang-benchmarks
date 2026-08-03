#!/usr/bin/env python3
"""Enumerate every forward pass in a trace and report its GPU wall time.

Each forward is delimited by a CPU-side annotation (SGLang: "Forward batch ...";
ATOM: a matching --forward-match string). Kernels are attributed to a forward via
the CUDA/HIP runtime correlation id of the launch that falls inside the CPU
annotation window, so the measurement does not depend on time-window heuristics.

GPU wall time is max(ts+dur) - min(ts) over the attributed kernels, which equals
the kernel sum only when the forward runs on a single stream with no overlap; the
ratio is reported so that assumption is visible rather than implied.

Usage:
  per_forward_wall.py TRACE.json.gz [--match SUBSTR] [--min-tokens N]
"""
import argparse
import gzip
import json
import re
import statistics
from collections import defaultdict

CPU_OP_CATS = {"cpu_op", "user_annotation", "cuda_runtime", "hip_runtime", "runtime"}
KERNEL_CATS = {"kernel", "Kernel"}


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--match", default="Forward batch",
                    help="substring identifying the per-forward CPU annotation")
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="only report forwards with at least this many tokens "
                         "(filters partial chunked-prefill steps)")
    args = ap.parse_args()

    trace = load(args.trace)
    events = trace.get("traceEvents", [])

    forwards = []      # (ts, dur, name)
    launches = []      # (ts, correlation)
    kernels = {}       # correlation -> (ts, dur, stream)
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        cat = e.get("cat", "")
        name = e.get("name", "")
        if cat in KERNEL_CATS:
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                kernels[corr] = (e["ts"], e["dur"], e.get("tid"))
        elif args.match in name and cat in CPU_OP_CATS:
            forwards.append((e["ts"], e["dur"], name))
        elif cat in ("cuda_runtime", "hip_runtime", "runtime"):
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                launches.append((e["ts"], corr))

    if not forwards:
        raise SystemExit(f"no CPU annotation matching {args.match!r} found")

    launches.sort()
    launch_ts = [t for t, _ in launches]
    import bisect

    rows = []
    for fts, fdur, fname in sorted(forwards):
        lo = bisect.bisect_left(launch_ts, fts)
        hi = bisect.bisect_right(launch_ts, fts + fdur)
        ks = [kernels[c] for _, c in launches[lo:hi] if c in kernels]
        if not ks:
            continue
        ksum = sum(d for _, d, _ in ks)
        span = max(t + d for t, d, _ in ks) - min(t for t, _, _ in ks)
        ntok = None
        m = re.search(r"toks?=(\d+)|(\d+)\s*tokens?", fname)
        if m:
            ntok = int(m.group(1) or m.group(2))
        rows.append(dict(name=fname, tokens=ntok, cpu_ms=fdur / 1000,
                         kernel_ms=ksum / 1000, wall_ms=span / 1000,
                         nkern=len(ks), nstream=len({s for _, _, s in ks})))

    if args.min_tokens:
        rows = [r for r in rows if (r["tokens"] or 0) >= args.min_tokens]

    print(f"{'#':>3} {'tokens':>7} {'kernels':>8} {'streams':>7} "
          f"{'Σkernel(ms)':>12} {'GPUwall(ms)':>12} {'overlap':>8}")
    for i, r in enumerate(rows, 1):
        ov = r["kernel_ms"] / r["wall_ms"] if r["wall_ms"] else 0
        print(f"{i:>3} {str(r['tokens']):>7} {r['nkern']:>8} {r['nstream']:>7} "
              f"{r['kernel_ms']:>12.2f} {r['wall_ms']:>12.2f} {ov:>7.2f}x")

    if len(rows) >= 2:
        w = [r["wall_ms"] for r in rows]
        sd = statistics.stdev(w)
        print(f"\nn={len(w)}  mean={statistics.mean(w):.2f} ms  "
              f"median={statistics.median(w):.2f} ms  stdev={sd:.2f} ms "
              f"({sd / statistics.mean(w) * 100:.2f}%)  "
              f"min={min(w):.2f}  max={max(w):.2f}  "
              f"range={(max(w) - min(w)) / statistics.mean(w) * 100:.1f}%")


if __name__ == "__main__":
    main()
