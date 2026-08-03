#!/usr/bin/env python3
"""forward_overlap.py

Per-forward wall time vs kernel time, and how much of the gap between them is
stream overlap rather than idle GPU.

per_forward_wall.py answers the same question from correlation-linked kernels,
which graph-replayed launches do not have -- it sees 11 of MI355X's 1649 decode
kernels. This one takes every kernel whose timestamp falls inside the forward's
GPU annotation, so it works on graph-ON traces, and it separates the two ways
Sigma_kernel and wall can disagree:

    Sigma_kernel   sum of every kernel's duration, streams counted separately
    union_busy     wall time with >= 1 kernel running anywhere
    overlap        Sigma_kernel / union_busy   (1.0 = nothing runs concurrently)
    idle           wall - union_busy           (GPU with nothing to run)

A forward can be slow because its kernels are slow (Sigma_kernel), because they
do not overlap (overlap ~ 1.0 while the other side is higher), or because the
GPU stalls between them (idle). The three are independent and the fix differs.

The profiler projects one GPU annotation onto every stream that has work in the
window, so the projections of a single forward are merged here the same way
analyze/sglang_trace.py merges them.

    python3 forward_overlap.py TRACE --match "DECODE bs=64"
"""

import argparse
import gzip
import json
import statistics
import sys


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def merged_windows(events, match):
    wins = []
    for e in events:
        if e.get("cat") != "gpu_user_annotation" or e.get("ph") != "X":
            continue
        if match not in e.get("name", ""):
            continue
        wins.append((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)), e["name"]))
    wins.sort()
    out = []
    for s, t, n in wins:
        if out and n == out[-1][2] and s < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], t), n)
        else:
            out.append((s, t, n))
    return out


def union(intervals):
    if not intervals:
        return 0.0
    intervals.sort()
    total, cs, ce = 0.0, *intervals[0]
    for s, e in intervals[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + ce - cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--match", required=True)
    a = ap.parse_args()

    events = load(a.trace).get("traceEvents", [])
    wins = merged_windows(events, a.match)
    if not wins:
        sys.exit(f"no gpu_user_annotation matching {a.match!r}")

    kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]

    print(f"{'#':>2} {'wall(ms)':>9} {'Sigma_k(ms)':>12} {'union(ms)':>10} "
          f"{'overlap':>8} {'idle(ms)':>9} {'streams':>8} {'kernels':>8}")
    walls = []
    for i, (s, e, _) in enumerate(wins, 1):
        inside = [k for k in kernels if s <= k["ts"] < e]
        sig = sum(k.get("dur", 0) for k in inside) / 1000
        uni = union([(k["ts"], k["ts"] + k.get("dur", 0)) for k in inside]) / 1000
        wall = (e - s) / 1000
        walls.append(wall)
        ns = len({k.get("args", {}).get("stream") for k in inside})
        print(f"{i:>2} {wall:>9.2f} {sig:>12.2f} {uni:>10.2f} "
              f"{sig / uni if uni else 0:>7.2f}x {wall - uni:>9.2f} {ns:>8} {len(inside):>8}")

    if len(walls) > 1:
        print(f"\nn={len(walls)}  median wall={statistics.median(walls):.2f} ms  "
              f"min={min(walls):.2f}  max={max(walls):.2f}")


if __name__ == "__main__":
    main()
