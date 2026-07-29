#!/usr/bin/env python3
"""Summarize GPU kernel time from a torch-profiler .trace.json.gz.

Usage:
  python3 trace_kernel_summary.py <trace.json.gz> [--filter substr] [--top N]

Prints per-kernel total/mean/count over GPU (kernel) events, sorted by total
time. Use --filter to focus (e.g. --filter main_kernel, --filter _attn_).
"""
import argparse
import gzip
import json
from collections import defaultdict


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        data = json.load(f)
    return data.get("traceEvents", data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--filter", default=None)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    events = load(args.trace)
    # GPU kernel events: cat == "kernel" (torch profiler). Fall back to ph=X on
    # a device stream if cat missing.
    tot = defaultdict(float)
    cnt = defaultdict(int)
    grand = 0.0
    for e in events:
        if e.get("cat") != "kernel":
            continue
        name = e.get("name", "?")
        dur = float(e.get("dur", 0.0))
        tot[name] += dur
        cnt[name] += 1
        grand += dur

    rows = sorted(tot.items(), key=lambda kv: kv[1], reverse=True)
    print(f"# total GPU kernel time (all): {grand/1000:.3f} ms over {sum(cnt.values())} launches")
    if args.filter:
        print(f"# filter: '{args.filter}'")
    print(f"{'total_ms':>12} {'mean_us':>10} {'count':>7}  name")
    shown = 0
    for name, t in rows:
        if args.filter and args.filter not in name:
            continue
        print(f"{t/1000:12.3f} {t/cnt[name]:10.2f} {cnt[name]:7d}  {name[:110]}")
        shown += 1
        if shown >= args.top:
            break


if __name__ == "__main__":
    main()
