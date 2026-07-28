#!/usr/bin/env python3
"""Dump raw trace events (incl. args: grid/block, stream, ...) for comm kernels.

Groups by (kernel name, grid, block) and reports count + duration stats, so an
SGLang vs ATOM launch-config difference (e.g. number of workgroups used by
quickreduce twoshot) becomes visible.

Usage: python comm_kernel_args.py --trace T.json.gz [--limit-print 3]
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics as st
from collections import defaultdict

COMM_RE = re.compile(r"reduce_scatter|all_?reduce|allgather|all_gather|nccl|rccl|"
                     r"cross_device|quick.*reduce|custom_all", re.I)


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        t = json.load(f)
    return t if isinstance(t, list) else t.get("traceEvents", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--limit-print", type=int, default=2)
    args = ap.parse_args()

    evs = load(args.trace)
    groups = defaultdict(list)
    samples = {}
    for e in evs:
        if not (isinstance(e, dict) and e.get("ph") == "X"
                and str(e.get("cat", "")).lower() == "kernel"):
            continue
        nm = e.get("name", "")
        if not COMM_RE.search(nm):
            continue
        a = e.get("args", {}) or {}
        key = (nm[:70], tuple(a.get("grid", [])), tuple(a.get("block", [])),
               a.get("registers per thread"), a.get("shared memory"))
        groups[key].append(float(e.get("dur", 0.0)))
        samples.setdefault(key, e)

    print(f"# {args.trace}")
    for key, ds in sorted(groups.items(), key=lambda kv: -sum(kv[1])):
        nm, grid, block, regs, shm = key
        print(f"\ncount={len(ds)}  sum_ms={sum(ds)/1e3:.2f}  mean_us={st.mean(ds):.1f} "
              f"median_us={st.median(ds):.1f}")
        print(f"  grid={grid} block={block} regs={regs} shmem={shm}")
        print(f"  name={nm}")
        if args.limit_print:
            print(f"  sample_event={json.dumps(samples[key])[:600]}")


if __name__ == "__main__":
    main()
