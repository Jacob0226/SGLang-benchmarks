#!/usr/bin/env python3
"""Is there a per-kernel duration floor inside the decode HIP graph?

Matters for split-K tuning: the router GEMM's split-K reduce kernel costs
4.32 us to reduce 18 KB, which is far too slow for the work involved. If every
kernel in the graph pays a ~4 us floor then NUM_KSPLIT>1 is charged a second
floor and a slower single-kernel config can still win.
"""

import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    trace = json.load(f)

# GPU kernels only: the profiler puts them on a device/stream pair.
kern = [
    e for e in trace["traceEvents"]
    if e.get("ph") == "X"
    and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    and e.get("dur") is not None
]
print(f"{len(kern)} gpu events")

by_name = defaultdict(list)
for e in kern:
    by_name[e["name"]].append(e["dur"])

meds = sorted((sorted(v)[len(v) // 2], len(v), n) for n, v in by_name.items())
print(f"{len(meds)} distinct kernels\n")

print("--- 20 fastest kernels by median duration ---")
for med, cnt, n in meds[:20]:
    print(f"  {med:8.3f} us  n={cnt:<5d} {n[:96]}")

allsorted = sorted(e["dur"] for e in kern)
print(f"\nabsolute min single event: {allsorted[0]:.3f} us")
for p in (0.1, 1, 5, 10, 25, 50):
    i = int(len(allsorted) * p / 100)
    print(f"  p{p:<5} {allsorted[i]:8.3f} us")

below = [m for m, _, _ in meds if m < 4.0]
print(f"\nkernels with median < 4.0 us: {len(below)} / {len(meds)}")
below2 = [m for m, _, _ in meds if m < 2.0]
print(f"kernels with median < 2.0 us: {len(below2)} / {len(meds)}")
