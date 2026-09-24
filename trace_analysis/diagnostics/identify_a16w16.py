#!/usr/bin/env python3
"""Identify what the 42-launch _gemm_a16_w16_kernel in the decode trace actually is.

Prints its grid/block dims and the enclosing python annotations, which together
pin the (M, N, K) of the GEMM.
"""

import gzip
import json
import sys
from collections import Counter, defaultdict

path = sys.argv[1]
needle = sys.argv[2] if len(sys.argv) > 2 else "_gemm_a16_w16_kernel"

with gzip.open(path, "rt") as f:
    trace = json.load(f)
events = trace["traceEvents"]

kern = [e for e in events if needle in (e.get("name") or "")]
print(f"{len(kern)} launches of *{needle}*")
if not kern:
    sys.exit(0)

names = Counter(e["name"] for e in kern)
for n, c in names.most_common():
    print(f"\n=== {c} launches ===\n{n}")
    sample = next(e for e in kern if e["name"] == n)
    args = sample.get("args", {})
    for k in ("grid", "block", "registers per thread", "shared memory",
              "warps per SM", "est. achieved occupancy %", "stream",
              "device", "queue id"):
        if k in args:
            print(f"  {k}: {args[k]}")
    dur = [e["dur"] for e in kern if e["name"] == n]
    dur.sort()
    print(f"  dur us: min={dur[0]} med={dur[len(dur)//2]} max={dur[-1]} "
          f"sum={sum(dur)/1000:.3f} ms")

# Walk the CPU-side launches to find the enclosing user annotation.
flow_start = {e["id"]: e for e in events
              if e.get("ph") == "s" and e.get("cat") in ("ac2g", "async_gpu")}
ext_to_kernel = defaultdict(list)
for e in kern:
    ext = e.get("args", {}).get("External id") or e.get("args", {}).get("correlation")
    if ext is not None:
        ext_to_kernel[ext].append(e)

cpu = [e for e in events
       if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation",
                                                  "gpu_user_annotation", "python_function")]
cpu.sort(key=lambda e: e["ts"])

# For a handful of kernels, find CPU ops whose time range encloses the launch.
print("\n=== enclosing CPU annotations (first 3 launches) ===")
launch_ops = [e for e in events
              if e.get("cat") == "cuda_runtime" or e.get("cat") == "hip_runtime"]
by_corr = {}
for e in launch_ops:
    c = e.get("args", {}).get("correlation")
    if c is not None:
        by_corr[c] = e

shown = 0
for k in kern:
    corr = k.get("args", {}).get("correlation")
    l = by_corr.get(corr)
    if l is None:
        continue
    t0 = l["ts"]
    encl = [e["name"] for e in cpu
            if e["ts"] <= t0 <= e["ts"] + e.get("dur", 0)
            and e.get("cat") in ("cpu_op", "user_annotation", "python_function")]
    print(f"\nlaunch @{t0}: {' > '.join(encl[-6:]) if encl else '(none)'}")
    shown += 1
    if shown >= 3:
        break
