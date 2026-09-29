#!/usr/bin/env python3
"""List decode kernels grouped by launches-per-forward.

GLM-5.3-Flash layer counts make the per-forward launch count a fingerprint:
42 = MoE layers (router, expert GEMMs, sorting), 34 = KDA linear-attention
layers, 11 = MLA/DSA sparse layers, 3 = dense MLP layers, 45 = all layers.

Used to find the router GEMM at conc64, where it leaves the triton path and is
dispatched through aiter.tuned_gemm instead.
"""

import gzip
import json
import sys
from collections import Counter, defaultdict

path = sys.argv[1]
want = int(sys.argv[2]) if len(sys.argv) > 2 else None

with gzip.open(path, "rt") as f:
    ev = json.load(f)["traceEvents"]

# step annotations are emitted on both the CPU and GPU timelines, so the
# forward count is half the number of step[...] events.
steps = [e for e in ev if isinstance(e.get("name"), str) and e["name"].startswith("step[")]
nfwd = max(1, len(steps) // 2)
print(f"{len(steps)} step events -> {nfwd} forwards")

kern = [
    e for e in ev
    if e.get("ph") == "X" and e.get("cat") == "kernel" and e.get("dur") is not None
]
agg = defaultdict(list)
for e in kern:
    agg[e["name"]].append(e["dur"])

rows = []
for name, durs in agg.items():
    per_fwd = len(durs) / nfwd
    durs.sort()
    rows.append((per_fwd, len(durs), durs[len(durs) // 2], sum(durs) / 1000.0, name))

rows.sort(key=lambda r: -r[3])
if want is not None:
    rows = [r for r in rows if abs(r[0] - want) < 0.01]
    print(f"\n=== kernels firing exactly {want}x per forward ===")
else:
    print("\n=== all kernels, by total time ===")

print(f"{'per_fwd':>8} {'n':>5} {'med_us':>8} {'sum_ms':>8}  name")
for per_fwd, n, med, tot, name in rows[:40]:
    print(f"{per_fwd:>8.1f} {n:>5} {med:>8.2f} {tot:>8.3f}  {name[:110]}")
