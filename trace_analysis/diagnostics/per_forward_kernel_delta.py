#!/usr/bin/env python3
"""Per-kernel accounting of a decode-forward regression between two traces.

Normalizes by the number of captured forwards (traces often differ: profile-by-stage
may capture 1 forward in one run and 2 in another), then attributes the wall-clock
delta per forward to individual kernels as (mean_b - mean_a) * launches_per_forward.

Usage:
  per_forward_kernel_delta.py --a BASE.json.gz --b VARIANT.json.gz \
      --anchor reduce_scatter_cross_device_store --anchor-count 155 [--top 20]

--anchor/--anchor-count give the forward count: a kernel with a known number of
launches per forward (default: the fused all-reduce, 155 per GLM-5.2 forward).
"""
import argparse
import gzip
import json
from collections import defaultdict


def load_kernels(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        trace = json.load(f)
    # GPU kernels carry a device stream; filter out CPU-side ops and metadata.
    out = defaultdict(list)
    for e in trace.get("traceEvents", []):
        if e.get("ph") != "X" or "dur" not in e:
            continue
        cat = e.get("cat", "")
        if cat not in ("kernel", "Kernel", "gpu_user_annotation", "gpu_memcpy"):
            continue
        if cat != "kernel" and cat != "Kernel":
            continue
        out[e["name"]].append(e["dur"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline trace")
    ap.add_argument("--b", required=True, help="variant trace")
    ap.add_argument("--labels", nargs=2, default=("A", "B"))
    ap.add_argument("--anchor", default="reduce_scatter_cross_device_store")
    ap.add_argument("--anchor-count", type=int, default=155)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    ka, kb = load_kernels(args.a), load_kernels(args.b)

    def forwards(k, which):
        n = sum(len(v) for name, v in k.items() if args.anchor in name)
        if not n:
            raise SystemExit(f"{which}: anchor {args.anchor!r} not found")
        return n / args.anchor_count

    fa, fb = forwards(ka, args.labels[0]), forwards(kb, args.labels[1])
    ta = sum(sum(v) for v in ka.values()) / fa / 1000
    tb = sum(sum(v) for v in kb.values()) / fb / 1000
    la, lb = args.labels

    print(f"{la}: {fa:g} forward(s), {ta:.3f} ms of kernel time per forward")
    print(f"{lb}: {fb:g} forward(s), {tb:.3f} ms of kernel time per forward")
    print(f"delta: {(tb - ta) * 1000:+.0f} us per forward ({100 * (tb / ta - 1):+.1f}%)\n")

    rows = []
    for name in set(ka) | set(kb):
        va, vb = ka.get(name, []), kb.get(name, [])
        na, nb = len(va) / fa, len(vb) / fb
        ma = sum(va) / len(va) if va else 0.0
        mb = sum(vb) / len(vb) if vb else 0.0
        contrib = (sum(vb) / fb if vb else 0.0) - (sum(va) / fa if va else 0.0)
        rows.append((contrib, name, na, nb, ma, mb))
    rows.sort(key=lambda r: -abs(r[0]))

    print(f"{'us/fwd':>9} {'launch/fwd':>12} {'mean us':>17}   kernel")
    print(f"{'delta':>9} {la[:5]:>6}{lb[:5]:>6} {la[:7]:>8}{lb[:8]:>9}")
    for contrib, name, na, nb, ma, mb in rows[: args.top]:
        print(f"{contrib:>+9.1f} {na:>6.0f}{nb:>6.0f} {ma:>8.2f}{mb:>9.2f}   {name[:72]}")

    top = sum(r[0] for r in rows[: args.top])
    print(f"\ntop {args.top} account for {top:+.1f} us of {(tb - ta) * 1000:+.1f} us "
          f"({100 * top / (tb - ta) if tb != ta else 0:.0f}%)")


if __name__ == "__main__":
    main()
