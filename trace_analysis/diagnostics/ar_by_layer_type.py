#!/usr/bin/env python3
"""Split decode all-reduce calls by layer type, forked vs not.

Answers one question: when ROCm dual stream forks only the 21 full-indexer
layers, does the all-reduce slow down on *those* layers only (a per-fork cost) or
on all 78 (a global property of multi-stream graph capture)?

GLM-5.2 layer map (78 layers, from the step3 LayerTypes sheet):
  full+MLP   0,1,2
  full+MoE   6,10,...,74 (every 4th)
  shared+MoE everything else -> no indexer, so no fork after the gating patch

Per decode forward the fused all-reduce fires 155 times: layer 0 has only its MLP
all-reduce, layers 1..77 have [attn, mlp]. So AR index -> layer is
  layer 0 -> 0 (mlp);  layer i>=1 -> attn 2i-1, mlp 2i.

Usage:
  ar_by_layer_type.py --label baseline TRACE --label dual TRACE [...]
"""
import argparse
import gzip
import json
import statistics as st

AR_KERNEL = "reduce_scatter_cross_device_store"
FULL_INDEXER = {0, 1, 2} | set(range(6, 78, 4))
ARS_PER_FORWARD = 155


def load_ar_events(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        trace = json.load(f)
    evs = [
        e
        for e in trace.get("traceEvents", [])
        if e.get("ph") == "X" and AR_KERNEL in e.get("name", "")
    ]
    evs.sort(key=lambda e: e["ts"])
    return evs


def ar_index_to_layer(i):
    if i == 0:
        return 0
    return (i + 1) // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", nargs=2, metavar=("NAME", "TRACE"),
                    required=True)
    args = ap.parse_args()

    rows = []
    for name, path in args.label:
        evs = load_ar_events(path)
        nfwd, rem = divmod(len(evs), ARS_PER_FORWARD)
        if nfwd == 0:
            raise SystemExit(f"{name}: only {len(evs)} AR calls, expected >= {ARS_PER_FORWARD}")
        forked, plain = [], []
        for f in range(nfwd):
            chunk = evs[f * ARS_PER_FORWARD : (f + 1) * ARS_PER_FORWARD]
            for i, e in enumerate(chunk):
                layer = ar_index_to_layer(i)
                (forked if layer in FULL_INDEXER else plain).append(e["dur"])
        rows.append((name, nfwd, len(evs) - nfwd * ARS_PER_FORWARD, forked, plain))
        if rem:
            print(f"[warn] {name}: {rem} trailing AR calls ignored (partial forward)")

    print(f"{'variant':<12} {'fwd':>4} {'full-indexer layers (forked)':>32} {'shared layers (not forked)':>30}")
    print(f"{'':<12} {'':>4} {'n':>5} {'mean':>8} {'median':>8} {'n':>7} {'mean':>8} {'median':>8}")
    for name, nfwd, _, forked, plain in rows:
        print(f"{name:<12} {nfwd:>4} {len(forked):>5} {st.mean(forked):>8.2f} "
              f"{st.median(forked):>8.2f} {len(plain):>7} {st.mean(plain):>8.2f} "
              f"{st.median(plain):>8.2f}")

    if len(rows) == 2:
        (n0, _, _, f0, p0), (n1, _, _, f1, p1) = rows
        print()
        print(f"delta {n1} vs {n0}:  forked {st.mean(f1) - st.mean(f0):+.2f} us/call, "
              f"not-forked {st.mean(p1) - st.mean(p0):+.2f} us/call")
        print("  -> a per-fork cost would show up only in the forked column;")
        print("     a global capture cost inflates both.")


if __name__ == "__main__":
    main()
