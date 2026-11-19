#!/usr/bin/env python3
"""Compare prefill (EXTEND / DRAFT) steps between two profile-by-stage EXTEND traces.

Each step is cut on the GPU side by its `step[EXTEND bs=.. toks=..]` gpu_user_annotation.
Prints, per step kind and token count, the GPU span of every captured step on both sides,
then the kernel time inside the EXTEND windows grouped by kernel name (per step, so traces
that captured a different number of steps still line up).

Usage:
  extend_step_compare.py A.trace.json.gz B.trace.json.gz [--labels 1106 1115] [--top 25]
"""
import argparse
import gzip
import json
import re
from collections import defaultdict

STEP_RE = re.compile(r"step\[(EXTEND|DRAFT) bs=(\d+) toks=(\d+)\]")


def load(path):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        return json.load(f)["traceEvents"]


def steps_and_kernels(path):
    ev = load(path)
    steps = []
    for e in ev:
        if e.get("cat") != "gpu_user_annotation":
            continue
        m = STEP_RE.fullmatch(e.get("name", ""))
        if m:
            steps.append((m[1], int(m[2]), int(m[3]), float(e["ts"]), float(e["dur"])))
    kernels = [(float(e["ts"]), float(e["dur"]), e["name"]) for e in ev if e.get("cat") == "kernel"]
    kernels.sort()
    return steps, kernels


def kernel_groups(steps, kernels, kind):
    """Kernel time per name summed over the `kind` windows, divided by the window count."""
    wins = sorted((s[3], s[3] + s[4]) for s in steps if s[0] == kind)
    tot = defaultdict(float)
    for ts, dur, name in kernels:
        if any(a <= ts < b for a, b in wins):
            tot[name] += dur
    n = max(len(wins), 1)
    return {k: v / n for k, v in tot.items()}, len(wins)


def short(name, width=90):
    name = re.sub(r"\s+", " ", name)
    return name if len(name) <= width else name[: width - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--labels", nargs=2, default=("A", "B"))
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    la, lb = args.labels

    sa, ka = steps_and_kernels(args.a)
    sb, kb = steps_and_kernels(args.b)

    print(f"GPU span per step (ms): {la} | {lb}")
    keys = sorted({(s[0], s[1], s[2]) for s in sa + sb})
    for k in keys:
        da = [s[4] / 1e3 for s in sa if s[:3] == k]
        db = [s[4] / 1e3 for s in sb if s[:3] == k]
        fa = " ".join(f"{x:7.1f}" for x in da) or "      -"
        fb = " ".join(f"{x:7.1f}" for x in db) or "      -"
        print(f"  {k[0]:6s} bs={k[1]} toks={k[2]:6d}:  {fa}  |  {fb}")

    for kind in ("EXTEND", "DRAFT"):
        ga, na = kernel_groups(sa, ka, kind)
        gb, nb = kernel_groups(sb, kb, kind)
        if not (na and nb):
            continue
        ta, tb = sum(ga.values()), sum(gb.values())
        print(f"\n{kind}: kernel time per step {la} {ta/1e3:.2f} ms ({na} steps) | "
              f"{lb} {tb/1e3:.2f} ms ({nb} steps) | delta {(tb-ta)/1e3:+.2f} ms ({(tb/ta-1)*100:+.1f}%)")
        names = sorted(set(ga) | set(gb), key=lambda n: -abs(gb.get(n, 0) - ga.get(n, 0)))
        print(f"  {'delta us':>9} {la:>9} {lb:>9}  kernel")
        for n in names[: args.top]:
            a, b = ga.get(n, 0.0), gb.get(n, 0.0)
            print(f"  {b-a:+9.0f} {a:9.0f} {b:9.0f}  {short(n)}")


if __name__ == "__main__":
    main()
