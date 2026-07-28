#!/usr/bin/env python3
"""
comm_trace_check.py

Independent (no-excel) verification of TP all-reduce / comm kernel cost inside a
single forward, for SGLang and ATOM traces.

For every forward window (delimited by gpu_user_annotation wrappers
"step[EXTEND ...]" / "step[DECODE ...]" for SGLang, "prefill[...]" / "decode[...]"
for ATOM) it reports:
  - the wrapper label (bs / toks)
  - total kernel time in the window
  - comm kernel count, Sigma, mean/median/min/max per call

Optionally dumps the full per-call duration list of the chosen forward so you can
see the layer-by-layer distribution and spot outliers (skew absorbed by
all-reduce, first-call warmup, ...).

Usage:
  python comm_trace_check.py --trace T.json.gz --stack sglang [--phase prefill]
                             [--match "toks=16"] [--dump] [--all-streams]
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
import statistics as st
from collections import Counter

COMM_RE = re.compile(
    r"reduce_scatter|all_?reduce|allgather|all_gather|nccl|rccl|cross_device|"
    r"quick.*reduce|custom_all",
    re.I,
)


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        t = json.load(f)
    return t if isinstance(t, list) else t.get("traceEvents", [])


def collect(evs, all_streams=False):
    kern = [e for e in evs
            if isinstance(e, dict) and e.get("ph") == "X"
            and str(e.get("cat", "")).lower() == "kernel"]
    dom = Counter((k["pid"], k["tid"]) for k in kern).most_common(1)[0][0]
    if not all_streams:
        kern = [k for k in kern if (k["pid"], k["tid"]) == dom]
    kern = [{"name": k.get("name", "?"), "ts": float(k["ts"]),
             "dur": float(k.get("dur", 0.0)),
             "stream": (k["pid"], k["tid"])} for k in kern]
    kern.sort(key=lambda k: k["ts"])
    ann = [{"name": a.get("name", "?"), "ts": float(a["ts"]),
            "dur": float(a.get("dur", 0.0))}
           for a in evs if isinstance(a, dict) and a.get("ph") == "X"
           and a.get("cat") == "gpu_user_annotation"
           and (a["pid"], a["tid"]) == dom]
    ann.sort(key=lambda a: a["ts"])
    return kern, ann, dom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--stack", choices=["sglang", "atom"], required=True)
    ap.add_argument("--phase", choices=["prefill", "decode"], default="prefill")
    ap.add_argument("--match", default=None,
                    help="only report windows whose label contains this substring")
    ap.add_argument("--dump", action="store_true",
                    help="dump per-call comm durations of the best-matching window")
    ap.add_argument("--dump-idx", type=int, default=None,
                    help="dump the N-th matching window instead of the first")
    ap.add_argument("--all-streams", action="store_true")
    args = ap.parse_args()

    kern, ann, dom = collect(load(args.trace), args.all_streams)
    if args.stack == "sglang":
        pref = ("step[",)
        want = "EXTEND" if args.phase == "prefill" else "DECODE"
    else:
        pref = ("prefill[", "decode[")
        want = "prefill[" if args.phase == "prefill" else "decode["
    marks = [a for a in ann if any(a["name"].startswith(p) for p in pref)]
    mts = [m["ts"] for m in marks]
    kts = [k["ts"] for k in kern]
    print(f"# trace={args.trace}")
    print(f"# dominant stream={dom}  kernels={len(kern)}  wrappers={len(marks)}")

    rows = []
    for i, m in enumerate(marks):
        if want not in m["name"]:
            continue
        if args.match and args.match not in m["name"]:
            continue
        s0 = m["ts"]
        j = bisect.bisect_right(mts, s0)
        s1 = mts[j] if j < len(mts) else (kts[-1] + kern[-1]["dur"] + 1)
        lo = bisect.bisect_left(kts, s0)
        hi = bisect.bisect_left(kts, s1)
        win = kern[lo:hi]
        comm = [k for k in win if COMM_RE.search(k["name"])]
        rows.append((i, m["name"], s1 - s0, sum(k["dur"] for k in win),
                     len(win), comm))

    hdr = (f"{'#':>4} {'label':<46} {'span_ms':>8} {'Sigk_ms':>8} {'nk':>5} "
           f"{'ncomm':>6} {'comm_ms':>8} {'mean_us':>8} {'med_us':>8} "
           f"{'min_us':>8} {'max_us':>8}")
    print(hdr)
    print("-" * len(hdr))
    for i, lab, span, tot, nk, comm in rows:
        d = [k["dur"] for k in comm]
        if d:
            print(f"{i:>4} {lab[:46]:<46} {span/1e3:>8.2f} {tot/1e3:>8.2f} {nk:>5} "
                  f"{len(d):>6} {sum(d)/1e3:>8.2f} {st.mean(d):>8.1f} "
                  f"{st.median(d):>8.1f} {min(d):>8.1f} {max(d):>8.1f}")
        else:
            print(f"{i:>4} {lab[:46]:<46} {span/1e3:>8.2f} {tot/1e3:>8.2f} {nk:>5} "
                  f"{0:>6}")

    if rows:
        alld = [k["dur"] for _, _, _, _, _, c in rows for k in c]
        if alld:
            print(f"\n# across {len(rows)} matching windows: ncomm={len(alld)} "
                  f"mean={st.mean(alld):.1f}us median={st.median(alld):.1f}us")
        names = Counter(k["name"][:90] for _, _, _, _, _, c in rows for k in c)
        print("# comm kernel names:")
        for n, c in names.most_common():
            print(f"    {c:>5}  {n}")

    if args.dump and rows:
        sel = rows[args.dump_idx or 0]
        print(f"\n=== per-call dump: window #{sel[0]} {sel[1]} ===")
        print(f"{'idx':>4} {'t_rel_ms':>10} {'dur_us':>9} {'stream':>16}  name")
        base = None
        for n, k in enumerate(sel[5]):
            base = k["ts"] if base is None else base
            print(f"{n:>4} {(k['ts']-base)/1e3:>10.3f} {k['dur']:>9.1f} "
                  f"{str(k['stream']):>16}  {k['name'][:60]}")


if __name__ == "__main__":
    main()
