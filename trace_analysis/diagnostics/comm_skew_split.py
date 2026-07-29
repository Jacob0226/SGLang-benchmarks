#!/usr/bin/env python3
"""
comm_skew_split.py

Split the cost of a TP all-reduce inside one forward into
  (a) pure communication  ~= per-call MIN over all TP ranks
      (all ranks' twoshot kernels end together, so the rank that STARTS last
       barely waits; its duration is the closest thing to real data movement)
  (b) skew / idle wait    == per-rank duration - that MIN

Feed it the N per-rank traces of the SAME run and a window label substring.

Usage:
  python comm_skew_split.py --stack sglang --match "bs=3" --traces T0 T1 T2 T3
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
import statistics as st
from collections import Counter

COMM_RE = re.compile(r"allreduce_prototype|quickreduce|reduce_scatter_cross_device", re.I)


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        t = json.load(f)
    return t if isinstance(t, list) else t.get("traceEvents", [])


def comm_seq(path, stack, match, phase, pick):
    evs = load(path)
    kern = [e for e in evs if isinstance(e, dict) and e.get("ph") == "X"
            and str(e.get("cat", "")).lower() == "kernel"]
    dom = Counter((k["pid"], k["tid"]) for k in kern).most_common(1)[0][0]
    kern = [{"name": k.get("name", "?"), "ts": float(k["ts"]),
             "dur": float(k.get("dur", 0.0))}
            for k in kern if (k["pid"], k["tid"]) == dom]
    kern.sort(key=lambda k: k["ts"])
    pref = ("step[",) if stack == "sglang" else ("prefill[", "decode[")
    want = ("EXTEND" if phase == "prefill" else "DECODE") if stack == "sglang" else ""
    ann = sorted([{"name": a.get("name", "?"), "ts": float(a["ts"])}
                  for a in evs if isinstance(a, dict) and a.get("ph") == "X"
                  and a.get("cat") == "gpu_user_annotation"
                  and (a["pid"], a["tid"]) == dom
                  and any(a.get("name", "").startswith(p) for p in pref)],
                 key=lambda a: a["ts"])
    mts = [m["ts"] for m in ann]
    kts = [k["ts"] for k in kern]
    hits = [i for i, m in enumerate(ann) if want in m["name"] and match in m["name"]]
    if not hits:
        raise SystemExit(f"no window matching {match!r} in {path}")
    i = hits[pick]
    s0 = mts[i]
    j = bisect.bisect_right(mts, s0)
    s1 = mts[j] if j < len(mts) else kts[-1] + kern[-1]["dur"] + 1
    lo, hi = bisect.bisect_left(kts, s0), bisect.bisect_left(kts, s1)
    win = kern[lo:hi]
    tot = sum(k["dur"] for k in win)
    seq = [(k["ts"], k["dur"]) for k in win if COMM_RE.search(k["name"])]
    return ann[i]["name"], tot, seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--stack", choices=["sglang", "atom"], default="sglang")
    ap.add_argument("--phase", choices=["prefill", "decode"], default="prefill")
    ap.add_argument("--match", default="bs=3")
    ap.add_argument("--pick", type=int, default=0)
    ap.add_argument("--dump", type=int, default=0, help="print first N per-call rows")
    args = ap.parse_args()

    seqs, labels, tots = [], [], []
    for p in args.traces:
        lab, tot, seq = comm_seq(p, args.stack, args.match, args.phase, args.pick)
        labels.append(lab); tots.append(tot); seqs.append([d for _, d in seq])
        print(f"rank{len(seqs)-1}: {lab}  Sigma_kernel={tot/1e3:.2f}ms  "
              f"ncomm={len(seq)}  comm_sum={sum(d for _, d in seq)/1e3:.2f}ms  "
              f"median={st.median([d for _, d in seq]):.1f}us")
    n = min(len(s) for s in seqs)
    if len(set(len(s) for s in seqs)) != 1:
        print(f"[warn] differing comm counts {[len(s) for s in seqs]}; using first {n}")
    mins = [min(s[i] for s in seqs) for i in range(n)]
    maxs = [max(s[i] for s in seqs) for i in range(n)]
    print(f"\n=== aligned over {len(seqs)} ranks, {n} comm calls ===")
    print(f"pure-comm  (Sigma of per-call MIN over ranks) = {sum(mins)/1e3:.2f} ms "
          f"(median per call {st.median(mins):.1f} us)")
    print(f"slowest    (Sigma of per-call MAX over ranks) = {sum(maxs)/1e3:.2f} ms")
    for ri, s in enumerate(seqs):
        wait = sum(s[i] - mins[i] for i in range(n))
        print(f"  rank{ri}: comm={sum(s[:n])/1e3:8.2f} ms   of which idle-wait="
              f"{wait/1e3:7.2f} ms  ({100*wait/sum(s[:n]):.1f}%)")
    if args.dump:
        print(f"\n{'idx':>4} " + " ".join(f"{'r'+str(i)+'_us':>9}" for i in range(len(seqs)))
              + f" {'min_us':>9}")
        for i in range(min(args.dump, n)):
            print(f"{i:>4} " + " ".join(f"{s[i]:>9.1f}" for s in seqs)
                  + f" {mins[i]:>9.1f}")


if __name__ == "__main__":
    main()
