#!/usr/bin/env python3
"""
window_kernel_diff.py

Compare per-kernel Sigma inside ONE forward window between two traces (e.g. two
TP ranks of the same run, or SGLang vs ATOM). Used to locate rank skew: if the
all-reduce kernel is much cheaper on one rank, that rank must be spending the
time somewhere else (it arrives last), and this tool says where.

--show-args additionally groups the matching kernels by launch config
(grid/block/registers/shared memory), which is how you tell "the two stacks call
the same kernel differently" apart from "the same launch is just slower".

Usage:
  python window_kernel_diff.py --a T0.json.gz --b T1.json.gz --stack sglang \
      --match "bs=3" [--top 25] [--stats quickreduce] [--show-args quickreduce]
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import statistics as st
from collections import Counter, defaultdict


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        t = json.load(f)
    return t if isinstance(t, list) else t.get("traceEvents", [])


def _window(path, stack, match, phase, pick=0):
    evs = load(path)
    kern = [e for e in evs if isinstance(e, dict) and e.get("ph") == "X"
            and str(e.get("cat", "")).lower() == "kernel"]
    dom = Counter((k["pid"], k["tid"]) for k in kern).most_common(1)[0][0]
    kern = [{"name": k.get("name", "?"), "ts": float(k["ts"]),
             "dur": float(k.get("dur", 0.0)), "args": k.get("args") or {}}
            for k in kern if (k["pid"], k["tid"]) == dom]
    kern.sort(key=lambda k: k["ts"])
    pref = ("step[",) if stack == "sglang" else ("prefill[", "decode[")
    ann = sorted([{"name": a.get("name", "?"), "ts": float(a["ts"])}
                  for a in evs if isinstance(a, dict) and a.get("ph") == "X"
                  and a.get("cat") == "gpu_user_annotation"
                  and (a["pid"], a["tid"]) == dom
                  and any(a.get("name", "").startswith(p) for p in pref)],
                 key=lambda a: a["ts"])
    want = ("EXTEND" if phase == "prefill" else "DECODE") if stack == "sglang" else ""
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
    agg = defaultdict(lambda: [0.0, 0])
    calls = defaultdict(list)
    cfg = defaultdict(list)
    for k in win:
        e = agg[k["name"]]
        e[0] += k["dur"]; e[1] += 1
        calls[k["name"]].append(k["dur"])
        a = k["args"]
        cfg[(k["name"], tuple(a.get("grid", [])), tuple(a.get("block", [])),
             a.get("registers per thread"), a.get("shared memory"))].append(k["dur"])
    return (ann[i]["name"], sum(k["dur"] for k in win), dict(agg), dict(calls),
            dict(cfg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--stack", choices=["sglang", "atom"], default="sglang")
    ap.add_argument("--stack-b", choices=["sglang", "atom"], default=None)
    ap.add_argument("--phase", choices=["prefill", "decode"], default="prefill")
    ap.add_argument("--match", default="bs=3")
    ap.add_argument("--match-b", default=None)
    ap.add_argument("--pick", type=int, default=0)
    ap.add_argument("--pick-b", type=int, default=None)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--stats", metavar="SUBSTR", action="append", default=None,
                    help="per-call stats (n/mean/median/min/max) for kernels whose "
                         "name contains SUBSTR; distinguishes 'more work' from "
                         "'same work, slower clocks'")
    ap.add_argument("--show-args", metavar="SUBSTR", action="append", default=None,
                    dest="show_args",
                    help="launch config (grid/block/regs/shmem) of kernels whose "
                         "name contains SUBSTR, grouped per config")
    args = ap.parse_args()

    la, ta, ka, ca, ga = _window(args.a, args.stack, args.match, args.phase,
                                 args.pick)
    lb, tb, kb, cb, gb = _window(args.b, args.stack_b or args.stack,
                                 args.match_b or args.match, args.phase,
                                 args.pick if args.pick_b is None else args.pick_b)
    LA, LB = args.labels
    print(f"{LA}: {la}  Sigma={ta/1e3:.2f}ms  ({len(ka)} distinct kernels)")
    print(f"{LB}: {lb}  Sigma={tb/1e3:.2f}ms  ({len(kb)} distinct kernels)")
    rows = []
    for nm in set(ka) | set(kb):
        sa, cnta = ka.get(nm, [0.0, 0])
        sb, cntb = kb.get(nm, [0.0, 0])
        rows.append((sa - sb, sa, cnta, sb, cntb, nm))
    rows.sort(key=lambda r: -abs(r[0]))
    print(f"\n{'d(A-B)_ms':>10} {'A_ms':>9} {'Acnt':>5} {'B_ms':>9} {'Bcnt':>5}  kernel")
    for d, sa, na, sb, nb, nm in rows[:args.top]:
        print(f"{d/1e3:>10.2f} {sa/1e3:>9.2f} {na:>5} {sb/1e3:>9.2f} {nb:>5}  {nm[:78]}")

    for sub in (args.stats or []):
        names = sorted(n for n in set(ca) | set(cb) if sub in n)
        for nm in names:
            print(f"\nper-call stats for {nm[:78]}")
            print(f"{'side':<10} {'n':>5} {'mean_us':>9} {'med_us':>9} "
                  f"{'min_us':>9} {'max_us':>9}")
            for lab, d in ((LA, ca), (LB, cb)):
                v = d.get(nm, [])
                if not v:
                    print(f"{lab:<10} {0:>5}")
                    continue
                print(f"{lab:<10} {len(v):>5} {st.mean(v):>9.1f} {st.median(v):>9.1f} "
                      f"{min(v):>9.1f} {max(v):>9.1f}")

    for sub in (args.show_args or []):
        print(f"\nlaunch configs for kernels matching {sub!r}")
        for lab, g in ((LA, ga), (LB, gb)):
            keys = [k for k in g if sub in k[0]]
            if not keys:
                print(f"  {lab}: (none)")
                continue
            for key in sorted(keys, key=lambda k: -sum(g[k])):
                nm, grid, block, regs, shm = key
                ds = g[key]
                print(f"  {lab}: n={len(ds):<4} sum_ms={sum(ds)/1e3:>8.2f} "
                      f"median_us={st.median(ds):>8.1f}  grid={grid} block={block} "
                      f"regs={regs} shmem={shm}")
                print(f"        {nm[:88]}")


if __name__ == "__main__":
    main()
