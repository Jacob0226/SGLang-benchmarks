#!/usr/bin/env python3
"""
verify_step3_against_trace.py

Independently re-derive, straight from the raw trace, what a
step3_layer_breakdown workbook claims, and diff the two.

The workbook is built as  Σ = (per-call avg inside ONE forward) x (the layers that
really run that call site), so for every kernel name we check:
  * AvgDuration_us  == the kernel's mean duration inside that same forward
  * Σ SumDuration_us == the kernel's total inside that forward
    (a gap here is legitimate only when the kernel also runs OUTSIDE the decoder
     layers, e.g. lm_head / sampling / logits all-reduce, which step3 does not
     cover — those show up as "in trace, not in xlsx")

Window selection is re-implemented here (annotation wrapper + substring match)
rather than imported, so a bug in the analyzer's own segmentation cannot hide.

Usage:
  python verify_step3_against_trace.py --step3 step3_....xlsx --trace T.json.gz \
      --stack sglang --forward-match "bs=3" [--pick median] [--tol 1.0]
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import sys
from collections import Counter, defaultdict

from openpyxl import load_workbook

PREFIXES = ("step[", "prefill[", "decode[")


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        t = json.load(f)
    return t if isinstance(t, list) else t.get("traceEvents", [])


def forward_kernels(path, match, pick):
    evs = load(path)
    kern = [e for e in evs if isinstance(e, dict) and e.get("ph") == "X"
            and str(e.get("cat", "")).lower() == "kernel"]
    if not kern:
        sys.exit("[ERROR] no kernel events")
    dom = Counter((k["pid"], k["tid"]) for k in kern).most_common(1)[0][0]
    kern = [{"name": k.get("name", "?"), "ts": float(k["ts"]),
             "dur": float(k.get("dur", 0.0))}
            for k in kern if (k["pid"], k["tid"]) == dom]
    kern.sort(key=lambda k: k["ts"])
    kts = [k["ts"] for k in kern]

    wraps = sorted([(float(a["ts"]), float(a.get("dur", 0.0)), a.get("name", ""))
                    for a in evs if isinstance(a, dict) and a.get("ph") == "X"
                    and a.get("cat") == "gpu_user_annotation"
                    and (a["pid"], a["tid"]) == dom
                    and str(a.get("name", "")).startswith(PREFIXES)])
    cands = []
    for i, (ts, dur, name) in enumerate(wraps):
        if match not in name:
            continue
        # SGLang's step[...] wrapper spans the whole forward; ATOM's prefill[...]
        # is a point marker (dur~0) that must be closed by the next wrapper.
        nxt = wraps[i + 1][0] if i + 1 < len(wraps) else kts[-1] + kern[-1]["dur"] + 1
        end = ts + dur if dur > 0.5 * (nxt - ts) else nxt
        lo, hi = bisect.bisect_left(kts, ts), bisect.bisect_left(kts, min(end, nxt))
        win = kern[lo:hi]
        cands.append((sum(k["dur"] for k in win), name, win))
    if not cands:
        sys.exit(f"[ERROR] no forward wrapper matching {match!r}")
    cands.sort(key=lambda c: c[0])
    if pick == "first":
        chosen = cands[0]
    elif pick == "max":
        chosen = cands[-1]
    elif pick == "min":
        chosen = cands[0]
    else:
        chosen = cands[len(cands) // 2]
    agg = defaultdict(lambda: [0.0, 0])
    for k in chosen[2]:
        e = agg[k["name"]]
        e[0] += k["dur"]; e[1] += 1
    return chosen[1], chosen[0], len(chosen[2]), dict(agg), len(cands)


def read_step3(path):
    ws = load_workbook(path, data_only=True).active
    rows = list(ws.iter_rows(values_only=True))
    h = {n: i for i, n in enumerate(rows[0]) if n}
    agg = defaultdict(lambda: [0.0, 0, set()])
    nrows = 0
    for r in rows[1:]:
        nm = r[h["KernelName"]]
        s = r[h["SumDuration_us"]]
        if nm is None or not isinstance(s, (int, float)):
            continue
        nrows += 1
        e = agg[nm]
        e[0] += s
        c = r[h["Count"]]
        e[1] += c if isinstance(c, int) else 0
        a = r[h["AvgDuration_us"]]
        if isinstance(a, (int, float)):
            e[2].add(round(a, 3))
    return nrows, dict(agg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step3", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--forward-match", required=True)
    ap.add_argument("--pick", default="median")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="avg mismatch tolerance in %% (default 1.0)")
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()

    label, wtot, wn, tr, ncand = forward_kernels(args.trace, args.forward_match,
                                                 args.pick)
    nrows, xl = read_step3(args.step3)
    xtot = sum(v[0] for v in xl.values())
    print(f"trace forward : {label}  ({args.pick}-of-{ncand})")
    print(f"                Σ={wtot/1e3:.2f} ms  kernels={wn}  "
          f"distinct names={len(tr)}")
    print(f"xlsx          : {args.step3.split('/')[-1]}")
    print(f"                Σ={xtot/1e3:.2f} ms  rows={nrows}  "
          f"distinct names={len(xl)}")

    bad_avg, only_trace, only_xlsx, ok = [], [], [], 0
    for nm, (xs, xc, xavgs) in xl.items():
        t = tr.get(nm)
        if t is None:
            only_xlsx.append((xs, nm))
            continue
        tavg = t[0] / t[1]
        for xa in xavgs:
            if tavg == 0 or abs(xa - tavg) / tavg * 100 > args.tol:
                bad_avg.append((abs(xa - tavg) / max(tavg, 1e-9) * 100, nm, xa, tavg))
                break
        else:
            ok += 1
    for nm, (ts, tc) in tr.items():
        if nm not in xl:
            only_trace.append((ts, nm))

    print(f"\nper-call avg check: {ok}/{len(xl)} kernel names match the trace "
          f"within {args.tol}%")
    if bad_avg:
        print(f"  MISMATCHED ({len(bad_avg)}):")
        for d, nm, xa, ta in sorted(bad_avg, reverse=True)[:args.top]:
            print(f"    {d:6.1f}%  xlsx={xa:9.1f}us  trace={ta:9.1f}us  {nm[:70]}")
    if only_xlsx:
        print(f"  in xlsx but NOT in this forward ({len(only_xlsx)}):")
        for s, nm in sorted(only_xlsx, reverse=True)[:args.top]:
            print(f"    {s/1e3:8.2f} ms  {nm[:70]}")
    if only_trace:
        tot = sum(s for s, _ in only_trace)
        print(f"  in trace but NOT in xlsx ({len(only_trace)}, Σ={tot/1e3:.2f} ms) "
              f"— expected for kernels outside the decoder layers:")
        for s, nm in sorted(only_trace, reverse=True)[:args.top]:
            print(f"    {s/1e3:8.2f} ms  {nm[:70]}")

    print(f"\nΣ per kernel name (trace vs xlsx), sorted by trace Σ:")
    print(f"  {'trace_ms':>9} {'n':>5} {'avg_us':>9} | {'xlsx_ms':>9} {'n':>5} "
          f"| {'dΣ_ms':>8} {'dΣ%':>7}  kernel")
    for nm, (ts, tc) in sorted(tr.items(), key=lambda kv: -kv[1][0])[:args.top]:
        xs, xc, _ = xl.get(nm, (0.0, 0, set()))
        d = xs - ts
        print(f"  {ts/1e3:>9.2f} {tc:>5} {ts/tc:>9.1f} | {xs/1e3:>9.2f} {xc:>5} "
              f"| {d/1e3:>8.2f} {100*d/ts:>6.1f}%  {nm[:60]}")


if __name__ == "__main__":
    main()
