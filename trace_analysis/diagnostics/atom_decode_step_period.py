#!/usr/bin/env python3
"""Split an ATOM graph-ON decode step into verify forward vs everything after it.

`decode[...]` only wraps the target verify forward; the MTP draft passes run
between its end and the next `decode[` start. This reports, per full-batch step:
period (decode[ start -> next decode[ start), verify span, draft kernel time in
the remainder, and idle GPU in the remainder. Median step is printed; the
period is what median ITL measures.

Usage:
  atom_decode_step_period.py TRACE.json.gz --bs 64
"""
import argparse
import bisect
import gzip
import json


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--bs", type=int, required=True, help="full decode batch size")
    args = p.parse_args()

    ev = json.load(gzip.open(args.trace, "rt"))
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    ks = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"),
                key=lambda e: e["ts"])
    steps = sorted((e for e in ev if e.get("ph") == "X"
                    and e.get("cat") == "gpu_user_annotation"
                    and e["name"].startswith("decode[")), key=lambda e: e["ts"])
    kts = [e["ts"] for e in ks]
    rows = []
    for a, b in zip(steps, steps[1:]):
        if not a["name"].startswith(f"decode[bs={args.bs} "):
            continue
        v_end = a["ts"] + a["dur"]
        lo, hi = bisect.bisect_left(kts, v_end), bisect.bisect_left(kts, b["ts"])
        draft = sum(e["dur"] for e in ks[lo:hi])
        rows.append((b["ts"] - a["ts"], a["dur"], b["ts"] - v_end, draft))
    if not rows:
        raise SystemExit(f"no full-batch decode[bs={args.bs} ...] step followed by another step")
    rows.sort()
    period, verify, after, draft = (x / 1e3 for x in rows[len(rows) // 2])
    print(f"steps={len(rows)} period={period:.2f} ms verify={verify:.2f} ms "
          f"after-verify={after:.2f} ms (draft kernels {draft:.2f} ms, idle {after - draft:.2f} ms)")


if __name__ == "__main__":
    main()
