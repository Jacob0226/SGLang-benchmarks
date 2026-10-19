"""Summarize a steptime.jsonl (sglang/srt/steptime.py) over a time range.

  python3 steptime_report.py FILE [--skip-min 25] [--last-min 0]

Prints the GPU-time split (decode / prefill / idle) and the prefill cost by DCP
path and by batch prefix size, aggregated over the selected windows.
"""

import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--skip-min", type=float, default=0, help="drop the first N minutes")
    ap.add_argument("--last-min", type=float, default=0, help="keep only the last N minutes")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.file) if l.strip()]
    t0 = rows[0]["ts"] - rows[0]["wall_s"]
    rows = [r for r in rows if r["ts"] - t0 >= args.skip_min * 60]
    if args.last_min:
        rows = [r for r in rows if rows[-1]["ts"] - r["ts"] <= args.last_min * 60]
    if not rows:
        raise SystemExit("no windows in range")

    wall = sum(r["wall_s"] for r in rows)
    # MTP rewrites batch.forward_mode to DRAFT_EXTEND_V2 inside the forward,
    # so spec decode rounds land in "other" (bs not recorded).
    dec = {k: sum(r["decode"][k] + r["other"].get(k, 0) for r in rows) for k in ("n", "ms", "bs")}
    ext = {k: sum(r["extend"][k] for r in rows) for k in ("n", "ms", "tokens", "prefix", "seqs")}
    busy = (dec["ms"] + ext["ms"]) / 1e3
    print(f"windows={len(rows)} wall={wall:.0f}s")
    print(f"decode : {dec['ms'] / 1e3:8.1f}s {100 * dec['ms'] / 1e3 / wall:5.1f}%  "
          f"n={dec['n']} avg step={dec['ms'] / max(dec['n'], 1):.1f}ms")
    print(f"prefill: {ext['ms'] / 1e3:8.1f}s {100 * ext['ms'] / 1e3 / wall:5.1f}%  "
          f"n={ext['n']} tokens={ext['tokens'] / 1e6:.1f}M "
          f"({ext['tokens'] / max(ext['ms'] / 1e3, 1e-9):.0f} tok/s while prefilling) "
          f"avg prefix/batch={ext['prefix'] / max(ext['n'], 1) / 1e3:.0f}k")
    print(f"idle   : {wall - busy:8.1f}s {100 * (wall - busy) / wall:5.1f}%  (GPU stream not in a timed forward)")

    for key, title in (("paths", "prefill by DCP path"), ("prefix_buckets", "prefill by batch prefix")):
        agg = defaultdict(lambda: {"n": 0, "ms": 0.0, "tokens": 0})
        for r in rows:
            for k, v in r[key].items():
                for f in ("n", "ms", "tokens"):
                    agg[k][f] += v[f]
        print(f"-- {title}")
        for k, v in sorted(agg.items(), key=lambda kv: -kv[1]["ms"]):
            print(f"   {k:16} {v['ms'] / 1e3:8.1f}s {100 * v['ms'] / max(ext['ms'], 1e-9):5.1f}%  "
                  f"n={v['n']:6} tokens={v['tokens'] / 1e6:7.2f}M "
                  f"{v['tokens'] / max(v['ms'] / 1e3, 1e-9):8.0f} tok/s")


if __name__ == "__main__":
    main()
