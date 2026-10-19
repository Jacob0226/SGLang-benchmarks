"""Summarize torch caching-allocator segments in a memory snapshot.

Usage: python segment_report.py snapshot.pickle [--min-mb 64] [--frames 5]
Shows reserved vs allocated per segment size class and the sites that created
the largest segments with the most unused bytes.
"""

import argparse
import pickle
from collections import defaultdict


def _site(frames, n):
    keep = [f for f in frames if "/sglang/" in f["filename"]] or frames
    return " <- ".join(
        f"{f['filename'].split('/python/')[-1].split('site-packages/')[-1]}:{f['line']}({f['name']})"
        for f in keep[:n]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--min-mb", type=int, default=64)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    snap = pickle.load(open(args.snapshot, "rb"))
    segs = snap["segments"]
    dev = max({s["device"] for s in segs}, key=lambda d: sum(s["total_size"] for s in segs if s["device"] == d))
    segs = [s for s in segs if s["device"] == dev]

    tot = sum(s["total_size"] for s in segs)
    alloc = sum(s["allocated_size"] for s in segs)
    print(f"device {dev}: {len(segs)} segments reserved={tot / 2**30:.2f}G allocated={alloc / 2**30:.2f}G unused={(tot - alloc) / 2**30:.2f}G")

    by_pool = defaultdict(lambda: [0, 0, 0])
    for s in segs:
        k = (s.get("segment_type"), str(s.get("segment_pool_id")))
        by_pool[k][0] += s["total_size"]
        by_pool[k][1] += s["total_size"] - s["allocated_size"]
        by_pool[k][2] += 1
    for k, (t, u, n) in sorted(by_pool.items(), key=lambda kv: -kv[1][1]):
        print(f"  pool {k}: n={n} reserved={t / 2**30:.2f}G unused={u / 2**30:.2f}G")

    big = [s for s in segs if s["total_size"] - s["allocated_size"] >= args.min_mb << 20]
    big.sort(key=lambda s: -(s["total_size"] - s["allocated_size"]))
    print(f"segments with >= {args.min_mb} MB unused: {len(big)}")
    by_site = defaultdict(lambda: [0, 0, 0])
    for s in big:
        site = _site(s.get("frames", []), args.frames)
        by_site[site][0] += s["total_size"] - s["allocated_size"]
        by_site[site][1] += s["total_size"]
        by_site[site][2] += 1
    for site, (u, t, n) in sorted(by_site.items(), key=lambda kv: -kv[1][0])[: args.top]:
        print(f"{u / 2**30:7.2f}G unused / {t / 2**30:7.2f}G in x{n:<4d} {site}")


if __name__ == "__main__":
    main()
