"""Which call sites grew the caching allocator (segment_alloc events in the trace).

Usage: python segment_growth.py snapshot.pickle [--frames 5] [--top 20]
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
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    snap = pickle.load(open(args.snapshot, "rb"))
    dev = max(range(len(snap["device_traces"])), key=lambda i: len(snap["device_traces"][i]))
    trace = snap["device_traces"][dev]

    grow = defaultdict(lambda: [0, 0, []])
    freed = 0
    for e in trace:
        if e["action"] == "segment_alloc":
            g = grow[_site(e.get("frames", []), args.frames)]
            g[0] += e["size"]
            g[1] += 1
            g[2].append(e["size"])
        elif e["action"] == "segment_free":
            freed += e["size"]
    total = sum(g[0] for g in grow.values())
    print(f"device {dev}: segment_alloc {total / 2**30:.2f}G, segment_free {freed / 2**30:.2f}G, trace events {len(trace)}")
    for site, (b, n, sizes) in sorted(grow.items(), key=lambda kv: -kv[1][0])[: args.top]:
        sz = sorted(sizes)
        print(
            f"{b / 2**30:7.2f}G x{n:<4d} min={sz[0] / 2**20:.0f}M max={sz[-1] / 2**20:.0f}M  {site}"
        )


if __name__ == "__main__":
    main()
