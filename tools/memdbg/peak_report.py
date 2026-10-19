"""Replay a torch.cuda.memory._snapshot() trace and report the live blocks at peak.

Usage: python peak_report.py snapshot.pickle [--top 25] [--frames 6]
Only allocations made after history recording started are visible.
"""

import argparse
import pickle
from collections import defaultdict


def _site(frames, n):
    keep = [
        f for f in frames if "/sglang/" in f["filename"] or "/aiter/" in f["filename"]
    ] or frames
    return " <- ".join(
        f"{f['filename'].split('/python/')[-1].split('site-packages/')[-1]}:{f['line']}({f['name']})"
        for f in keep[:n]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()

    snap = pickle.load(open(args.snapshot, "rb"))
    dev = max(range(len(snap["device_traces"])), key=lambda i: len(snap["device_traces"][i]))
    trace = snap["device_traces"][dev]

    live, cur, peak, peak_idx = {}, 0, 0, -1
    for i, e in enumerate(trace):
        if e["action"] == "alloc":
            live[e["addr"]] = e
            cur += e["size"]
            if cur > peak:
                peak, peak_idx = cur, i
        elif e["action"] in ("free_completed",) and e["addr"] in live:
            cur -= live.pop(e["addr"])["size"]

    live = {}
    for e in trace[: peak_idx + 1]:
        if e["action"] == "alloc":
            live[e["addr"]] = e
        elif e["action"] == "free_completed":
            live.pop(e["addr"], None)

    print(f"device {dev}: {len(trace)} events, peak live (traced) = {peak / 2**30:.2f} GiB")
    by_site = defaultdict(lambda: [0, 0])
    for e in live.values():
        s = _site(e.get("frames", []), args.frames)
        by_site[s][0] += e["size"]
        by_site[s][1] += 1
    for s, (b, n) in sorted(by_site.items(), key=lambda kv: -kv[1][0])[: args.top]:
        print(f"{b / 2**30:8.3f} GiB  x{n:<5d} {s}")


if __name__ == "__main__":
    main()
