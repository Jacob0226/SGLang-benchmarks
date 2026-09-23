"""Where does the one long collective sit, and what is it waiting for?

A capture can show 455 collective launches of which 454 are microseconds and one
is milliseconds. That single launch decides whether a bucket table reads
"communication is expensive", so it is worth knowing exactly where it is: first
launch of the capture (profiler arming), first of every forward (per-step
resync), or somewhere in the middle (a real stall).

Prints every launch with its index, its position relative to the first kernel in
the trace, and which forward it falls in, then zooms in on the long ones.

    python3 locate_long_collective.py <trace.json.gz> --kernel all_reduce \
        [--match "DECODE bs="] [--threshold-us 1000]
"""

import argparse
import gzip
import json


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--match", default="DECODE bs=")
    ap.add_argument("--threshold-us", type=float, default=1000.0)
    args = ap.parse_args()

    ev = load(args.trace).get("traceEvents", [])
    ker = sorted([e for e in ev if e.get("ph") == "X"
                  and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")],
                 key=lambda e: e["ts"])
    cpu = sorted([e for e in ev if e.get("ph") == "X"
                  and e.get("cat") in ("user_annotation", "cpu_op")
                  and args.match in e.get("name", "")], key=lambda e: e["ts"])
    if not ker:
        raise SystemExit("no kernels")
    t0 = ker[0]["ts"]

    ar = [e for e in ker if args.kernel in e.get("name", "")]
    print(f"=== {args.trace.split('/')[-1]}")
    print(f"{len(ar)} launches of {args.kernel!r}; "
          f"{len(cpu)} CPU-side {args.match!r} annotations")
    print(f"CPU annotation starts (ms from first kernel): "
          f"{[round((e['ts']-t0)/1000, 2) for e in cpu]}")

    longs = [(i, e) for i, e in enumerate(ar) if e.get("dur", 0) >= args.threshold_us]
    print(f"\nlaunches >= {args.threshold_us:.0f} us: {len(longs)} of {len(ar)}")
    for i, e in longs:
        print(f"   index {i:>4} of {len(ar)}   rel={(e['ts']-t0)/1000:9.3f} ms   "
              f"dur={e['dur']:9.1f} us   stream={e.get('tid')}")

    if not longs:
        return
    i, big = longs[0]

    # What ran on the GPU while it spun, and what came right before it.
    s, t = big["ts"], big["ts"] + big["dur"]
    during = [k for k in ker if k is not big and k["ts"] < t and k["ts"] + k.get("dur", 0) > s]
    busy = sum(min(t, k["ts"] + k.get("dur", 0)) - max(s, k["ts"]) for k in during)
    print(f"\nwhile the {big['dur']:.0f} us launch at index {i} was running:")
    print(f"   {len(during)} other kernels overlapped it, covering {busy/1000:.3f} ms "
          f"({100*busy/big['dur']:.1f}% of its duration)")
    for k in sorted(during, key=lambda k: -k.get("dur", 0))[:5]:
        print(f"     {k.get('dur',0):8.1f} us  stream={k.get('tid')}  {k['name'][:60]}")

    same = [k for k in ker if k.get("tid") == big.get("tid")]
    pos = same.index(big)
    print(f"\non its own stream ({big.get('tid')}), the 4 kernels before and after:")
    for k in same[max(0, pos - 4):pos + 5]:
        mark = "  <== the long one" if k is big else ""
        print(f"   rel={(k['ts']-t0)/1000:9.3f} ms  dur={k.get('dur',0):9.1f} us  "
              f"{k['name'][:52]}{mark}")

    # Gap on the GPU immediately before it: an idle GPU before a collective is
    # the signature of waiting for something off this rank.
    prev_end = max((k["ts"] + k.get("dur", 0) for k in ker if k["ts"] + k.get("dur", 0) <= s),
                   default=None)
    if prev_end is not None:
        print(f"\nGPU-wide idle gap immediately before it: {(s - prev_end)/1000:.3f} ms")


if __name__ == "__main__":
    main()
