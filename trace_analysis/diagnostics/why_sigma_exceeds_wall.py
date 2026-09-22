"""Explain a forward whose Sigma_kernel exceeds the ITL the benchmark measured.

Three different "durations" get quoted for the same forward and they are all
real, which is how a bucket table reading 10.9 ms can sit next to a measured
6.1 ms ITL:

  CPU annotation (cat=user_annotation)  how long the launch loop took
  GPU annotation (cat=gpu_user_annotation), ONE COPY PER STREAM, and
     forward_overlap.py merges them -> the span from the earliest stream's
     start to the latest stream's end, which is >= what the main stream row
     shows in a trace viewer
  Sigma_kernel   every kernel's duration added up, streams counted separately,
     so concurrent kernels are counted more than once

This prints all of them for one forward, per stream, and then singles out the
kernels whose duration is mostly spent overlapping something else -- which is
what a spin-waiting collective looks like.

    python3 why_sigma_exceeds_wall.py <trace.json.gz> --match "DECODE bs=4" \
        [--kernel all_reduce]
"""

import argparse
import gzip
import json
from collections import defaultdict


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def union_len(intervals):
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total, cs, ce = 0.0, *intervals[0]
    for s, e in intervals[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + ce - cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--match", required=True)
    ap.add_argument("--kernel", default=None,
                    help="substring of one kernel to break out (e.g. all_reduce)")
    ap.add_argument("--forward", type=int, default=0,
                    help="which matching forward (0-based, default 0)")
    args = ap.parse_args()

    ev = load(args.trace).get("traceEvents", [])

    cpu, gpu = [], []
    for e in ev:
        if e.get("ph") != "X" or args.match not in e.get("name", ""):
            continue
        if e.get("cat") == "gpu_user_annotation":
            gpu.append(e)
        elif e.get("cat") in ("user_annotation", "cpu_op"):
            cpu.append(e)

    print(f"=== annotations matching {args.match!r}")
    print(f"CPU-side copies: {len(cpu)}   Sigma dur = "
          f"{sum(e.get('dur', 0) for e in cpu)/1000:.2f} ms")
    for e in sorted(cpu, key=lambda x: x["ts"])[:6]:
        print(f"   cpu   ts={e['ts']/1000:12.3f} ms  dur={e.get('dur',0)/1000:8.3f} ms")

    print(f"GPU-side copies: {len(gpu)} (one per stream that ran work in the forward)")
    per_stream = defaultdict(list)
    for e in gpu:
        per_stream[e.get("tid")].append((e["ts"], e["ts"] + e.get("dur", 0)))
    # group the forward: sort stream windows by start, take the args.forward-th
    # cluster on the widest stream as the anchor
    anchors = sorted((min(s for s, _ in v), max(t for _, t in v), tid)
                     for tid, v in per_stream.items())
    if not anchors:
        raise SystemExit("no gpu_user_annotation found for that match")

    # cluster all GPU windows into forwards by gap
    allw = sorted((e["ts"], e["ts"] + e.get("dur", 0), e.get("tid")) for e in gpu)
    forwards, cur = [], [allw[0]]
    for w in allw[1:]:
        if w[0] - max(x[1] for x in cur) > 2000:   # >2 ms gap = next forward
            forwards.append(cur)
            cur = [w]
        else:
            cur.append(w)
    forwards.append(cur)
    print(f"clustered into {len(forwards)} forward(s)")

    fw = forwards[min(args.forward, len(forwards) - 1)]
    f_start, f_end = min(w[0] for w in fw), max(w[1] for w in fw)
    print(f"\n=== forward #{args.forward}: merged GPU span "
          f"{(f_end - f_start)/1000:.3f} ms   ({len(fw)} stream windows)")
    by_tid = defaultdict(lambda: [float("inf"), 0.0])
    for s, e, tid in fw:
        by_tid[tid][0] = min(by_tid[tid][0], s)
        by_tid[tid][1] = max(by_tid[tid][1], e)
    print(f"{'stream':>8} {'start_rel_ms':>13} {'end_rel_ms':>11} {'span_ms':>9}")
    for tid, (s, e) in sorted(by_tid.items(), key=lambda kv: -(kv[1][1] - kv[1][0]))[:10]:
        print(f"{str(tid):>8} {(s-f_start)/1000:>13.3f} {(e-f_start)/1000:>11.3f} "
              f"{(e-s)/1000:>9.3f}")
    if len(by_tid) > 10:
        print(f"   ... {len(by_tid)-10} more streams")

    # kernels inside the forward window
    ker = [e for e in ev
           if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
           and e["ts"] >= f_start and e["ts"] + e.get("dur", 0) <= f_end]
    sigma = sum(k.get("dur", 0) for k in ker) / 1000
    uni = union_len([(k["ts"], k["ts"] + k.get("dur", 0)) for k in ker]) / 1000
    print(f"\nkernels in window : {len(ker)}")
    print(f"Sigma_kernel      : {sigma:.3f} ms   (streams counted separately)")
    print(f"union_busy        : {uni:.3f} ms   (>=1 kernel running)")
    print(f"overlap factor    : {sigma/uni if uni else 0:.2f}x")
    print(f"merged GPU span   : {(f_end-f_start)/1000:.3f} ms")
    print(f"idle in span      : {(f_end-f_start)/1000 - uni:.3f} ms")

    if args.kernel:
        sel = [k for k in ker if args.kernel in k.get("name", "")]
        others = [(k["ts"], k["ts"] + k.get("dur", 0)) for k in ker
                  if args.kernel not in k.get("name", "")]
        oth_union = union_len(others)
        tot = sum(k.get("dur", 0) for k in sel) / 1000
        durs = sorted(k.get("dur", 0) for k in sel)
        print(f"\n=== kernel filter {args.kernel!r}: {len(sel)} launches, "
              f"Sigma = {tot:.3f} ms")
        if durs:
            print(f"    per launch us: min={durs[0]:.1f} "
                  f"median={durs[len(durs)//2]:.1f} max={durs[-1]:.1f}")
            streams = defaultdict(float)
            for k in sel:
                streams[k.get("tid")] += k.get("dur", 0)
            print(f"    streams: " + ", ".join(f"{t}:{v/1000:.3f}ms"
                                               for t, v in sorted(streams.items())))
            # how much of this kernel's time runs while OTHER kernels also run
            covered = 0.0
            for k in sel:
                s, e = k["ts"], k["ts"] + k.get("dur", 0)
                inter = union_len([(max(s, a), min(e, b)) for a, b in others
                                   if a < e and b > s])
                covered += inter
            print(f"    of that Sigma, {covered/1000:.3f} ms ({100*covered/(tot*1000) if tot else 0:.1f}%) "
                  f"runs concurrently with other kernels")
            print(f"    all OTHER kernels: Sigma={sum(k.get('dur',0) for k in ker if args.kernel not in k.get('name',''))/1000:.3f} ms, "
                  f"union={oth_union/1000:.3f} ms")


if __name__ == "__main__":
    main()
