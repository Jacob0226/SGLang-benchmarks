#!/usr/bin/env python3
"""
trim_trace_to_forward.py

Trim an SGLang torch-profiler trace down to ONE forward pass, so analyze_trace.py
reports a single-forward breakdown (matching ATOM's single-forward scope) instead
of aggregating every forward in the trace.

The forward boundary is taken from SGLang's `step[EXTEND ...]` / `step[DECODE ...]`
gpu_user_annotation wrappers: we keep all events whose timestamp falls in
[chosen_wrapper.ts, next_wrapper.ts). All event categories (kernel,
python_function, gpu_user_annotation, cpu_op, metadata) are preserved so
analyze_trace's nn.Module reconstruction still works.

Usage:
  python trim_trace_to_forward.py --in TRACE.json.gz --out TRIMMED.json.gz \
      --phase prefill   # keeps a step[EXTEND toks=16..] forward (default: full 16k chunk)
  python trim_trace_to_forward.py --in TRACE.json.gz --out TRIMMED.json.gz --phase decode
"""
import argparse
import bisect
import gzip
import json
import sys
from collections import Counter


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def dump(obj, path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=["prefill", "decode"], required=True)
    ap.add_argument("--which", choices=["full16k", "median", "first"], default="full16k",
                    help="which forward to keep (prefill: full16k picks toks=16..)")
    args = ap.parse_args()

    t = load(args.inp)
    is_list = isinstance(t, list)
    evs = t if is_list else t.get("traceEvents", [])

    # wrapper markers on the dominant kernel stream
    kern = [e for e in evs if isinstance(e, dict) and e.get("ph") == "X"
            and str(e.get("cat", "")).lower() == "kernel"]
    dom = Counter((k["pid"], k["tid"]) for k in kern).most_common(1)[0][0]
    pref = "step[EXTEND" if args.phase == "prefill" else "step[DECODE"
    # Collect BOTH the CPU-side (cat=user_annotation, bounds nn.Module python_function
    # events) and the GPU-side (cat=gpu_user_annotation, bounds the kernels) wrappers.
    # CPU dispatch runs ahead of the GPU, so we take the UNION [min ts, max ts+dur]
    # of the matching CPU+GPU wrappers to capture both the module tree and the kernels.
    def match(e):
        n = str(e.get("name", ""))
        if not n.startswith(pref):
            return False
        if args.phase == "prefill" and args.which == "full16k":
            return "toks=16" in n
        return True
    wraps = [e for e in evs if isinstance(e, dict) and e.get("ph") == "X"
             and e.get("cat") in ("user_annotation", "gpu_user_annotation")
             and match(e)]
    if not wraps:
        # decode / fallback: any step[ on either annotation stream
        wraps = [e for e in evs if isinstance(e, dict) and e.get("ph") == "X"
                 and e.get("cat") in ("user_annotation", "gpu_user_annotation")
                 and str(e.get("name", "")).startswith("step[")]
    if not wraps:
        sys.exit(f"[ERROR] no step[ wrapper found for phase {args.phase}")
    # A single forward = ONE CPU wrapper (bounds nn.Module) paired with the GPU wrapper
    # that runs it (bounds kernels). CPU dispatch leads the GPU, so window = union of the
    # matched CPU+GPU pair: [cpu.ts, gpu.ts+gpu.dur]. Names like "step[DECODE bs=64]" are
    # identical across forwards, so we pair by time proximity, NOT by parsing the name.
    cpu_w = sorted([w for w in wraps if w["cat"] == "user_annotation"],
                   key=lambda e: float(e["ts"]))
    gpu_w = sorted([w for w in wraps if w["cat"] == "gpu_user_annotation"],
                   key=lambda e: float(e["ts"]))
    if not cpu_w:
        sys.exit("[ERROR] no CPU-side (user_annotation) step wrapper; nn.Module tree "
                 "unavailable (cuda-graph trace?). Use the no-cuda-graph trace.")

    def toks(e):
        import re
        m = re.search(r"toks?=(\d+)", str(e.get("name", "")))
        return int(m.group(1)) if m else -1

    # pick the target CPU wrapper
    if args.phase == "prefill" and args.which == "full16k":
        # CPU wraps already filtered to toks=16 by match(); take the largest-toks one
        cpu = max(cpu_w, key=toks)
    elif args.which == "first":
        cpu = cpu_w[0]
    else:  # median / decode default: middle instance (skip warmup edge forwards)
        cpu = cpu_w[len(cpu_w) // 2]

    # pair with the GPU wrapper whose span best overlaps this CPU dispatch
    def gpu_key(g):
        gs, ge = float(g["ts"]), float(g["ts"]) + float(g.get("dur", 0) or 0)
        cs = float(cpu["ts"])
        return abs(gs - cs) if ge >= cs else float("inf")
    gpu = min(gpu_w, key=gpu_key) if gpu_w else None

    s0 = float(cpu["ts"])
    s1 = float(cpu["ts"]) + float(cpu.get("dur", 0) or 0)
    if gpu is not None:
        s1 = max(s1, float(gpu["ts"]) + float(gpu.get("dur", 0) or 0))
    tgt = cpu
    print(f"[INFO] keeping forward '{cpu['name']}' "
          f"(CPU 1 + GPU {'1' if gpu is not None else '0'} paired) "
          f"window [{s0:.0f}, {s1:.0f}] ({(s1-s0)/1000:.1f} ms)", file=sys.stderr)

    kept = []
    for e in evs:
        if not isinstance(e, dict):
            continue
        ph = e.get("ph")
        if ph == "X":
            ts = float(e.get("ts", 0)); dur = float(e.get("dur", 0) or 0)
            if ts < s1 and ts + dur > s0:   # keep any event OVERLAPPING the window
                kept.append(e)
        elif ph in ("M",) or "ts" not in e:  # metadata (process/thread names)
            kept.append(e)
    print(f"[INFO] kept {len(kept)}/{len(evs)} events", file=sys.stderr)

    if is_list:
        dump(kept, args.out)
    else:
        t["traceEvents"] = kept
        dump(t, args.out)
    print(f"[INFO] trimmed trace written: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
