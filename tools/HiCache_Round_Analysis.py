#!/usr/bin/env python3
"""Compare ONE profiled round across platforms (e.g. MI355X vs B200), auto-
skipping the leading noise and the inter-round idle gap.

A round-bounded profiler trace (cascade_L2_torchprofile.sh --profile-start-round N)
looks like this on the timeline:

    [tail of previous round: a few prefill/decode]   [big idle gap]   [THE round]
    \------------------ red box (noise) -----------/                 \-- green --/

The idle gap is the bench's inter-round transition (client receiving the
previous round's responses, appending the next sub-question, hitting the round
barrier, re-dispatching). During it the GPU is idle. We do NOT want to count the
red box.

This script:
  1. Loads each trace, finds GPU kernel events.
  2. Detects the LARGEST idle gap between consecutive GPU events.
  3. Defines "the round" = GPU-active span AFTER that gap (green box).
  4. Within the round, reports GPU-busy time per category and the total prefill
     tokens (sum of step[EXTEND ... toks=] annotations in the window), so you get
     an apples-to-apples per-token compute comparison — IMMUNE to the chunked-
     prefill `bs` double-counting that compare_hicache_trace_overhead.py suffers
     from (it never counts requests; it sums tokens directly).

Modes:
  1) one MI355X (ROCm) trace        -> single-platform report
  2) one B200 (NV) trace            -> single-platform report
  3) both                           -> adds a us/tok COMPARISON (total / compute / cache / comm)

Outputs (under --out-dir):
  summary.csv      round wall, uncached(prefill) tokens, GPU util
  categories.csv   per-category per-token
  kernels.csv      per-kernel report: COMPUTE(+comm) block over CACHE block,
                   platforms side by side, uncached-token counts at the top
  comparison.csv   us/tok ratio (only when 2 traces given)

Usage:
  HiCache_Round_Analysis.py --trace MI355X a.json.gz --trace B200 b.json.gz --out-dir out/
  # optional: --min-gap-ms 200  (ignore gaps smaller than this when picking the round)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

STEP_RE = re.compile(r"step\[(?P<mode>\w+) bs=(?P<bs>\d+)(?: toks=(?P<toks>\d+))?\]")


@dataclass
class Event:
    name: str
    cat: str
    ts: float
    dur: float
    args: dict

    @property
    def end(self) -> float:
        return self.ts + self.dur


def load_events(path: Path) -> list[Event]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("traceEvents", []) if isinstance(data, dict) else data
    evs: list[Event] = []
    for e in raw:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        try:
            ts = float(e.get("ts", 0.0))
            dur = float(e.get("dur", 0.0))
        except (TypeError, ValueError):
            continue
        if dur <= 0:
            continue
        evs.append(
            Event(str(e.get("name", "")), str(e.get("cat", "")), ts, dur, e.get("args") or {})
        )
    evs.sort(key=lambda e: e.ts)
    return evs


def category_for(ev: Event) -> str:
    hay = "\n".join([ev.name, ev.cat, str(ev.args.get("kernel", ""))])
    if re.search(r"transfer_kernel_impl|transfer_kv_per_layer|kvcacheio", hay, re.I):
        return "hicache_transfer_kernel"
    if re.search(r"Memcpy HtoD|Host -> Device", hay, re.I):
        return "memcpy_h2d"
    if re.search(r"Memcpy DtoH|Device -> Host", hay, re.I):
        return "memcpy_d2h"
    if re.search(r"nccl|all[_ ]?reduce|AllReduce|AllGather|all[_ ]?gather|quickreduce", hay, re.I):
        return "comm"
    # GPU compute kernels, split into attention / moe-gemm / other.
    if ev.cat.lower() == "kernel":
        if re.search(r"fmha|attn|attention|flash|mla", hay, re.I):
            return "gpu_attention"
        if re.search(r"moe|fmoe|gemm|matmul|wvSplitK|Cijk|gmm|grouped", hay, re.I):
            return "gpu_moe_gemm"
        if re.search(r"rope|norm|rms|silu|act|elementwise|reduce", hay, re.I):
            return "gpu_elementwise"
        return "gpu_kernel_other"
    if ev.cat.lower() in ("gpu_memcpy", "gpu_memset"):
        return "gpu_memcpy_other"
    return "cpu_other"


def is_gpu(ev: Event) -> bool:
    return ev.cat.lower() in ("kernel", "gpu_memcpy", "gpu_memset")


def stream_class(cat: str) -> str:
    """Group GPU kernels by which stream/role they belong to, so prefill
    compute is separated from the cache-IO transfer kernels (different streams)."""
    if cat in ("hicache_transfer_kernel", "memcpy_h2d", "memcpy_d2h", "gpu_memcpy_other"):
        return "cache"
    if cat == "comm":
        return "comm"
    return "compute"


def clean_kernel_name(ev: Event) -> str:
    """Short, comparable kernel name. Preserves the HiCache transfer direction
    (load pf->lf vs backup lf->pf) and trims C++ template/arg noise."""
    n = ev.name or str(ev.args.get("kernel", ""))
    if "transfer_kernel_impl" in n:
        ipf = n.find("get_global_offset_pf")
        ilf = n.find("get_global_offset_lf")
        if ipf != -1 and (ilf == -1 or ipf < ilf):
            return "transfer_kv(load pf->lf)"
        return "transfer_kv(backup lf->pf)"
    # Strip mangled prefix like "_Z20"
    n = re.sub(r"^_Z\d+", "", n)
    # Cut at first template/arg paren to collapse variants.
    for sep in ("<", "("):
        i = n.find(sep)
        if i > 0:
            n = n[:i]
    return n.strip()[:80]


def detect_round(events: list[Event], min_gap_us: float):
    """Round = GPU-active span after the largest inter-event idle gap."""
    gpu = sorted((e for e in events if is_gpu(e)), key=lambda e: e.ts)
    if not gpu:
        gpu = events
    best_gap = 0.0
    split_idx = 0
    prev_end = gpu[0].end
    for i in range(1, len(gpu)):
        gap = gpu[i].ts - prev_end
        if gap > best_gap:
            best_gap = gap
            split_idx = i
        if gpu[i].end > prev_end:
            prev_end = gpu[i].end
    if best_gap < min_gap_us:
        # No clear gap → analyze the whole trace.
        return gpu[0].ts, max(e.end for e in gpu), best_gap
    return gpu[split_idx].ts, max(e.end for e in gpu), best_gap


def busy_union_us(events: list[Event]) -> float:
    """Wall-clock GPU-busy time (union of intervals; handles overlapping streams)."""
    iv = sorted((e.ts, e.end) for e in events)
    if not iv:
        return 0.0
    total = 0.0
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    total += ce - cs
    return total


def analyze(label: str, path: Path, min_gap_us: float) -> dict:
    events = load_events(path)
    r0, r1, gap = detect_round(events, min_gap_us)
    in_round = [e for e in events if e.ts >= r0 and e.ts <= r1]

    # Total prefill tokens in the round (sum of EXTEND step toks annotations).
    toks = 0
    n_extend = 0
    for e in in_round:
        if e.cat != "user_annotation":
            continue
        m = STEP_RE.search(e.name)
        if m and m.group("mode") == "EXTEND":
            toks += int(m.group("toks") or 0)
            n_extend += 1

    # GPU-busy per category (sum of durations) + GPU wall-busy (union).
    cats: dict[str, dict] = {}
    kernels: dict[tuple, dict] = {}   # (stream_class, clean_name) -> stats
    class_wall: dict[str, list] = {}  # stream_class -> list of events (for union)
    gpu_events = [e for e in in_round if is_gpu(e)]
    for e in gpu_events:
        c = category_for(e)
        s = cats.setdefault(c, {"count": 0, "sum_us": 0.0})
        s["count"] += 1
        s["sum_us"] += e.dur
        klass = stream_class(c)
        kn = clean_kernel_name(e)
        ks = kernels.setdefault((klass, kn), {"count": 0, "sum_us": 0.0})
        ks["count"] += 1
        ks["sum_us"] += e.dur
        class_wall.setdefault(klass, []).append(e)
    gpu_wall = busy_union_us(gpu_events)
    # Per-stream-class wall-busy (union within that class' events).
    class_busy = {k: busy_union_us(v) for k, v in class_wall.items()}

    return {
        "label": label,
        "path": str(path),
        "round_start_us": r0,
        "round_end_us": r1,
        "round_wall_ms": (r1 - r0) / 1000.0,
        "skipped_gap_ms": gap / 1000.0,
        "extend_steps": n_extend,
        "prefill_tokens": toks,
        "gpu_wall_busy_ms": gpu_wall / 1000.0,
        "gpu_busy_util_pct": 100.0 * gpu_wall / (r1 - r0) if r1 > r0 else 0.0,
        "cats": cats,
        "kernels": kernels,
        "class_busy_ms": {k: v / 1000.0 for k, v in class_busy.items()},
    }


def write_kernels_report(results: list[dict], path: Path) -> None:
    """Per-kernel report. Layout (matches the requested format):

      <plat1> uncached tokens: N1            <plat2> uncached tokens: N2
      === COMPUTE (incl. comm) ===
      <plat1 compute+comm kernels>   |   <plat2 compute+comm kernels>
      === CACHE ===
      <plat1 cache kernels>          |   <plat2 cache kernels>

    Compute(+comm) and cache are in separate blocks (viewed independently), and
    the platforms sit side by side. per_token_us uses each platform's uncached
    (prefill) token count as the denominator (shown at the top).
    """
    from itertools import zip_longest

    COLS = ["platform", "stream_class", "kernel", "count", "total_ms", "avg_us", "per_token_us"]
    SP = [""]  # spacer column between platforms

    def rows_for(r: dict, classes: set) -> list[list]:
        tk = r["prefill_tokens"] or 1
        ks = [(k, kn, s) for (k, kn), s in r["kernels"].items() if k in classes]
        ks.sort(key=lambda x: -x[2]["sum_us"])
        return [
            [r["label"], k, kn, s["count"], round(s["sum_us"] / 1000.0, 3),
             round(s["sum_us"] / s["count"], 3), round(s["sum_us"] / tk, 4)]
            for k, kn, s in ks
        ]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        # Top: uncached (prefill) token counts = the per_token_us denominator.
        hdr = []
        for i, r in enumerate(results):
            if i:
                hdr += SP
            hdr += [f"{r['label']} uncached tokens:", r["prefill_tokens"], "", "", "", "", ""]
        w.writerow(hdr)
        w.writerow([])

        for block_name, classes in (("=== COMPUTE (incl. comm) ===", {"compute", "comm"}),
                                    ("=== CACHE ===", {"cache"})):
            w.writerow([block_name])
            chrow = []
            for i, _ in enumerate(results):
                if i:
                    chrow += SP
                chrow += COLS
            w.writerow(chrow)
            per = [rows_for(r, classes) for r in results]
            for tup in zip_longest(*per):
                row = []
                for i, cells in enumerate(tup):
                    if i:
                        row += SP
                    row += list(cells) if cells else [""] * len(COLS)
                w.writerow(row)
            w.writerow([])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-gap-ms", type=float, default=200.0,
                   help="Ignore idle gaps smaller than this when locating the round (default 200ms).")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for label, path in args.trace:
        print(f"[INFO] analyzing {label}: {path}", file=sys.stderr)
        results.append(analyze(label, Path(path), args.min_gap_ms * 1000.0))

    # summary.csv
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["platform", "round_wall_ms", "skipped_gap_ms", "extend_steps",
                    "prefill_tokens", "gpu_wall_busy_ms", "gpu_busy_util_pct"])
        for r in results:
            w.writerow([r["label"], round(r["round_wall_ms"], 1), round(r["skipped_gap_ms"], 1),
                        r["extend_steps"], r["prefill_tokens"], round(r["gpu_wall_busy_ms"], 1),
                        round(r["gpu_busy_util_pct"], 1)])

    # categories.csv (per-category, per prefill token)
    with open(out / "categories.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["platform", "category", "count", "sum_ms", "per_token_us"])
        for r in results:
            tk = r["prefill_tokens"] or 1
            for c, s in sorted(r["cats"].items(), key=lambda kv: -kv[1]["sum_us"]):
                w.writerow([r["label"], c, s["count"], round(s["sum_us"] / 1000.0, 3),
                            round(s["sum_us"] / tk, 4)])

    # kernels.csv — per-kernel report: compute(+comm) block over cache block,
    # platforms side by side, uncached-token counts at the top.
    write_kernels_report(results, out / "kernels.csv")

    # Console report
    print("\n==== PER-STREAM-CLASS wall-busy (compute vs cache vs comm) ====")
    for r in results:
        cb = r["class_busy_ms"]
        print(f"[{r['label']}] " + " | ".join(f"{k}={cb.get(k,0):.0f}ms" for k in ("compute", "cache", "comm")))

    print("\n==== TOP KERNELS by stream class (total_ms, avg_us) ====")
    for r in results:
        print(f"\n[{r['label']}]  (prefill tokens={r['prefill_tokens']})")
        for klass in ("compute", "cache", "comm"):
            ks = [(kn, s) for (k, kn), s in r["kernels"].items() if k == klass]
            ks.sort(key=lambda x: -x[1]["sum_us"])
            print(f"  -- {klass} --")
            for kn, s in ks[:8]:
                print(f"     {s['sum_us']/1000.0:>8.1f}ms  avg={s['sum_us']/s['count']:>8.2f}us  n={s['count']:<6} {kn}")

    print("\n==== ROUND SUMMARY (red-box noise + idle gap auto-skipped) ====")
    for r in results:
        print(f"\n[{r['label']}]  skipped idle gap={r['skipped_gap_ms']:.0f} ms")
        print(f"  round wall: {r['round_wall_ms']:.0f} ms | prefill tokens: {r['prefill_tokens']} "
              f"| EXTEND steps: {r['extend_steps']}")
        print(f"  GPU wall-busy: {r['gpu_wall_busy_ms']:.0f} ms ({r['gpu_busy_util_pct']:.0f}% of round)")
        tk = r["prefill_tokens"] or 1
        for c, s in sorted(r["cats"].items(), key=lambda kv: -kv[1]["sum_us"])[:8]:
            print(f"    {c:<24}{s['sum_us']/1000.0:>9.1f} ms{s['sum_us']/tk:>9.3f} us/tok  (n={s['count']})")
    # Cross-platform comparison (only when >=2 traces given). Compares the
    # first two by us/tok for total GPU + each stream class, with ratio.
    if len(results) >= 2:
        a, b = results[0], results[1]

        def per_tok(r, ms):
            return ms * 1000.0 / (r["prefill_tokens"] or 1)

        metrics = [("total_gpu", lambda r: r["gpu_wall_busy_ms"])]
        for cls in ("compute", "cache", "comm"):
            metrics.append((cls, lambda r, c=cls: r["class_busy_ms"].get(c, 0.0)))

        print(f"\n==== COMPARISON  us/tok  ({a['label']} vs {b['label']}) ====")
        print(f"{'metric':<12}{a['label']:>12}{b['label']:>12}{'ratio':>9}")
        with open(out / "comparison.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", f"{a['label']}_us_per_tok", f"{b['label']}_us_per_tok",
                        f"ratio_{a['label']}_over_{b['label']}"])
            for name, fn in metrics:
                va, vb = per_tok(a, fn(a)), per_tok(b, fn(b))
                ratio = va / vb if vb else 0.0
                print(f"{name:<12}{va:>12.2f}{vb:>12.2f}{ratio:>9.2f}")
                w.writerow([name, round(va, 3), round(vb, 3), round(ratio, 3)])
        if len(results) > 2:
            print("[note] comparison shown for the first two traces only.")

    print(f"\n[INFO] wrote summary.csv / categories.csv / kernels.csv"
          + (" / comparison.csv" if len(results) >= 2 else "") + f" under {out}")


if __name__ == "__main__":
    main()
