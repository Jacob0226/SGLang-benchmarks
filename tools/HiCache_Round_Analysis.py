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
import time
from dataclasses import dataclass
from pathlib import Path

_T0 = time.time()


def _log(msg: str) -> None:
    """Progress line to stderr (so a long trace load doesn't look hung)."""
    print(f"[+{time.time() - _T0:6.1f}s] {msg}", file=sys.stderr, flush=True)

# Reuse analyze_trace.py (same tools/ dir) for the per-layer kernel-sequence +
# python call-site mapping (External-id linking). Optional: if unavailable, the
# layer breakdown section is skipped.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import analyze_trace as AT
except Exception:
    AT = None

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
    # Decompress in chunks with a live byte counter, then parse — a big trace's
    # gzip+JSON step gives no feedback otherwise and looks like it hung.
    if str(path).endswith(".gz"):
        _log(f"  decompressing {Path(path).name} ...")
        chunks = []
        total = 0
        last_logged = 0
        with gzip.open(path, "rb") as gf:
            while True:
                c = gf.read(64 * 1024 * 1024)
                if not c:
                    break
                chunks.append(c)
                total += len(c)
                if total - last_logged >= 512 * 1024 * 1024:  # log every ~512 MB
                    last_logged = total
                    _log(f"    decompressed {total / 1e6:7.0f} MB")
        blob = b"".join(chunks)
        del chunks
    else:
        with open(path, "rb") as f:
            blob = f.read()
    _log(f"  parsing JSON ({len(blob) / 1e6:.0f} MB) ...")
    data = json.loads(blob)
    del blob
    raw = data.get("traceEvents", []) if isinstance(data, dict) else data

    _log(f"  building event list from {len(raw):,} records ...")
    evs: list[Event] = []
    n = len(raw) or 1
    step = max(1, n // 10)
    for i, e in enumerate(raw):
        if (i + 1) % step == 0:
            _log(f"    scanned {100 * (i + 1) // n:3d}%  ({len(evs):,} kept)")
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
    _log(f"  sorting {len(evs):,} events ...")
    evs.sort(key=lambda e: e.ts)
    return evs, raw


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


def clean_name_str(n: str) -> str:
    """Short, comparable kernel name from a raw name string."""
    n = n or ""
    if "transfer_kernel_impl" in n:
        # Direction is encoded by the two template offset-functions, in order:
        #   1st = SOURCE layout, 2nd = DEST layout.  Verified identical on both
        #   ROCm (mangled) and CUDA/B200 (demangled) traces:
        #     load   : get_global_offset_pf      -> get_global_offset_lf   (pf->lf)
        #     backup : get_global_offset_lf_tbl  -> get_global_offset_pf   (lf->pf)
        # pf = page-first host pool, lf = layer-first device buffer.
        toks = re.findall(r"get_global_offset_(pf|lf_tbl|lf)", n)
        src = toks[0] if toks else "?"
        # Append the FULL raw kernel name (mangled on ROCm, demangled on CUDA).
        if src == "pf":
            return f"transfer_kv(load pf->lf): {n}"
        return f"transfer_kv(backup lf->pf): {n}"
    n = re.sub(r"^_Z\d+", "", n)  # strip mangled length prefix
    for sep in ("<", "("):
        i = n.find(sep)
        if i > 0:
            n = n[:i]
    return n.strip()[:80]


def clean_kernel_name(ev: Event) -> str:
    return clean_name_str(ev.name or str(ev.args.get("kernel", "")))


def _shorten_src(src: str) -> str:
    if not src:
        return ""
    # keep the path relative to sglang + the func name
    if "/sglang/" in src:
        src = src.split("/sglang/", 1)[1]
    return src[:90]


def layer_breakdown(raw: list, name_avg: dict) -> tuple:
    """Extract ONE representative decoder layer's GPU kernels in order, with a
    ROBUST per-kernel duration and the python source call-site.

    Delegates to analyze_trace.analyze_layer_structure so this matches the
    comparison_combined breakdown exactly: the duration is the median across
    all instances of the layer type at that position (artifact-resistant),
    NOT a single noisy instance.  Returns (layer_class, rows) where
    rows = [(order, kernel, avg_us, this_us, python_src)].
    """
    if AT is None:
        return None, []
    try:
        stream = AT.auto_detect_stream(raw)
        kernels_raw = AT.extract_gpu_kernels(raw, stream)
        cls, layer_types, callsites, robust = AT.analyze_layer_structure(
            raw, kernels_raw)
        if not cls or not layer_types:
            return cls, []
        # Show the layer type that covers the most layers (the dominant body).
        lt = max(layer_types, key=lambda t: t["count"])
        rows = []
        for order, (kidx, _section, _leaf) in enumerate(lt["kernel_breakdown"]):
            k = kernels_raw[kidx]
            nm = clean_name_str(k.get("name", ""))
            this_us = float(k.get("dur", 0.0))
            avg = robust.get(kidx, this_us)  # robust per-position median
            rows.append((order, nm, round(avg, 3), round(this_us, 3),
                         _shorten_src(callsites.get(kidx, ""))))
        return cls, rows
    except Exception as exc:  # best-effort; layer breakdown is optional
        print(f"[WARN] layer breakdown failed: {exc}", file=sys.stderr)
        return None, []


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
    _log(f"[{label}] loading trace ...")
    events, raw = load_events(path)
    _log(f"[{label}] detecting round (largest GPU idle gap) ...")
    r0, r1, gap = detect_round(events, min_gap_us)
    in_round = [e for e in events if e.ts >= r0 and e.ts <= r1]
    _log(f"[{label}] aggregating {len(in_round):,} in-round events ...")

    # Total prefill tokens in the round (sum of EXTEND step toks annotations),
    # plus the distribution of EXTEND steps by (batch size, tokens) — different
    # steps use different batch sizes, so the data volume per kernel differs;
    # this context matters when comparing per-kernel costs across platforms.
    from collections import Counter
    toks = 0
    n_extend = 0
    extend_dist: Counter = Counter()
    for e in in_round:
        if e.cat != "user_annotation":
            continue
        m = STEP_RE.search(e.name)
        if m and m.group("mode") == "EXTEND":
            tk = m.group("toks") or "0"
            toks += int(tk)
            n_extend += 1
            extend_dist[f"EXTEND bs={m.group('bs')} toks={tk}"] += 1

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
        ks = kernels.setdefault((klass, kn), {"count": 0, "sum_us": 0.0, "durs": []})
        ks["count"] += 1
        ks["sum_us"] += e.dur
        ks["durs"].append(e.dur)
        class_wall.setdefault(klass, []).append(e)
    gpu_wall = busy_union_us(gpu_events)
    # Per-stream-class wall-busy (union within that class' events).
    class_busy = {k: busy_union_us(v) for k, v in class_wall.items()}

    _log(f"[{label}] done ({n_extend} EXTEND steps, {toks:,} tokens)")
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
        "extend_dist": dict(extend_dist),
    }


_SP = [""]  # spacer column between platforms in side-by-side blocks


def _sidebyside(w, results, block_name, columns, rows_per_platform):
    """Write a side-by-side block: each platform's rows aligned in parallel cols."""
    from itertools import zip_longest
    w.writerow([block_name])
    chrow = []
    for i, _ in enumerate(results):
        if i:
            chrow += _SP
        chrow += columns
    w.writerow(chrow)
    for tup in zip_longest(*rows_per_platform):
        row = []
        for i, cells in enumerate(tup):
            if i:
                row += _SP
            row += list(cells) if cells else [""] * len(columns)
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

    # ---- ONE merged summary.csv: round summary + comparison + kernels + layer ----
    def per_tok(r, ms):
        return ms * 1000.0 / (r["prefill_tokens"] or 1)

    def cls_ms(r, c):
        return r["gpu_wall_busy_ms"] if c == "total_gpu" else r["class_busy_ms"].get(c, 0.0)

    def kernel_rows(r, classes):
        tk = r["prefill_tokens"] or 1
        ks = [(kn, s) for (k, kn), s in r["kernels"].items() if k in classes]
        ks.sort(key=lambda x: -x[1]["sum_us"])
        # Per-call cost uses the MEDIAN (robust to profiler-artifact outliers
        # and to varying prefill-chunk sizes), matching the layer-breakdown's
        # robust per-kernel duration.  total_ms / per_token_us stay as the real
        # round-wide sum (true total contribution, used for ranking).
        import statistics as _st
        return [[r["label"], kn, s["count"], round(s["sum_us"] / 1000.0, 3),
                 round(_st.median(s["durs"]), 3), round(s["sum_us"] / tk, 4)]
                for kn, s in ks]

    KCOLS = ["platform", "kernel", "count", "total_ms", "median_us", "per_token_us"]
    labels = [r["label"] for r in results]
    two = len(results) >= 2

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        # Section 1: round summary + us/tok comparison (ratio when 2 platforms)
        w.writerow(["=== ROUND SUMMARY ==="])
        w.writerow(["metric"] + labels + ([f"ratio_{labels[0]}/{labels[1]}"] if two else []))

        def srow(name, vals, ratio=None):
            w.writerow([name] + vals + ([ratio] if two and ratio is not None else ([""] if two else [])))

        srow("uncached_tokens", [r["prefill_tokens"] for r in results])
        srow("round_wall_ms", [round(r["round_wall_ms"], 1) for r in results])
        srow("skipped_gap_ms", [round(r["skipped_gap_ms"], 1) for r in results])
        srow("gpu_busy_util_pct", [round(r["gpu_busy_util_pct"], 1) for r in results])
        srow("extend_steps", [r["extend_steps"] for r in results])
        for c in ("total_gpu", "compute", "cache", "comm"):
            vals = [round(per_tok(r, cls_ms(r, c)), 2) for r in results]
            ratio = round(vals[0] / vals[1], 2) if two and vals[1] else None
            srow(f"{c}_us_per_tok", vals, ratio)
        w.writerow([])

        # EXTEND-step distribution: how many steps of each (batch size, tokens).
        # Different steps process different amounts of data, so the same kernel
        # is run on different sizes -> essential context for kernel comparison.
        from itertools import zip_longest as _zl
        w.writerow(["=== EXTEND STEP DISTRIBUTION (bs/toks -> #steps; data volume varies per step) ==="])
        ehdr = []
        for i, r in enumerate(results):
            if i:
                ehdr += _SP
            ehdr += [f"step ({r['label']})", "#steps"]
        w.writerow(ehdr)
        ext_lists = [sorted(r["extend_dist"].items(), key=lambda x: (-x[1], x[0]))
                     for r in results]
        for tup in _zl(*ext_lists):
            row = []
            for i, cell in enumerate(tup):
                if i:
                    row += _SP
                row += [cell[0], cell[1]] if cell else ["", ""]
            w.writerow(row)
        total_row = []
        for i, r in enumerate(results):
            if i:
                total_row += _SP
            total_row += (["Total", r["extend_steps"]] if i == 0
                          else ["", r["extend_steps"]])
        w.writerow(total_row)
        w.writerow([])

        # Section 2 & 3: kernels (compute+comm, then cache), side by side
        _sidebyside(w, results, "=== KERNELS: COMPUTE (incl. comm) ===", KCOLS,
                    [kernel_rows(r, {"compute", "comm"}) for r in results])
        _sidebyside(w, results, "=== KERNELS: CACHE ===", KCOLS,
                    [kernel_rows(r, {"cache"}) for r in results])

    # Console report
    print("\n==== PER-STREAM-CLASS wall-busy (compute vs cache vs comm) ====")
    for r in results:
        cb = r["class_busy_ms"]
        print(f"[{r['label']}] " + " | ".join(f"{k}={cb.get(k,0):.0f}ms" for k in ("compute", "cache", "comm")))

    print("\n==== TOP KERNELS by stream class (total_ms, median_us) ====")
    import statistics as _st
    for r in results:
        print(f"\n[{r['label']}]  (prefill tokens={r['prefill_tokens']})")
        for klass in ("compute", "cache", "comm"):
            ks = [(kn, s) for (k, kn), s in r["kernels"].items() if k == klass]
            ks.sort(key=lambda x: -x[1]["sum_us"])
            print(f"  -- {klass} --")
            for kn, s in ks[:8]:
                print(f"     {s['sum_us']/1000.0:>8.1f}ms  med={_st.median(s['durs']):>8.2f}us  n={s['count']:<6} {kn}")

    print("\n==== ROUND SUMMARY (red-box noise + idle gap auto-skipped) ====")
    for r in results:
        print(f"\n[{r['label']}]  skipped idle gap={r['skipped_gap_ms']:.0f} ms")
        print(f"  round wall: {r['round_wall_ms']:.0f} ms | prefill tokens: {r['prefill_tokens']} "
              f"| EXTEND steps: {r['extend_steps']}")
        print(f"  GPU wall-busy: {r['gpu_wall_busy_ms']:.0f} ms ({r['gpu_busy_util_pct']:.0f}% of round)")
        for lbl, cnt in sorted(r["extend_dist"].items(), key=lambda x: (-x[1], x[0])):
            print(f"    -- {lbl}: {cnt}")
        tk = r["prefill_tokens"] or 1
        for c, s in sorted(r["cats"].items(), key=lambda kv: -kv[1]["sum_us"])[:8]:
            print(f"    {c:<24}{s['sum_us']/1000.0:>9.1f} ms{s['sum_us']/tk:>9.3f} us/tok  (n={s['count']})")
    if two:
        print(f"\n==== COMPARISON us/tok ({labels[0]} vs {labels[1]}) ====")
        for c in ("total_gpu", "compute", "cache", "comm"):
            va, vb = per_tok(results[0], cls_ms(results[0], c)), per_tok(results[1], cls_ms(results[1], c))
            print(f"  {c:<12}{va:>9.2f}{vb:>9.2f}  ratio={va/vb:.2f}" if vb else f"  {c:<12}{va:>9.2f}")

    print(f"\n[INFO] wrote single summary.csv (round + comparison + kernels + layer) under {out}")


if __name__ == "__main__":
    main()
