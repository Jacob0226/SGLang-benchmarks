#!/usr/bin/env python3
from __future__ import annotations
"""
auto_detect_layer.py

Analyze PyTorch profiler traces to discover model layer structure and kernel breakdown.

Three-step analysis:
  Step 1: Parse cuda-graph-ON trace → kernel statistics (count, sum, avg, percentage)
  Step 2: Parse cuda-graph-OFF trace → layer structure (types, sub-modules)
  Step 3: Combine → one row per call site of ONE forward, with the layer types that
          run it and how many layers that is (so Σ covers the whole forward)

Usage:
    # Full analysis (both traces)
    python auto_detect_layer.py --graph-on on.trace.json.gz --graph-off off.trace.json.gz

    # Step 1 only: kernel statistics
    python auto_detect_layer.py --graph-on on.trace.json.gz

    # Step 2 only: layer structure
    python auto_detect_layer.py --graph-off off.trace.json.gz

    # Export step 1 to CSV
    python auto_detect_layer.py --graph-on on.trace.json.gz --csv kernels.csv

    # Restrict to ONE forward pass (per-forward numbers, comparable with
    # compare_glm52_sglang_atom.py); without it every kernel's Avg is a
    # whole-trace average that blends forwards of different batch/token sizes
    python auto_detect_layer.py --graph-on on.trace.json.gz \
        --graph-off off.trace.json.gz --forward-match "bs=3"
"""

import argparse
import bisect
import csv
import gzip
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_trace(path: str) -> dict | list:
    p = Path(path)
    if not p.exists():
        sys.exit(f"[ERROR] File not found: {path}")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# GPU kernel extraction
# ---------------------------------------------------------------------------

def extract_gpu_kernels(trace: dict | list, stream: int | None = None) -> list[dict]:
    """Extract GPU kernel events (cat='kernel' only, no memcpy/memset)."""
    events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
    kernels = []
    for ev in events:
        if not isinstance(ev, dict) or ev.get("ph") != "X":
            continue
        if ev.get("cat", "").lower() != "kernel":
            continue
        if stream is not None and ev.get("tid") != stream:
            continue
        ts = float(ev.get("ts", 0))
        dur = float(ev.get("dur", 0))
        kernels.append({
            "name": ev.get("name", "<unknown>"),
            "ts": ts, "ts_end": ts + dur, "dur": dur,
            "cat": ev.get("cat", ""),
            "pid": ev.get("pid", ""), "tid": ev.get("tid", ""),
            "args": ev.get("args", {}),
        })
    kernels.sort(key=lambda k: k["ts"])
    return kernels


FORWARD_PREFIXES = ("step[", "prefill[", "decode[")


def find_forward_windows(trace: dict | list, cat: str, match: str,
                         stream: int | None = None) -> list[tuple[float, float, str]]:
    """Forward-pass windows from profiler annotations, as [(ts0, ts1, name)].

    `cat` is "gpu_user_annotation" (GPU-side, bounds kernels) or "user_annotation"
    (CPU-side, bounds nn.Module / python_function events).  Only wrappers whose
    name starts with a forward prefix (step[ / prefill[ / decode[) and contains
    `match` are returned.  Annotations without a duration are closed by the next
    wrapper on the same track.
    """
    events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
    wraps = []
    for ev in events:
        if not isinstance(ev, dict) or ev.get("ph") != "X":
            continue
        if ev.get("cat") != cat:
            continue
        name = ev.get("name", "")
        if not name.startswith(FORWARD_PREFIXES):
            continue
        if cat == "gpu_user_annotation" and stream is not None and ev.get("tid") != stream:
            continue
        wraps.append((float(ev.get("ts", 0)), float(ev.get("dur", 0)), name))
    wraps.sort()
    out = []
    for i, (ts, dur, name) in enumerate(wraps):
        end = ts + dur if dur > 0 else (wraps[i + 1][0] if i + 1 < len(wraps)
                                        else float("inf"))
        if match in name:
            out.append((ts, end, name))
    return out


def select_forward_window(windows: list[tuple[float, float, str]],
                          kernels: list[dict] | None,
                          pick: str) -> tuple[float, float, str]:
    """Choose one window out of the candidates.

    pick=first/last → by time; pick=min/median/max → by Σ kernel duration inside
    the window (needs `kernels`), which is how compare_glm52_sglang_atom.py picks
    its representative forward.
    """
    if len(windows) == 1 or kernels is None or pick in ("first", "last"):
        return windows[0] if pick != "last" else windows[-1]
    scored = sorted(
        ((sum(k["dur"] for k in kernels if w[0] <= k["ts"] < w[1]), w)
         for w in windows), key=lambda sw: sw[0])
    if pick == "min":
        return scored[0][1]
    if pick == "max":
        return scored[-1][1]
    return scored[len(scored) // 2][1]


def auto_detect_stream(trace: dict | list,
                        single_stream_threshold: float = 0.8) -> int | None:
    """Return a single dominant stream id, or None if kernels are spread
    across many streams (e.g. GLM5 on B200 where each layer runs on its
    own stream).  Passing stream=None to extract_gpu_kernels collects all
    streams, which is the correct behaviour for multi-stream models.
    """
    events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
    counts: Counter = Counter()
    for ev in events:
        if not isinstance(ev, dict) or ev.get("ph") != "X":
            continue
        if ev.get("cat", "").lower() != "kernel":
            continue
        counts[ev.get("tid")] += 1
    if not counts:
        sys.exit("[ERROR] No GPU kernels found in trace.")
    total = sum(counts.values())
    top_stream, top_count = counts.most_common(1)[0]
    if top_count / total >= single_stream_threshold:
        return top_stream
    print(f"[INFO] Kernels spread across {len(counts)} streams "
          f"(top stream has {top_count/total:.0%} of {total} kernels); "
          f"collecting ALL streams.", file=sys.stderr)
    return None


def fmt_dur(us: float) -> str:
    if us >= 1_000_000:
        return f"{us/1_000_000:.3f} s"
    if us >= 1_000:
        return f"{us/1_000:.3f} ms"
    return f"{us:.1f} us"


# ===================================================================
# Step 1: Kernel statistics from cuda-graph-ON trace
# ===================================================================

def compute_kernel_stats(kernels: list[dict]) -> list[dict]:
    """Aggregate kernel statistics: count, sum, avg, percentage."""
    stats: dict[str, dict] = {}
    total_dur = 0.0
    for k in kernels:
        name = k["name"]
        if name not in stats:
            stats[name] = {"name": name, "count": 0, "sum_dur": 0.0}
        stats[name]["count"] += 1
        stats[name]["sum_dur"] += k["dur"]
        total_dur += k["dur"]

    result = []
    for s in stats.values():
        s["avg_dur"] = s["sum_dur"] / s["count"] if s["count"] > 0 else 0
        s["pct"] = s["sum_dur"] / total_dur * 100 if total_dur > 0 else 0
        result.append(s)

    result.sort(key=lambda s: s["sum_dur"], reverse=True)
    return result


def print_step1(stats: list[dict]) -> None:
    total_dur = sum(s["sum_dur"] for s in stats)
    total_count = sum(s["count"] for s in stats)

    print(f"\n{'='*120}")
    print(f" Step 1: Kernel Statistics (cuda-graph-ON)")
    print(f" Total: {total_count} kernel calls, {fmt_dur(total_dur)}")
    print(f"{'='*120}")
    print(f"  {'Kernel Name':<80s} {'Count':>6} {'Sum(us)':>10} {'Avg(us)':>10} {'Pct':>6}")
    print(f"  {'-'*80} {'-'*6} {'-'*10} {'-'*10} {'-'*6}")

    for s in stats:
        name = s["name"][:80]
        print(f"  {name:<80s} {s['count']:>6} {s['sum_dur']:>10.1f} "
              f"{s['avg_dur']:>10.1f} {s['pct']:>5.1f}%")

    print(f"{'='*120}\n")


def _write_xlsx(headers: list[str], rows: list[list], path: str) -> None:
    """Write data to Excel with Arial font and auto-width columns."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active

    arial = Font(name="Arial", size=10)
    arial_bold = Font(name="Arial", size=10, bold=True)
    header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2",
                              fill_type="solid")

    for c, val in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=val)
        cell.font = arial_bold
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
    ws.freeze_panes = "A2"

    for r, row_data in enumerate(rows, 2):
        for c, val in enumerate(row_data, 1):
            if isinstance(val, str):
                try:
                    val = float(val) if val and "." in val and val.replace(".", "").replace("-", "").isdigit() else val
                except (ValueError, TypeError):
                    pass
            cell = ws.cell(row=r, column=c, value=val if val != "" else None)
            cell.font = arial

    for c in range(1, len(headers) + 1):
        max_len = len(str(headers[c - 1]))
        for r in range(2, min(len(rows) + 2, 50)):
            val = ws.cell(row=r, column=c).value
            if val:
                max_len = max(max_len, min(len(str(val)), 60))
        ws.column_dimensions[get_column_letter(c)].width = max_len + 2

    wb.save(path)


def write_step1(stats: list[dict], path: str) -> None:
    headers = ["Name", "Count", "SumDuration_us", "AvgDuration_us", "Percentage"]
    rows = [[s["name"], s["count"], round(s["sum_dur"], 1),
             round(s["avg_dur"], 3), round(s["pct"], 2)] for s in stats]
    _write_xlsx(headers, rows, path)
    print(f"[INFO] Step 1 written to: {path}", file=sys.stderr)


def _strip_jit_hash(name: str) -> str:
    """Strip JIT/Triton compilation hashes from kernel names.

    JIT-compiled kernels (Triton, tilelang) often embed a run-specific
    hex hash in their name so the same kernel may appear as:
      graph-OFF:  act_quant_kernel_abc12345__kernel
      graph-ON:   act_quant_kernel_def67890__kernel
    Stripping the hash allows cross-trace name matching.
    """
    return re.sub(r'_[0-9a-fA-F]{6,}(?=_|$)', '', name)



def _lookup_stat(name: str,
                 stat_lookup: dict, stat_lookup_norm: dict) -> tuple[dict | None, str]:
    """Three-level kernel stat lookup; returns (stat_dict_or_None, method_str)."""
    s = stat_lookup.get(name)
    if s is not None:
        return s, "exact"
    s = stat_lookup_norm.get(_strip_jit_hash(name))
    if s is not None:
        return s, "norm"
    return None, ""


def write_step3(agg_table: list[dict], kernels_off: list[dict],
                kernel_stats: list[dict] | None, path: str,
                callsite_map: dict | None = None) -> None:
    """Export step 3 breakdown to Excel: one row per call site, aggregated over
    every layer of ONE forward.

    Structure (sections, modules, call order, call sites) comes from the graph-OFF
    trace; per-launch timing comes from the graph-ON stats, matched by kernel name
    (exact or hash-stripped). Count is the number of layers that really run this
    call site, measured across all layers, so Σ = avg x Count covers the forward
    without extrapolating from one representative layer.

    Output columns:
      LayerType,        ← which layer types run this call site, e.g. "A+B"
      LayerCount,       ← how many layers that is
      Index, Section, LeafModule,
      KernelName,       ← graph-ON name if matched, else graph-OFF name
      AvgDuration_us,   ← graph-ON avg_dur if matched, else graph-OFF dur
      Count, SumDuration_us, Percentage,
      MatchMethod,      ← "exact" / "norm" / "none"
      GraphOFF_KernelName, GraphOFF_Duration_us,  ← graph-OFF reference
      CallSite,
      TraceCount_fwd, TraceSum_ms_fwd  ← graph-ON ground truth for the whole
          forward, per kernel NAME (repeated on every row sharing the name, so do
          NOT sum this column). Σ of the rows should now match it; the coverage
          report at the end lists any kernel where it does not.
    """
    # Build stat lookups (exact + hash-normalized)
    stat_lookup: dict[str, dict] = {}
    stat_lookup_norm: dict[str, dict] = {}
    if kernel_stats:
        for s in kernel_stats:
            stat_lookup[s["name"]] = s
            norm = _strip_jit_hash(s["name"])
            if norm not in stat_lookup_norm:
                stat_lookup_norm[norm] = s

    headers = ["LayerType", "LayerCount", "Index", "Section", "LeafModule",
               "KernelName", "AvgDuration_us",
               "Count", "SumDuration_us", "Percentage",
               "MatchMethod",
               "GraphOFF_KernelName", "GraphOFF_Duration_us",
               "CallSite",
               "TraceCount_fwd", "TraceSum_ms_fwd"]

    rows = []
    grand_total = 0.0  # sum of per-callsite graph-ON Σ (one forward), for Percentage
    for pos, e in enumerate(agg_table):
        kidx = e["kidx"]
        k = kernels_off[kidx]
        off_name = k["name"]
        off_dur = round(k["dur"], 1)
        cs = (callsite_map or {}).get(kidx, "")

        if kernel_stats:
            s, method = _lookup_stat(off_name, stat_lookup, stat_lookup_norm)
        else:
            s, method = None, "none"

        if s:
            # This row is ONE call site; e["count"] is how many layers of the forward
            # actually reach it, so avg (per launch, graph-ON) x count is its share of
            # the forward. Nothing is extrapolated from a representative layer.
            kernel_name = s["name"]
            avg_dur = round(s["avg_dur"], 3)
            count = e["count"]
            sum_dur = round(s["avg_dur"] * e["count"], 1)
            grand_total += sum_dur
            pct = None  # filled after grand_total is known
        else:
            kernel_name = off_name
            avg_dur = off_dur
            count = ""
            sum_dur = ""
            pct = ""
            method = "none"

        row = [e["types"], e["count"], pos,
               e["section"], e["leaf"] or "(self)",
               kernel_name, avg_dur,
               count, sum_dur, pct,
               method,
               off_name, off_dur,
               cs,
               s["count"] if s else "",
               round(s["sum_dur"] / 1000.0, 3) if s else ""]
        rows.append(row)

    for r in rows:
        if r[9] is None:  # pct placeholder
            r[9] = round(100.0 * r[8] / grand_total, 2) if grand_total else 0.0

    _write_xlsx(headers, rows, path)
    n_matched = sum(1 for r in rows if r and r[10] != "none" and r[10] != "")
    n_total = sum(1 for r in rows if r and r[2] != "")
    print(f"[INFO] Step 3 written to: {path} "
          f"({n_total} kernels, {n_matched} matched to graph-ON)",
          file=sys.stderr)
    _print_coverage(rows, kernel_stats)


def _print_coverage(rows: list[list], kernel_stats: list[dict] | None) -> None:
    """Report kernels whose per-call-site Σ misses the real per-forward Σ, which now
    only happens for kernels that also run OUTSIDE the decoder layers."""
    if not kernel_stats:
        return
    struct: dict[str, float] = {}
    for r in rows:
        if not r or not isinstance(r[8], (int, float)):
            continue
        struct[r[5]] = struct.get(r[5], 0.0) + r[8]
    truth = {s["name"]: s["sum_dur"] for s in kernel_stats}
    covered = sum(struct.values())
    total = sum(truth.values())
    # noise floor: sub-0.1 ms kernels (aranges, fills, cat) drift by a launch or two
    # without saying anything about the forward's cost
    off = [(struct.get(n, 0.0) - t, n, struct.get(n, 0.0), t)
           for n, t in truth.items()
           if t > 0 and abs(struct.get(n, 0.0) - t) / t > 0.05
           and abs(struct.get(n, 0.0) - t) > 100.0]
    print(f"[INFO] Coverage: structural Σ = {fmt_dur(covered)} of the forward's "
          f"{fmt_dur(total)} ({100*covered/total:.1f}%)", file=sys.stderr)
    if off:
        print(f"[WARN] {len(off)} kernels whose per-call-site Σ differs from the real "
              f"per-forward Σ by >5% and >0.1 ms (see TraceSum_ms_fwd):",
              file=sys.stderr)
        for d, n, s, t in sorted(off, key=lambda x: -abs(x[0]))[:10]:
            print(f"         {d/1000:+8.2f} ms  struct={s/1000:7.2f} "
                  f"trace={t/1000:7.2f}  {n[:60]}", file=sys.stderr)


# ===================================================================
# Step 2: Layer structure from cuda-graph-OFF trace
# ===================================================================

def find_decoder_layer_class(trace: dict | list) -> str | None:
    """Find the nn.Module class that represents a DecoderLayer."""
    events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
    module_events = [
        ev for ev in events
        if isinstance(ev, dict) and ev.get("ph") == "X"
        and ev.get("name", "").startswith("nn.Module: ") and ev.get("dur", 0) > 0
    ]
    if not module_events:
        return None

    class_info: dict[str, dict] = {}
    for ev in module_events:
        cls = re.sub(r'_\d+$', '', ev["name"].replace("nn.Module: ", ""))
        if cls not in class_info:
            class_info[cls] = {"instances": set(), "total_dur": 0, "count": 0}
        class_info[cls]["instances"].add(ev["name"])
        class_info[cls]["total_dur"] += ev.get("dur", 0)
        class_info[cls]["count"] += 1

    candidates = [(cls, len(info["instances"]), info["total_dur"] / info["count"])
                  for cls, info in class_info.items() if len(info["instances"]) >= 3]
    if not candidates:
        return None

    named = [c for c in candidates if "Decoder" in c[0] or "Layer" in c[0]]
    pool = named if named else candidates
    return max(pool, key=lambda c: c[1] * c[2])[0]


def get_one_forward_pass(trace: dict | list, cls_name: str,
                         window: tuple[float, float, str] | None = None) -> list[dict]:
    """Get nn.Module events for one forward pass (2nd pass to skip warmup).

    With `window` (from find_forward_windows on the CPU-side annotations) only
    module events inside that forward are considered, so the structure comes from
    the same batch shape the timing stats were taken from.
    """
    events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
    all_events = sorted(
        [ev for ev in events
         if isinstance(ev, dict) and ev.get("ph") == "X"
         and re.sub(r'_\d+$', '', ev.get("name", "").replace("nn.Module: ", "")) == cls_name
         and (window is None or window[0] <= ev["ts"] < window[1])],
        key=lambda ev: ev["ts"],
    )
    instances = set(ev["name"] for ev in all_events)
    n_layers = len(instances)
    if n_layers == 0:
        return []
    n_passes = len(all_events) // n_layers
    pass_idx = min(1, n_passes - 1)  # 2nd pass if available
    return all_events[pass_idx * n_layers: (pass_idx + 1) * n_layers]


def find_direct_children(trace: dict | list, parent_ev: dict,
                          target_classes: set[str] | None = None) -> list[dict]:
    """Find direct nn.Module children of a parent module event."""
    events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
    pts, pend = parent_ev["ts"], parent_ev["ts"] + parent_ev.get("dur", 0)

    # Get all nn.Module events within parent's time range (excluding parent)
    children = []
    for ev in events:
        if not isinstance(ev, dict) or ev.get("ph") != "X":
            continue
        if not ev.get("name", "").startswith("nn.Module: "):
            continue
        if ev is parent_ev or ev["name"] == parent_ev["name"]:
            continue
        ets = ev.get("ts", 0)
        eend = ets + ev.get("dur", 0)
        if ets >= pts and eend <= pend + 1:
            cls = re.sub(r'_\d+$', '', ev["name"].replace("nn.Module: ", ""))
            if target_classes is None or cls in target_classes:
                children.append(ev)

    # Filter to direct children only (not nested inside another child)
    children.sort(key=lambda e: e["ts"])
    direct = []
    for c in children:
        cts, cend = c["ts"], c["ts"] + c.get("dur", 0)
        is_nested = any(
            o["ts"] < cts and o["ts"] + o.get("dur", 0) > cend
            for o in children if o is not c
        )
        if not is_nested:
            direct.append(c)

    return direct


def build_ext_id_map(trace: dict | list, kernels: list[dict]) -> tuple:
    """Build kernel lookup maps and sorted runtime events.

    Returns (ext_to_kidx, corr_to_kidx, runtime, rt_ts) where:
      ext_to_kidx: External id → kernel index
      corr_to_kidx: correlation → kernel index (for kernels without External id)
      runtime: sorted cuda_runtime events (with External id or correlation)
      rt_ts: timestamps for binary search
    """
    events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
    ext_to_kidx = {}
    corr_to_kidx = {}
    for i, k in enumerate(kernels):
        eid = k.get("args", {}).get("External id")
        corr = k.get("args", {}).get("correlation")
        if eid is not None and eid not in ext_to_kidx:
            ext_to_kidx[eid] = i
        elif eid is None and corr is not None and corr not in corr_to_kidx:
            corr_to_kidx[corr] = i

    runtime = sorted(
        [ev for ev in events
         if isinstance(ev, dict) and ev.get("cat") == "cuda_runtime"
         and ev.get("ph") == "X"
         and (ev.get("args", {}).get("External id") is not None
              or ev.get("args", {}).get("correlation") is not None)],
        key=lambda ev: ev["ts"],
    )
    rt_ts = [ev["ts"] for ev in runtime]
    return ext_to_kidx, corr_to_kidx, runtime, rt_ts


def build_callsite_map(trace: dict | list, kernels: list[dict],
                        ext_to_kidx: dict, runtime: list[dict],
                        target_kidxs: set[int] | None = None) -> dict:
    """
    Build kernel_index → call site string mapping.
    Finds the innermost python_function from sglang source code
    that contains each kernel's cuda_runtime launch event.

    If target_kidxs is given, only resolve those kernel indices (fast).
    """
    events = trace if isinstance(trace, list) else trace.get("traceEvents", [])

    # Build per-thread sorted list of python_function events
    py_by_thread: dict[int, list] = {}
    py_ts_by_thread: dict[int, list] = {}
    for ev in events:
        if not isinstance(ev, dict) or ev.get("cat") != "python_function":
            continue
        if ev.get("ph") != "X" or ".py" not in ev.get("name", ""):
            continue
        tid = ev.get("tid")
        if tid not in py_by_thread:
            py_by_thread[tid] = []
        py_by_thread[tid].append(ev)

    for tid in py_by_thread:
        py_by_thread[tid].sort(key=lambda e: e["ts"])
        py_ts_by_thread[tid] = [e["ts"] for e in py_by_thread[tid]]

    # Build ext_id and correlation → runtime event mappings
    ext_to_runtime = {}
    corr_to_runtime = {}
    for rt in runtime:
        args = rt.get("args", {})
        eid = args.get("External id")
        corr = args.get("correlation")
        if eid is not None and eid not in ext_to_runtime:
            ext_to_runtime[eid] = rt
        if corr is not None and corr not in corr_to_runtime:
            corr_to_runtime[corr] = rt

    # Resolve call sites
    callsite_map = {}
    resolve_list = target_kidxs if target_kidxs else range(len(kernels))

    for i in resolve_list:
        k = kernels[i]
        args = k.get("args", {})
        eid = args.get("External id")
        rt = ext_to_runtime.get(eid) if eid is not None else None
        if rt is None:
            corr = args.get("correlation")
            rt = corr_to_runtime.get(corr) if corr is not None else None
        if rt is None:
            continue
        rt_ts_val = rt["ts"]
        rt_tid = rt["tid"]

        if rt_tid not in py_by_thread:
            continue

        py_list = py_by_thread[rt_tid]
        ts_list = py_ts_by_thread[rt_tid]

        # Binary search to find approximate position
        pos = bisect.bisect_right(ts_list, rt_ts_val) - 1

        # Scan backwards. Don't break on gaps — outer functions with
        # longer duration may have started earlier and still contain rt_ts.
        # Limit scan to 2000 events or 50ms back to keep it fast.
        best = None
        best_dur = float("inf")
        scan_limit = max(0, pos - 2000)
        ts_limit = rt_ts_val - 50_000  # 50ms back max

        for j in range(max(0, pos), scan_limit - 1, -1):
            pf = py_list[j]
            pf_ts = pf["ts"]
            if pf_ts > rt_ts_val:
                continue
            if pf_ts < ts_limit:
                break  # too far back in time
            pf_end = pf_ts + pf.get("dur", 0)
            if pf_end < rt_ts_val:
                continue  # this one doesn't contain rt, but keep scanning
            # This event contains the runtime call
            name = pf["name"]
            dur = pf.get("dur", 0)
            if "/sglang/" in name and dur < best_dur:
                best = name
                best_dur = dur

        if best:
            callsite_map[i] = best

    return callsite_map


def find_layer_sections(pf_events: list[dict], layer_ev: dict) -> list[dict]:
    """Find section-level functions/modules within a decoder layer.

    Sections are direct children of the decoder layer's forward method,
    such as prepare_attn, DeepseekV2AttentionMLA, prepare_mlp, etc.

    Args:
        pf_events: python_function events within the layer's time range,
                   sorted by (ts, -dur).
        layer_ev: the decoder layer nn.Module event.

    Returns sorted list of {"name": str, "ts": float, "end": float}
    """
    if not pf_events:
        return []

    layer_dur = layer_ev.get("dur", 1)

    # Find the model's forward function (sglang source, large duration)
    fwd = None
    for ev in pf_events:
        name = ev.get("name", "")
        if ("/sglang/" in name and ": forward" in name
                and ev.get("dur", 0) > layer_dur * 0.5):
            fwd = ev
            break

    if fwd is None:
        return []

    fts = fwd["ts"]
    fend = fts + fwd.get("dur", 0)

    # Collect events contained in forward
    children = [ev for ev in pf_events
                if ev is not fwd
                and ev["ts"] >= fts
                and ev["ts"] + ev.get("dur", 0) <= fend + 1]

    # Keep only direct children (not nested inside another child)
    children.sort(key=lambda e: e["ts"])
    direct = []
    for c in children:
        cts, cend = c["ts"], c["ts"] + c.get("dur", 0)
        if not any(o["ts"] < cts and o["ts"] + o.get("dur", 0) > cend
                   for o in children if o is not c):
            direct.append(c)

    # Filter to significant functions and extract clean names
    sections = []
    for c in direct:
        name = c.get("name", "")
        dur = c.get("dur", 0)
        # Skip tiny helpers and torch internals
        if dur < 1:
            continue
        if ("torch/" in name or "nn/modules/" in name
                or "<built-in" in name) and not name.startswith("nn.Module:"):
            continue

        if name.startswith("nn.Module: "):
            clean = re.sub(r'_\d+$', '', name.replace("nn.Module: ", ""))
        else:
            match = re.search(r':\s*(\w+)\s*$', name)
            clean = match.group(1) if match else name

        sections.append({"name": clean, "ts": c["ts"],
                         "end": c["ts"] + dur})

    sections.sort(key=lambda s: s["ts"])
    return sections


def find_pf_leaf(all_pf: list[dict], all_pf_ts: list[float],
                 rt_ts: float, section_name: str) -> str:
    """Find innermost python_function as fallback leaf label.

    Used when no nn.Module wraps a kernel. Looks for the innermost
    non-torch, non-module python_function that contains the runtime
    event, excluding the section function itself.

    Returns "filename.py(line)" or empty string.
    """
    pos = bisect.bisect_right(all_pf_ts, rt_ts) - 1
    best_name = None
    best_dur = float("inf")

    for j in range(max(0, pos), max(0, pos - 2000), -1):
        pf = all_pf[j]
        pf_ts = pf["ts"]
        if pf_ts > rt_ts:
            continue
        if pf_ts < rt_ts - 50_000:
            break
        pf_end = pf_ts + pf.get("dur", 0)
        if pf_end < rt_ts:
            continue
        name = pf.get("name", "")
        dur = pf.get("dur", 0)
        # Skip torch internals, nn.Module wrappers, built-ins, triton runtime
        if ("torch/" in name or "nn/modules/" in name
                or "<built-in" in name or name.startswith("nn.Module:")
                or "triton/runtime/" in name or "triton/backends/" in name):
            continue
        if dur < best_dur:
            best_name = name
            best_dur = dur

    if not best_name:
        return ""

    # Extract "filename.py(line)" from full path
    m = re.search(r'([^/]+\.py\(\d+\))', best_name)
    if not m:
        return ""

    short = m.group(1)
    # Skip if it's the section's own function (e.g. forward, prepare_attn)
    func_match = re.search(r':\s*(\w+)\s*$', best_name)
    if func_match and func_match.group(1) == section_name:
        return ""

    return short


def get_section_name(sections: list[dict], ts: float) -> str:
    """Find which section contains the given timestamp."""
    for s in sections:
        if s["ts"] <= ts <= s["end"]:
            return s["name"]
    return "(layer)"


def get_kernels_for_module(module_ev: dict, ext_to_kidx: dict,
                            corr_to_kidx: dict,
                            runtime: list[dict], rt_ts: list[float]) -> list[int]:
    """Get GPU kernel indices launched by a module event."""
    mod_ts = module_ev["ts"]
    mod_end = mod_ts + module_ev.get("dur", 0)
    lo = bisect.bisect_left(rt_ts, mod_ts)
    hi = bisect.bisect_right(rt_ts, mod_end)
    kidxs = set()
    for j in range(lo, hi):
        args = runtime[j].get("args", {})
        eid = args.get("External id")
        if eid is not None and eid in ext_to_kidx:
            kidxs.add(ext_to_kidx[eid])
        else:
            corr = args.get("correlation")
            if corr is not None and corr in corr_to_kidx:
                kidxs.add(corr_to_kidx[corr])
    return sorted(kidxs)


def find_deep_kernel_labels(module_events: list[dict], layer_ev: dict,
                             kidxs: list[int], kernels: list[dict],
                             ext_to_rt: dict, corr_to_rt: dict) -> dict:
    """Find deepest nn.Module path for each kernel within a layer.

    Returns {kernel_idx: "ParentModule > LeafModule"}.
    """
    pts = layer_ev["ts"]
    pend = pts + layer_ev.get("dur", 0)

    # Collect all nn.Module descendants within layer's time range
    descendants = []
    for ev in module_events:
        if ev is layer_ev:
            continue
        ets = ev.get("ts", 0)
        eend = ets + ev.get("dur", 0)
        if ets >= pts and eend <= pend + 1:
            cls = re.sub(r'_\d+$', '', ev["name"].replace("nn.Module: ", ""))
            descendants.append({
                "cls": cls, "ts": ets, "end": eend, "dur": ev.get("dur", 0),
            })

    if not descendants:
        return {}

    # Build hierarchy paths: sort by duration desc (parents first)
    descendants.sort(key=lambda m: m["dur"], reverse=True)
    for i, m in enumerate(descendants):
        parent_path = None
        parent_dur = float("inf")
        for j in range(i):
            p = descendants[j]
            if (p["ts"] <= m["ts"] and p["end"] >= m["end"] - 1
                    and p["dur"] < parent_dur):
                parent_path = p["path"]
                parent_dur = p["dur"]
        m["path"] = f"{parent_path} > {m['cls']}" if parent_path else m["cls"]

    # For each kernel, find deepest enclosing module via runtime event timestamp
    labels = {}
    for ki in kidxs:
        k = kernels[ki]
        args = k.get("args", {})
        eid = args.get("External id")
        rt = ext_to_rt.get(eid) if eid is not None else None
        if rt is None:
            corr = args.get("correlation")
            rt = corr_to_rt.get(corr) if corr is not None else None
        if rt is None:
            continue
        rt_ts_val = rt["ts"]

        best = None
        best_dur = float("inf")
        for m in descendants:
            if m["ts"] <= rt_ts_val <= m["end"] and m["dur"] < best_dur:
                best = m
                best_dur = m["dur"]

        if best:
            labels[ki] = best["path"]

    return labels


def analyze_layer_structure(trace: dict | list, kernels: list[dict],
                            window: tuple[float, float, str] | None = None):
    """
    Step 2: Discover layer structure.
    Returns (cls_name, layer_types, callsite_map, fine_types, agg_table):
      layer_types: layers grouped by sub-module signature (what the model looks
        like), [ { 'count': int, 'layers': [name, ...], 'sub_modules': [cls, ...],
                  'kernel_breakdown': [ (kernel_idx, section, leaf), ... ] }, ... ]
      callsite_map: { kernel_idx: "file.py(line): func_name" }
      fine_types, agg_table: layers grouped by what they actually run, and every
        call site aggregated over all layers — see _aggregate_call_sites()
    """
    cls_name = find_decoder_layer_class(trace)
    if cls_name is None:
        return None, [], {}, [], []

    forward_pass = get_one_forward_pass(trace, cls_name, window)
    if not forward_pass:
        return cls_name, [], {}, [], []

    ext_to_kidx, corr_to_kidx, runtime, rt_ts = build_ext_id_map(trace, kernels)

    # Pre-compute module events and runtime mapping for deep labeling
    events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
    module_events = [
        ev for ev in events
        if isinstance(ev, dict) and ev.get("ph") == "X"
        and ev.get("name", "").startswith("nn.Module: ")
    ]
    module_events.sort(key=lambda e: e["ts"])

    ext_to_rt = {}
    corr_to_rt = {}
    for rt in runtime:
        args = rt.get("args", {})
        eid = args.get("External id")
        corr = args.get("correlation")
        if eid is not None and eid not in ext_to_rt:
            ext_to_rt[eid] = rt
        if corr is not None and corr not in corr_to_rt:
            corr_to_rt[corr] = rt

    # Discover significant child classes by sampling multiple layers
    # (different layer types may have different children)
    significant_classes = set()
    sample_indices = set()
    # Sample a few layers from different positions
    for idx in [0, 1, len(forward_pass) // 2, len(forward_pass) - 1]:
        sample_indices.add(min(idx, len(forward_pass) - 1))
    for idx in sample_indices:
        sample = forward_pass[idx]
        all_children = find_direct_children(trace, sample)
        for c in all_children:
            cls = re.sub(r'_\d+$', '', c["name"].replace("nn.Module: ", ""))
            avg_dur = c.get("dur", 0)
            if avg_dur > sample.get("dur", 1) * 0.05:
                significant_classes.add(cls)

    print(f"[INFO] Found nn.Module: {cls_name} "
          f"({len(forward_pass)} layers per forward pass)", file=sys.stderr)
    print(f"[INFO] Sub-module classes: {', '.join(sorted(significant_classes))}",
          file=sys.stderr)

    # Pre-filter python_function events for section detection
    all_pf = sorted(
        [ev for ev in events
         if isinstance(ev, dict) and ev.get("ph") == "X"
         and ev.get("cat") == "python_function" and ev.get("dur", 0) > 0],
        key=lambda e: (e["ts"], -e.get("dur", 0)),
    )
    all_pf_ts = [e["ts"] for e in all_pf]

    # Analyze each layer
    layer_data = []
    for layer_ev in forward_pass:
        layer_name = layer_ev["name"].replace("nn.Module: ", "")
        children = find_direct_children(trace, layer_ev, significant_classes)
        sub_mods = [re.sub(r'_\d+$', '', c["name"].replace("nn.Module: ", ""))
                    for c in children]

        # Find sections (high-level code blocks in the layer's forward)
        lts = layer_ev["ts"]
        lend = lts + layer_ev.get("dur", 0)
        pf_lo = bisect.bisect_left(all_pf_ts, lts)
        pf_hi = bisect.bisect_right(all_pf_ts, lend)
        sections = find_layer_sections(all_pf[pf_lo:pf_hi], layer_ev)

        # Get kernel-to-module mapping (deep — find leaf module)
        all_kidxs = get_kernels_for_module(layer_ev, ext_to_kidx,
                                            corr_to_kidx, runtime, rt_ts)
        kernel_labels = find_deep_kernel_labels(
            module_events, layer_ev, all_kidxs, kernels,
            ext_to_rt, corr_to_rt)

        breakdown = []
        for ki in all_kidxs:
            # Determine section from python_function context
            k_args = kernels[ki].get("args", {})
            eid = k_args.get("External id")
            rt = ext_to_rt.get(eid) if eid is not None else None
            if rt is None:
                corr = k_args.get("correlation")
                rt = corr_to_rt.get(corr) if corr is not None else None
            section = get_section_name(sections, rt["ts"]) if rt else "(layer)"

            # Leaf module from nn.Module hierarchy
            leaf = kernel_labels.get(ki, "")
            # Strip section prefix from leaf if it starts with the section name
            if leaf.startswith(section + " > "):
                leaf = leaf[len(section) + 3:]
            elif leaf == section:
                leaf = ""

            # Fallback: use innermost python_function when no nn.Module
            if not leaf and rt:
                leaf = find_pf_leaf(all_pf, all_pf_ts, rt["ts"], section)

            breakdown.append((ki, section, leaf))
        layer_data.append({
            "name": layer_name,
            "sub_modules": tuple(sub_mods),
            "kernel_breakdown": breakdown,
            "n_kernels": len(all_kidxs),
        })

    # === Fallback: recover unlinked kernels via GPU-timestamp interpolation ===
    # Some kernels (e.g. launched via cudaLaunchKernelExC) lack matching
    # cuda_runtime events, so get_kernels_for_module cannot find them.
    # We handle two sub-cases:
    #   (a) Kernels that fall WITHIN the forward-pass GPU time window →
    #       assign to the same layer/section as the nearest preceding linked kernel.
    #   (b) Kernels that fall OUTSIDE (after) the forward-pass window →
    #       these are inter-layer / model-level ops (allreduce, residual norms, etc.)
    #       that execute between decoder layers.  Collect them into a synthetic
    #       "(inter-layer)" layer_data entry so they show up labelled in Step 3.
    linked_entries = []  # (gpu_ts, kernel_idx, layer_data_idx, section)
    for ldi, ld in enumerate(layer_data):
        for ki, section, _leaf in ld["kernel_breakdown"]:
            linked_entries.append((kernels[ki]["ts"], ki, ldi, section))
    linked_entries.sort()

    if linked_entries:
        assigned_kidxs = {e[1] for e in linked_entries}
        fp_gpu_start = linked_entries[0][0]
        fp_gpu_end = max(kernels[e[1]]["ts_end"] for e in linked_entries)
        entry_ts = [e[0] for e in linked_entries]

        unlinked_by_layer: dict[int, list] = {}
        inter_layer_kidxs: list[int] = []   # case (b): outside forward-pass window

        for ki in range(len(kernels)):
            if ki in assigned_kidxs:
                continue
            k = kernels[ki]
            if k["ts"] < fp_gpu_start:
                continue  # before forward pass — ignore
            if k["ts"] > fp_gpu_end:
                # case (b): inter-layer / post-layer kernel
                inter_layer_kidxs.append(ki)
                continue
            # case (a): within window — assign to nearest preceding linked kernel
            pos = bisect.bisect_right(entry_ts, k["ts"]) - 1
            if pos < 0:
                ldi = linked_entries[0][2]
                inferred_section = linked_entries[0][3]
            else:
                ldi = linked_entries[pos][2]
                inferred_section = linked_entries[pos][3]
            unlinked_by_layer.setdefault(ldi, []).append(
                (ki, inferred_section))

        n_recovered = 0
        for ldi, items in unlinked_by_layer.items():
            ld = layer_data[ldi]
            existing = list(ld["kernel_breakdown"])
            for ki, section in items:
                existing.append((ki, section, ""))
            existing.sort(key=lambda item: kernels[item[0]]["ts"])
            ld["kernel_breakdown"] = existing
            n_recovered += len(items)

        if n_recovered:
            print(f"[INFO] Recovered {n_recovered} unlinked kernels via "
                  f"GPU-timestamp interpolation", file=sys.stderr)

        if inter_layer_kidxs:
            print(f"[INFO] Skipping {len(inter_layer_kidxs)} kernels outside "
                  f"decoder-layer GPU window (inter-layer / post-layer ops)",
                  file=sys.stderr)

    # Group by sub_module signature
    groups: OrderedDict = OrderedDict()
    for i, ld in enumerate(layer_data):
        key = ld["sub_modules"]
        if key not in groups:
            groups[key] = {"layers": [], "indices": [], "example_breakdown": None}
        groups[key]["layers"].append(ld["name"])
        groups[key]["indices"].append(i)
        if groups[key]["example_breakdown"] is None and i >= 1:
            groups[key]["example_breakdown"] = ld["kernel_breakdown"]
    # Fallback: use first if no 2nd instance
    for key, g in groups.items():
        if g["example_breakdown"] is None:
            idx = g["indices"][0]
            g["example_breakdown"] = layer_data[idx]["kernel_breakdown"]

    layer_types = []
    for sub_mods, g in groups.items():
        layer_types.append({
            "sub_modules": list(sub_mods),
            "count": len(g["layers"]),
            "layers": g["layers"],
            "indices": g["indices"],
            "kernel_breakdown": g["example_breakdown"],
        })

    fine_types, agg_table = _aggregate_call_sites(layer_data, kernels)
    if len(fine_types) > len(layer_types):
        shape = ", ".join(f"{t['label']}={t['count']}" for t in fine_types)
        print(f"[INFO] {len(layer_types)} layer types by sub-module signature, "
              f"{len(fine_types)} once the kernels they run are compared "
              f"({shape} layers)", file=sys.stderr)

    # Build callsite map only for the aggregated call sites (fast)
    target_kidxs = {e["kidx"] for e in agg_table}
    print(f"[INFO] Resolving call sites for {len(target_kidxs)} kernels...",
          file=sys.stderr)
    callsite_map = build_callsite_map(trace, kernels, ext_to_kidx, runtime,
                                      target_kidxs)

    return cls_name, layer_types, callsite_map, fine_types, agg_table


def _aggregate_call_sites(layer_data: list[dict], kernels: list[dict]):
    """Group layers by what they actually run, and aggregate every call site over
    ALL layers of the forward.

    The sub-module signature used above only sees a layer's direct children, so
    layers differing deeper collapse together: in GLM-5.2 one layer's DSA top-k is
    shared by the next three (`index_topk_freq`), and those three never run the
    indexer, yet all 75 MoE layers look identical from the outside. Grouping by the
    (section, leaf, kernel) multiset separates them.

    Aggregating over every layer also means Count is MEASURED (the indexer chain
    really runs in 21 of 78 layers) instead of extrapolated from one layer, so a
    call site is listed once with the number of layers that run it.

    Returns (fine_types, agg_table) where agg_table entries are ordered by their
    mean relative position inside a layer:
        {section, leaf, kernel, kidx (a representative launch), count, labels}
    """
    def key_of(item):
        kidx, section, leaf = item
        return (section, leaf, kernels[kidx]["name"])

    fine: OrderedDict = OrderedDict()
    for i, ld in enumerate(layer_data):
        sig = tuple(sorted(Counter(key_of(it)
                                   for it in ld["kernel_breakdown"]).items()))
        fine.setdefault(sig, []).append(i)

    fine_types = []
    for n, (sig, idxs) in enumerate(fine.items()):
        fine_types.append({
            "label": chr(ord("A") + n),
            "count": len(idxs),
            "indices": idxs,
            "sub_modules": list(layer_data[idxs[0]]["sub_modules"]),
            "keys": {k for k, _c in sig},
        })
    # describe each type as a diff against the most common one
    def names(keys):
        return sorted({(leaf or sec) for sec, leaf, _k in keys})

    baseline = max(fine_types, key=lambda t: t["count"]) if fine_types else None
    for t in fine_types:
        if t is baseline:
            t["diff"] = ["(most common)"]
            continue
        t["diff"] = ([f"+{n}" for n in names(t["keys"] - baseline["keys"])]
                     + [f"-{n}" for n in names(baseline["keys"] - t["keys"])])
    label_of = {i: t["label"] for t in fine_types for i in t["indices"]}

    agg: dict = {}
    for i, ld in enumerate(layer_data):
        seen: Counter = Counter()
        n = max(len(ld["kernel_breakdown"]), 1)
        for pos, item in enumerate(ld["kernel_breakdown"]):
            k = key_of(item)
            nth = seen[k]
            seen[k] += 1
            e = agg.setdefault((k, nth), {
                "section": k[0], "leaf": k[1], "kernel": k[2],
                "kidx": item[0], "count": 0, "labels": set(), "pos": []})
            e["count"] += 1
            e["labels"].add(label_of[i])
            e["pos"].append(pos / n)

    for e in agg.values():
        e["order"] = sum(e["pos"]) / len(e["pos"])
        e["types"] = ("all" if len(e["labels"]) == len(fine_types)
                      else "+".join(sorted(e["labels"])))
    # Sort by mean position, but keep each section in one block: a call site that
    # only exists in a few layers (e.g. the dense MLP) would otherwise land in the
    # middle of another section's rows.
    sec_order: dict = {}
    for e in agg.values():
        sec_order[e["section"]] = min(sec_order.get(e["section"], 9e9), e["order"])
    agg_table = sorted(agg.values(),
                       key=lambda e: (sec_order[e["section"]], e["order"]))
    return fine_types, agg_table


def print_step2(cls_name: str, layer_types: list[dict],
                fine_types: list[dict] | None = None) -> None:
    total_layers = sum(lt["count"] for lt in layer_types)
    print(f"\n{'='*100}")
    print(f" Step 2: Layer Structure (cuda-graph-OFF)")
    print(f" Model Layer: {cls_name}")
    print(f" Layers per forward pass: {total_layers}")
    print(f" Distinct layer types: {len(layer_types)}")
    print(f"{'='*100}")

    for i, lt in enumerate(layer_types):
        label = chr(ord("A") + i)
        layers = lt["layers"]
        if len(layers) <= 5:
            name_str = ", ".join(layers)
        else:
            name_str = f"{layers[0]} .. {layers[-1]}"

        print(f"\n  Type {label}: {lt['count']} layers")
        print(f"  Layers: {name_str}")
        print(f"  Sub-modules: {' + '.join(lt['sub_modules'])}")

    if fine_types and len(fine_types) > len(layer_types):
        print(f"\n  Sub-modules alone merge layers that run different kernels; "
              f"by what they actually run there are {len(fine_types)} types:")
        for t in fine_types:
            diff = ", ".join(t["diff"][:5])
            if len(t["diff"]) > 5:
                diff += f", ... (+{len(t['diff']) - 5} more)"
            print(f"    {t['label']}: {t['count']:>3} layers  "
                  f"idx {_index_ranges(t['indices'])}  vs {diff}")

    print(f"\n{'='*100}\n")


def _index_ranges(idxs: list[int]) -> str:
    """Collapse layer indices into ranges/steps, e.g. '0-2' or '6,10,..,74'."""
    if len(idxs) <= 3:
        return ",".join(str(i) for i in idxs)
    steps = {b - a for a, b in zip(idxs, idxs[1:])}
    if steps == {1}:
        return f"{idxs[0]}-{idxs[-1]}"
    if len(steps) == 1:
        return f"{idxs[0]},{idxs[1]},..,{idxs[-1]} (every {steps.pop()})"
    return f"{idxs[0]},{idxs[1]},..,{idxs[-1]}"


# ===================================================================
# Step 3: Per-layer kernel breakdown with sub-module labels
# ===================================================================

def print_step3(cls_name: str, agg_table: list[dict],
                kernels: list[dict], kernel_stats: list[dict] | None,
                callsite_map: dict | None = None) -> None:
    """
    Print the call-site breakdown of one forward: sections in call order, each
    call site once, with the layer types that run it and its share of the forward.
    Structure comes from the graph-OFF trace; per-launch avg from graph-ON step 1
    when the kernel name matches.
    """
    stat_lookup: dict[str, dict] = {}
    stat_lookup_norm: dict[str, dict] = {}
    if kernel_stats:
        for s in kernel_stats:
            stat_lookup[s["name"]] = s
            norm = _strip_jit_hash(s["name"])
            stat_lookup_norm.setdefault(norm, s)

    print(f"\n{'='*100}")
    print(f" Step 3: Call-site Breakdown of One Forward ({cls_name})")
    print(f"{'='*100}")

    current_sec = None
    sec_sum = 0.0
    total_sum = 0.0

    print(f"  {'#':>3}  {'Types':>9} {'Layers':>6}  {'Detail':<34s}  "
          f"{'Avg':>9}  {'Sum':>10}  Kernel Name")
    print(f"  {'-'*3}  {'-'*9} {'-'*6}  {'-'*34}  {'-'*9}  {'-'*10}  {'-'*40}")

    for pos, e in enumerate(agg_table):
        k = kernels[e["kidx"]]
        s, _m = (_lookup_stat(k["name"], stat_lookup, stat_lookup_norm)
                 if kernel_stats else (None, "none"))
        avg = s["avg_dur"] if s else k["dur"]
        row_sum = avg * e["count"]
        total_sum += row_sum

        if e["section"] != current_sec:
            if current_sec is not None:
                print(f"  {'':>3}  {'':>16}  {'Subtotal':>34}  {'':>9}  "
                      f"{fmt_dur(sec_sum):>10}\n")
            current_sec = e["section"]
            sec_sum = 0.0
            print(f"  ---- {current_sec} ----")
        sec_sum += row_sum

        print(f"  {pos:>3}  {e['types']:>9} {e['count']:>6}  "
              f"{(e['leaf'] or '(self)')[:34]:<34s}  {fmt_dur(avg):>9}  "
              f"{fmt_dur(row_sum):>10}  {k['name'][:60]}")
        cs = callsite_map.get(e["kidx"], "") if callsite_map else ""
        if cs:
            print(f"  {'':>3}  {'':>16}  {'':>34}  {'':>9}  {'':>10}  caller: {cs}")

    if current_sec is not None:
        print(f"  {'':>3}  {'':>16}  {'Subtotal':>34}  {'':>9}  "
              f"{fmt_dur(sec_sum):>10}")
    print(f"\n  {'':>3}  {'':>16}  {'TOTAL':>34}  {'':>9}  "
          f"{fmt_dur(total_sum):>10}  ({len(agg_table)} call sites)")

    print(f"\n{'='*100}\n")


# ===================================================================
# CLI
# ===================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze PyTorch profiler traces for layer structure and kernel breakdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--graph-on", metavar="TRACE",
                   help="Cuda-graph-ON trace (for kernel statistics)")
    p.add_argument("--graph-off", metavar="TRACE",
                   help="Cuda-graph-OFF trace (for layer structure)")
    p.add_argument("--stream", type=int, default=None,
                   help="GPU stream ID (default: auto-detect)")
    p.add_argument("--forward-match", metavar="SUBSTR", default=None,
                   help="restrict statistics to ONE forward pass: the profiler "
                        "annotation wrapping it must contain SUBSTR, e.g. "
                        "'bs=3' or 'DECODE bs=64'. Without this, Step 1 averages "
                        "each kernel over the WHOLE trace, which mixes forwards of "
                        "different batch/token sizes (a 7.6k-token and a 16.4k-token "
                        "prefill chunk get blended into one average).")
    p.add_argument("--forward-pick", choices=["first", "last", "min", "median", "max"],
                   default="median",
                   help="which matching forward to use when several match "
                        "(min/median/max are by Σ kernel duration; default: median)")
    p.add_argument("--out", metavar="DIR",
                   help="Export Excel files to directory (step1_kernel_stats.xlsx, step3_layer_breakdown.xlsx)")
    p.add_argument("--tag", metavar="TAG", default="",
                   help="Append tag to output filenames, e.g. --tag _ATOM → step3_layer_breakdown_ATOM.xlsx")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.graph_on and not args.graph_off:
        sys.exit("Provide at least one: --graph-on or --graph-off")

    # Prepare output dir
    out_dir = None
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

    kernel_stats = None
    cls_name = None
    layer_types = None
    kernels_off = None

    # --- Step 1: Kernel statistics from graph-ON trace ---
    if args.graph_on:
        print(f"[INFO] Step 1: Loading graph-ON trace: {args.graph_on}", file=sys.stderr)
        trace_on = load_trace(args.graph_on)
        stream_on = args.stream if args.stream is not None else auto_detect_stream(trace_on)
        kernels_on = extract_gpu_kernels(trace_on, stream=stream_on)
        stream_desc = f"stream {stream_on}" if stream_on is not None else "all streams"
        print(f"[INFO] Found {len(kernels_on)} kernels on {stream_desc}",
              file=sys.stderr)

        if args.forward_match:
            wins = find_forward_windows(trace_on, "gpu_user_annotation",
                                        args.forward_match, stream_on)
            if not wins:
                sys.exit(f"[ERROR] no forward annotation matching "
                         f"{args.forward_match!r} in {args.graph_on}")
            w = select_forward_window(wins, kernels_on, args.forward_pick)
            kernels_on = [k for k in kernels_on if w[0] <= k["ts"] < w[1]]
            print(f"[INFO] Step 1 restricted to ONE forward: {w[2]} "
                  f"({args.forward_pick}-of-{len(wins)}) → {len(kernels_on)} kernels, "
                  f"Σ={fmt_dur(sum(k['dur'] for k in kernels_on))}", file=sys.stderr)

        kernel_stats = compute_kernel_stats(kernels_on)
        print_step1(kernel_stats)
        del trace_on, kernels_on  # free memory; stats are all we need

        if out_dir:
            write_step1(kernel_stats, str(out_dir / f"step1_kernel_stats{args.tag}.xlsx"))

    # --- Step 2: Layer structure from graph-OFF trace ---
    if args.graph_off:
        print(f"[INFO] Step 2: Loading graph-OFF trace: {args.graph_off}",
              file=sys.stderr)
        trace_off = load_trace(args.graph_off)
        stream_off = args.stream if args.stream is not None else auto_detect_stream(trace_off)
        kernels_off = extract_gpu_kernels(trace_off, stream=stream_off)
        stream_desc = f"stream {stream_off}" if stream_off is not None else "all streams"
        print(f"[INFO] Found {len(kernels_off)} kernels on {stream_desc}",
              file=sys.stderr)

        window_off = None
        if args.forward_match:
            # CPU-side annotations bound the nn.Module events; the label differs
            # slightly from the graph-ON run (e.g. toks=16341 vs toks=16368), so
            # match on the shared part (e.g. "bs=3").
            wins_off = find_forward_windows(trace_off, "user_annotation",
                                            args.forward_match)
            if wins_off:
                window_off = wins_off[min(1, len(wins_off) - 1)]
                print(f"[INFO] Step 2 structure restricted to forward: "
                      f"{window_off[2]} (of {len(wins_off)} matching)",
                      file=sys.stderr)
            else:
                print(f"[WARN] no CPU-side forward annotation matching "
                      f"{args.forward_match!r} in the graph-OFF trace; "
                      f"falling back to the 2nd forward pass", file=sys.stderr)

        (cls_name, layer_types, callsite_map,
         fine_types, agg_table) = analyze_layer_structure(
            trace_off, kernels_off, window_off)
        if cls_name:
            print_step2(cls_name, layer_types, fine_types)

            # --- Step 3: Combined breakdown ---
            if agg_table:
                print_step3(cls_name, agg_table, kernels_off, kernel_stats,
                            callsite_map)
                if out_dir:
                    write_step3(agg_table, kernels_off, kernel_stats,
                                str(out_dir / f"step3_layer_breakdown{args.tag}.xlsx"),
                                callsite_map)
        else:
            print("[WARN] No nn.Module DecoderLayer events found.", file=sys.stderr)


if __name__ == "__main__":
    main()
