#!/usr/bin/env python3
"""
auto_detect_layer.py

Analyze PyTorch profiler traces to discover model layer structure and kernel breakdown.

Three-step analysis:
  Step 1: Parse cuda-graph-ON trace → kernel statistics (count, sum, avg, percentage)
  Step 2: Parse cuda-graph-OFF trace → layer structure (types, sub-modules)
  Step 3: Combine → per-layer kernel breakdown with sub-module labels

Usage:
    # Full analysis (both traces)
    python auto_detect_layer.py --graph-on on.trace.json.gz --graph-off off.trace.json.gz

    # Step 1 only: kernel statistics
    python auto_detect_layer.py --graph-on on.trace.json.gz

    # Step 2 only: layer structure
    python auto_detect_layer.py --graph-off off.trace.json.gz

    # Export step 1 to CSV
    python auto_detect_layer.py --graph-on on.trace.json.gz --csv kernels.csv
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


def find_busiest_stream(trace: dict | list) -> int:
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
    return counts.most_common(1)[0][0]


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


def write_step1_csv(stats: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Name", "Count", "SumDuration_us", "AvgDuration_us", "Percentage"])
        for s in stats:
            w.writerow([s["name"], s["count"], f"{s['sum_dur']:.1f}",
                        f"{s['avg_dur']:.3f}", f"{s['pct']:.2f}"])
    print(f"[INFO] Step 1 CSV written to: {path}", file=sys.stderr)


def write_step3_csv(layer_types: list[dict], kernels: list[dict],
                     kernel_stats: list[dict] | None, path: str,
                     callsite_map: dict | None = None) -> None:
    """Export step 3 breakdown to CSV."""
    stat_lookup = {}
    if kernel_stats:
        for s in kernel_stats:
            stat_lookup[s["name"]] = s

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        headers = ["LayerType", "LayerCount", "Index", "Module",
                    "KernelName", "Duration_us"]
        if kernel_stats:
            headers += ["GraphON_AvgDuration_us", "GraphON_Count",
                        "GraphON_SumDuration_us", "GraphON_Percentage"]
        headers.append("CallSite")
        w.writerow(headers)

        for i, lt in enumerate(layer_types):
            if i > 0:
                w.writerow([])  # blank row between layer types

            label = chr(ord("A") + i)
            sub_mod_str = " + ".join(lt["sub_modules"])
            breakdown = lt.get("kernel_breakdown", [])
            for pos, (kidx, mod_label) in enumerate(breakdown):
                k = kernels[kidx]
                cs = callsite_map.get(kidx, "") if callsite_map else ""
                row = [f"{label}: {sub_mod_str}", lt["count"], pos,
                       mod_label, k["name"], f"{k['dur']:.1f}"]
                if kernel_stats:
                    s = stat_lookup.get(k["name"])
                    if s:
                        row += [f"{s['avg_dur']:.3f}", s["count"],
                                f"{s['sum_dur']:.1f}", f"{s['pct']:.2f}"]
                    else:
                        row += ["", "", "", ""]
                row.append(cs)
                w.writerow(row)

    print(f"[INFO] Step 3 CSV written to: {path}", file=sys.stderr)


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


def get_one_forward_pass(trace: dict | list, cls_name: str) -> list[dict]:
    """Get nn.Module events for one forward pass (2nd pass to skip warmup)."""
    events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
    all_events = sorted(
        [ev for ev in events
         if isinstance(ev, dict) and ev.get("ph") == "X"
         and re.sub(r'_\d+$', '', ev.get("name", "").replace("nn.Module: ", "")) == cls_name],
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
    """Build External id → kernel index mapping and sorted runtime events."""
    events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
    ext_to_kidx = {}
    for i, k in enumerate(kernels):
        eid = k.get("args", {}).get("External id")
        if eid is not None and eid not in ext_to_kidx:
            ext_to_kidx[eid] = i

    runtime = sorted(
        [ev for ev in events
         if isinstance(ev, dict) and ev.get("cat") == "cuda_runtime"
         and ev.get("ph") == "X"
         and ev.get("args", {}).get("External id") is not None],
        key=lambda ev: ev["ts"],
    )
    rt_ts = [ev["ts"] for ev in runtime]
    return ext_to_kidx, runtime, rt_ts


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

    # Build ext_id → runtime event mapping
    ext_to_runtime = {}
    for rt in runtime:
        eid = rt.get("args", {}).get("External id")
        if eid is not None and eid not in ext_to_runtime:
            ext_to_runtime[eid] = rt

    # Resolve call sites
    callsite_map = {}
    resolve_list = target_kidxs if target_kidxs else range(len(kernels))

    for i in resolve_list:
        k = kernels[i]
        eid = k.get("args", {}).get("External id")
        if eid is None or eid not in ext_to_runtime:
            continue
        rt = ext_to_runtime[eid]
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
        # Limit scan to 500 events or 50ms back to keep it fast.
        best = None
        best_dur = float("inf")
        scan_limit = max(0, pos - 500)
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


def get_kernels_for_module(module_ev: dict, ext_to_kidx: dict,
                            runtime: list[dict], rt_ts: list[float]) -> list[int]:
    """Get GPU kernel indices launched by a module event."""
    mod_ts = module_ev["ts"]
    mod_end = mod_ts + module_ev.get("dur", 0)
    lo = bisect.bisect_left(rt_ts, mod_ts)
    hi = bisect.bisect_right(rt_ts, mod_end)
    kidxs = set()
    for j in range(lo, hi):
        eid = runtime[j].get("args", {}).get("External id")
        if eid in ext_to_kidx:
            kidxs.add(ext_to_kidx[eid])
    return sorted(kidxs)


def analyze_layer_structure(trace: dict | list, kernels: list[dict]):
    """
    Step 2: Discover layer structure.
    Returns (cls_name, layer_types, callsite_map) where layer_types is:
      [ { 'name': str, 'count': int, 'layers': [name, ...],
          'sub_modules': [cls, ...],
          'kernel_breakdown': [ (kernel_idx, module_label), ... ] }, ... ]
    and callsite_map is { kernel_idx: "file.py(line): func_name" }
    """
    cls_name = find_decoder_layer_class(trace)
    if cls_name is None:
        return None, [], {}

    forward_pass = get_one_forward_pass(trace, cls_name)
    if not forward_pass:
        return cls_name, [], {}

    ext_to_kidx, runtime, rt_ts = build_ext_id_map(trace, kernels)

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

    # Analyze each layer
    layer_data = []
    for layer_ev in forward_pass:
        layer_name = layer_ev["name"].replace("nn.Module: ", "")
        children = find_direct_children(trace, layer_ev, significant_classes)
        sub_mods = [re.sub(r'_\d+$', '', c["name"].replace("nn.Module: ", ""))
                    for c in children]

        # Get kernel-to-module mapping
        all_kidxs = get_kernels_for_module(layer_ev, ext_to_kidx, runtime, rt_ts)
        kernel_labels = {}  # kidx -> module_class
        for cev in children:
            ccls = re.sub(r'_\d+$', '', cev["name"].replace("nn.Module: ", ""))
            ckidxs = get_kernels_for_module(cev, ext_to_kidx, runtime, rt_ts)
            for ki in ckidxs:
                if ki not in kernel_labels:
                    kernel_labels[ki] = ccls

        breakdown = [(ki, kernel_labels.get(ki, "(layer)")) for ki in all_kidxs]
        layer_data.append({
            "name": layer_name,
            "sub_modules": tuple(sub_mods),
            "kernel_breakdown": breakdown,
            "n_kernels": len(all_kidxs),
        })

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

    # Build callsite map only for kernels in breakdowns (fast)
    target_kidxs = set()
    for lt in layer_types:
        for kidx, _ in lt.get("kernel_breakdown", []):
            target_kidxs.add(kidx)
    print(f"[INFO] Resolving call sites for {len(target_kidxs)} kernels...",
          file=sys.stderr)
    callsite_map = build_callsite_map(trace, kernels, ext_to_kidx, runtime,
                                      target_kidxs)

    return cls_name, layer_types, callsite_map


def print_step2(cls_name: str, layer_types: list[dict]) -> None:
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

    print(f"\n{'='*100}\n")


# ===================================================================
# Step 3: Per-layer kernel breakdown with sub-module labels
# ===================================================================

def print_step3(cls_name: str, layer_types: list[dict],
                kernels: list[dict], kernel_stats: list[dict] | None,
                callsite_map: dict | None = None) -> None:
    """
    Print kernel breakdown for each layer type.
    Uses graph-OFF kernel data with sub-module labels.
    If kernel_stats (from graph-ON step 1) is provided, also shows
    the graph-ON avg duration for cross-reference.
    """
    # Build name->avg lookup from step 1
    stat_lookup = {}
    if kernel_stats:
        for s in kernel_stats:
            stat_lookup[s["name"]] = s

    print(f"\n{'='*100}")
    print(f" Step 3: Layer Kernel Breakdown")
    print(f"{'='*100}")

    for i, lt in enumerate(layer_types):
        label = chr(ord("A") + i)
        breakdown = lt["kernel_breakdown"]
        if not breakdown:
            continue

        print(f"\n  Type {label} ({lt['count']} layers): "
              f"{' + '.join(lt['sub_modules'])}")
        print(f"  {'-'*90}")

        current_module = None
        module_dur = 0.0
        total_dur = 0.0

        header = f"  {'#':>3}  {'Module':<30s}  {'Duration':>10}  {'Kernel Name'}"
        if kernel_stats:
            header += f"  {'(graph-ON avg)':>14}"
        print(header)
        print(f"  {'-'*3}  {'-'*30}  {'-'*10}  {'-'*50}")

        for pos, (kidx, mod_label) in enumerate(breakdown):
            k = kernels[kidx]
            dur = k["dur"]
            total_dur += dur

            if mod_label != current_module:
                if current_module is not None:
                    print(f"  {'':>3}  {'Subtotal':>30}  {fmt_dur(module_dur):>10}")
                    print()
                current_module = mod_label
                module_dur = 0.0

            module_dur += dur
            name_display = k["name"][:60]
            cs = callsite_map.get(kidx, "") if callsite_map else ""
            line = f"  {pos:>3}  {mod_label:<30s}  {fmt_dur(dur):>10}  {name_display}"

            if kernel_stats and k["name"] in stat_lookup:
                avg = stat_lookup[k["name"]]["avg_dur"]
                line += f"  {fmt_dur(avg):>14}"

            print(line)
            if cs:
                print(f"  {'':>3}  {'':>30}  {'':>10}  caller: {cs}")

        # Last module subtotal
        if current_module is not None:
            print(f"  {'':>3}  {'Subtotal':>30}  {fmt_dur(module_dur):>10}")

        print(f"\n  {'':>3}  {'TOTAL':>30}  {fmt_dur(total_dur):>10}  "
              f"({len(breakdown)} kernels)")

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
    p.add_argument("--csv", metavar="DIR",
                   help="Export CSVs to directory (step1_kernel_stats.csv, step3_layer_breakdown.csv)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.graph_on and not args.graph_off:
        sys.exit("Provide at least one: --graph-on or --graph-off")

    # Prepare CSV output dir
    csv_dir = None
    if args.csv:
        csv_dir = Path(args.csv)
        csv_dir.mkdir(parents=True, exist_ok=True)

    kernel_stats = None
    cls_name = None
    layer_types = None
    kernels_off = None

    # --- Step 1: Kernel statistics from graph-ON trace ---
    if args.graph_on:
        print(f"[INFO] Step 1: Loading graph-ON trace: {args.graph_on}", file=sys.stderr)
        trace_on = load_trace(args.graph_on)
        stream_on = args.stream or find_busiest_stream(trace_on)
        kernels_on = extract_gpu_kernels(trace_on, stream=stream_on)
        print(f"[INFO] Found {len(kernels_on)} kernels on stream {stream_on}",
              file=sys.stderr)

        kernel_stats = compute_kernel_stats(kernels_on)
        print_step1(kernel_stats)

        if csv_dir:
            write_step1_csv(kernel_stats, str(csv_dir / "step1_kernel_stats.csv"))

        del trace_on  # free memory

    # --- Step 2: Layer structure from graph-OFF trace ---
    if args.graph_off:
        print(f"[INFO] Step 2: Loading graph-OFF trace: {args.graph_off}",
              file=sys.stderr)
        trace_off = load_trace(args.graph_off)
        stream_off = args.stream or find_busiest_stream(trace_off)
        kernels_off = extract_gpu_kernels(trace_off, stream=stream_off)
        print(f"[INFO] Found {len(kernels_off)} kernels on stream {stream_off}",
              file=sys.stderr)

        cls_name, layer_types, callsite_map = analyze_layer_structure(
            trace_off, kernels_off)
        if cls_name:
            print_step2(cls_name, layer_types)

            # --- Step 3: Combined breakdown ---
            if layer_types:
                print_step3(cls_name, layer_types, kernels_off, kernel_stats,
                            callsite_map)
                if csv_dir:
                    write_step3_csv(layer_types, kernels_off, kernel_stats,
                                    str(csv_dir / "step3_layer_breakdown.csv"),
                                    callsite_map)
        else:
            print("[WARN] No nn.Module DecoderLayer events found.", file=sys.stderr)


if __name__ == "__main__":
    main()
