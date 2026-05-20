#!/usr/bin/env python3
"""
summarize_cascade.py — produce two CSV summaries from a cascade run:

  per_mode    -- one row per round inside a single $LOG_DIR. Joins
                 bench_multiturn.jsonl (per-round TTFT / hit rate from
                 the client side) with cache_tiers.csv (per-round L2
                 fill level + L3 prefetch / backup bandwidth from the
                 cache_monitor.py sidecar).

  cross_mode  -- one row per round across all four cache modes inside
                 a $BASE_LOG_DIR (none / L1 / L2 / L3_file). Same shape
                 as the cascade_summary_*.csv style we used for the
                 single-shot manual analysis, plus L3_file extras
                 (prefetch BW p50/p99, L2 fill level).

The script tolerates missing files: a mode that wasn't run, a
cache_tiers.csv from before the metrics extension, or a partial bench
that didn't write all rounds. Missing cells render as "" so the CSVs
are still loadable.

Usage:
    summarize_cascade.py per_mode   <log_dir>
    summarize_cascade.py cross_mode <base_log_dir>
"""

import csv
import json
import os
import sys
from collections import defaultdict


# ============================== Loaders ==============================

def load_bench_rounds(log_dir):
    """Return dict round_index_0based -> {avg_ttft, hit_rate}.

    Takes the LAST jsonl line so reruns of bench_multiturn (which append)
    show their newest summary, not stale data.
    """
    path = os.path.join(log_dir, "bench_multiturn.jsonl")
    if not os.path.isfile(path):
        return {}
    last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    if not last or "round" not in last:
        return {}
    out = {}
    for k, v in last["round"].items():
        # k looks like "round_3"; turn it into 3.
        try:
            idx = int(k.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        out[idx] = {
            "avg_ttft": v.get("average_ttft"),
            "hit_rate": v.get("cache_hit_rate"),
        }
    return out


def load_cache_tier_lasts(log_dir):
    """Return dict round_index_0based -> last sample row from cache_tiers.csv.

    The cache_monitor sidecar samples every 10s and per-row L1_pct etc.
    are running within-round percentages, so taking the LAST row whose
    round_index matches gives that round's final state.
    """
    path = os.path.join(log_dir, "cache_tiers.csv")
    if not os.path.isfile(path):
        return {}
    last_per_round = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["round_index"])
            except (KeyError, ValueError):
                continue
            if idx < 0:
                # round_index = -1 means pre-cascade idle; skip.
                continue
            last_per_round[idx] = row
    return last_per_round


# ============================== Summaries ==============================

# Columns we'd like to surface from cache_tiers.csv when present. Older
# cache_monitor.py outputs won't have these; we just leave the cell
# empty if the column isn't in the CSV.
CACHE_TIER_COLS = (
    "L1_pct",
    "L2_pct",
    "L3_pct",
    "any_pct",
    "hicache_host_used_pct",
    "prefetch_count",
    "prefetch_bw_avg_gbps",
    "prefetch_bw_p50_gbps",
    "prefetch_bw_p99_gbps",
    "backup_bw_avg_gbps",
    "backup_bw_p50_gbps",
    "backup_bw_p99_gbps",
    "round_prefetched_delta",
    "round_backuped_delta",
)


def fmt_ttft(v):
    return "" if v is None else f"{v:.3f}"


def fmt_rate(v):
    return "" if v is None else f"{v:.4f}"


def fmt_passthrough(v):
    return "" if v is None or v == "" else v


def write_per_mode(log_dir, out_path):
    rounds = load_bench_rounds(log_dir)
    tiers = load_cache_tier_lasts(log_dir)
    if not rounds:
        # Nothing to summarize; don't emit a file at all so the user
        # isn't confused by an empty CSV.
        return False

    n_rounds = max(rounds) + 1
    fields = ["#Round", "avg_ttft_sec", "hit_rate"] + list(CACHE_TIER_COLS)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in range(n_rounds):
            br = rounds.get(r, {})
            tr = tiers.get(r, {})
            row = [
                r + 1,
                fmt_ttft(br.get("avg_ttft")),
                fmt_rate(br.get("hit_rate")),
            ]
            for col in CACHE_TIER_COLS:
                row.append(fmt_passthrough(tr.get(col)))
            w.writerow(row)
    return True


# ============================== Cross-mode ==============================

# In a cascade chain, $BASE_LOG_DIR has these subdirs:
#   none/                     -- always
#   L1/                       -- always
#   L2_size_<N>/              -- one or more sizes (current layout)
#   L3file_L2_size_<N>/       -- one or more sizes (current layout)
# Legacy (pre-FairCompare_0520_v3) layout used a nested form:
#   L2/size_<N>/
#   L3_file/size_<N>/
# We accept both so old result trees still summarize. For L2/L3 we pick
# the largest <N> if multiple sizes exist (matches what the cascade
# chain produces for a single --hicache-size run).

# Prefix used to encode each mode's L2 (host) pool size as a flat dir.
_MODE_DIR_PREFIX = {
    "L2": "L2_size_",
    "L3_file": "L3file_L2_size_",
}


def _largest_size_subdir(parent):
    """Return the (size, full_path) with the largest int suffix under parent,
    looking for entries named 'size_<N>'. Returns (None, None) if none found.
    Used only for the legacy nested layout."""
    sizes = []
    for entry in os.listdir(parent):
        full = os.path.join(parent, entry)
        if entry.startswith("size_") and os.path.isdir(full):
            try:
                sz = int(entry.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            sizes.append((sz, full))
    if not sizes:
        return None, None
    sizes.sort(reverse=True)
    return sizes[0]


def _resolve_mode_dir(base, mode):
    if mode in ("none", "L1"):
        direct = os.path.join(base, mode)
        return direct if os.path.isdir(direct) else None

    prefix = _MODE_DIR_PREFIX.get(mode)
    if prefix is not None:
        sizes = []
        for entry in os.listdir(base):
            full = os.path.join(base, entry)
            if entry.startswith(prefix) and os.path.isdir(full):
                try:
                    sz = int(entry[len(prefix):])
                except ValueError:
                    continue
                sizes.append((sz, full))
        if sizes:
            sizes.sort(reverse=True)
            return sizes[0][1]

    # Legacy nested layout: <base>/<mode>/size_<N>/
    legacy_parent = os.path.join(base, mode)
    if os.path.isdir(legacy_parent):
        _, legacy_dir = _largest_size_subdir(legacy_parent)
        if legacy_dir is not None:
            return legacy_dir

    return None


def write_cross_mode(base_dir, out_path):
    modes = ("none", "L1", "L2", "L3_file")
    per_mode_data = {}
    for mode in modes:
        mode_dir = _resolve_mode_dir(base_dir, mode)
        if mode_dir is None:
            per_mode_data[mode] = ({}, {})
            continue
        per_mode_data[mode] = (
            load_bench_rounds(mode_dir),
            load_cache_tier_lasts(mode_dir),
        )

    n_rounds = max(
        (max(rounds) + 1 for rounds, _ in per_mode_data.values() if rounds),
        default=0,
    )
    if n_rounds == 0:
        return False

    # Cross-mode header: per-mode TTFT/hit_rate for all modes, plus
    # L3_file-only cache-tier extras (prefetch BW + L2 fill level —
    # these signals are L3-mode-specific and don't make sense to
    # duplicate per-mode).
    header = ["#Round"]
    for mode in modes:
        header += [f"{mode}_avg_TTFT_sec", f"{mode}_hit_rate"]
    header += [
        "L3_file_prefetch_bw_avg_gbps",
        "L3_file_prefetch_bw_p50_gbps",
        "L3_file_prefetch_bw_p99_gbps",
        "L3_file_host_used_pct",
        "L3_file_prefetched_delta",
    ]

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in range(n_rounds):
            row = [r + 1]
            for mode in modes:
                rounds_d, _ = per_mode_data[mode]
                d = rounds_d.get(r, {})
                row.append(fmt_ttft(d.get("avg_ttft")))
                row.append(fmt_rate(d.get("hit_rate")))
            _, l3_tiers = per_mode_data["L3_file"]
            tr = l3_tiers.get(r, {})
            row += [
                fmt_passthrough(tr.get("prefetch_bw_avg_gbps")),
                fmt_passthrough(tr.get("prefetch_bw_p50_gbps")),
                fmt_passthrough(tr.get("prefetch_bw_p99_gbps")),
                fmt_passthrough(tr.get("hicache_host_used_pct")),
                fmt_passthrough(tr.get("round_prefetched_delta")),
            ]
            w.writerow(row)
    return True


# ============================== CLI ==============================

def main():
    if len(sys.argv) != 3:
        sys.stderr.write(__doc__)
        sys.exit(2)
    cmd, target = sys.argv[1], sys.argv[2]
    if cmd == "per_mode":
        out = os.path.join(target, "round_summary.csv")
        ok = write_per_mode(target, out)
        if ok:
            print(f">>> per-mode summary: {out}")
        else:
            print(f">>> per-mode summary: skipped (no bench_multiturn.jsonl in {target})")
    elif cmd == "cross_mode":
        out = os.path.join(target, "cascade_summary.csv")
        ok = write_cross_mode(target, out)
        if ok:
            print(f">>> cross-mode summary: {out}")
        else:
            print(f">>> cross-mode summary: skipped (no per-mode results under {target})")
    else:
        sys.stderr.write(f"unknown cmd: {cmd}\n")
        sys.stderr.write(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
