#!/usr/bin/env python3
"""Aggregate HiCache.sh bench_multiturn.jsonl files into a single CSV.

Walks the results tree produced by HiCache.sh and emits one CSV row per
(docker, model, tag, cache_mode, hicache_size, [round]) tuple, with
TTFT / ITL / latency percentiles, throughput, and cache-hit-rate.

Path layout (must match HiCache.sh):
  results/<docker>/<model>-HiCache[-<tag>]/<cache_mode>/[size_<N>/]bench_multiturn.jsonl

Usage:
  python3 parse_hicache_multiturn.py
  python3 parse_hicache_multiturn.py --root-dir ~/SGLang-benchmarks/results
  python3 parse_hicache_multiturn.py --include-rounds   # add per-round columns
  python3 parse_hicache_multiturn.py --tag MI355X       # filter by --tag
  python3 parse_hicache_multiturn.py --pivot-size       # pivot on hicache_size

Two output files (next to --root-dir by default):
  <root>/hicache_multiturn_summary.csv   long table, one row per run
  <root>/hicache_multiturn_pivot.csv     wide pivot on hicache_size (only with --pivot-size)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

# Make `tools/calc_kv_per_token.py` importable so we can compute
# bytes/token/rank from the model's config.json on the fly.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "tools"))
try:
    import calc_kv_per_token as _kvc
except ImportError:
    _kvc = None

PATH_RE = re.compile(
    r"results/(?P<docker>[^/]+)/"
    r"(?P<model_tag>[^/]+)-HiCache(?:-(?P<tag>[^/]+))?/"
    r"(?P<cache_mode>[^/]+)"
    r"(?:/size_(?P<size>\d+))?/bench_multiturn\.jsonl$"
)

# Logical ordering used in the CSV: baselines first (no caching → GPU radix
# only), then HiCache variants in increasing capacity order (L1+L2 → L1+L2
# +file L3 → L1+L2+RDMA L3). Modes not listed get a large index and sort
# alphabetically at the end.
CACHE_MODE_ORDER = [
    "no_cache",
    "L1",
    "L2",
    "L3_file",
    "L3_hf3fs",
    "L3_mooncake",
    "L3_nixl",
    # Legacy names kept for back-compat with older runs:
    "no_radix",
    "radix",
    "hicache",
    "hicache_file",
    "hicache_hf3fs",
    "hicache_mooncake",
]


def cache_mode_rank(name: str) -> int:
    try:
        return CACHE_MODE_ORDER.index(name)
    except ValueError:
        return len(CACHE_MODE_ORDER) + hash(name) % 1000

# Order of columns in the long-table CSV.
LONG_COLS = [
    "docker",
    "model",
    "tag",
    "cache_mode",
    "hicache_size_gb",
    # Workload sizing (read from bench_meta.json + computed via calc_kv_per_token)
    "num_clients",
    "num_rounds",
    "request_length",
    "kv_kb_per_token_per_rank",
    "device_pool_gb",
    "working_set_tokens",
    "working_set_pct_l1",          # <— what % of GPU L1 the workload covers
    # Bench result summary
    "total_requests",
    "request_rate",
    "avg_prompt_len",
    "p99_prompt_len",
    "avg_output_len",
    "p99_output_len",
    "avg_ttft_ms",
    "median_ttft_ms",
    "p90_ttft_ms",
    "p99_ttft_ms",
    "max_ttft_ms",
    "avg_itl_ms",
    "median_itl_ms",
    "p90_itl_ms",
    "p99_itl_ms",
    "max_itl_ms",
    "avg_latency_ms",
    "median_latency_ms",
    "p90_latency_ms",
    "p99_latency_ms",
    "max_latency_ms",
    "input_token_throughput",
    "output_token_throughput",
    "request_throughput",
    "cache_hit_rate",
    "timestamp",
    "jsonl_path",
]


def find_jsonls(root: Path) -> Iterable[Path]:
    """Yield bench_multiturn.jsonl files anywhere under `root`."""
    for path in root.rglob("bench_multiturn.jsonl"):
        if path.stat().st_size == 0:
            continue
        yield path


def parse_path(path: Path, root: Path) -> dict:
    """Extract docker / model / tag / cache_mode / hicache_size from path."""
    rel = str(path.relative_to(root.parent if root.name == "results" else root))
    # Normalize: prefix "results/..." for the regex.
    # Whatever the user passed as --root-dir, find the segment "results/..." in the absolute path.
    abs_str = str(path.resolve())
    idx = abs_str.find("/results/")
    if idx == -1:
        return {}
    sub = abs_str[idx + 1 :]  # drop leading "/"
    m = PATH_RE.search(sub)
    if not m:
        return {}
    return {
        "docker": m.group("docker"),
        "model": m.group("model_tag"),
        "tag": m.group("tag") or "",
        "cache_mode": m.group("cache_mode"),
        "hicache_size_gb": int(m.group("size")) if m.group("size") else "",
    }


def s_to_ms(v):
    """sec → ms; pass non-numerics through."""
    try:
        return round(float(v) * 1000, 3)
    except (TypeError, ValueError):
        return v


def derive_workload_meta(jsonl_path: Path, record: dict) -> dict:
    """Read bench_meta.json next to bench_multiturn.jsonl and compute working
    set + % of L1. Falls back to round-dict-based estimation if no meta file
    is present (older runs)."""
    out = {
        "num_clients": "",
        "num_rounds": "",
        "request_length": "",
        "kv_kb_per_token_per_rank": "",
        "device_pool_gb": "",
        "working_set_tokens": "",
        "working_set_pct_l1": "",
    }
    meta_path = jsonl_path.parent / "bench_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        out["num_clients"]    = meta.get("num_clients", "")
        out["num_rounds"]     = meta.get("num_rounds", "")
        out["request_length"] = meta.get("request_length", "")
        out["device_pool_gb"] = meta.get("device_pool_gb", "")

        if _kvc and meta.get("model_path"):
            try:
                r = _kvc.kv_per_token(
                    meta["model_path"],
                    tp=meta.get("tp_size", 8),
                    kv_dtype=meta.get("kv_cache_dtype", "fp8_e4m3"),
                )
                bpt = r["bytes_per_token_per_rank"]
                out["kv_kb_per_token_per_rank"] = round(bpt / 1024, 3)
                if meta.get("num_clients") and meta.get("num_rounds") and meta.get("request_length"):
                    ws_tokens = (meta["num_clients"]
                                 * meta["num_rounds"]
                                 * meta["request_length"])
                    out["working_set_tokens"] = ws_tokens
                    if meta.get("device_pool_gb", 0):
                        l1_tokens = meta["device_pool_gb"] * (1024 ** 3) / bpt
                        if l1_tokens > 0:
                            out["working_set_pct_l1"] = round(
                                100 * ws_tokens / l1_tokens, 1)
            except Exception as e:
                print(f"[warn] {jsonl_path.parent.name}: kv calc failed: {e}")
    else:
        # Older runs without bench_meta.json: count rounds from JSONL,
        # back out request_length from average_prompt_len, num_clients
        # from total_requests / num_rounds.
        s = record.get("summary", {})
        r = record.get("round") or {}
        round_keys = [k for k in r if k.startswith("round_")]
        if round_keys and s.get("total_requests"):
            num_rounds = max(int(k.split("_")[1]) for k in round_keys) + 1
            num_clients = int(s["total_requests"] / num_rounds)
            avg_prompt_len = s.get("average_prompt_len") or 0
            if num_rounds > 0 and avg_prompt_len > 0:
                request_length = round(avg_prompt_len * 2 / (num_rounds + 1))
            else:
                request_length = 0
            out["num_clients"]    = num_clients
            out["num_rounds"]     = num_rounds
            out["request_length"] = request_length
            if request_length:
                out["working_set_tokens"] = num_clients * num_rounds * request_length
    return out


def flatten_summary(record: dict, meta: dict, jsonl_path: Path) -> dict:
    s = record.get("summary", {})
    workload = derive_workload_meta(jsonl_path, record)
    row = {
        **meta,
        **workload,
        "total_requests": s.get("total_requests"),
        "request_rate": s.get("request_rate"),
        "avg_prompt_len": s.get("average_prompt_len"),
        "p99_prompt_len": s.get("p99_prompt_len"),
        "avg_output_len": s.get("average_output_len"),
        "p99_output_len": s.get("p99_output_len"),
        "avg_ttft_ms":     s_to_ms(s.get("average_ttft")),
        "median_ttft_ms":  s_to_ms(s.get("median_ttft")),
        "p90_ttft_ms":     s_to_ms(s.get("p90_ttft")),
        "p99_ttft_ms":     s_to_ms(s.get("p99_ttft")),
        "max_ttft_ms":     s_to_ms(s.get("max_ttft")),
        "avg_itl_ms":      s_to_ms(s.get("average_itl")),
        "median_itl_ms":   s_to_ms(s.get("median_itl")),
        "p90_itl_ms":      s_to_ms(s.get("p90_itl")),
        "p99_itl_ms":      s_to_ms(s.get("p99_itl")),
        "max_itl_ms":      s_to_ms(s.get("max_itl")),
        "avg_latency_ms":     s_to_ms(s.get("average_latency")),
        "median_latency_ms":  s_to_ms(s.get("median_latency")),
        "p90_latency_ms":     s_to_ms(s.get("p90_latency")),
        "p99_latency_ms":     s_to_ms(s.get("p99_latency")),
        "max_latency_ms":     s_to_ms(s.get("max_latency")),
        "input_token_throughput":  s.get("input_token_throughput"),
        "output_token_throughput": s.get("output_token_throughput"),
        "request_throughput":      s.get("throughput"),
        "cache_hit_rate":          s.get("cache_hit_rate"),
        "timestamp":               record.get("timestamp", ""),
        "jsonl_path":              str(jsonl_path),
    }
    return row


def collect_rows(root: Path, tag_filter: str | None) -> list[dict]:
    rows = []
    for jsonl in find_jsonls(root):
        meta = parse_path(jsonl, root)
        if not meta:
            print(f"[skip] cannot parse path: {jsonl}")
            continue
        if tag_filter and meta.get("tag", "") != tag_filter:
            continue
        with open(jsonl) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[skip] bad json {jsonl}:{line_no}: {e}")
                    continue
                rows.append((jsonl, meta, rec))
    rows.sort(
        key=lambda x: (
            x[1].get("docker", ""),
            x[1].get("model", ""),
            x[1].get("tag", ""),
            cache_mode_rank(x[1].get("cache_mode", "")),
            int(x[1].get("hicache_size_gb") or 0),
        )
    )
    return rows


def emit_long(rows, out_path: Path, include_rounds: bool):
    cols = list(LONG_COLS)
    if include_rounds:
        # find max round count across all records
        max_rounds = 0
        for _, _, rec in rows:
            r = rec.get("round") or {}
            for k in r:
                if k.startswith("round_"):
                    try:
                        max_rounds = max(max_rounds, int(k.split("_")[1]) + 1)
                    except (IndexError, ValueError):
                        pass
        for i in range(max_rounds):
            cols += [f"round{i}_ttft_ms", f"round{i}_hit_rate", f"round{i}_n"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for jsonl, meta, rec in rows:
            row = flatten_summary(rec, meta, jsonl)
            if include_rounds:
                rdict = rec.get("round") or {}
                for i in range(max_rounds):
                    rd = rdict.get(f"round_{i}", {})
                    row[f"round{i}_ttft_ms"] = s_to_ms(rd.get("average_ttft"))
                    row[f"round{i}_hit_rate"] = rd.get("cache_hit_rate")
                    row[f"round{i}_n"] = rd.get("request_count")
            w.writerow(row)
    print(f"Wrote {len(rows)} rows → {out_path}")


def emit_pivot(rows, out_path: Path):
    """Wide pivot: rows = (docker, model, cache_mode), cols = size_<N>_<metric>."""
    metrics = [
        ("median_ttft_ms",     "median_ttft_ms"),
        ("p99_ttft_ms",        "p99_ttft_ms"),
        ("input_token_throughput", "input_tput"),
        ("request_throughput", "req_tput"),
        ("cache_hit_rate",     "hit"),
    ]
    sizes = sorted({m["hicache_size_gb"] for _, m, _ in rows if m.get("hicache_size_gb") not in ("", None)})
    cols = ["docker", "model", "tag", "cache_mode"]
    for sz in sizes:
        for _, label in metrics:
            cols.append(f"size_{sz}_{label}")
    # group by (docker, model, tag, cache_mode)
    groups: dict[tuple, dict] = {}
    for jsonl, meta, rec in rows:
        key = (meta["docker"], meta["model"], meta.get("tag", ""), meta["cache_mode"])
        if key not in groups:
            groups[key] = {"docker": key[0], "model": key[1], "tag": key[2], "cache_mode": key[3]}
        sz = meta.get("hicache_size_gb")
        if sz in ("", None):
            continue
        row_data = flatten_summary(rec, meta, jsonl)
        for src, label in metrics:
            groups[key][f"size_{sz}_{label}"] = row_data.get(src)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for k in sorted(groups):
            w.writerow(groups[k])
    print(f"Wrote pivot ({len(groups)} rows) → {out_path}")


def main():
    p = argparse.ArgumentParser()
    default_root = Path(os.environ.get("HOME", "~")).expanduser() / "SGLang-benchmarks" / "results"
    p.add_argument("--root-dir", default=str(default_root),
                   help="Top-level results dir (default: %(default)s)")
    p.add_argument("--out", default="",
                   help="Output CSV path. Default: <root>/hicache_multiturn_summary.csv")
    p.add_argument("--pivot-out", default="",
                   help="Pivot CSV path. Default: <root>/hicache_multiturn_pivot.csv")
    p.add_argument("--tag", default=None,
                   help="Only include runs whose --tag matches (e.g. MI355X).")
    p.add_argument("--include-rounds", action="store_true",
                   help="Add round_0_ttft/hit_rate/n columns.")
    p.add_argument("--pivot-size", action="store_true",
                   help="Also write a wide pivot keyed on hicache_size.")
    args = p.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"--root-dir not found: {root}")

    rows = collect_rows(root, args.tag)
    if not rows:
        print(f"No bench_multiturn.jsonl found under {root}")
        return

    out_path = Path(args.out) if args.out else root / "hicache_multiturn_summary.csv"
    emit_long(rows, out_path, args.include_rounds)

    if args.pivot_size:
        pivot_path = Path(args.pivot_out) if args.pivot_out else root / "hicache_multiturn_pivot.csv"
        emit_pivot(rows, pivot_path)


if __name__ == "__main__":
    main()
