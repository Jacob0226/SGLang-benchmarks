#!/usr/bin/env python3
"""Side-by-side KV / prefix-cache metrics for AgentX result dirs.

usage: agentic_cache_metrics.py <result-dir>...
Reads aiperf_artifacts/server_metrics_export.csv (the SGLang /metrics scrape AIPerf
records during the run) and prints one column per result dir.
"""

import csv
import os
import sys

ROWS = [
    # (label, metric, stat): stat is a CSV column, or "counter" for the summed counter value.
    ("prefix hit (cached/prompt)", "ratio", None),
    ("prompt tokens (M)", "sglang:prompt_tokens", "counter"),
    ("cached tokens (M)", "sglang:cached_tokens", "counter"),
    ("  from device (M)", "sglang:cached_tokens@device", "counter"),
    ("  from host/HiCache (M)", "sglang:cached_tokens@host", "counter"),
    ("computed prefill tokens (M)", "computed", None),
    ("evicted tokens (M)", "sglang:evicted_tokens", "counter"),
    ("full token usage avg", "sglang:full_token_usage", "avg"),
    ("full token usage p90", "sglang:full_token_usage", "p90"),
    ("full token usage max", "sglang:full_token_usage", "max"),
    ("mamba usage avg", "sglang:mamba_usage", "avg"),
    ("mamba usage max", "sglang:mamba_usage", "max"),
    ("running reqs avg", "sglang:num_running_reqs", "avg"),
    ("queued reqs avg", "sglang:num_queue_reqs", "avg"),
    ("queued reqs max", "sglang:num_queue_reqs", "max"),
    ("retracted reqs max", "sglang:num_retracted_reqs", "max"),
    ("gen throughput avg (tok/s)", "sglang:gen_throughput", "avg"),
    ("hicache host used (M tok)", "sglang:hicache_host_used_tokens", "avg"),
    ("KV memory (GB)", "sglang:kv_cache_memory_usage_gb", "avg"),
]


def load(result_dir):
    path = os.path.join(result_dir, "aiperf_artifacts", "server_metrics_export.csv")
    with open(path) as f:
        lines = [line for line in f if not line.startswith("#")]
    stats = {}
    # Counters arrive once per label set; every TP rank repeats the same value, so
    # take the max over ranks and sum over the remaining labels (e.g. cache source).
    per_label = {}
    for row in csv.DictReader(lines):
        metric = row["Metric"]
        if row["Type"] == "counter":
            # Counter rows carry fewer stat columns than the header, so their first
            # label (the cache source for cached_tokens) lands under "p10".
            source = row.get("p10") or ""
            key = (metric, source)
            per_label[key] = max(per_label.get(key, 0.0), _num(row["avg"]))
        elif row["Type"] == "gauge":
            for col in ("avg", "max", "p50", "p90"):
                stats.setdefault((metric, col), _num(row[col]))
    for (metric, source), value in per_label.items():
        stats[(metric, "counter")] = stats.get((metric, "counter"), 0.0) + value
        if source:
            stats[(f"{metric}@{source}", "counter")] = value
    return stats


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def fmt(stats, metric, stat):
    if metric == "ratio":
        prompt = stats.get(("sglang:prompt_tokens", "counter"))
        cached = stats.get(("sglang:cached_tokens", "counter"))
        return f"{cached / prompt:.1%}" if prompt else "-"
    if metric == "computed":
        prompt = stats.get(("sglang:prompt_tokens", "counter"), 0.0)
        cached = stats.get(("sglang:cached_tokens", "counter"), 0.0)
        return f"{(prompt - cached) / 1e6:.1f}"
    value = stats.get((metric, stat))
    if value is None:
        return "-"
    if stat == "counter" or "host_used" in metric:
        return f"{value / 1e6:.1f}"
    return f"{value:.2f}"


dirs = sys.argv[1:]
all_stats = [load(d) for d in dirs]
names = [os.path.basename(os.path.dirname(d.rstrip("/"))).replace("bench-Agentic-", "") for d in dirs]
width = max(14, *(len(n) for n in names))
print(f"{'':30s}" + "".join(f"{n:>{width + 2}s}" for n in names))
for label, metric, stat in ROWS:
    print(f"{label:30s}" + "".join(f"{fmt(s, metric, stat):>{width + 2}s}" for s in all_stats))
