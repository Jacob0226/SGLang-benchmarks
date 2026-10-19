"""KV and mamba pool demand of AgentX runs, from aiperf's server_metrics_export.csv.

usage: agentic_pool_demand.py <label>=<result_dir> [...]
result_dir is the per-concurrency directory holding server.log and aiperf_artifacts/.
Token counts are logical (pre-DCP) tokens; GB are per GPU.
"""

import csv
import glob
import json
import re
import sys

METRICS = (
    "kv_used_tokens",
    "kv_evictable_tokens",
    "max_total_num_tokens",
    "hicache_host_used_tokens",
    "hicache_host_total_tokens",
    "mamba_used_tokens",
    "mamba_evictable_tokens",
    "num_running_reqs",
)


def load_metrics(path):
    out = {}
    with open(path) as f:
        rows = [line for line in f if not line.startswith("#")]
    for row in csv.DictReader(rows):
        name = row["Metric"].removeprefix("sglang:")
        if name in METRICS and row.get("tp_rank") in ("0", "") and name not in out:
            out[name] = {k: float(row[k]) for k in ("avg", "p50", "p99", "max")}
    return out


def pool_geometry(server_log):
    text = open(server_log, errors="ignore").read()
    kv = re.search(r"TP0\] KV Cache is allocated\. dtype: \S+, #tokens: (\d+), KV size: ([\d.]+) GB", text)
    mamba = re.search(
        r"TP0\] Mamba Cache is allocated\. max_mamba_cache_size: (\d+), conv_state size: ([\d.]+)GB, ssm_state size: ([\d.]+)GB",
        text,
    )
    dcp = re.search(r"'dcp_size': (\d+)", text)
    local_tokens, kv_gb = int(kv.group(1)), float(kv.group(2))
    slots = int(mamba.group(1))
    return {
        "dcp": int(dcp.group(1)) if dcp else 1,
        "kv_gb_per_local_token": kv_gb / local_tokens,
        "mamba_slots": slots,
        "mamba_gb_per_slot": (float(mamba.group(2)) + float(mamba.group(3))) / slots,
    }


def mean_isl(result_dir):
    j = json.load(open(glob.glob(result_dir + "/*_local-mi355x.json")[0]))
    return j["request_metrics"]["tokens"]["input"].get("mean")


print(
    "label | running avg/max | KV in-flight p50/p99 (M tok, GB/GPU) | KV held avg: device of capacity + host (M tok) "
    "| mamba in-flight p50/p99 (slots, GB/GPU) | mamba cached avg | est. working set (M tok, GB/GPU)"
)
for arg in sys.argv[1:]:
    label, d = arg.split("=", 1)
    m = load_metrics(d + "/aiperf_artifacts/server_metrics_export.csv")
    g = pool_geometry(d + "/server.log")
    gb_tok = g["kv_gb_per_local_token"] / g["dcp"]
    run, used, evict = m["num_running_reqs"], m["kv_used_tokens"], m["kv_evictable_tokens"]
    # Host accounting is already in logical slots under DCP.
    host = m.get("hicache_host_used_tokens", {"avg": 0.0})["avg"]
    mu, me = m["mamba_used_tokens"], m["mamba_evictable_tokens"]
    conc = int(re.search(r"conc(\d+)", d).group(1))
    # Every lane keeps its context alive between turns, so the cache must hold
    # about conc in-flight-sized contexts to reach the theoretical hit rate.
    per_req = used["avg"] / max(run["avg"], 1e-9)
    ws = per_req * conc
    print(
        f"{label} | {run['avg']:.0f}/{run['max']:.0f} "
        f"| {used['p50'] / 1e6:.2f}/{used['p99'] / 1e6:.2f} ({used['p50'] * gb_tok:.0f}/{used['p99'] * gb_tok:.0f} GB) "
        f"| {(used['avg'] + evict['avg']) / 1e6:.1f} of {m['max_total_num_tokens']['avg'] / 1e6:.1f} + host {host / 1e6:.1f} "
        f"| {mu['p50']:.0f}/{mu['p99']:.0f} ({mu['p50'] * g['mamba_gb_per_slot']:.0f}/{mu['p99'] * g['mamba_gb_per_slot']:.0f} GB) "
        f"| {me['avg']:.0f} of {g['mamba_slots']} "
        f"| {ws / 1e6:.1f} ({ws * gb_tok:.0f} GB) "
        f"[KV {gb_tok * 1e6:.2f} KB/tok, mamba {g['mamba_gb_per_slot'] * 1e3:.1f} MB/slot, mean ISL {mean_isl(d):.0f}]"
    )
