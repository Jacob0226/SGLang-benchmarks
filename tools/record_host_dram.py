#!/usr/bin/env python3
"""Record the host DRAM HiCache actually pinned into the AgentX result JSON.

The harness reports `allocated_cpu_dram_gb`, but that is InferenceX's budget
(available node DRAM, capped at 3 TB, scaled by dram-utilization and the
fraction of the node's GPUs in use). The SGLang HiCache recipe never receives
it: it passes --hicache-ratio, so the pool size follows the device pool instead.
The only record of what was really pinned is server.log, which logs two
allocations per rank -- the anchor MLA host pool and the DSA indexer sidecar
that inherits its slot count.

  record_host_dram.py <result-dir> ...        # one run, or a tree of runs
"""
import argparse
import json
import os
import re
import sys

KV_POOL_RE = re.compile(
    r"hierarchical KV host pool: (\d+) tokens, ([\d.]+) GB host memory"
)
INDEXER_RE = re.compile(r"Allocating ([\d.]+) GB host memory for DSA indexer")


def parse_server_log(path):
    with open(path, errors="replace") as fh:
        log = fh.read()

    kv = [(int(t), float(gb)) for t, gb in KV_POOL_RE.findall(log)]
    if not kv:
        return None
    indexer = [float(gb) for gb in INDEXER_RE.findall(log)]

    # One line per rank, and every rank allocates the same pool.
    ranks = len(kv)
    tokens, kv_gb = kv[0]
    indexer_gb = indexer[0] if indexer else 0.0
    per_rank = kv_gb + indexer_gb

    return {
        "source": "server.log",
        "tp_ranks": ranks,
        "host_pool_tokens_per_rank": tokens,
        "kv_pool_gb_per_rank": round(kv_gb, 2),
        "dsa_indexer_gb_per_rank": round(indexer_gb, 2),
        "gb_per_rank": round(per_rank, 2),
        "gb_total": round(per_rank * ranks, 2),
        "bytes_per_token": round(per_rank * 1e9 / tokens),
    }


def result_json_in(run_dir):
    names = [n for n in os.listdir(run_dir)
             if n.endswith(".json") and not n.startswith("mooncake")]
    return os.path.join(run_dir, names[0]) if len(names) == 1 else None


def annotate(run_dir):
    server_log = os.path.join(run_dir, "server.log")
    if not os.path.exists(server_log):
        print(f"  skip {os.path.basename(run_dir)}: no server.log")
        return None
    result = result_json_in(run_dir)
    if result is None:
        # Still in flight, or the recipe died before writing a result.
        print(f"  skip {os.path.basename(run_dir)}: no result JSON yet")
        return None

    measured = parse_server_log(server_log)
    if measured is None:
        print(f"  skip {os.path.basename(run_dir)}: no host pool in server.log")
        return None

    try:
        with open(result) as fh:
            payload = json.load(fh)
        payload["host_dram_measured"] = measured
        with open(result, "w") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
    except OSError as exc:
        # The recipe writes results as root inside the container; one
        # unwritable run should not abort a backfill over the whole tree.
        print(f"  skip {os.path.basename(run_dir)}: {exc.strerror}")
        return None
    return measured


def run_dirs(path):
    if os.path.exists(os.path.join(path, "server.log")):
        return [path]
    found = []
    for root, _, files in os.walk(path):
        if "server.log" in files:
            found.append(root)
    return sorted(found)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="result dir, or a tree containing them")
    args = p.parse_args()

    annotated = 0
    for path in args.paths:
        for run_dir in run_dirs(path):
            measured = annotate(run_dir)
            if measured is None:
                continue
            annotated += 1
            print(f"  {os.path.basename(run_dir)}: "
                  f"{measured['gb_per_rank']} GB/rank x {measured['tp_ranks']} = "
                  f"{measured['gb_total']} GB "
                  f"({measured['kv_pool_gb_per_rank']} KV + "
                  f"{measured['dsa_indexer_gb_per_rank']} indexer)")
    print(f"annotated {annotated} run(s)")
    return 0 if annotated else 1


if __name__ == "__main__":
    sys.exit(main())
