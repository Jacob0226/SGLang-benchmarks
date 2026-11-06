#!/usr/bin/env python3
"""Per decode cycle (one VERIFY step plus its DRAFT) kernel time by category, for TP-0
DECODE traces from GLM.sh --prof. Cycles are cut at the CPU start of each step[VERIFY ...]
annotation; every kernel launched by the CPU inside a cycle counts toward it.

usage: decode_cycle_breakdown.py LABEL=trace.json.gz [LABEL=trace.json.gz ...] [--top N]
"""
import argparse
import bisect
import collections
import gzip
import json
import re

CATS = [
    ("indexer", r"mqa_logits|topk|kpool|hadamard|indexer"),
    ("sparse_attn", r"sparse_mla|fmha|flash_?attn|mla_|decode_attention|flashmla|trtllm_.*attn|xqa"),
    ("moe", r"moe|bmm_|finalize|expandInputRows|routing|permute|fused_experts"),
    ("allreduce", r"reduce|AllReduce|all_reduce|quickreduce|nccl|allgather|all_gather"),
    ("kda", r"kda|causal_conv|chunk_|gated_delta|delta_rule|mamba_state|fla_|recurrent"),
    ("mhc", r"mhc"),
    ("gemm", r"gemm|Cijk|nvjet|flydsl|cutlass|cublas|xdl|sm100_|wvSplitK|splitk"),
    ("sampling", r"sampl|argmax|softmax|verify|tree|accept|top_k|top_p|rejection"),
    ("elementwise", r"elementwise|vectorized|copy|fill|cat_|index|rms|norm|rotary|rope|act_and_mul|silu"),
]


def cat(name):
    for c, p in CATS:
        if re.search(p, name, re.I):
            return c
    return "other"


def analyze(path):
    ev = json.load(gzip.open(path))
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    verify = sorted(
        e["ts"] for e in ev
        if e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith("step[VERIFY")
    )
    bs = re.search(r"bs=(\d+)", next(
        e["name"] for e in ev if str(e.get("name", "")).startswith("step[VERIFY")
    )).group(1)
    kernels = [e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"]
    launches = {}
    for e in ev:
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {}):
            launches[e["args"]["correlation"]] = e["ts"]
    per = collections.defaultdict(lambda: collections.Counter())
    names = collections.Counter()
    counts = collections.Counter()
    spans = collections.defaultdict(lambda: [float("inf"), 0.0])
    for k in kernels:
        t = launches.get(k.get("args", {}).get("correlation"))
        if t is None:
            continue
        i = bisect.bisect_right(verify, t) - 1
        if i < 0 or i >= len(verify) - 1:
            continue
        c = cat(k["name"])
        per[i][c] += k["dur"] / 1e3
        per[i]["_total"] += k["dur"] / 1e3
        names[k["name"][:100]] += k["dur"] / 1e3
        counts[c] += 1
        spans[i][0] = min(spans[i][0], k["ts"])
        spans[i][1] = max(spans[i][1], k["ts"] + k["dur"])
    n = len(per)
    avg = collections.Counter()
    for i in per:
        for c, v in per[i].items():
            avg[c] += v / n
    cycle = (verify[-1] - verify[0]) / 1e3 / (len(verify) - 1)
    return dict(bs=bs, n=n, cycle=cycle, avg=avg, names={k: v / n for k, v in names.items()},
                counts={k: v / n for k, v in counts.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    res = {}
    for spec in a.traces:
        label, path = spec.split("=", 1)
        res[label] = analyze(path)
    cols = [c for c, _ in CATS] + ["other", "_total"]
    print(f"{'':14}{'bs':>4}{'cyc':>4}{'wall':>7}" + "".join(f"{c[:9]:>10}" for c in cols))
    for label, r in res.items():
        print(f"{label:14}{r['bs']:>4}{r['n']:>4}{r['cycle']:7.2f}"
              + "".join(f"{r['avg'][c]:10.2f}" for c in cols))
    labels = list(res)
    if len(labels) >= 2:
        a0, b0 = labels[0], labels[1]
        print(f"\nper-category delta ({a0} - {b0}) ms per cycle:")
        for c in sorted(cols, key=lambda c: -(res[a0]["avg"][c] - res[b0]["avg"][c])):
            print(f"  {c:12}{res[a0]['avg'][c] - res[b0]['avg'][c]:+8.2f}")
    for label, r in res.items():
        print(f"\ntop kernels {label} (ms per cycle):")
        for name, v in sorted(r["names"].items(), key=lambda x: -x[1])[: a.top]:
            print(f"  {v:7.3f}  [{cat(name)}] {name}")


if __name__ == "__main__":
    main()
