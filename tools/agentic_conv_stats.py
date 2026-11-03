#!/usr/bin/env python3
"""Conversation / turn counts of an Agentic trace-replay run (aiperf profile_export.jsonl).

usage: agentic_conv_stats.py <result_dir_or_profile_export.jsonl>
"""
import bisect
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

p = Path(sys.argv[1])
if p.is_dir():
    p = p / "aiperf_artifacts" / "profile_export.jsonl"

by_phase = defaultdict(list)
for line in p.open():
    rec = json.loads(line)
    md = rec["metadata"]
    m = rec.get("metrics", {})
    by_phase[md.get("benchmark_phase", "?")].append(
        (
            md["conversation_id"],
            md.get("source_trace_id"),
            md.get("turn_index", 0),
            md.get("source_kind"),
            m.get("usage_prompt_tokens", {}).get("value", 0),
            m.get("usage_completion_tokens", {}).get("value", 0),
            md.get("request_start_ns", 0),
            md.get("request_end_ns", 0),
        )
    )


def q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))]


for phase, rows in by_phase.items():
    convs = defaultdict(list)
    for conv, trace, turn, kind, isl, osl, t0, t1 in rows:
        convs[conv].append((turn, isl, osl, t0, t1))
    traces = {r[1] for r in rows}
    kinds = defaultdict(int)
    for r in rows:
        kinds[r[3]] += 1
    turns = [len(v) for v in convs.values()]
    deltas = []
    for v in convs.values():
        v.sort()
        deltas += [b[1] - a[1] for a, b in zip(v, v[1:]) if b[1] >= a[1]]
    print(f"== phase {phase}")
    print(f"requests {len(rows)}  conversations {len(convs)}  source traces {len(traces)}")
    print(f"request kinds {dict(kinds)}")
    print(
        f"requests per conversation: mean {statistics.mean(turns):.1f}  p50 {q(turns, .5)}"
        f"  p90 {q(turns, .9)}  max {max(turns)}"
    )
    if deltas:
        print(
            f"ISL growth between consecutive turns: mean {statistics.mean(deltas):,.0f}"
            f"  p50 {q(deltas, .5):,}  p90 {q(deltas, .9):,}"
        )
    # Reuse distance: requests (any conversation) that finished between turn t ending
    # and turn t+1 starting. Each of them leaves its own states in the cache meanwhile.
    ends = sorted(r[7] for r in rows if r[7])
    span_s = (max(ends) - min(r[6] for r in rows if r[6])) / 1e9
    gaps_n, gaps_s = [], []
    for v in convs.values():
        for a, b in zip(v, v[1:]):
            if a[4] and b[3] and b[3] >= a[4]:
                lo = bisect.bisect_right(ends, a[4])
                hi = bisect.bisect_left(ends, b[3])
                gaps_n.append(hi - lo)
                gaps_s.append((b[3] - a[4]) / 1e9)
    print(f"request rate {len(ends) / span_s:.2f}/s over {span_s:.0f} s")
    events = sorted([(r[6], 1) for r in rows if r[6]] + [(r[7], -1) for r in rows if r[7]])
    live = peak = 0
    for _, d in events:
        live += d
        peak = max(peak, live)
    per_trace = defaultdict(set)
    for r in rows:
        per_trace[r[1]].add(r[0])
    nconv = [len(v) for v in per_trace.values()]
    print(
        f"peak requests in flight {peak}; conversations per source trace: mean "
        f"{statistics.mean(nconv):.1f}  p50 {q(nconv, .5)}  max {max(nconv)}"
    )
    if gaps_n:
        print(
            f"requests finished between consecutive turns: mean {statistics.mean(gaps_n):.0f}"
            f"  p50 {q(gaps_n, .5)}  p90 {q(gaps_n, .9)}  p99 {q(gaps_n, .99)}"
        )
        print(
            f"time between consecutive turns (s): mean {statistics.mean(gaps_s):.1f}"
            f"  p50 {q(gaps_s, .5):.1f}  p90 {q(gaps_s, .9):.1f}  p99 {q(gaps_s, .99):.1f}"
        )
