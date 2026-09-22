"""List the profiler forward annotations in a trace, with counts and Σ duration.

--forward-match has to name an annotation that actually exists, and the label
differs between stacks (a speculative-decoding run has no plain "DECODE bs=N"
forward at all). Printing the annotations first is cheaper than guessing.

    python3 trace_analysis/diagnostics/list_forwards.py <trace.json.gz> [...]
"""

import collections
import gzip
import json
import re
import sys


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def main(paths):
    for path in paths:
        trace = load(path)
        events = trace.get("traceEvents", trace if isinstance(trace, list) else [])
        # Forward annotations are cpu_op / user_annotation records whose name
        # carries the stage and shape, e.g. "DECODE bs=64" or "EXTEND bs=3 toks=16384".
        stats = collections.defaultdict(lambda: [0, 0.0])
        for e in events:
            if e.get("ph") != "X":
                continue
            cat = e.get("cat", "")
            if cat not in ("user_annotation", "cpu_op", "python_function"):
                continue
            name = e.get("name", "")
            if not re.search(r"\b(EXTEND|DECODE|TARGET_VERIFY|DRAFT|IDLE)\b", name):
                continue
            if len(name) > 120:
                continue
            s = stats[name]
            s[0] += 1
            s[1] += e.get("dur", 0) / 1000.0

        print(f"=== {path.split('/')[-1]}")
        if not stats:
            print("   (no stage-like annotations found)")
        for name, (cnt, ms) in sorted(stats.items(), key=lambda kv: -kv[1][1])[:25]:
            print(f"   {cnt:5d} x  Σ{ms:10.2f} ms   {name}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
