#!/usr/bin/env python3
"""Per-point wall-clock breakdown for an AgentX sweep.

The result JSON records what happened inside the measurement window and
nothing about getting there. That hides exactly where a warm compiled-kernel
cache pays off: weight load, kernel compile and CUDA-graph capture all happen
before the window opens, and warmup happens after that.

    ./phase_times.py <sweep-dir> [more sweep-dirs...]

Columns are seconds:
  startup   first server log line -> "The server is fired up and ready to roll"
  warmup    AIPerf warmup phase, start to drain
  window    the measurement window itself (fixed by --duration)
  total     sweep.log's own start -> exit stamps, so it includes teardown
"""

import argparse
import glob
import os
import re
import sys
from datetime import datetime, timedelta

HMS = re.compile(r"(\d{2}):(\d{2}):(\d{2})")


def _secs(text):
    m = HMS.search(text or "")
    if not m:
        return None
    h, mi, s = (int(g) for g in m.groups())
    return h * 3600 + mi * 60 + s


def _span(a, b):
    """b - a in seconds, tolerating one midnight wrap."""
    if a is None or b is None:
        return None
    d = b - a
    return d + 86400 if d < 0 else d


def _first_last(path, needle):
    first = last = None
    try:
        with open(path, errors="ignore") as fh:
            for line in fh:
                if needle in line:
                    t = _secs(line)
                    if t is not None:
                        first = first if first is not None else t
                        last = t
    except OSError:
        pass
    return first, last


def point_times(point_dir):
    server = os.path.join(point_dir, "server.log")
    recipe = os.path.join(point_dir, "recipe.log")

    server_first = None
    try:
        with open(server, errors="ignore") as fh:
            for line in fh:
                t = _secs(line)
                if t is not None:
                    server_first = t
                    break
    except OSError:
        pass
    ready, _ = _first_last(server, "The server is fired up and ready to roll")

    warm_start, _ = _first_last(recipe, "Phase warmup (warmup) started")
    _, warm_end = _first_last(recipe, "Phase warmup (warmup) complete")
    prof_start, _ = _first_last(recipe, "Phase profiling (profiling) started")
    _, prof_end = _first_last(recipe, "Phase profiling (profiling) sending complete")

    return {
        "startup": _span(server_first, ready),
        "warmup": _span(warm_start, warm_end),
        "window": _span(prof_start, prof_end),
    }


def sweep_totals(sweep_dir):
    """conc -> total seconds, from the driver's own start/exit stamps."""
    out = {}
    log = os.path.join(sweep_dir, "sweep.log")
    start = {}
    try:
        with open(log, errors="ignore") as fh:
            for line in fh:
                stamp = _secs(line)
                if stamp is None:
                    continue
                m = re.search(r"starting conc=(\d+)", line)
                if m:
                    start[m.group(1)] = stamp
                    continue
                m = re.search(r"conc=(\d+) exit=", line)
                if m and m.group(1) in start:
                    out[m.group(1)] = _span(start[m.group(1)], stamp)
    except OSError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()

    head = f"{'sweep':<26}{'conc':>5}{'startup':>9}{'warmup':>8}{'window':>8}{'total':>8}"
    print(head)
    print("-" * len(head))
    for sweep in args.dirs:
        totals = sweep_totals(sweep)
        label = os.path.basename(sweep.rstrip("/"))[-24:]
        points = sorted(
            glob.glob(os.path.join(sweep, "glm5.3flash_tp*_conc*")),
            key=lambda p: int(re.search(r"conc(\d+)", p).group(1)),
        )
        for p in points:
            conc = re.search(r"conc(\d+)", p).group(1)
            t = point_times(p)
            row = [t["startup"], t["warmup"], t["window"], totals.get(conc)]
            cells = "".join(f"{v:>8}" if v is not None else f"{'-':>8}" for v in row)
            print(f"{label:<26}{conc:>5}{cells[:8]:>9}{cells[8:]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
