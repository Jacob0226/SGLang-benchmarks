#!/usr/bin/env bash
# One rank waits 5 ms at a barrier. Which rank was it waiting for, and what was
# that rank doing? Same-node CUPTI timestamps are on one clock, so the four
# traces can be laid on a common axis.
# Output: ~/who_is_late.txt
exec > /home/jacchang/who_is_late.txt 2>&1
cd /home/jacchang/SGLang-benchmarks || exit 1

D=results/nvidia_GLM-5.3-Flash-NVFP4/lmsysorg_sglang-v0.5.20-cu130/prof-Fixed-NVFP4-TP4-steps5/prof_in8192_out16_conc4_p8

python3 - "$D" <<'PY'
import glob, gzip, json, os

d = os.sys.argv[1]
ranks = {}
for p in sorted(glob.glob(f"{d}/*-NV-TP-*-DECODE.trace.json.gz")):
    r = p.split("-TP-")[1].split("-")[0]
    ev = json.load(gzip.open(p, "rt"))["traceEvents"]
    ker = sorted([e for e in ev if e.get("ph") == "X"
                  and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")],
                 key=lambda e: e["ts"])
    ranks[r] = ker
    ar = [e for e in ker if "all_reduce" in e.get("name", "")]
    longs = [e for e in ar if e.get("dur", 0) >= 500]
    print(f"rank{r}: {len(ker)} kernels, {len(ar)} all_reduce, "
          f"{len(longs)} of them >=500us")
    for e in longs:
        print(f"        long one: abs_ts={e['ts']:.1f}  dur={e['dur']:.1f} us")

# Anchor on rank 0's long wait and ask what every rank was doing in that window.
r0 = ranks["0"]
big = max((e for e in r0 if "all_reduce" in e.get("name", "")),
          key=lambda e: e.get("dur", 0))
s, t = big["ts"], big["ts"] + big["dur"]
print(f"\n=== window where rank0 waits: abs {s:.1f} -> {t:.1f}  ({big['dur']/1000:.3f} ms)")
for r, ker in ranks.items():
    inside = [k for k in ker if k["ts"] < t and k["ts"] + k.get("dur", 0) > s]
    busy = sum(min(t, k["ts"] + k.get("dur", 0)) - max(s, k["ts"]) for k in inside
               if "all_reduce" not in k.get("name", ""))
    waiting = sum(min(t, k["ts"] + k.get("dur", 0)) - max(s, k["ts"]) for k in inside
                  if "all_reduce" in k.get("name", ""))
    # last kernel that ENDED before the window, and first that STARTS inside
    before = [k for k in ker if k["ts"] + k.get("dur", 0) <= s]
    gap = (s - max(k["ts"] + k.get("dur", 0) for k in before)) / 1000 if before else float("nan")
    print(f"\n  rank{r}: {len(inside)} kernels overlap the window | "
          f"real work {busy/1000:7.3f} ms | in a collective {waiting/1000:7.3f} ms | "
          f"idle before window {gap:.3f} ms")
    for k in sorted(inside, key=lambda k: -k.get("dur", 0))[:4]:
        print(f"      {k.get('dur',0):9.1f} us  {k['name'][:64]}")
PY
