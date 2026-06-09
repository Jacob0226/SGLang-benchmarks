import gzip, json, sys
from collections import defaultdict

path = sys.argv[1]
print("loading", path)
with gzip.open(path) as f:
    data = json.load(f)
ev = data["traceEvents"]

# GPU kernel events: cat in {kernel, gpu_memcpy, gpu_memset}
gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
ks = [e for e in ev if e.get("cat") in gpu_cats and "dur" in e and "ts" in e]
if not ks:
    print("no gpu kernels found; cats present:", sorted({e.get("cat") for e in ev if e.get("cat")})[:20])
    sys.exit(0)

ks.sort(key=lambda e: e["ts"])
span0 = ks[0]["ts"]
span1 = max(e["ts"] + e["dur"] for e in ks)
span = span1 - span0

# merge busy intervals
busy = 0
gaps = []
cur_s, cur_e = ks[0]["ts"], ks[0]["ts"] + ks[0]["dur"]
for e in ks[1:]:
    s, en = e["ts"], e["ts"] + e["dur"]
    if s > cur_e:
        gaps.append((cur_e, s - cur_e))  # (gap_start, gap_len)
        busy += cur_e - cur_s
        cur_s, cur_e = s, en
    else:
        cur_e = max(cur_e, en)
busy += cur_e - cur_s
idle = span - busy

print(f"GPU span      : {span/1e3:.1f} ms")
print(f"GPU busy      : {busy/1e3:.1f} ms ({100*busy/span:.1f}%)")
print(f"GPU idle      : {idle/1e3:.1f} ms ({100*idle/span:.1f}%)")
print(f"num kernels   : {len(ks)}")

gaps.sort(key=lambda g: -g[1])
print("\nTop 10 GPU idle gaps (ms) and CPU ops overlapping each gap:")
# index cpu ops (cat cpu_op / python) by ts for overlap lookup
cpu = [e for e in ev if e.get("cat") in ("cpu_op", "user_annotation", "python_function") and "dur" in e]
cpu.sort(key=lambda e: e["ts"])
import bisect
cpu_ts = [e["ts"] for e in cpu]
for gstart, glen in gaps[:10]:
    gend = gstart + glen
    # find cpu ops overlapping [gstart, gend]
    lo = bisect.bisect_left(cpu_ts, gstart - 5_000_000)
    names = defaultdict(float)
    for e in cpu[lo:]:
        if e["ts"] > gend:
            break
        ov_s = max(e["ts"], gstart); ov_e = min(e["ts"] + e["dur"], gend)
        if ov_e > ov_s:
            names[e["name"][:55]] += (ov_e - ov_s)
    top = sorted(names.items(), key=lambda x: -x[1])[:4]
    tops = ", ".join(f"{n}={d/1e3:.1f}ms" for n, d in top)
    print(f"  gap {glen/1e3:7.1f} ms | {tops}")
