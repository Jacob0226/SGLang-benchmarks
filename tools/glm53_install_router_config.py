#!/usr/bin/env python3
"""Install GEMM-A16W16-N=288-K=4096.json into aiter's gfx950 config dir.

The MoE router GEMM (M=4, N=288, K=4096, bf16, 42 launches per decode forward)
has no tuned config for this shape, so it falls back to DEFAULT.json's M_LEQ_8
and costs 27.2 us on the 0923 image. The swept winner does it in 4.9 us.

A specialized config file replaces DEFAULT.json wholesale in the lookup, so the
file written here is DEFAULT plus overrides on the small-M buckets only; every
other bucket keeps DEFAULT's values.
"""

import argparse
import copy
import json
import os
import shutil

from aiter.ops.triton.utils.config_utils import resolve_config_dir

TUNED = {
    "BLOCK_SIZE_M": 4,
    "BLOCK_SIZE_N": 16,
    "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1,
    "NUM_KSPLIT": 4,
    "num_warps": 2,
    "num_stages": 3,
    "waves_per_eu": 0,
    "matrix_instr_nonkdim": 16,
    "cache_modifier": None,
}
OVERRIDE_BUCKETS = ["M_LEQ_1", "M_LEQ_4", "M_LEQ_8", "M_LEQ_16"]

ap = argparse.ArgumentParser()
ap.add_argument("--revert", action="store_true")
args = ap.parse_args()

cfg_dir = resolve_config_dir("gemm", "GEMM-A16W16", backend="triton")
target = os.path.join(cfg_dir, "GEMM-A16W16-N=288-K=4096.json")
print(f"config dir = {cfg_dir}")

if args.revert:
    if os.path.exists(target):
        os.remove(target)
        print(f"removed {target}")
    else:
        print("nothing to revert")
    raise SystemExit(0)

with open(os.path.join(cfg_dir, "DEFAULT.json")) as f:
    default = json.load(f)
print(f"DEFAULT buckets: {list(default.keys())}")

out = copy.deepcopy(default)
for b in OVERRIDE_BUCKETS:
    out[b] = dict(TUNED)

with open(target, "w") as f:
    json.dump(out, f, indent=4)
    f.write("\n")
os.chmod(target, 0o644)
print(f"wrote {target}")

# Also drop a copy next to the analysis output so it survives the container.
keep = "/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm/GEMM-A16W16-N=288-K=4096.json"
shutil.copy(target, keep)
try:
    os.chmod(keep, 0o666)
except OSError:
    pass

# ------------------------------------------------------------------ verify
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16 import (  # noqa: E402
    _get_config as _get_triton_config,
)
from aiter.ops.triton.utils.gemm_config_utils import _get_gemm_config_cached

_get_gemm_config_cached.cache_clear()

print("\n--- lookup check ---")
for M in (1, 4, 8, 16, 64, 8192):
    c, tuned = _get_triton_config(M, 288, 4096)
    print(
        f"  M={M:5d} is_tuned={tuned}  "
        f"BM={c['BLOCK_SIZE_M']} BN={c['BLOCK_SIZE_N']} BK={c['BLOCK_SIZE_K']} "
        f"KSPLIT={c['NUM_KSPLIT']} warps={c['num_warps']}"
    )

# The other shape that shares DEFAULT must be untouched.
print("\n--- unrelated shape still on DEFAULT ---")
c, tuned = _get_triton_config(4, 512, 4096)
print(f"  N=512 K=4096 M=4 is_tuned={tuned} BM={c['BLOCK_SIZE_M']} BK={c['BLOCK_SIZE_K']}")
print("INSTALL_DONE")
