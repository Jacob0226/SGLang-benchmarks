#!/usr/bin/env python3
"""
compute_profile_params.py — derive cascade-profile launch params from
two user-facing knobs (--L1-size, --L2-size in GB per rank), so you
don't have to hand-tune --mem-fraction-static or --num-rounds.

Outputs shell-eval-able lines:
    WEIGHTS_GB_PER_RANK=...
    HBM_GB_PER_RANK=...
    KV_GB_PER_ROUND_PER_RANK=...
    MEM_FRACTION_STATIC=...
    WARMUP_ROUNDS=...
    NUM_ROUNDS=...
    L1_SIZE_GB=...
    L2_SIZE_GB=...
    PROFILE_TARGET_ROUND_1IDX=...

Intended use from bash:
    eval "$(python3 compute_profile_params.py --model M --L1-size 40 --L2-size 80)"

Why these specific numbers
--------------------------
* HBM per rank: read from `nvidia-smi --query-gpu=memory.total` (NVIDIA)
  or `rocm-smi --showmeminfo vram --json` (ROCm). Both report bytes;
  we round down to GiB.
* Weights per rank: sum of *.safetensors file sizes in $MODEL_PATH,
  divided by tp_size. This is the on-disk size which equals the
  in-VRAM weight footprint for any single dtype (fp8/bf16/fp16).
* Framework reserve per rank: HBM locked up by PyTorch CUDA/HIP
  context + RCCL/NCCL communicator buffers + driver JIT cache BEFORE
  SGLang runs `torch.cuda.mem_get_info()` and computes its mem_fraction
  budget. SGLang prints this implicitly as `Load weight begin. avail
  mem=X GB` — comparing X to `rocm-smi`/`nvidia-smi` HBM gives the
  reserve. Empirically:
    MI355X TP=8 (RCCL + QuickReduce):  ~6 GB/rank
    B200 TP=8 (NCCL):                  ~4-6 GB/rank (estimate)
  Stable for a fixed (HW, driver, RCCL/NCCL, TP, env) tuple; varies
  little across models. Default `--framework-reserve-gb 8` is slightly
  conservative on both platforms; for precise calibration pass
  `--reserve-from-server-log <path>` to a previous run's server.log.
* Buffer per rank: 12 GB default minimum HBM reserved for activations
  + cuda graph + KV scratch + framework bookkeeping. Tightening below
  8 GB risks OOM.
* mem_fraction_static = (weights + L1_KV) / (HBM - framework_reserve).
  SGLang's own server_args.py defines it as
      mem_fraction_static = (weights + KV cache pool) / HBM
  where HBM here is what `torch.cuda.mem_get_info()` returns AFTER
  framework init — i.e. SGLang's "usable" HBM, not the raw vendor
  total. Subtracting --framework-reserve-gb from the rocm-smi/nvidia-smi
  total before dividing makes the helper's mem_fraction land on
  SGLang's internal denominator, so the actual KV pool ends up equal
  to --L1-size (instead of being short by the framework reserve, as
  happened in cascade-FairCompare_0520_v3 with 32.25 GB vs 35 GB).
  The activation / cuda graph reservation lives in the OTHER
  (1 - mem_fraction_static) portion, not inside the numerator. Putting
  --buffer-gb inside the numerator inflates mem_fraction_static and
  causes SGLang to allocate a KV pool larger than --L1-size by
  buffer_gb (12 GB by default) — which then violates the HiCache
  "host_pool > device_pool" invariant the moment --L2-size is set to
  any value near --L1-size + 12 GB.
  Validation: we still enforce
  `(1 - mem_fraction) * usable_HBM >= buffer_gb` so the activation
  budget can't get squeezed to zero. Anything that would push
  mem-fraction-static above --max-mem-fraction (default 0.92, no
  headroom for fragmentation) is rejected; >0.85 is warned.
* KV per token (replicated MLA): default 34 KB for DSR1-0528.
  Override with --kv-bytes-per-token for other models. Math:
    DSR1: kv_lora_rank(512) + qk_rope_head_dim(64) = 576 B/layer
          × 61 layers × 1 byte (FP8 KV) = 35,136 B ≈ 34 KB
* KV per round per rank = num_clients × request_length × kv_per_token.
  For DSR1 default workload (300 × 4096 × 34 KB) = 39.85 GiB ≈ 40 GB.
  This is REPLICATED across TP ranks because MLA's c_kv is shared
  across heads, not sharded by head.
* WARMUP_ROUNDS = ceil(L1_size / KV_per_round). The (warmup+1)-th round
  is the first one where L1 is ALREADY 100% full at the round's first
  prefill step, so every insertion in that round triggers an L1->L2
  eviction. That's the profile target ("steady-state L1->L2 cliff").

  Why ceil and not floor: if L1 isn't a multiple of KV_per_round,
  floor(L1/KV) leaves L1 with `(L1 mod KV)` GB free at the end of
  warmup. The target round's early prefill steps would then just fill
  that remaining space (no eviction); only the later prefills would
  evict. With --num-steps 5, that early window often dominates the
  trace and the L1->L2 kernels never show up.
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys


def detect_hbm_per_rank_gb():
    """Return one GPU's HBM in GiB. Tries NVIDIA first, then AMD."""
    # NVIDIA: simple, deterministic.
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if out:
            mb = int(out.splitlines()[0].strip())
            return mb // 1024  # MiB → GiB
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    # AMD: rocm-smi has multiple output schemas across versions; --json is
    # the most stable. Look for the largest "Total" key under any device.
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            stderr=subprocess.DEVNULL,
        ).decode()
        data = json.loads(out)
        biggest_bytes = 0
        for dev, info in (data or {}).items():
            for k, v in (info or {}).items():
                # Keys vary: "VRAM Total Memory (B)" or just numeric "Total"
                if "total" in k.lower():
                    try:
                        biggest_bytes = max(biggest_bytes, int(v))
                    except (TypeError, ValueError):
                        continue
        if biggest_bytes > 0:
            return biggest_bytes // (1024 ** 3)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        pass
    raise RuntimeError(
        "Could not detect HBM per rank via nvidia-smi or rocm-smi --json"
    )


_LOAD_WEIGHT_BEGIN_RE = re.compile(r"Load weight begin\. avail mem=([\d.]+)\s*GB")


def extract_sglang_usable_hbm_gb(server_log_path):
    """Return the first 'Load weight begin. avail mem=X GB' value from a
    SGLang server.log (in GB). That value IS SGLang's usable HBM per rank
    on the box that produced this log — i.e. rocm-smi/nvidia-smi total
    minus everything PyTorch / RCCL / NCCL / driver JIT pre-allocated.
    Returns None if no matching line is found."""
    try:
        with open(server_log_path, "r", errors="ignore") as f:
            for line in f:
                m = _LOAD_WEIGHT_BEGIN_RE.search(line)
                if m:
                    return float(m.group(1))
    except OSError:
        return None
    return None


def model_weights_gb(model_path):
    """Sum sizes of *.safetensors in MODEL_PATH (in GiB)."""
    total_bytes = 0
    pattern = os.path.join(model_path, "*.safetensors")
    for f in glob.glob(pattern):
        total_bytes += os.path.getsize(f)
    if total_bytes == 0:
        # Some HF dumps put weights in nested model-* subdirs. Fall back
        # to a recursive search rather than failing immediately.
        for f in glob.glob(os.path.join(model_path, "**", "*.safetensors"),
                           recursive=True):
            total_bytes += os.path.getsize(f)
    return total_bytes / (1024 ** 3)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--model", required=True,
                   help="Model dir (must contain *.safetensors)")
    p.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    p.add_argument("--L1-size", dest="L1_size", type=int, required=True,
                   help="Target L1 (GPU radix cache) size in GB per rank")
    p.add_argument("--L2-size", dest="L2_size", type=int, required=True,
                   help="Target L2 (HiCache host pool) size in GB per rank; "
                        "must be >= L1-size")
    p.add_argument("--num-clients", type=int, default=300,
                   help="Cascade --num-clients (used to compute KV per round)")
    p.add_argument("--request-length", type=int, default=4096,
                   help="Cascade --request-length (per-turn user tokens)")
    p.add_argument("--kv-bytes-per-token", type=int, default=34 * 1024,
                   help="KV cache bytes per token (replicated, all layers). "
                        "Default 34 KiB for DSR1-0528 MLA + FP8 KV.")
    p.add_argument("--buffer-gb", type=int, default=12,
                   help="Per-rank HBM reserved for activations / cuda graph / "
                        "KV scratch")
    p.add_argument("--framework-reserve-gb", type=float, default=8.0,
                   help="HBM/rank locked up by PyTorch CUDA/HIP context + "
                        "RCCL/NCCL communicator buffers + driver JIT cache "
                        "BEFORE SGLang computes its mem_fraction_static "
                        "budget. Subtracted from rocm-smi/nvidia-smi HBM "
                        "before dividing. Default 8 GB is conservative for "
                        "MI355X TP=8 (~6 GB measured) and B200 TP=8 "
                        "(~4-6 GB est.). Use --reserve-from-server-log to "
                        "calibrate precisely from a previous run.")
    p.add_argument("--reserve-from-server-log", type=str, default=None,
                   help="Path to a previous SGLang server.log. Reads the "
                        "first 'Load weight begin. avail mem=X GB' line and "
                        "computes framework reserve = HBM - X, overriding "
                        "--framework-reserve-gb. Use this once per "
                        "(HW, driver, TP) combination to get exact "
                        "calibration; the reserve is stable across models "
                        "for a fixed env.")
    p.add_argument("--rounds-profile", type=int, default=1,
                   help="How many rounds the profiler will cover")
    p.add_argument("--rounds-margin", type=int, default=1,
                   help="Extra rounds tacked onto NUM_ROUNDS so the bench "
                        "doesn't end exactly when the profiler returns")
    p.add_argument("--max-mem-fraction", type=float, default=0.92,
                   help="Reject params that would push mem-fraction-static "
                        "above this; 0.92 leaves room for fragmentation")
    args = p.parse_args()

    if args.L2_size < args.L1_size:
        sys.stderr.write(
            f"ERROR: --L2-size ({args.L2_size}) must be >= --L1-size "
            f"({args.L1_size}). HiCache hierarchy requires L2 ≥ L1; if L2 "
            f"is smaller it can't hold L1 evictions and the spill defeats "
            f"the purpose.\n"
        )
        sys.exit(1)

    weights_gb = model_weights_gb(args.model)
    if weights_gb <= 0:
        sys.stderr.write(
            f"ERROR: no .safetensors found under {args.model}\n"
        )
        sys.exit(1)
    weights_per_rank_gb = weights_gb / args.tp

    hbm_per_rank_gb = detect_hbm_per_rank_gb()

    # Framework reserve: HBM that PyTorch + RCCL/NCCL + driver lock up
    # BEFORE SGLang sees mem_get_info(). If --reserve-from-server-log is
    # given, derive the exact value for this (HW, driver, TP) combo.
    framework_reserve_gb = args.framework_reserve_gb
    reserve_source = "default"
    if args.reserve_from_server_log:
        sglang_usable_hbm = extract_sglang_usable_hbm_gb(
            args.reserve_from_server_log
        )
        if sglang_usable_hbm is None:
            sys.stderr.write(
                f"WARN: no 'Load weight begin. avail mem=' line in "
                f"{args.reserve_from_server_log}; falling back to "
                f"--framework-reserve-gb={args.framework_reserve_gb}.\n"
            )
        else:
            framework_reserve_gb = hbm_per_rank_gb - sglang_usable_hbm
            reserve_source = (
                f"measured from {args.reserve_from_server_log} "
                f"(SGLang avail={sglang_usable_hbm:.2f} GB)"
            )

    usable_hbm_gb = hbm_per_rank_gb - framework_reserve_gb
    if usable_hbm_gb <= 0:
        sys.stderr.write(
            f"ERROR: framework_reserve_gb={framework_reserve_gb:.1f} >= "
            f"HBM/rank={hbm_per_rank_gb}; nothing left for the model.\n"
        )
        sys.exit(2)

    # SGLang's mem_fraction_static = (weights + KV pool) / usable_HBM.
    # Using rocm-smi's raw HBM as the denominator (instead of usable_HBM)
    # makes mem_fraction land too low, producing a KV pool that's short
    # by `framework_reserve_gb * mem_fraction`. Buffer lives in the OTHER
    # (1 - mem_fraction_static) portion; including it in the numerator
    # inflates the KV pool by buffer_gb and trips the HiCache
    # "host_pool > device_pool" assertion when --L2-size is sized to
    # L1 + small margin. See cascade-FairCompare_0520_v2 / v3 postmortems.
    static_gb = weights_per_rank_gb + args.L1_size
    mem_fraction = static_gb / usable_hbm_gb
    remaining_gb = usable_hbm_gb - static_gb  # for activations + cuda graph

    if mem_fraction > args.max_mem_fraction:
        sys.stderr.write(
            f"ERROR: derived mem-fraction-static={mem_fraction:.3f} exceeds "
            f"--max-mem-fraction={args.max_mem_fraction}.\n"
            f"  weights/rank={weights_per_rank_gb:.1f}GB + "
            f"L1={args.L1_size}GB = {static_gb:.1f}GB; "
            f"usable HBM/rank={usable_hbm_gb:.1f}GB "
            f"(raw {hbm_per_rank_gb}GB - reserve {framework_reserve_gb:.1f}GB).\n"
            f"  Reduce --L1-size, raise --tp, or use a smaller-weight model.\n"
        )
        sys.exit(2)
    if remaining_gb < args.buffer_gb:
        sys.stderr.write(
            f"ERROR: only {remaining_gb:.1f}GB/rank left for activations + "
            f"cuda graph after weights ({weights_per_rank_gb:.1f}GB) + "
            f"L1 ({args.L1_size}GB), need >= --buffer-gb={args.buffer_gb}GB.\n"
            f"  Reduce --L1-size, raise --tp, or lower --buffer-gb if you "
            f"know the activation footprint.\n"
        )
        sys.exit(2)
    if mem_fraction > 0.85:
        sys.stderr.write(
            f"WARN: mem-fraction-static={mem_fraction:.2f} is tight; "
            f"with large CUDA graphs you may OOM. Consider --buffer-gb 16 "
            f"or shrink --L1-size by 4-8 GB if it OOMs.\n"
        )

    kv_bytes_per_round_per_rank = (
        args.num_clients * args.request_length * args.kv_bytes_per_token
    )
    kv_gb_per_round_per_rank = kv_bytes_per_round_per_rank / (1024 ** 3)

    # Profile target = first round whose ENTIRE prefill runs against a
    # 100%-full L1, so every insertion triggers an L1->L2 eviction.
    # That requires warmup to have written at least L1 bytes into the
    # cache, i.e. warmup × KV >= L1, i.e. warmup = ceil(L1 / KV).
    #
    # Using floor here is a subtle bug: if L1 isn't a multiple of KV,
    # the (floor+1)-th round STARTS with `L1 mod KV` GB free in L1, so
    # the early prefill steps just fill that residual headroom without
    # evicting. A --num-steps 5 trace then frequently captures only
    # those non-evicting steps and misses the L1->L2 kernels entirely
    # (e.g. L1=60, KV=40: floor gives target=R2 which is half "fill
    # remaining 20 GB" + half "evict 20 GB to L2"; ceil gives target=R3
    # which is pure eviction).
    #
    # max(1, ...) is a safety net; ceil of any positive ratio is >= 1.
    warmup_rounds = max(1, math.ceil(args.L1_size / kv_gb_per_round_per_rank))
    profile_target_round_1idx = warmup_rounds + 1
    num_rounds = warmup_rounds + args.rounds_profile + args.rounds_margin

    # Stderr diagnostics: not eval'd by the bash caller (whitelist regex
    # in cascade_dsr1_lite.sh only captures stdout `^[A-Z0-9_]+=` lines),
    # but visible in chain.log for postmortem.
    sys.stderr.write(
        f"INFO: HBM/rank={hbm_per_rank_gb}GB, "
        f"framework_reserve={framework_reserve_gb:.2f}GB ({reserve_source}), "
        f"usable={usable_hbm_gb:.2f}GB, "
        f"weights={weights_per_rank_gb:.1f}GB, L1={args.L1_size}GB, "
        f"derived mem_fraction_static={mem_fraction:.4f} -> "
        f"emitted as {mem_fraction:.2f}\n"
    )

    # Shell-eval-able output. All numeric so eval is safe.
    print(f"WEIGHTS_GB_PER_RANK={weights_per_rank_gb:.1f}")
    print(f"HBM_GB_PER_RANK={hbm_per_rank_gb}")
    print(f"USABLE_HBM_GB_PER_RANK={usable_hbm_gb:.2f}")
    print(f"FRAMEWORK_RESERVE_GB={framework_reserve_gb:.2f}")
    print(f"KV_GB_PER_ROUND_PER_RANK={kv_gb_per_round_per_rank:.2f}")
    print(f"MEM_FRACTION_STATIC={mem_fraction:.2f}")
    print(f"WARMUP_ROUNDS={warmup_rounds}")
    print(f"NUM_ROUNDS={num_rounds}")
    print(f"L1_SIZE_GB={args.L1_size}")
    print(f"L2_SIZE_GB={args.L2_size}")
    print(f"PROFILE_TARGET_ROUND_1IDX={profile_target_round_1idx}")


if __name__ == "__main__":
    main()
