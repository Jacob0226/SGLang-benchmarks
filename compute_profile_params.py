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
* Buffer per rank: 12 GB default for activations + cuda graph + KV
  scratch + framework bookkeeping. Tightening below 8 GB risks OOM.
* mem_fraction_static = (weights + L1_KV + buffer) / HBM. Anything
  >0.92 is rejected (no headroom for fragmentation), >0.85 is warned.
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
    needed_gb = weights_per_rank_gb + args.L1_size + args.buffer_gb
    mem_fraction = needed_gb / hbm_per_rank_gb

    if mem_fraction > args.max_mem_fraction:
        sys.stderr.write(
            f"ERROR: derived mem-fraction-static={mem_fraction:.3f} exceeds "
            f"--max-mem-fraction={args.max_mem_fraction}.\n"
            f"  weights/rank={weights_per_rank_gb:.1f}GB + "
            f"L1={args.L1_size}GB + buffer={args.buffer_gb}GB = "
            f"{needed_gb:.1f}GB; HBM/rank={hbm_per_rank_gb}GB.\n"
            f"  Reduce --L1-size, raise --tp, or use a smaller-weight model.\n"
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

    # Shell-eval-able output. All numeric so eval is safe.
    print(f"WEIGHTS_GB_PER_RANK={weights_per_rank_gb:.1f}")
    print(f"HBM_GB_PER_RANK={hbm_per_rank_gb}")
    print(f"KV_GB_PER_ROUND_PER_RANK={kv_gb_per_round_per_rank:.2f}")
    print(f"MEM_FRACTION_STATIC={mem_fraction:.2f}")
    print(f"WARMUP_ROUNDS={warmup_rounds}")
    print(f"NUM_ROUNDS={num_rounds}")
    print(f"L1_SIZE_GB={args.L1_size}")
    print(f"L2_SIZE_GB={args.L2_size}")
    print(f"PROFILE_TARGET_ROUND_1IDX={profile_target_round_1idx}")


if __name__ == "__main__":
    main()
