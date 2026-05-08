#!/usr/bin/env python3
"""Compute the per-token KV cache footprint of an HF model checkpoint.

Reads <model_path>/config.json and prints, for various KV dtypes:
  - bytes per token for the full model (sum over all attention layers)
  - bytes per token per TP rank (after sharding / replication rules)
  - tokens needed in the working set to fill a given GPU KV pool

Usage:
  python3 calc_kv_per_token.py /data/huggingface/hub/deepseek-ai/DeepSeek-R1-0528
  python3 calc_kv_per_token.py /data/huggingface/hub/openai/gpt-oss-120b --tp 8 --device-pool-gb 160

Architecture handling:
  - MLA (DeepSeek V2/V3/R1):
      cache stores compressed kv_lora_rank + qk_rope_head_dim per layer
      cache is REPLICATED across TP ranks (each rank holds the full cache)
  - GQA / MHA (Llama, Qwen, GPT-OSS, …):
      cache stores 2 × num_kv_heads × head_dim per layer (K + V)
      cache is SHARDED across TP by num_kv_heads (1 head per rank when
      num_kv_heads == TP). When num_kv_heads < TP, KV is REPLICATED
      across rank groups (typical when TP > num_kv_heads).
  - Sliding-window layers (gpt-oss "sliding_attention"):
      cache size capped at sliding_window per token; reported separately
      as a fixed per-sequence cost rather than per-token growing cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DTYPE_BYTES = {
    "bf16":     2,
    "fp16":     2,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "fp4":      0.5,
}


def load_config(model_path: Path) -> dict:
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"config.json not found at {cfg_path}")
    with open(cfg_path) as f:
        return json.load(f)


def compute_mla(cfg: dict, tp: int) -> dict:
    """DeepSeek MLA: stores compressed kv_lora + k_pe; cache replicated across TP."""
    num_layers = cfg["num_hidden_layers"]
    kv_lora_rank = cfg.get("kv_lora_rank", 512)
    qk_rope_head_dim = cfg.get("qk_rope_head_dim", 64)
    elements_per_token = (kv_lora_rank + qk_rope_head_dim) * num_layers
    return {
        "attn_type": "MLA (DeepSeek)",
        "num_layers_full": num_layers,
        "num_layers_sliding": 0,
        "elements_per_token_full": elements_per_token,
        "elements_per_token_sliding": 0,
        "shard_factor": 1,                 # replicated → no /TP
        "shard_note": f"replicated across TP={tp} (MLA stores compressed kv_lora — each rank holds full cache)",
        "fixed_elements_per_seq": 0,
    }


def compute_gqa(cfg: dict, tp: int) -> dict:
    """Llama / Qwen / GPT-OSS / etc.: K and V stored separately, sharded by num_kv_heads."""
    num_kv_heads = cfg.get("num_key_value_heads") or cfg.get("num_attention_heads")
    head_dim = cfg.get("head_dim") or (cfg["hidden_size"] // cfg["num_attention_heads"])

    layer_types = cfg.get("layer_types") or []
    if layer_types and len(layer_types) == cfg["num_hidden_layers"]:
        full_layers = sum(1 for t in layer_types if t != "sliding_attention")
        sliding_layers = sum(1 for t in layer_types if t == "sliding_attention")
    else:
        full_layers = cfg["num_hidden_layers"]
        sliding_layers = 0

    sliding_window = cfg.get("sliding_window", 0)

    # Per-token (per layer): K + V = 2 × kv_heads × head_dim
    per_token_per_layer = 2 * num_kv_heads * head_dim
    elements_per_token_full = per_token_per_layer * full_layers

    # Sliding layers: cache is capped, so per-token contribution is 0; the
    # fixed per-sequence cost is sliding_window × per_token_per_layer × layers.
    fixed_elements = per_token_per_layer * sliding_layers * sliding_window

    # Sharding: GQA shards by num_kv_heads. If num_kv_heads >= TP, factor = TP.
    # If num_kv_heads < TP, KV gets replicated across (TP / num_kv_heads) groups,
    # so per-rank cost = full / num_kv_heads (not / TP).
    if tp <= num_kv_heads:
        shard = tp
        shard_note = f"sharded by KV heads: {num_kv_heads} kv_heads / TP={tp} = {num_kv_heads // tp} head/rank"
    else:
        shard = num_kv_heads
        shard_note = f"replicated across rank groups: {num_kv_heads} kv_heads < TP={tp}"

    layer_breakdown = f"{full_layers} full"
    if sliding_layers:
        layer_breakdown += f" + {sliding_layers} sliding-window (window={sliding_window})"

    return {
        "attn_type": "GQA" if num_kv_heads < cfg.get("num_attention_heads", num_kv_heads) else "MHA",
        "num_layers_full": full_layers,
        "num_layers_sliding": sliding_layers,
        "elements_per_token_full": elements_per_token_full,
        "elements_per_token_sliding": 0,
        "shard_factor": shard,
        "shard_note": shard_note,
        "fixed_elements_per_seq": fixed_elements,
        "layer_breakdown": layer_breakdown,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "sliding_window": sliding_window,
    }


def detect_attention_type(cfg: dict) -> str:
    """MLA if kv_lora_rank present, else GQA/MHA."""
    if cfg.get("kv_lora_rank") or cfg.get("q_lora_rank"):
        return "mla"
    return "gqa"


# ============================== Library API ==============================
# These are the only functions external scripts (HiCache.sh wrappers,
# parse_hicache_multiturn.py, …) should call. Keep the return shape stable.

def kv_per_token(model_path: str | Path, tp: int = 8,
                 kv_dtype: str = "fp8_e4m3") -> dict:
    """Compute per-token KV cache cost for a model checkpoint.

    Returns a dict with at least:
      attn_type:                "MLA (DeepSeek)" | "GQA" | "MHA"
      bytes_per_token_full:     full-model bytes/token (sum over layers)
      bytes_per_token_per_rank: per-rank bytes/token (after TP sharding)
      shard_factor:             how many ranks the KV is split across
      fixed_bytes_per_seq:      sliding-window layer fixed cost (0 if none)
      num_layers_full / sliding
    """
    cfg = load_config(Path(model_path).expanduser().resolve())
    attn = detect_attention_type(cfg)
    info = compute_mla(cfg, tp) if attn == "mla" else compute_gqa(cfg, tp)
    dtype_b = DTYPE_BYTES[kv_dtype]
    bpt_full = info["elements_per_token_full"] * dtype_b
    return {
        "attn_type": info["attn_type"],
        "shard_factor": info["shard_factor"],
        "shard_note": info["shard_note"],
        "num_layers_full": info["num_layers_full"],
        "num_layers_sliding": info["num_layers_sliding"],
        "elements_per_token_full": info["elements_per_token_full"],
        "bytes_per_token_full": bpt_full,
        "bytes_per_token_per_rank": bpt_full / info["shard_factor"],
        "fixed_bytes_per_seq": info.get("fixed_elements_per_seq", 0) * dtype_b,
        "kv_dtype": kv_dtype,
    }


def tokens_for_l1(device_pool_bytes: int, bytes_per_token_per_rank: float) -> int:
    """How many tokens fit in a GPU KV pool of `device_pool_bytes`."""
    if bytes_per_token_per_rank <= 0:
        return 0
    return int(device_pool_bytes / bytes_per_token_per_rank)


# ==========================================================================


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_path", help="Path to local HF checkpoint dir (must contain config.json)")
    ap.add_argument("--tp", type=int, default=8, help="Tensor parallel size (default 8)")
    ap.add_argument("--kv-dtype", default="fp8_e4m3", choices=list(DTYPE_BYTES),
                    help="KV cache dtype (default fp8_e4m3 to match SGLang's --kv-cache-dtype)")
    ap.add_argument("--device-pool-gb", type=float, default=0,
                    help="If set, also report tokens to fill an L1 cache of this size per rank")
    args = ap.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    cfg = load_config(model_path)
    print(f"=== {model_path.name} ===")
    print(f"  architectures: {cfg.get('architectures', ['?'])}")
    print(f"  num_hidden_layers: {cfg.get('num_hidden_layers')}")
    print(f"  hidden_size:       {cfg.get('hidden_size')}")
    print(f"  num_attention_heads: {cfg.get('num_attention_heads')}")

    attn = detect_attention_type(cfg)
    if attn == "mla":
        info = compute_mla(cfg, args.tp)
        print(f"  kv_lora_rank:       {cfg.get('kv_lora_rank')}")
        print(f"  qk_rope_head_dim:   {cfg.get('qk_rope_head_dim')}")
    else:
        info = compute_gqa(cfg, args.tp)
        print(f"  num_key_value_heads: {info['num_kv_heads']}")
        print(f"  head_dim:            {info['head_dim']}")
        print(f"  layers:              {info['layer_breakdown']}")
        if info["sliding_window"]:
            print(f"  sliding_window:      {info['sliding_window']}")

    bpt_full_elements = info["elements_per_token_full"]
    bpt_full_bytes_bf16 = bpt_full_elements * 2
    bpt_full_bytes = bpt_full_elements * DTYPE_BYTES[args.kv_dtype]
    bpt_per_rank_bytes = bpt_full_bytes / info["shard_factor"]

    print(f"\nAttention: {info['attn_type']}")
    print(f"Sharding:  {info['shard_note']}")
    print()

    print(f"Per-token KV (FULL model, growing context only):")
    print(f"  elements/token  = {bpt_full_elements:,}")
    print(f"  bytes/token  bf16 = {fmt_bytes(bpt_full_bytes_bf16)}")
    print(f"  bytes/token  {args.kv_dtype} = {fmt_bytes(bpt_full_bytes)}")
    print()

    print(f"Per-token KV (PER RANK at TP={args.tp}, {args.kv_dtype}):")
    print(f"  bytes/token = {fmt_bytes(bpt_per_rank_bytes)}")

    if info.get("fixed_elements_per_seq"):
        fixed_bytes = info["fixed_elements_per_seq"] * DTYPE_BYTES[args.kv_dtype]
        print(f"\nFixed per-sequence KV (sliding-window layers, NOT growing):")
        print(f"  bytes/sequence = {fmt_bytes(fixed_bytes)} (per rank: {fmt_bytes(fixed_bytes / info['shard_factor'])})")

    if args.device_pool_gb > 0:
        pool_bytes = args.device_pool_gb * 1024 ** 3
        tokens = pool_bytes / bpt_per_rank_bytes
        print(f"\nTokens to fill {args.device_pool_gb} GB GPU L1 (per rank, {args.kv_dtype}):")
        print(f"  total working-set tokens = {int(tokens):,}")
        # multiturn translation
        for nr, rl in [(10, 2048), (10, 4096), (20, 4096)]:
            nc = int(tokens / (nr * rl))
            if nc > 0:
                print(f"    multiturn (rounds={nr}, req_len={rl}) → num_clients ≈ {nc}")


if __name__ == "__main__":
    main()
