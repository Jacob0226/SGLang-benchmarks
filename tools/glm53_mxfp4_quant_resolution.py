"""Resolve GLM-5.3-Flash Quark MXFP4 per-layer quant decisions without a GPU.

Reproduces the table in GLM53-Flash-ROCm-PrStack_Repro.md "The MXFP4 checkpoint
needs two more PRs": for a given sglang tree, ask what quantization method each
runtime layer name would be given, and whether the vision qkv is excluded.

A layer that should be block FP8 but resolves to fp4/per_group is the silent
failure the document describes.

    PYTHONPATH=<tree>/python python3 glm53_mxfp4_quant_resolution.py <config.json>
"""

import json
import sys
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.quark.quark import QuarkConfig
from sglang.srt.layers.quantization.quark.utils import should_ignore_layer
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration

CHECKS = [
    # (runtime layer name, what the checkpoint says it must be)
    ("model.layers.7.self_attn.q_a_proj", "fp8_e4m3/per_block"),
    ("model.layers.7.self_attn.kv_a_proj_with_mqa", "fp8_e4m3/per_block"),
    ("model.layers.7.self_attn.o_proj", "fp8_e4m3/per_block"),
    ("model.layers.45.mlp.experts.7.up_proj", "fp8_e4m3/per_block"),
    ("model.layers.3.mlp.experts.0.up_proj", "fp4/per_group"),
    ("model.layers.20.mlp.experts.0.down_proj", "fp4/per_group"),
]

EXCLUDE_CHECKS = [
    ("visual.blocks.0.attn.qkv_proj", True),
    ("visual.blocks.23.attn.qkv_proj", True),
    ("visual.blocks.0.mlp.gate_up_proj", True),
    ("model.layers.7.self_attn.q_a_proj", False),
]


def spec(cfg):
    if cfg is None:
        return "None (falls back to global)"
    w = cfg.get("weight") or {}
    bs = w.get("block_size")
    return f"{w.get('dtype')}/{w.get('qscheme')}" + (f" {bs}" if bs else "")


def main(path):
    raw = json.load(open(path))
    qc = dict(raw["quantization_config"])
    text = raw.get("text_config", raw)
    hf = SimpleNamespace(
        model_type=raw.get("model_type", "glm5_next"),
        num_hidden_layers=text.get("num_hidden_layers"),
        num_nextn_predict_layers=text.get("num_nextn_predict_layers", 0),
    )
    qc["packed_modules_mapping"] = Glm5NextForConditionalGeneration.packed_modules_mapping

    config = QuarkConfig(
        quant_config=qc,
        hf_config=hf,
        kv_cache_group=list((qc.get("export") or {}).get("kv_cache_group") or []),
        is_prequantized=True,
    )
    config.apply_weight_name_mapper(
        Glm5NextForConditionalGeneration.hf_to_sglang_mapper
    )

    lqc = config.quant_config.get("layer_quant_config") or {}
    print(f"layers={hf.num_hidden_layers} nextn={hf.num_nextn_predict_layers}")
    print(f"layer_quant_config entries: {len(lqc)}")
    print(f"exclude entries (after mapper): {len(config.exclude_layers)}")
    print(f"global: {spec(config.quant_config.get('global_quant_config'))}\n")

    dummy = torch.nn.Linear(1, 1)
    fails = 0

    # _find_matched_config is what get_quant_method actually consults: an
    # explicit entry if there is one, otherwise the global config. Querying
    # _find_matched_layer_config instead reports None for every layer that
    # legitimately inherits the global scheme.
    print(f"{'runtime layer':<48} {'expected':<22} {'resolved':<30} ok")
    for name, expected in CHECKS:
        got = spec(config._find_matched_config(name, dummy))
        dtype, qscheme = expected.split("/")
        ok = dtype in got and qscheme in got
        fails += not ok
        print(f"{name:<48} {expected:<22} {got:<30} {'OK' if ok else 'FAIL'}")

    print(f"\n{'runtime layer':<48} {'excluded?':<22} {'resolved':<30} ok")
    for name, expected in EXCLUDE_CHECKS:
        got = should_ignore_layer(
            name, ignore=config.exclude_layers, fused_mapping=config.packed_modules_mapping
        )
        ok = got == expected
        fails += not ok
        print(f"{name:<48} {str(expected):<22} {str(got):<30} {'OK' if ok else 'FAIL'}")

    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
