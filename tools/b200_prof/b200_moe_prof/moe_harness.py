#!/usr/bin/env python3
"""Standalone flashinfer trtllm-gen NVFP4 MoE microbenchmark.

Isolates the GLM-5.2 NVFP4 MoE (up/gate gemm1 -> swiglu -> down gemm2 -> finalize)
at the deployed TP4 prefill shape so Nsight Compute can profile the
`bmm_Bfloat16_E2m1...sm100f` down kernel without attaching to a live server.

Numerics are NOT meaningful (random weights/activations); only tensor shapes,
dtypes and layouts match production, which is all that matters for kernel
scheduling / memory-traffic profiling.

Shape defaults (GLM-5.2-NVFP4, TP4, per-rank), override via env:
  T=16304  H=6144  I=512  E=256  TOPK=8
"""
import os
import torch
import flashinfer
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe

T = int(os.environ.get("T", 16304))       # tokens in the big prefill step
H = int(os.environ.get("H", 6144))         # hidden_size
I = int(os.environ.get("I", 512))          # per-rank moe_intermediate (2048 / TP4)
E = int(os.environ.get("E", 256))          # n_routed_experts
TOPK = int(os.environ.get("TOPK", 8))      # num_experts_per_tok
NGROUP = int(os.environ.get("NGROUP", 1))
TOPKGROUP = int(os.environ.get("TOPKGROUP", 1))
ROUTED_SCALE = float(os.environ.get("ROUTED_SCALE", 2.5))
WARMUP = int(os.environ.get("WARMUP", 3))
ITERS = int(os.environ.get("ITERS", 6))
SF_VEC = 16  # nvfp4 block-scale vector size

DEV = "cuda"
torch.cuda.set_device(int(os.environ.get("CUDA_VISIBLE_DEV_IDX", 0)))
torch.manual_seed(0)


def q_2d(x, gsf):
    """Quantize a 2D [rows, cols] bf16 tensor to nvfp4: returns (packed uint8, fp8 scale)."""
    packed, scale = flashinfer.nvfp4_quantize(
        x, gsf, sfLayout=flashinfer.SfLayout.layout_128x4,
        do_shuffle=False, sf_vec_size=SF_VEC,
    )
    return packed, scale.view(torch.float8_e4m3fn)


def q_3d(x, gsf):
    """Quantize a batched [E, rows, cols] bf16 tensor to nvfp4 per expert."""
    packed, scale = flashinfer.nvfp4_batched_quantize(x, gsf, sf_vec_size=SF_VEC)
    return packed, scale.view(torch.float8_e4m3fn)


def main():
    print(f"[shape] T={T} H={H} I={I} E={E} TOPK={TOPK} "
          f"n_group={NGROUP} topk_group={TOPKGROUP} routed_scale={ROUTED_SCALE}")

    gsf = torch.tensor(1.0, device=DEV, dtype=torch.float32)

    # ---- hidden states (activations) ----
    hs = torch.randn(T, H, device=DEV, dtype=torch.bfloat16) * 0.1
    hs_fp4, hs_scale = q_2d(hs, gsf)

    # ---- gemm1 (up/gate) weights: [E, 2I, H] -> packed [E, 2I, H//2] ----
    w13 = torch.randn(E, 2 * I, H, device=DEV, dtype=torch.bfloat16) * 0.05
    w13_fp4, w13_scale = q_3d(w13, gsf)

    # ---- gemm2 (down) weights: [E, H, I] -> packed [E, H, I//2] ----
    w2 = torch.randn(E, H, I, device=DEV, dtype=torch.bfloat16) * 0.05
    w2_fp4, w2_scale = q_3d(w2, gsf)

    # ---- routing (sigmoid + bias, DeepSeekV3-style / noaux_tc) ----
    routing_logits = torch.randn(T, E, device=DEV, dtype=torch.bfloat16)
    routing_bias = torch.zeros(E, device=DEV, dtype=torch.bfloat16)

    # ---- per-expert dequant/requant scalars (values irrelevant for profiling) ----
    ones_e = torch.ones(E, device=DEV, dtype=torch.float32)

    print(f"[dtypes] hs_fp4={hs_fp4.dtype}{tuple(hs_fp4.shape)} "
          f"hs_scale={hs_scale.dtype}{tuple(hs_scale.shape)}")
    print(f"[dtypes] w13={w13_fp4.dtype}{tuple(w13_fp4.shape)} "
          f"w13_scale={tuple(w13_scale.shape)}")
    print(f"[dtypes] w2={w2_fp4.dtype}{tuple(w2_fp4.shape)} "
          f"w2_scale={tuple(w2_scale.shape)}")

    def run():
        return trtllm_fp4_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=routing_bias,
            hidden_states=hs_fp4,
            hidden_states_scale=hs_scale,
            gemm1_weights=w13_fp4,
            gemm1_weights_scale=w13_scale,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=w2_fp4,
            gemm2_weights_scale=w2_scale,
            gemm2_bias=None,
            output1_scale_scalar=ones_e,
            output1_scale_gate_scalar=ones_e,
            output2_scale_scalar=ones_e,
            num_experts=E,
            top_k=TOPK,
            n_group=NGROUP,
            topk_group=TOPKGROUP,
            intermediate_size=I,
            local_expert_offset=0,
            local_num_experts=E,
            routed_scaling_factor=ROUTED_SCALE,
            routing_method_type=2,  # DeepSeekV3 (sigmoid + bias + grouped top-k)
            do_finalize=True,
            tune_max_num_tokens=T,
        )

    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        out = run()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / ITERS
    o = out[0] if isinstance(out, (list, tuple)) else out
    print(f"[ok] output={tuple(o.shape)} {o.dtype}  avg {ms*1000:.1f} us / full-MoE-iter")


if __name__ == "__main__":
    main()
