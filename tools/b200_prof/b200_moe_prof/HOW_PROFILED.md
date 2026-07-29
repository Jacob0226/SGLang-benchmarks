# How these B200 GLM-5.2-NVFP4 MoE profiles were produced

Target: the MoE **down projection (gemm2)** trtllm-gen kernel
`bmm_Bfloat16_E2m1...sm100f` for **GLM-5.2-NVFP4, TP4, prefill (EXTEND)**, and the
neighboring gemm1 (up/gate) + finalize kernels.

## Environment
- GPU: 1x NVIDIA B200 (device 0 of an 8x B200 box), HBM3e.
- Container: `jacchang_GLM` = `lmsysorg/sglang:v0.5.15.post1-cu130-runtime`.
  - sglang 0.5.15.post1, flashinfer 0.6.12, Nsight Compute (ncu) 2025.3.1.0, CUDA 13.0.
- Model config (from `nvidia/GLM-5.2-NVFP4` config.json):
  hidden=6144, moe_intermediate=2048, n_routed_experts=256, top_k=8,
  n_group=1, topk_group=1, routed_scaling_factor=2.5, scoring=sigmoid (noaux_tc).

## Why a standalone harness instead of profiling the live server
ncu serializes/replays every kernel and cannot cleanly attach to the multi-process
TP4 sglang server (it would stall past the watchdog). So, per the profiling
instructions' fallback, `moe_harness.py` calls
`flashinfer.fused_moe.trtllm_fp4_block_scale_moe` **once at the deployed per-rank
shape**. This launches the identical trtllm-gen kernels
(`bmm_E2m1..._swiGlu` gemm1, `bmm_Bfloat16_E2m1...` gemm2/down, `finalizeKernelVecLoad`).
Weights/activations are random — numerics are meaningless, but tensor shapes,
dtypes and NVFP4/fp8-scale layouts match production exactly, which is all that
determines kernel scheduling and memory traffic.

Per-rank shape used (TP4):
- tokens T = 16384  (nearest multiple of 128 to the ~16304-token big prefill step;
  16304 is padded to 16384 by nvfp4 scale layout, which trips a flashinfer assert)
- hidden H = 6144, intermediate I = 512 (= 2048 / TP4)
- experts E = 256, top_k = 8, routing_method_type = 2 (DeepSeekV3 / sigmoid+bias+grouped)
- NVFP4 (E2m1) weights + activations, fp8_e4m3 block scales (sf_vec_size = 16), do_finalize=True

## Commands (run inside container `jacchang_GLM`, workdir = this folder)

### 1. b200_moe.ncu-rep  — full Nsight Compute profile (main artifact)
Warm up 4 iters (settle cubin load; flashinfer autotune is OFF by default so the
tile config is fixed and deterministic — exactly 3 matching kernels per iter:
gemm1 / gemm2 / finalize), skip those 12 warmup launches, then capture 9 clean
steady-state kernels (3 instances each).

```bash
T=16384 WARMUP=4 ITERS=4 CUDA_VISIBLE_DEVICES=0 \
ncu --set full \
    --target-processes all \
    -k "regex:bmm_Bfloat16|bmm_E2m1|finalize" \
    --launch-skip 12 --launch-count 9 \
    -o b200_moe -f \
    python3 moe_harness.py
```

Selected kernels captured (tile/CTA config is encoded in the name):
- gemm1 up/gate: `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x1_..._swiGlu_..._sm100f`
- **gemm2 down** : `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x128x256u2_s6_et128x128_m256x128x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f`
- finalize      : `moe::finalizeKernelVecLoad<...bfloat16_t...4,1>`

### 2. b200_gpu.txt  — device specs for roofline normalization
```bash
nvidia-smi -q > b200_gpu.txt
```

### 3. b200_kernel_cfg.txt  — which trtllm-gen configs were selected
Distinct kernel names extracted from the report (name = the config):
```bash
{ echo "# shape / config header ..."; \
  ncu --import b200_moe.ncu-rep --page raw --csv \
    | tail -n +2 | awk -F'","' '{print $5}' | sort -u; } > b200_kernel_cfg.txt
```

### 4. b200_moe.sass.txt  — SASS of the selected down cubin (optional)
The trtllm-gen cubins ship precompiled in the `flashinfer_cubin` package. The down
kernel's cubin is the `Bmm_Bfloat16_E2m1E2m1_..._t128x128x256u2_s6_..._m256x128x64_...`
file in `flashinfer_cubin/cubins/<hash>/batched_gemm-*/`.
```bash
cuobjdump -sass <that>.cubin > b200_moe.sass.txt
# TMA + tensor-core check:
grep -ioE "UTMALDG|LDGSTS|TCGEN05|OMMA|HMMA|QMMA|IMMA" b200_moe.sass.txt | sort | uniq -c
#   -> UTMALDG (TMA bulk async copy) + OMMA (tcgen05 tensor MMA on sm100) present
```

## Quick sanity numbers for the down kernel (bmm_Bfloat16, T=16384, under ncu locked clocks)
- Duration ~722 us (ncu clock-locking inflates vs the ~435 us deployed wall-clock)
- Memory Throughput 57.9% / DRAM 42.6% of peak  (memory-bound)
- Achieved Occupancy 16.4%, 168 registers/thread
- Highest-utilized pipeline: TMEM / tensor 35.8%
- SASS confirms TMA (UTMALDG) + tcgen05 tensor cores (OMMA) + async pipeline

## Files in this archive
- `b200_moe.ncu-rep`   — ncu --set full report (main deliverable)
- `b200_gpu.txt`       — nvidia-smi -q
- `b200_kernel_cfg.txt`— selected trtllm-gen kernel/config names
- `b200_moe.sass.txt`  — SASS of the selected down-GEMM cubin
- `moe_harness.py`     — the standalone reproduction script
- `HOW_PROFILED.md`    — this file
