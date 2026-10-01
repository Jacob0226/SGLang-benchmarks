# GLM-5.3-Flash AgentX: MI355X (MXFP4) vs B200 (NVFP4), TP4

**Status: conc 1 / 4 / 8 / 12 done, conc 16 running (ETA 2026-10-01 10:00 UTC).**

MI355X lands within 0.3–4% of B200 on throughput per GPU at every point measured so far, and 11–18% behind on P90 interactivity. All four points ran with zero scheduler crashes and zero OOM warnings at `--mem-fraction-static 0.85 --chunked-prefill-size 16384`, the settings B200 had to lower to 0.75 / 8192.

## Performance

| conc | Throughput per GPU (tok/s)<br>MI355X | B200 | MI355X / B200 | P90 interactivity (tok/s/user)<br>MI355X | B200 | MI355X / B200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5,408 | 5,638 | 0.96 | 211.4 | 240.3 | 0.88 |
| 4 | 8,284 | 8,308 | 1.00 | 180.3 | 210.3 | 0.86 |
| 8 | 14,607 | 14,801 | 0.99 | 154.8 | 188.2 | 0.82 |
| 12 | 16,689 | 17,397 | 0.96 | 144.4 | 162.5 | 0.89 |
| 16 | *running* | 26,977 | | *running* | 120.4 | |

P90 interactivity is 1 / (P90 of ITL), in tok/s/user.

Latency detail:

| conc | ITL p50 (ms)<br>MI355X | B200 | ITL p90 (ms)<br>MI355X | TTFT p90 (s)<br>MI355X | B200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.27 | 2.55 | 4.73 | 1.86 | 2.98 |
| 4 | 3.53 | 2.87 | 5.55 | 1.74 | 3.13 |
| 8 | 4.09 | 2.98 | 6.46 | 1.55 | 3.11 |
| 12 | 4.35 | 3.24 | 6.93 | 1.74 | 3.04 |

The shape is consistent: MI355X decodes slower per token (ITL p50 +21–37%) but prefills faster (TTFT p90 about half of B200's), and the two roughly cancel in throughput per GPU.

## Health

| conc | scheduler crashes | OOM warnings | errors among profiled | profiled / all records | MTP acceptance length |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0.4% | 277 / 289 | 4.57 |
| 4 | 0 | 0 | 0.3% | 647 / 693 | 4.39 |
| 8 | 0 | 0 | 0.1% | 1,354 / 1,443 | 4.31 |
| 12 | 0 | 0 | 0.1% | 1,726 / 1,859 | 4.25 |

The gap between profiled and all is warmup, not failures: warmup runs with `max_tokens=1`, and aiperf files those as `InvalidInferenceResultError`. The error column is `check_health.py`'s profiled-only rate, which is what aiperf's 10% gate applies to.

### Output sanity check: OSL is pinned by `ignore_eos`

The AgentX scenario injects `ignore_eos=true` into every request (aiperf logs `injecting extra_inputs.ignore_eos=true`), so each request runs to the `max_tokens` its trace recorded. OSL sitting at the cap is the intended behaviour, and a raw OSL histogram cannot detect a non-terminating server. `Agentic/check_osl.py` checks what can:

| conc | OSL p50 | OSL p90 | at cap | stopped short | over cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 722 | 4,548 | 86% | 14% | 0 |
| 4 | 354 | 2,625 | 87% | 13% | 0 |
| 8 | 383 | 2,356 | 88% | 12% | 0 |
| 12 | 346 | 2,005 | 87% | 13% | 0 |

The OSL median falls with concurrency because more lanes sample more subagent turns, which are short. The B200 reference of OSL median 130 / p90 1,789 should be compared at the same concurrency; if it came from a point other than conc 16, B200 stopped short of `max_tokens` far more often than MI355X, and the two platforms generated different token counts. That is checkable from `osl_mismatch_diff_pct` in B200's `profile_export.jsonl`.

## Configuration

| | MI355X | B200 |
| --- | --- | --- |
| checkpoint | `amd/GLM-5.3-Flash-Quark-MXFP4` | `nvidia/GLM-5.3-Flash-NVFP4` |
| image | `rocm/sgl-dev:v0.5.20-rocm10-mi35x-20260928` | `lmsysorg/sglang:v0.5.20-cu130` |
| quantization | Quark MXFP4, auto-detected | `modelopt_fp4` |
| DSA prefill / decode | tilelang / tilelang | trtllm / trtllm |
| MoE runner | aiter | flashinfer_trtllm |
| KV dtype | bfloat16 | fp8_e4m3 |
| mem-fraction-static / chunked-prefill | **0.85 / 16384** | 0.75 / 8192 |
| KV pool per rank | 11.5M tokens | — |
| KDA state slots per rank | 1,197 | 518 |
| `SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION` | 0.04 | n/a |

Identical on both: TP4 / EP1, MTP steps 5 / 6 draft tokens, acceptance by real verification (`ACC_MODE=real`), radix cache on with `--mamba-radix-cache-strategy extra_buffer` (resolved on `linear_attn_backend=triton`, `page_size=64`), `--max-running-requests 2×conc`, HiCache off, `MODEL_PREFIX=glm5.3flash` (unfiltered 1M-context corpus), 3600 s window, InferenceX `bee3405ca`, aiperf `754356e9`.

InferenceX `bee3405ca` is the last commit whose layout the driver's preflight accepts: the next commit removes `utils/agentic-benchmark/requirements.txt`, and 2026-09-27 moves everything under `inferencex-e2e/`. The agentic client, `benchmark_lib.sh`'s agentic half and the aiperf pin do not change between the B200 recipe's date (2026-09-24) and `bee3405ca`.

## Reproduce

Inside the MI355X container, with `fuser` available (`apt-get install -y psmisc`; this image does not ship it):

```bash
cd SGLang-benchmarks/Agentic
export HIP_VISIBLE_DEVICES=0,1,2,3
export AIPERF_RUNTIME_DIR=/data/ix-agentic-runtime HF_HUB_CACHE=/data/huggingface/hub
# The tokenised corpus is ~6 GB; keep it off a nearly full root disk.
export AIPERF_DATASET_MMAP_BASE_PATH=/dev/shm/aiperf/base AIPERF_DATASET_MMAP_CACHE_DIR=/dev/shm/aiperf/cache
./ix_agentx_glm53flash.sh --conc "1 4 8 12 16"

./ix_agentx_summarize.py --hw mi355x <sweep-dir>
./check_health.py <sweep-dir>
./check_osl.py <sweep-dir>
./archive_agentx.sh <sweep-dir>
```

The B200 curve is the same driver with `--platform b200 --mem-fraction 0.75 --chunked-prefill 8192`.

The per-point result JSONs, workload summaries and command lines are archived under `Agentic/archive/amd_GLM-5.3-Flash-Quark-MXFP4/rocm_sgl-dev-v0.5.20-rocm10-mi35x-20260928/bench-Agentic-TP4_EP1-mtp5/`, and `ix_agentx_summarize.py` reads that directory directly.
