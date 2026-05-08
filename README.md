# SGLang-benchmarks

Benchmark scripts and analysis tools for SGLang on B200 / MI355X / MI325X.

## Branches

| Branch | Focus |
|---|---|
| `main` | Stable shared utilities (analyze_trace, compare_breakdown, parse_*) |
| `GLM5_FP8_InferenceMax` | GLM-5-FP8 dual-stream / NSA / decode-layer optimization |
| `HiCache` | HiCache (hierarchical KV cache) benchmarks for DeepSeek-R1-0528 / GPT-OSS |

## Top-level scripts

| Script | What it benchmarks |
|---|---|
| `GLM.sh` | GLM-5-FP8 throughput / latency / profiling sweep (kept here as a layout reference) |
| `HiCache.sh` | SGLang HiCache (L1+L2+L3) sweep on DSR1-0528 / GPT-OSS, MI355X + B200 |

## Quick start — HiCache

```bash
# Default: DeepSeek-R1-0528, hicache (L1+L2 host memory), multi-turn benchmark
./HiCache.sh

# Sweep all 4 cache modes (no_radix → radix → hicache → hicache_file)
./HiCache.sh --sweep --tag 0508_sweep

# GPT-OSS 120B (MxFP4) variant
./HiCache.sh --model-preset gpt-oss-120b

# Tag results with the docker image so MI355X / B200 runs don't clobber
./HiCache.sh --docker rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260412   # MI355X
./HiCache.sh --docker lmsysorg/sglang:v0.5.9-cu130                      # B200
```

See `bash HiCache.sh --help` for the full flag list.

## Analysis tools

| Tool | Purpose |
|---|---|
| `parse_perf_metrics_to_csv.py` | `sglang.bench_serving` log → CSV (E2E / TTFT / ITL) |
| `parse_bench.py` | Parse benchmark logs → summary CSV |
| `parse_torch_profiler.py` | Streaming parse of large `.trace.json.gz` → kernel CSV |
| `analyze_trace.py` | 3-step trace analysis (kernel stats → layer structure → per-layer breakdown) |
| `compare_breakdown.py` | Compare two breakdown CSVs side-by-side (e.g. MI355X vs B200) |
| `fetch_kernels_from_torch_profiler.py` | Fetch kernels by timestamp range from a trace |

See `tools/README.md` for CI-reproduction and environment-setup helpers.
