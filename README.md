# SGLang-benchmarks

Benchmark drivers, profiling analysis, and micro-benchmarks for SGLang on MI355X /
MI325 (ROCm) and B200 (CUDA). Run everything from this directory.

| directory | content |
|-----------|---------|
| `trace_analysis/` | PyTorch-profiler trace analysis: SGLang and ATOM analyzers, the comparison workbooks built on their step3 output, and per-window / per-stream diagnostics. Start with its README. |
| `tools/` | micro-benchmarks, stream/graph experiments, CI reproduction, environment setup |
| `amd_gemm_tuning/` | GEMM tuning configs and results |
| `ix_bench_serving/` | vendored `bench_serving` variant |
| `MultiStream_Patches/` | patches for dual-stream MoE experiments |
| `*.sh` at the root | end-to-end serve + bench + profile drivers (`GLM.sh`, `ATOM_GLM.sh`, …) |

Bench log parsing: `parse_perf_metrics_to_csv.py --root-dir <results_dir>`.
