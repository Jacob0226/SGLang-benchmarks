# tools/

Profiling, analysis, and micro-benchmark scripts for GLM-5 decode-layer optimization on B200 / MI355X.

## Trace analysis

| Script | Description | Example |
|--------|-------------|---------|
| `analyze_trace_overlap.py` | Sweep-line analysis of CUDA kernel overlap, bubble time, and stream concurrency from Torch Profiler traces. Supports HTML reports and two-trace comparison. | `python3 analyze_trace_overlap.py --trace1 dual.trace.json.gz --trace2 single.trace.json.gz --html cmp.html` |
| `extract_stream.py` | Extract and inspect kernels from specific CUDA streams within a trace. List streams, filter by time range, or compare two streams side-by-side. | `python3 extract_stream.py trace.json.gz --list-streams` |

## Micro-benchmarks

| Script | Description | Target |
|--------|-------------|--------|
| `glm5_decode_layer.py` | Single decode-layer micro-benchmark using real aiter/CK/TileLang/Triton kernels matching the actual SGLang decode trace. | MI355X |
| `glm5_proposalA_test.py` | Proposal A: overlap kv_a_norm / W_kc / RoPE / Cat with NSA indexer to fill the idle gap between fork and join. | MI355X |
| `glm5_proposalA_test_v2.py` | Proposal A v2: same idea, outputs traces to `trace_v2/` for comparison. | MI355X |

## Stream & overlap tests

| Script | Description | Target |
|--------|-------------|--------|
| `graph_stream_test.py` | Verify that CUDA/HIP Graph capture preserves N independent stream assignments. Auto-detects NVIDIA vs AMD. | B200 / MI355X |
| `test_dual_stream_sweep.py` | GLM5 MoE dual-stream unit test — replicates shared vs routed expert work on two streams. Compares single-stream vs dual-stream graph capture. | B200 / MI355X |
| `test_gemm_vs_elemwise_overlap.py` | Diagnostic: measures dual-stream overlap with correct fork-join pattern for GEMM+GEMM, GEMM+elementwise, and elem+elem combos. | B200 / MI355X |
| `test_graph_multi_stream_nv.py` | Test multi-stream GEMM overlap at various token counts (1–128). Each path simulates a full MLP: gate_up GEMM → mul → down GEMM. | B200 |
