A=/sgl-workspace/aiter
O=/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm_0928
sed -n 175,235p $A/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py > $O/getconfig2.txt 2>&1
chmod 666 $O/getconfig2.txt
