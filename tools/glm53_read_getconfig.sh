A=/sgl-workspace/aiter
O=/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm_0928
{
echo "===== _get_config in _triton_kernels/gemm/basic/gemm_a16w16.py ====="
awk '/^def _get_config/,/^def [a-z_]+\(|^@/{print NR": "$0}' $A/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py | head -60
echo
echo "===== grep _get_config definition ====="
grep -n "_get_config" $A/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py
} > $O/getconfig.txt 2>&1; chmod 666 $O/getconfig.txt
