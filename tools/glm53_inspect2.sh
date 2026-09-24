A=/sgl-workspace/aiter
O=/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm
{
echo "===== where is gemm_a16w16 public api ====="
find $A/aiter -name "*.py" | xargs grep -ln "def gemm_a16w16" 2>/dev/null
echo
echo "===== def gemm_a16w16 signatures ====="
find $A/aiter -name "*.py" | xargs grep -n "def gemm_a16w16" 2>/dev/null
echo
echo "===== triton op wrapper file ====="
F=$(find $A/aiter/ops/triton -name "*.py" | xargs grep -ln "_gemm_a16_w16_kernel\|gemm_a16w16" 2>/dev/null | grep -v _triton_kernels | head -3)
echo "FILES: $F"
for f in $F; do echo "----- $f -----"; cat $f; done
echo
echo "===== gemm_config_utils.py ====="
cat $A/aiter/ops/triton/utils/gemm_config_utils.py
echo
echo "===== custom.py wv_splitk ====="
sed -n 1,60p $A/aiter/ops/custom.py
} > $O/api_inspect2.txt 2>&1
chmod 666 $O/api_inspect2.txt
echo DONE >> $O/api_inspect2.txt
