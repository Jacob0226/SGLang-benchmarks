A=/sgl-workspace/aiter
O=/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm
mkdir -p $O && chmod 777 $O
{
echo "===== gemm_a16w16.py ====="
cat $A/aiter/ops/triton/gemm_a16w16.py
echo
echo "===== _triton_kernels/gemm/basic/gemm_a16w16.py (head) ====="
sed -n 1,60p $A/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py
echo
echo "===== config utils ====="
grep -rn "M_LEQ\|def get_config\|_get_config_file\|json.load" $A/aiter/ops/triton/utils/core.py $A/aiter/ops/triton/utils/*.py 2>/dev/null | head -30
echo
echo "===== skinny gemm ====="
grep -rn "def wv_splitk\|def wvSplitK\|wv_splitk_small\|def gemm_a16w16" $A/aiter/ops/custom.py $A/aiter/ops/*.py 2>/dev/null | head -20
echo
echo "===== versions ====="
python -c "import triton,torch;print('triton',triton.__version__);print('torch',torch.__version__)"
} > $O/api_inspect.txt 2>&1
chmod 666 $O/api_inspect.txt
echo DONE >> $O/api_inspect.txt
