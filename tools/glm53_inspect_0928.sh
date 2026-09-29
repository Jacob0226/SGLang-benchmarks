A=/sgl-workspace/aiter
O=/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm_0928
mkdir -p $O && chmod 777 $O
{
echo "===== versions ====="
python -c "import triton,torch;print('triton',triton.__version__);print('torch',torch.__version__)"
python -c "import sys;print('python',sys.version.split()[0])"
cat /opt/rocm/.info/version 2>/dev/null
echo
echo "===== aiter HEAD ====="
git config --global --add safe.directory $A 2>/dev/null
git -C $A log --oneline -1
echo "PR5599 in HEAD?"
git -C $A merge-base --is-ancestor 6b35e2b9c HEAD 2>/dev/null && echo "  YES" || echo "  NO"
echo
echo "===== does upstream now ship N=288 config? ====="
ls $A/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/ | grep -E "N=288-" && echo "  ^ EXISTS UPSTREAM" || echo "  still missing"
echo
echo "===== all gemm_a16w16 tuned shapes on gfx950 ====="
ls $A/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/
echo
echo "===== DEFAULT.json ====="
cat $A/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/DEFAULT.json
echo
echo "===== glm5_3 fmoe csvs present? ====="
ls $A/aiter/configs/model_configs/ | grep -i "glm5" || echo none
echo
echo "===== sglang version ====="
python -c "import sglang,os;print(sglang.__version__ if hasattr(sglang,'__version__') else '?');print(os.path.dirname(sglang.__file__))"
git -C /sgl-workspace/sglang log --oneline -1 2>/dev/null
} > $O/inspect.txt 2>&1
chmod 666 $O/inspect.txt
echo DONE >> $O/inspect.txt
