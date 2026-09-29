A=/sgl-workspace/aiter
O=/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm_0928
{
echo "===== solMap ====="
grep -n "solMap" $A/aiter/tuned_gemm.py | head
echo "--- solMap definition ---"
awk '/^solMap *=/,/^}/' $A/aiter/tuned_gemm.py
echo
echo "===== get_GEMM_A16W16_config ====="
awk '/def get_GEMM_A16W16_config/,/^def [a-z]/' $A/aiter/tuned_gemm.py | head -80
echo
echo "===== how /tmp/aiter_configs is populated (jit/core.py) ====="
sed -n 440,485p $A/aiter/jit/core.py
echo
echo "===== AITER_CONFIG_GEMM_BF16 ====="
sed -n 178,190p $A/aiter/jit/core.py
sed -n 288,302p $A/aiter/jit/core.py
echo
echo "===== current /tmp/aiter_configs ====="
ls -la /tmp/aiter_configs/ 2>/dev/null | head
echo "N=288 rows in /tmp copy: $(awk -F, 'NR>1 && $4==288' /tmp/aiter_configs/bf16_tuned_gemm.csv 2>/dev/null | wc -l)"
} > $O/tunedgemm.txt 2>&1
chmod 666 $O/tunedgemm.txt
