A=/sgl-workspace/aiter
O=/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm_0928
{
echo "===== get_padded_m ====="
F=$(grep -rln "def get_padded_m" $A/aiter/ | head -1); echo "in: $F"
sed -n "$(grep -n 'def get_padded_m' $F | cut -d: -f1),+45p" $F
echo
echo "===== rest of get_GEMM_A16W16_config (fallback) ====="
sed -n "$(grep -n 'def get_GEMM_A16W16_config(' $A/aiter/tuned_gemm.py | cut -d: -f1),+75p" $A/aiter/tuned_gemm.py | tail -50
} > $O/paddedm.txt 2>&1; chmod 666 $O/paddedm.txt
