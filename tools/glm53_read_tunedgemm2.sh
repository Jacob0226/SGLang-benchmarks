A=/sgl-workspace/aiter
O=/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm_0928
{
echo "===== get_GEMM_A16W16_config_ + get_GEMM_A16W16_config bodies ====="
sed -n "$(grep -n 'def get_GEMM_A16W16_config_' $A/aiter/tuned_gemm.py | cut -d: -f1),+70p" $A/aiter/tuned_gemm.py
echo
echo "===== triton_gemm ====="
sed -n "$(grep -n '^def triton_gemm' $A/aiter/tuned_gemm.py | cut -d: -f1),+35p" $A/aiter/tuned_gemm.py
echo
echo "===== N=288 rows in the merged runtime table ====="
head -1 /tmp/aiter_configs/bf16_tuned_gemm.csv
awk -F, 'NR>1 && $4==288' /tmp/aiter_configs/bf16_tuned_gemm.csv
} > $O/tunedgemm2.txt 2>&1; chmod 666 $O/tunedgemm2.txt
