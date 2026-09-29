A=/sgl-workspace/aiter
O=/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm_0928
{
C=$A/aiter/configs/model_configs/glm53_bf16_tuned_gemm.csv
echo "===== glm53_bf16_tuned_gemm.csv header ====="
head -1 $C
echo
echo "===== rows mentioning N=288 (col order per header) ====="
awk -F, 'NR==1{next} $2==288 || $3==288 {print}' $C | head -20
echo "(count: $(awk -F, 'NR==1{next} $2==288 || $3==288' $C | wc -l))"
echo
echo "===== distinct N values ====="
awk -F, 'NR==1{next}{print $2}' $C | sort -n -u | tr '\n' ' '
echo
echo "===== distinct M values ====="
awk -F, 'NR==1{next}{print $1}' $C | sort -n -u | tr '\n' ' '
echo
echo "===== total rows ====="
wc -l $C
echo
echo "===== is it loaded at runtime? who reads bf16_tuned_gemm.csv ====="
grep -rn "bf16_tuned_gemm" $A/aiter/*.py $A/aiter/jit/*.py 2>/dev/null | head -10
} > $O/bf16_csv.txt 2>&1
chmod 666 $O/bf16_csv.txt
echo DONE >> $O/bf16_csv.txt
