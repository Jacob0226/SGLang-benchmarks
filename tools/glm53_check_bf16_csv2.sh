A=/sgl-workspace/aiter
O=/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm_0928
{
for C in $A/aiter/configs/model_configs/glm53_bf16_tuned_gemm.csv $A/aiter/configs/bf16_tuned_gemm.csv; do
  echo "########## $C ##########"
  [ -f "$C" ] || { echo "  (missing)"; continue; }
  echo "rows: $(($(wc -l < $C) - 1))"
  echo "-- header --"; head -1 $C
  echo "-- distinct (N,K) --"
  awk -F, 'NR>1{print $4"x"$5}' $C | sort -u | tr '\n' ' '; echo
  echo "-- rows with N=288 --"
  awk -F, 'NR>1 && $4==288' $C | head -20
  echo "  count: $(awk -F, 'NR>1 && $4==288' $C | wc -l)"
  echo "-- distinct M --"
  awk -F, 'NR>1{print $3}' $C | sort -n -u | tr '\n' ' '; echo
  echo "-- libtypes used --"
  awk -F, 'NR>1{print $11}' $C | sort | uniq -c
  echo
done
echo "########## what gets copied to /tmp/aiter_configs ##########"
grep -n "AITER_CONFIG_GEMM_BF16\|aiter_configs" $A/aiter/jit/core.py | head -12
} > $O/bf16_csv2.txt 2>&1; chmod 666 $O/bf16_csv2.txt; echo DONE >> $O/bf16_csv2.txt
