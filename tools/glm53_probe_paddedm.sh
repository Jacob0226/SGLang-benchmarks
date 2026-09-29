O=/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm_0928
cd /sgl-workspace
python - > $O/paddedm_probe.txt 2>&1 <<'PY'
from aiter.ops.gemm_op_common import get_padded_m
N, K = 288, 4096
BS = [1,2,4,8,12,16,24,32,40,48,56,64,72,80,96,112,128,160,192,256]
print(f"{'M':>5} {'gl=0':>6} {'gl=1':>6}")
for M in BS:
    r = []
    for gl in (0, 1):
        try:
            r.append(get_padded_m(M, N, K, gl))
        except Exception as e:
            r.append(f"ERR:{type(e).__name__}")
    print(f"{M:>5} {str(r[0]):>6} {str(r[1]):>6}")
PY
chmod 666 $O/paddedm_probe.txt
