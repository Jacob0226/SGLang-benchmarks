#!/bin/bash
# Log every GEMM-A16W16 config lookup the server performs, to find out what
# (M, N, K) the router GEMM is actually asked for at each decode batch size.
#
# conc4 (bs=4) picks up the installed GEMM-A16W16-N=288-K=4096.json but
# conc64 (bs=64) still runs DEFAULT's M_LEQ_64 tile, in the same process. Since
# _get_config always forwards N and K, the M=64 call must be asking for a
# different shape. This names it instead of guessing.
set -uo pipefail
exec 200>/tmp/glm53_bench.lock
flock -n 200 || { echo "another GPU run holds the lock; refusing"; exit 0; }

OUT=/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm_0928
mkdir -p "$OUT/cfglog" && chmod 777 "$OUT/cfglog"
rm -f "$OUT"/cfglog/*.log
LOG=/home/jacchang/SGLang-benchmarks/tmp/logs/config_lookup_probe.log
exec > "$LOG" 2>&1

F=/sgl-workspace/aiter/aiter/ops/triton/utils/gemm_config_utils.py
[ -f "$F.orig" ] || cp "$F" "$F.orig"

python - <<'PY'
import re
F = "/sgl-workspace/aiter/aiter/ops/triton/utils/gemm_config_utils.py"
src = open(F + ".orig").read()

probe = '''

# --- temporary lookup probe (glm53_trace_config_lookups.sh) ---
import os as _os
import threading as _th

_probe_seen = set()
_probe_lock = _th.Lock()
_probe_path = (
    "/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm_0928/"
    f"cfglog/{_os.getpid()}.log"
)


def _probe(config_name, M, N, K, cfg, is_tuned):
    key = (config_name, M, N, K)
    with _probe_lock:
        if key in _probe_seen:
            return
        _probe_seen.add(key)
        try:
            with open(_probe_path, "a") as f:
                f.write(
                    f"{config_name}\\tM={M}\\tN={N}\\tK={K}\\ttuned={is_tuned}\\t"
                    f"BM={cfg.get('BLOCK_SIZE_M')}\\tBN={cfg.get('BLOCK_SIZE_N')}\\t"
                    f"BK={cfg.get('BLOCK_SIZE_K')}\\tKS={cfg.get('NUM_KSPLIT')}\\t"
                    f"warps={cfg.get('num_warps')}\\twpe={cfg.get('waves_per_eu')}\\t"
                    f"cm={cfg.get('cache_modifier')}\\n"
                )
        except Exception:
            pass
# --- end probe ---
'''

# Wrap the public get_gemm_config so every call is recorded once per shape.
old = """    config, is_tuned = _get_gemm_config_cached(
        config_name, M, N, K, bounds, specialized_filename, backend, B
    )
    return copy.deepcopy(config), is_tuned"""
new = """    config, is_tuned = _get_gemm_config_cached(
        config_name, M, N, K, bounds, specialized_filename, backend, B
    )
    out = copy.deepcopy(config)
    try:
        _probe(config_name, M, N, K, out, is_tuned)
    except Exception:
        pass
    return out, is_tuned"""
assert old in src, "anchor not found"
src = src.replace(old, new) + probe
open(F, "w").write(src)
print("patched", F)
PY

export PYTHONUNBUFFERED=1 SGLANG_USE_AITER=1
RUN_CACHE=/home/jacchang/SGLang-benchmarks/tmp/cache-glm53-0928-rocm10
export AITER_JIT_DIR="$RUN_CACHE/aiter" FLYDSL_RUNTIME_CACHE_DIR="$RUN_CACHE/flydsl"
export TILELANG_CACHE_DIR="$RUN_CACHE/tilelang" TRITON_CACHE_DIR="$RUN_CACHE/triton"
export TORCH_EXTENSIONS_DIR="$RUN_CACHE/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$RUN_CACHE/torchinductor"
export XDG_CACHE_HOME="$RUN_CACHE/xdg" SGLANG_JIT_CACHE_DIR="$RUN_CACHE/sglang_jit"

cd /sgl-workspace
python3 -m sglang.launch_server \
    --model /data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4 --tp 4 \
    --host localhost --port 8234 --trust-remote-code \
    --tool-call-parser glm47 --reasoning-parser glm45 --watchdog-timeout 7200 \
    --mem-fraction-static 0.85 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
    --disable-radix-cache --kv-cache-dtype bfloat16 \
    --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
    --moe-runner-backend aiter --context-length 131072 --tokenizer-worker-num 8 &
SRV=$!

for i in $(seq 1 120); do
    code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8234/health || true)
    [ "$code" = "200" ] && { echo "server ready after ${i}0s"; break; }
    sleep 10
done

kill -TERM $SRV 2>/dev/null
pkill -TERM -f 'sglang::' 2>/dev/null
sleep 10
pkill -KILL -f 'sglang.launch_server' 2>/dev/null

cp "$F.orig" "$F"
echo "restored $F"
chmod -R a+rw "$OUT/cfglog" 2>/dev/null
echo "PROBE_DONE"
