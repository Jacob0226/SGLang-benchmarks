#!/usr/bin/env bash
set -euo pipefail

# GLM-style benchmark matrix for DeepSeek-R1 on ROCm AITER.
# Matrix:
# 1) page_size=1  + SGLANG_AITER_FP8_PREFILL_ATTN=0
# 2) page_size=1  + PR25556 path (patched checkout + FP8 prefill enabled)
# 3) page_size=64 + SGLANG_AITER_FP8_PREFILL_ATTN=0
# 4) page_size=64 + PR25556 path (patched checkout + FP8 prefill enabled)

IMAGE="${IMAGE:-lmsysorg/sglang-rocm:v0.5.12-rocm720-mi35x-20260517}"
MODEL_PATH="${MODEL_PATH:-/data/huggingface/hub/deepseek-ai/DeepSeek-R1-0528}"
PR_CHECKOUT="${PR_CHECKOUT:-/home/jacchang/PR/sglang-pr25556}"
TAG="${TAG:-0521_Verify}"
RESULT_ROOT="${RESULT_ROOT:-/home/jacchang/SGLang-benchmarks/results/pagesize_pr25556_matrix_${TAG}}"
DATA_ROOT="${DATA_ROOT:-/data}"
TP="${TP:-8}"
HOST_PORT="${HOST_PORT:-31000}"
PROMPT_MULTIPLIER="${PROMPT_MULTIPLIER:-5}"
CONCURRENCIES_STR="${CONCURRENCIES_STR:-4 16 64 256}"
read -r -a CONCURRENCIES <<< "${CONCURRENCIES_STR}"
CASES="${CASES:-page1_fp8off page1_pr25556 page64_fp8off page64_pr25556}"
GSM8K_ENABLED="${GSM8K_ENABLED:-1}"
GSM8K_NUM_QUESTIONS="${GSM8K_NUM_QUESTIONS:-1200}"
GSM8K_PARALLEL="${GSM8K_PARALLEL:-1200}"
IN_TOKENS=4096
OUT_TOKENS=1

mkdir -p "${RESULT_ROOT}"
echo "Results root: ${RESULT_ROOT}"

run_case() {
  local case_name="$1"
  local page_size="$2"
  local fp8_prefill="$3"   # 0/1
  local use_pr="$4"        # 0/1
  local case_dir="${RESULT_ROOT}/${case_name}"
  local port
  local py_path
  local code_dir

  mkdir -p "${case_dir}"
  port="${HOST_PORT}"
  py_path="/opt/tilelang"
  code_dir="/sgl-workspace/sglang"

  if [[ "${use_pr}" == "1" ]]; then
    code_dir="/sglang-checkout"
    py_path="/sglang-checkout/python:/opt/tilelang"
  fi

  echo ""
  echo "============================================================"
  echo "CASE: ${case_name}"
  echo "  page_size=${page_size}, SGLANG_AITER_FP8_PREFILL_ATTN=${fp8_prefill}, use_pr=${use_pr}"
  echo "============================================================"

  docker run --rm \
    --privileged \
    --network=host \
    --ipc=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size 32g \
    --ulimit nofile=65536:65536 \
    -v "/home/jacchang:/home/jacchang" \
    -v "/home/jacchang/PR/sglang-pr25556:/sglang-checkout:ro" \
    -v "${DATA_ROOT}:/data" \
    -w "${code_dir}" \
    -e PYTHONPATH="${py_path}" \
    -e SAFETENSORS_FAST_GPU=1 \
    -e SGLANG_USE_AITER=1 \
    -e ROCM_QUICK_REDUCE_QUANTIZATION=NONE \
    -e SGLANG_AITER_FP8_PREFILL_ATTN="${fp8_prefill}" \
    "${IMAGE}" \
    bash -lc "
      set -euo pipefail
      LOG_DIR='${case_dir}'
      SERVER_LOG=\"\${LOG_DIR}/server.log\"
      touch \"\${SERVER_LOG}\"

      cleanup() {
        pkill -f 'sglang.launch_server' 2>/dev/null || true
        sleep 5
      }
      trap cleanup EXIT

      echo 'Launching server...' | tee -a \"\${SERVER_LOG}\"
      python3 -u -m sglang.launch_server \
        --model-path '${MODEL_PATH}' \
        --tp '${TP}' \
        --host 0.0.0.0 \
        --port '${port}' \
        --mem-fraction-static 0.85 \
        --watchdog-timeout 1800 \
        --kv-cache-dtype fp8_e4m3 \
        --attention-backend aiter \
        --page-size '${page_size}' \
        --chunked-prefill-size 32768 \
        --max-prefill-tokens 32768 \
        > \"\${SERVER_LOG}\" 2>&1 &

      deadline=\$(( \$(date +%s) + 1200 ))
      while [[ \"\$(curl -s -o /dev/null -w '%{http_code}' 'http://127.0.0.1:${port}/health' || true)\" != '200' ]]; do
        if [[ \$(date +%s) -ge \${deadline} ]]; then
          echo 'Server startup timeout' | tee -a \"\${SERVER_LOG}\"
          exit 1
        fi
        sleep 5
      done
      echo 'Server ready' | tee -a \"\${SERVER_LOG}\"

      python3 -m sglang.bench_serving \
        --host 127.0.0.1 \
        --port '${port}' \
        --model '${MODEL_PATH}' \
        --dataset-name random \
        --random-input 1024 \
        --random-output 64 \
        --random-range-ratio 1.0 \
        --max-concurrency 4 \
        --num-prompt 8 \
        --output-file /dev/null \
        > \"\${LOG_DIR}/warmup.log\" 2>&1

      if [[ '${GSM8K_ENABLED}' == '1' ]]; then
        GSM8K_LOG=\"\${LOG_DIR}/Accuracy_GSM8K.log\"
        GSM8K_JSONL=\"\${LOG_DIR}/Accuracy_GSM8K.jsonl\"
        if [[ -s \"\${GSM8K_JSONL}\" ]]; then
          echo \"Skip GSM8K: found existing \${GSM8K_JSONL}\" | tee -a \"\${GSM8K_LOG}\"
        else
        GSM8K_SCRIPT=''
        for c in \"${code_dir}/benchmark/gsm8k/bench_sglang.py\" \
                 '/sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py'; do
          if [[ -f \"\${c}\" ]]; then
            GSM8K_SCRIPT=\"\${c}\"
            break
          fi
        done

        if [[ -n \"\${GSM8K_SCRIPT}\" ]]; then
          echo \"Running GSM8K: num_questions=${GSM8K_NUM_QUESTIONS}, parallel=${GSM8K_PARALLEL}\" | tee -a \"\${GSM8K_LOG}\"
          (
            cd \"\${LOG_DIR}\"
            python3 \"\${GSM8K_SCRIPT}\" \
              --host 127.0.0.1 \
              --port '${port}' \
              --num-questions '${GSM8K_NUM_QUESTIONS}' \
              --parallel '${GSM8K_PARALLEL}' \
              --result-file \"\${GSM8K_JSONL}\" \
              2>&1 | tee -a \"\${GSM8K_LOG}\"
          ) || true
        else
          echo 'WARNING: GSM8K script not found, skip accuracy check' | tee -a \"\${GSM8K_LOG}\"
        fi
        fi
      fi

      for c in ${CONCURRENCIES[*]}; do
        num_prompt=\$(( c * ${PROMPT_MULTIPLIER} ))
        out_json=\"\${LOG_DIR}/bench_in${IN_TOKENS}_out${OUT_TOKENS}_conc\${c}.jsonl\"
        out_log=\"\${LOG_DIR}/bench_in${IN_TOKENS}_out${OUT_TOKENS}_conc\${c}.log\"
        if [[ -s \"\${out_json}\" ]]; then
          echo \"Skip conc=\${c}: found existing \${out_json}\" | tee -a \"\${out_log}\"
          continue
        fi
        echo \"Running conc=\${c}, prompts=\${num_prompt}\" | tee -a \"\${out_log}\"
        python3 -m sglang.bench_serving \
          --host 127.0.0.1 \
          --port '${port}' \
          --model '${MODEL_PATH}' \
          --dataset-name random \
          --random-input '${IN_TOKENS}' \
          --random-output '${OUT_TOKENS}' \
          --random-range-ratio 1.0 \
          --max-concurrency \"\${c}\" \
          --num-prompt \"\${num_prompt}\" \
          --output-file \"\${out_json}\" \
          >> \"\${out_log}\" 2>&1
      done
    "

  if [[ "${GSM8K_ENABLED}" == "1" ]]; then
    local gsm8k_log="${case_dir}/Accuracy_GSM8K.log"
    local acc="NA"
    if [[ -f "${gsm8k_log}" ]]; then
      acc="$(python3 - <<'PY' "${gsm8k_log}"
import re, sys
p = sys.argv[1]
txt = open(p, "r", errors="ignore").read()
m = re.findall(r"Accuracy:\s*([0-9.]+)", txt)
print(m[-1] if m else "NA")
PY
)"
    fi
    local summary_csv="${RESULT_ROOT}/gsm8k_summary.csv"
    if [[ ! -f "${summary_csv}" ]]; then
      echo "case,method,page_size,gsm8k_accuracy" > "${summary_csv}"
    fi
    local method_name="PR25556"
    if [[ "${fp8_prefill}" == "0" ]]; then
      method_name="SGLANG_AITER_FP8_PREFILL_ATTN=0"
    fi
    echo "${case_name},${method_name},${page_size},${acc}" >> "${summary_csv}"
  fi
}

for case_name in ${CASES}; do
  case "${case_name}" in
    page1_fp8off) run_case "page1_fp8off" 1 0 0 ;;
    page1_pr25556) run_case "page1_pr25556" 1 1 1 ;;
    page64_fp8off) run_case "page64_fp8off" 64 0 0 ;;
    page64_pr25556) run_case "page64_pr25556" 64 1 1 ;;
    *) echo "Unknown case in CASES: ${case_name}" >&2; exit 1 ;;
  esac
done

echo ""
echo "All cases finished. Results in: ${RESULT_ROOT}"
