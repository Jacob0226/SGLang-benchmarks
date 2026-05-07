import re
import csv
import argparse
from datetime import date as _date
from pathlib import Path

START_MARK = "============ Serving Benchmark Result ============"
END_MARK = "=================================================="

kv_pattern = re.compile(r"^(.*?):\s+(.*)$")
# 支援檔名格式：bench_in1000_out1000_conc1.log / bench_in1000_out1000_conc_1.log
meta_pattern = re.compile(r"in(\d+)_out(\d+)_conc_?(\d+)")
# 從 input_dir 路徑抓 TP 大小，例如 ".../GLM-5-FP8-bench-0507_TP4"
tp_pattern = re.compile(r"[Tt][Pp](\d+)")
# 從路徑名稱嗅 precision / spec_method
precision_pattern = re.compile(r"\b(fp4|fp8|fp16|bf16)\b", re.IGNORECASE)
mtp_pattern = re.compile(r"(?:^|[_\W])mtp(?:[_\W]|$)", re.IGNORECASE)

# 第一份 table 的 derived 欄位 (沿用原本格式，附加在 CSV 尾巴)
INTERACTIVITY_COL = "Interactivity (tok/s/user)"
PER_GPU_COL = "Token Throughput per GPU (token/s/gpu)"
DERIVED_COLUMNS = [INTERACTIVITY_COL, PER_GPU_COL]

# 第二份 table 的欄位順序，刻意與
# Analysis/B200_GLM5_FP8_inferenceX/B200_GLM5_all_variants.csv 完全一致，
# 方便直接 concat / pivot。run_url 一律填 "Local run"。
LOCAL_TABLE_COLUMNS = [
    "variant",
    "precision",
    "spec_method",
    "framework",
    "decode_tp",
    "num_decode_gpu",
    "input_len",
    "output_len",
    "concurrency",
    "interactivity (tok/s/user, median_intvty)",
    "token_throughput_per_gpu (tok/s/gpu, tput_per_gpu)",
    "output_tput_per_gpu (tok/s/gpu)",
    "mean_intvty (tok/s/user)",
    "median_ttft (s)",
    "median_tpot (s)",
    "median_e2el (s)",
    "date",
    "image",
    "run_url",
]

LOCAL_RUN_URL = "Local run"


def convert_value(v: str):
    v = v.strip()
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def parse_log(log_path: Path):
    records = []
    current = None
    in_block = False
    column_order = []

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if START_MARK in line:
                    current = {}
                    in_block = True
                    continue
                if END_MARK in line and in_block:
                    current["source_file"] = log_path.name
                    input_len, output_len, concurrency = get_bench_meta(log_path)
                    current["input_len"] = input_len
                    current["output_len"] = output_len
                    current["concurrency"] = concurrency
                    records.append(current)
                    current = None
                    in_block = False
                    continue
                if not in_block or not line:
                    continue
                m = kv_pattern.match(line)
                if m:
                    key = m.group(1).strip()
                    value = convert_value(m.group(2))
                    if key not in column_order:
                        column_order.append(key)
                    current[key] = value
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    return records, column_order


def get_bench_meta(path: Path):
    """從檔名提取 input_len/output_len/concurrency，找不到則回傳 0。"""
    match = meta_pattern.search(path.name)
    if not match:
        return 0, 0, 0
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def detect_tp_from_path(input_dir: Path):
    """從 input_dir 名稱猜 TP 大小 (例如 'GLM-5-FP8-bench-0507_TP4' -> 4)。
    找不到則回傳 None，讓呼叫端套用預設或 CLI 指定的值。"""
    for part in (input_dir.name, *(p.name for p in input_dir.parents)):
        m = tp_pattern.search(part)
        if m:
            return int(m.group(1))
    return None


def detect_precision_from_path(input_dir: Path):
    """從路徑名稱抓 precision (fp4/fp8/...)，找不到回傳空字串。"""
    for part in (input_dir.name, *(p.name for p in input_dir.parents)):
        m = precision_pattern.search(part)
        if m:
            return m.group(1).lower()
    return ""


def detect_spec_method_from_path(input_dir: Path):
    """路徑含獨立 'mtp' token 視為 MTP，否則 'none'。"""
    for part in (input_dir.name, *(p.name for p in input_dir.parents)):
        if mtp_pattern.search(part):
            return "mtp"
    return "none"


def _safe_float(value):
    """轉 float; 任何 TypeError/ValueError 都回 None。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_derived_metrics(record: dict, tp: int):
    """加 2 個 derived 欄位 (給第一份 table 用):
       Interactivity (tok/s/user)            = 1000 / Median TPOT (ms)
       Token Throughput per GPU (token/s/gpu) = Output token throughput (tok/s) / TP
    任何來源欄缺值或無法轉成數字時，欄位填空字串。"""
    tpot = record.get("Median TPOT (ms)")
    try:
        record[INTERACTIVITY_COL] = round(1000.0 / float(tpot), 4) if tpot else ""
    except (TypeError, ValueError):
        record[INTERACTIVITY_COL] = ""

    out_tps = record.get("Output token throughput (tok/s)")
    try:
        if out_tps and tp:
            record[PER_GPU_COL] = round(float(out_tps) / int(tp), 4)
        else:
            record[PER_GPU_COL] = ""
    except (TypeError, ValueError, ZeroDivisionError):
        record[PER_GPU_COL] = ""


def build_local_row(
    record: dict,
    *,
    variant: str,
    precision: str,
    spec_method: str,
    framework: str,
    decode_tp: int,
    num_decode_gpu: int,
    run_date: str,
    image: str,
):
    """根據單一 benchmark record 組出 InferenceX 格式的 row。
    Throughput 欄位都 / num_decode_gpu，得到 tok/s/gpu。
    interactivity / mean_intvty = 1000 / TPOT(ms)。
    各 latency 欄位由 ms 轉 s。"""
    median_tpot_ms = _safe_float(record.get("Median TPOT (ms)"))
    mean_tpot_ms = _safe_float(record.get("Mean TPOT (ms)"))
    median_ttft_ms = _safe_float(record.get("Median TTFT (ms)"))
    median_e2el_ms = _safe_float(record.get("Median E2E Latency (ms)"))
    out_tps = _safe_float(record.get("Output token throughput (tok/s)"))
    total_tps = _safe_float(record.get("Total token throughput (tok/s)"))

    median_intvty = (
        round(1000.0 / median_tpot_ms, 4) if median_tpot_ms else ""
    )
    mean_intvty = round(1000.0 / mean_tpot_ms, 4) if mean_tpot_ms else ""

    tput_per_gpu = (
        round(total_tps / num_decode_gpu, 4)
        if (total_tps is not None and num_decode_gpu)
        else ""
    )
    output_tput_per_gpu = (
        round(out_tps / num_decode_gpu, 4)
        if (out_tps is not None and num_decode_gpu)
        else ""
    )

    median_ttft_s = (
        round(median_ttft_ms / 1000.0, 4) if median_ttft_ms is not None else ""
    )
    median_tpot_s = (
        round(median_tpot_ms / 1000.0, 6) if median_tpot_ms is not None else ""
    )
    median_e2el_s = (
        round(median_e2el_ms / 1000.0, 4) if median_e2el_ms is not None else ""
    )

    return {
        "variant": variant,
        "precision": precision,
        "spec_method": spec_method,
        "framework": framework,
        "decode_tp": decode_tp,
        "num_decode_gpu": num_decode_gpu,
        "input_len": record.get("input_len", ""),
        "output_len": record.get("output_len", ""),
        "concurrency": record.get("concurrency", ""),
        "interactivity (tok/s/user, median_intvty)": median_intvty,
        "token_throughput_per_gpu (tok/s/gpu, tput_per_gpu)": tput_per_gpu,
        "output_tput_per_gpu (tok/s/gpu)": output_tput_per_gpu,
        "mean_intvty (tok/s/user)": mean_intvty,
        "median_ttft (s)": median_ttft_s,
        "median_tpot (s)": median_tpot_s,
        "median_e2el (s)": median_e2el_s,
        "date": run_date,
        "image": image,
        "run_url": LOCAL_RUN_URL,
    }


def write_csv(records, column_order, output_csv, *, local_table_kwargs):
    """寫出兩份 table 到同一份 CSV：
    1) 完整原始欄位 (front + log columns + DERIVED_COLUMNS)
    2) 一行空白後，附上 InferenceX 格式的 local-run table
    """
    if not records:
        print("No records found to write.")
        return

    front_columns = ["input_len", "output_len", "concurrency", "source_file"]
    middle_columns = [
        c for c in column_order
        if c not in front_columns and c not in DERIVED_COLUMNS
    ]
    ordered_columns = front_columns + middle_columns + DERIVED_COLUMNS

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        # Table 1 — 原始完整 metrics
        writer = csv.DictWriter(f, fieldnames=ordered_columns, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)

        # 一行空白當分隔，Excel/pandas 都能辨識成兩個區塊
        f.write("\n")

        # Table 2 — InferenceX 格式 (run_url = "Local run")
        local_writer = csv.DictWriter(f, fieldnames=LOCAL_TABLE_COLUMNS)
        local_writer.writeheader()
        for r in records:
            local_writer.writerow(build_local_row(r, **local_table_kwargs))


def main():
    parser = argparse.ArgumentParser(description="Parse benchmark logs sorted by concurrency number")
    parser.add_argument("--input_dir", "-i", required=True, help="Input directory")
    parser.add_argument("--output", "-o", required=True, help="Output csv file")
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Tensor parallel size for per-GPU throughput. "
             "If omitted, auto-detect from input_dir name (e.g. '..._TP4'); fallback=8.",
    )

    # 第二份 table 的 metadata；都有 sane 的 auto-detect / 預設值
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="第二份 table 的 variant 名 (e.g. 'MI355X_GLM5.1_FP8_TP8')。"
             "預設 = input_dir 的目錄名。",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["fp4", "fp8", "fp16", "bf16"],
        help="精度。預設從 input_dir 路徑自動偵測 (fp4/fp8/fp16/bf16)。",
    )
    parser.add_argument(
        "--spec-method",
        dest="spec_method",
        type=str,
        default=None,
        choices=["none", "mtp"],
        help="Speculative decoding method。預設：路徑含 'mtp' -> 'mtp'，否則 'none'。",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="sglang",
        help="框架名稱。預設 'sglang'。",
    )
    parser.add_argument(
        "--num-decode-gpu",
        dest="num_decode_gpu",
        type=int,
        default=None,
        help="Decode GPU 數量。預設 = --tp。",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Docker image tag。預設空字串。",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Run date (YYYY-MM-DD)。預設今天。",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)

    # 解析 TP 大小: CLI > 路徑 auto-detect > 預設 8
    if args.tp is not None:
        tp_size = args.tp
        tp_source = "cli"
    else:
        detected = detect_tp_from_path(input_path)
        if detected is not None:
            tp_size = detected
            tp_source = "auto-detected from path"
        else:
            tp_size = 8
            tp_source = "default fallback"
    print(f"TP size: {tp_size} ({tp_source})")

    # 解析第二份 table 的 metadata
    variant = args.variant or input_path.name
    precision = args.precision or detect_precision_from_path(input_path)
    spec_method = args.spec_method or detect_spec_method_from_path(input_path)
    framework = args.framework
    num_decode_gpu = args.num_decode_gpu if args.num_decode_gpu is not None else tp_size
    image = args.image
    run_date = args.date or _date.today().isoformat()

    print(f"Local-table variant      : {variant}")
    print(f"Local-table precision    : {precision or '(unset)'}")
    print(f"Local-table spec_method  : {spec_method}")
    print(f"Local-table framework    : {framework}")
    print(f"Local-table decode_tp    : {tp_size}")
    print(f"Local-table num_decode_gpu: {num_decode_gpu}")
    print(f"Local-table image        : {image or '(empty)'}")
    print(f"Local-table date         : {run_date}")

    local_table_kwargs = dict(
        variant=variant,
        precision=precision,
        spec_method=spec_method,
        framework=framework,
        decode_tp=tp_size,
        num_decode_gpu=num_decode_gpu,
        run_date=run_date,
        image=image,
    )

    # 1. 取得檔案並過濾
    log_files = [
        f for f in input_path.glob("*.log")
        if "warmup" not in f.name
        and "server" not in f.name
        and "Accuracy" not in f.name
        and "Finish" not in f.name
    ]

    # 2. 依照 input_len -> output_len -> concurrency 進行數值排序
    log_files.sort(key=get_bench_meta)

    all_records = []
    master_column_order = ["source_file"]

    # 3. 依序讀取
    for log_file in log_files:
        print(f"Processing ({get_bench_meta(log_file)[2]}): {log_file.name}")
        records, column_order = parse_log(log_file)
        for r in records:
            compute_derived_metrics(r, tp_size)
        all_records.extend(records)
        for col in column_order:
            if col not in master_column_order:
                master_column_order.append(col)

    write_csv(
        all_records,
        master_column_order,
        args.output,
        local_table_kwargs=local_table_kwargs,
    )

    print("-" * 30)
    print(f"Total files: {len(log_files)}")
    print(f"Result saved to: {args.output}")


if __name__ == "__main__":
    main()

'''

python ~/0129/parse_perf_metrics_to_csv.py \
    --input_dir ~/0129/logs_20260129_025052_DeepSeek-R1 \
    --output ~/0129/logs_20260129_025052_DeepSeek-R1/all_benchmarks.csv

# 第二份 table 全部用 auto-detect (variant=dir 名, precision/MTP 從路徑抓):
python ~/SGLang-benchmarks/parse_perf_metrics_to_csv.py \
    --input_dir ~/SGLang-benchmarks/run_logs/GLM-5-FP8-bench-0507_TP4 \
    --output ~/SGLang-benchmarks/run_logs/all_benchmarks.csv

# 完全自訂 metadata:
python ~/SGLang-benchmarks/parse_perf_metrics_to_csv.py \
    --input_dir ~/SGLang-benchmarks/run_logs/foo \
    --output ~/SGLang-benchmarks/run_logs/foo.csv \
    --tp 8 --num-decode-gpu 8 \
    --variant MI355X_GLM5.1_FP8_StepAB_TP8 \
    --precision fp8 --spec-method none --framework sglang \
    --image rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503 \
    --date 2026-05-07

'''
