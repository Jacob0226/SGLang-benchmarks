import re
import csv
import argparse
from pathlib import Path

START_MARK = "============ Serving Benchmark Result ============"
END_MARK = "=================================================="

kv_pattern = re.compile(r"^(.*?):\s+(.*)$")
# 支援檔名格式：bench_in1000_out1000_conc1.log / bench_in1000_out1000_conc_1.log
meta_pattern = re.compile(r"in(\d+)_out(\d+)_conc_?(\d+)")
# 從 input_dir 路徑抓 TP 大小，例如 ".../GLM-5-FP8-bench-0507_TP4"
tp_pattern = re.compile(r"[Tt][Pp](\d+)")

# Derived columns (appended at end of CSV)
INTERACTIVITY_COL = "Interactivity (tok/s/user)"
PER_GPU_COL = "Token Throughput per GPU (token/s/gpu)"
DERIVED_COLUMNS = [INTERACTIVITY_COL, PER_GPU_COL]

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
                    # 將來源檔名也存進去，方便追蹤
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

def compute_derived_metrics(record: dict, tp: int):
    """加 2 個 derived 欄位:
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

def write_csv(records, column_order, output_csv):
    if not records:
        print("No records found to write.")
        return
    
    # 固定把 metadata 欄位放在 CSV 最前面，把 derived 欄位放在最後面
    front_columns = ["input_len", "output_len", "concurrency", "source_file"]
    middle_columns = [
        c for c in column_order
        if c not in front_columns and c not in DERIVED_COLUMNS
    ]
    ordered_columns = front_columns + middle_columns + DERIVED_COLUMNS

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_columns, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)

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
            tp_source = f"auto-detected from path"
        else:
            tp_size = 8
            tp_source = "default fallback"
    print(f"TP size: {tp_size} ({tp_source})")

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

    write_csv(all_records, master_column_order, args.output)

    print("-" * 30)
    print(f"Total files: {len(log_files)}")
    print(f"Result saved to: {args.output}")

if __name__ == "__main__":
    main()

'''

python ~/0129/parse_perf_metrics_to_csv.py \
    --input_dir ~/0129/logs_20260129_025052_DeepSeek-R1 \
    --output ~/0129/logs_20260129_025052_DeepSeek-R1/all_benchmarks.csv

'''
