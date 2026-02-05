import re
import csv
import argparse
from pathlib import Path

START_MARK = "============ Serving Benchmark Result ============"
END_MARK = "=================================================="

kv_pattern = re.compile(r"^(.*?):\s+(.*)$")
# 用來提取檔名中 conc_ 後面數字的正則
conc_pattern = re.compile(r"conc_(\+?\d+)")

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

def get_conc_number(path: Path):
    """從檔名提取 conc_X 的數字，找不到則回傳 0"""
    match = conc_pattern.search(path.name)
    return int(match.group(1)) if match else 0

def write_csv(records, column_order, output_csv):
    if not records:
        print("No records found to write.")
        return
    
    # 確保 source_file 出現在 CSV 的第一欄
    if "source_file" not in column_order:
        column_order.insert(0, "source_file")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=column_order)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

def main():
    parser = argparse.ArgumentParser(description="Parse benchmark logs sorted by concurrency number")
    parser.add_argument("--input_dir", "-i", required=True, help="Input directory")
    parser.add_argument("--output", "-o", required=True, help="Output csv file")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    
    # 1. 取得檔案並過濾
    log_files = [
        f for f in input_path.glob("*.log") 
        if "warmup" not in f.name 
        and "server" not in f.name 
        and "Accuracy" not in f.name
    ]

    # 2. 依照 conc_XXX 的數字進行「數值排序」
    log_files.sort(key=get_conc_number)

    all_records = []
    master_column_order = ["source_file"]

    # 3. 依序讀取
    for log_file in log_files:
        print(f"Processing ({get_conc_number(log_file)}): {log_file.name}")
        records, column_order = parse_log(log_file)
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