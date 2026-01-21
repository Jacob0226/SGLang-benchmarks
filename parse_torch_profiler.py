import gzip
import json
import csv
import argparse
from collections import defaultdict

def analyze_trace_file(file_path, output_csv):
    kernel_stats = defaultdict(lambda: {"count": 0, "total_dur": 0.0})

    # Step 1: Load the trace file
    print(f"Loading {file_path}")
    with gzip.open(file_path, "rt") as f:
        trace = json.load(f)
        for event in trace["traceEvents"]:
            if event.get("cat") == "kernel" and "dur" in event and "name" in event:
                name = event["name"]
                dur = event["dur"]  # Duration in microseconds
                kernel_stats[name]["count"] += 1
                kernel_stats[name]["total_dur"] += dur

    total_duration_all = sum(stat["total_dur"] for stat in kernel_stats.values())

    # Step 2: Write CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "TotalCalls", "TotalDuration_us", "Ave_us", "Percentage"])

        for name, stat in sorted(kernel_stats.items(), key=lambda x: x[1]["total_dur"], reverse=True):
            count = stat["count"]
            total_dur = stat["total_dur"]
            ave = total_dur / count if count else 0
            percentage = (total_dur / total_duration_all * 100) if total_duration_all else 0
            writer.writerow([name, count, int(total_dur), round(ave, 3), round(percentage, 2)])

    print(f"Done. Output written to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze PyTorch profiler trace JSON (gzipped) for kernel timing.")
    parser.add_argument("--file", help="Path to .pt.trace.json.gz file")
    parser.add_argument("--out", help="Path to the generated csv file")
    
    args = parser.parse_args()
    if args.out == None:
        output_csv = args.file.split(".pt.trace.json.gz")[0]+'.csv'
    else:
        output_csv = args.out
    analyze_trace_file(args.file, output_csv)

'''
python $HOME/script/ray_engine/parse_torch_profiler.py \
    -file $HOME/script/ray_engine/prof/0617/meta-llama/Llama-4-Scout-17B-16E-Instruct_TP8/\
TCH_p8_i240000_o128_concur4_v1/\
atl1g2r4u9gpu.atl.do.cpe.ice.amd.com_105608.1750134599558712368.pt.trace.json.gz \
    -o  $HOME/script/ray_engine/prof/0617/meta-llama/Llama-4-Scout-17B-16E-Instruct_TP8/\
TCH_p8_i240000_o128_concur4_v1/\
TCH_p8_i240000_o128_concur4_v1.csv

python $HOME/script/ray_engine/parse_torch_profiler.py \
    -file $HOME/script/ray_engine/prof/0617/meta-llama/Llama-4-Scout-17B-16E-Instruct_TP8/\
TCH_p8_i240000_o1_concur4_v1/\
atl1g2r4u9gpu.atl.do.cpe.ice.amd.com_23054.1750237017760800312.pt.trace.json.gz \
    -o  $HOME/script/ray_engine/prof/0617/meta-llama/Llama-4-Scout-17B-16E-Instruct_TP8/\
TCH_p8_i240000_o1_concur4_v1/\
TCH_p8_i240000_o1_concur4_v1.csv

'''
