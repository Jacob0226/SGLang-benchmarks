#!/usr/bin/env python3
import os
import time
import argparse
import subprocess
from datetime import datetime

def main():
    # ---------- Parse arguments ----------
    parser = argparse.ArgumentParser(description="Run sglang benchmark for models.")
    parser.add_argument("--model", type=str, choices=["GROK1-INT4", "GROK1-FP8", "GROK2"], required=True,
                        help="Select which model to run: GROK1-INT4, GROK1-FP8 or GROK2.")
    args = parser.parse_args()
    model = args.model

    # ---------- Basic settings ----------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    request_rates = [1, 2, 4, 8]
    max_num_prompts_options = [2400]
    completed_file = "completed_combinations.log"

    # Ensure the completed log file exists
    if not os.path.exists(completed_file):
        open(completed_file, "w").close()

    # ---------- Main loop ----------
    for rate in request_rates:
        for max_num_prompts in max_num_prompts_options:
            for i in range(1, 4):  # Run each combination 3 times
                num_prompts = min(300 * rate, max_num_prompts)
                combination = f"{model}_{rate}_{num_prompts}_run{i}"

                # Check if this combination has been completed
                with open(completed_file, "r") as f:
                    if combination in f.read():
                        print(f"Skipping already completed combination: {combination}")
                        continue

                # Define the log file for this run
                logfile = f"sglang_client_log_{model.lower()}_{rate}_max{max_num_prompts}_run{i}_{timestamp}.log"
                print(f"Running benchmark with model={model}, rate={rate}, "
                      f"max={max_num_prompts}, num_prompts={num_prompts}, run={i}")
                
                # ---------- Determine the command based on model ----------
                if model == "GROK2":
                    cmd = (
                        f"python3 -m sglang.bench_serving "
                        f"--backend sglang --dataset-name random "
                        f"--random-input 8192 --random-output 1024 "
                        f"--num-prompts {num_prompts} "
                        f"--tokenizer /data2/grok-2/tokenizer.tok.json "
                        f"--request-rate {rate} "
                        f"--output-file online-GROK2.jsonl"
                    )
                elif model == "GROK1-FP8":
                    cmd = (
                        f"python3 -m sglang.bench_serving "
                        f"--backend sglang --dataset-name random "
                        f"--random-input 1024 --random-output 1024 "
                        f"--num-prompts {num_prompts} "
                        f"--tokenizer /data/Xenova/grok-1-tokenizer/ "
                        f"--request-rate {rate} "
                        f"--output-file online-GROK1-FP8.jsonl"
                    )

                else:  # GROK1
                    cmd = (
                        f"python3 -m sglang.bench_serving "
                        f"--backend sglang --dataset-name random "
                        f"--random-input 1024 --random-output 1024 "
                        f"--num-prompts {num_prompts} "
                        f"--tokenizer /data/Xenova/grok-1-tokenizer/ "
                        f"--request-rate {rate} "
                        f"--output-file online-GROK1.jsonl"
                    )

                # ---------- Execute the command ----------
                with open(logfile, "a") as log:
                    log.write(f"Executing: {cmd}\n")
                    log.flush()
                    process = subprocess.Popen(cmd, shell=True, stdout=log, stderr=log)
                    process.wait()

                # ---------- Record the completed combination ----------
                with open(completed_file, "a") as f:
                    f.write(f"{combination}\n")

                # ---------- Wait between runs ----------
                time.sleep(15)

if __name__ == "__main__":
    main()

