#!/usr/bin/env python3
import argparse
import subprocess
import datetime
import os
import sys

# Supported model configurations
SUPPORTED_MODELS = ["GROK1-INT4-KV_AUTO", "GROK1-INT4-KV_FP8", "GROK1-FP8", "GROK2",
                    "GROK2.8T", "LLAMA3.1-70B", "LLAMA3.1-8B"]

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Launch SGLang server with specific model and attention backend."
    )
    parser.add_argument(
        "--model",
        default="GROK1",
        choices=SUPPORTED_MODELS,
        help=f"Model to use. Supported: {', '.join(SUPPORTED_MODELS)} (default: GROK1)"
    )
    parser.add_argument(
        "--attn-backend",
        default="aiter",
        help="Attention backend to use (default: aiter)"
    )
    parser.add_argument(
        "--profiling",
        default="off",
        choices=["off", "profile"],
        help="Enable profiling if set to 'profile' (default: off)"
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print CMD and environment variables without executing"
    )

    args = parser.parse_args()

    # Generate timestamp and logfile name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = f"sglang_server_log_{timestamp}.json"

    model = args.model
    attn_backend = args.attn_backend
    profiling = args.profiling
    enable_profiling = profiling == "profile"

    # Default environment and parameters
    env = os.environ.copy()
    extra_args = ""
    tokenizer_path = None
    tp = 8
    quant = "fp8"

    # Set environment and paths based on model
    if model == "GROK1-INT4-KV_AUTO":
        env["RCCL_MSCCL_ENABLE"] = "0"
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_INT4_WEIGHT"] = "1"
        model_path = "/data/grok-1-W4A8KV8/"
        tokenizer_path = "/data/Xenova/grok-1-tokenizer"
        tp = 8
        quant = "fp8"
        extra_args = "--mem-fraction-static 0.5 --attention-backend aiter"
    
    elif model == "GROK1-INT4-KV_FP8":
        env["RCCL_MSCCL_ENABLE"] = "0"
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_INT4_WEIGHT"] = "1"
        model_path = "/data/grok-1-W4A8KV8/"
        tokenizer_path = "/data/Xenova/grok-1-tokenizer"
        tp = 8
        quant = "fp8"
        extra_args = "--mem-fraction-static 0.5 --attention-backend aiter --kv-cache-dtype fp8_e4m3"

    elif model == "GROK1-FP8":
        env["RCCL_MSCCL_ENABLE"] = "0"
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_INT4_WEIGHT"] = "0"
        model_path = "/data/lmzheng-grok-1/"
        tokenizer_path = "/data/Xenova/grok-1-tokenizer"
        tp = 8
        quant = "fp8"
        extra_args = "--mem-fraction-static 0.5"

    elif model == "GROK2":
        env["RCCL_MSCCL_ENABLE"] = "0"
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_INT4_WEIGHT"] = "0"
        env["SGLANG_ROCM_DISABLE_LINEARQUANT"] = "1"
        model_path = "/data/huggingface/hub/xai-org/grok-2"
        tokenizer_path = "/data/huggingface/hub/xai-org/grok-2/tokenizer.tok.json"
        tp = 8
        quant = "mxfp4" #"fp8"
        extra_args = "--page-size 16"

    elif model == "GROK2.8T":
        env["RCCL_MSCCL_ENABLE"] = "0"
        env["SGLANG_AITER_MLA_PERSIST"] = "1"
        env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        env["SGLANG_INT4_WEIGHT"] = "0"
        env["SGLANG_MOE_PADDING"] = "1"
        env["SGLANG_ROCM_DISABLE_LINEARQUANT"] = "0"
        env["SGLANG_ROCM_FUSED_DECODE_MLA"] = "1"
        env["SGLANG_SET_CPU_AFFINITY"] = "1"
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_USE_ROCM700A"] = "1"
        model_path =     "/data/huggingface/hub/xai-org/dummy_grok_2t/"
        tokenizer_path = "/data/huggingface/hub/alvarobartt/grok-2-tokenizer/"
        tp = 8
        # quant = "fp8"
        # --disable-custom-all-reduce 
        extra_args = "--load-format dummy --mem-fraction-static 0.7"

    elif model == "LLAMA3.1-70B":
        env["RCCL_MSCCL_ENABLE"] = "1"
        model_path = "amd/Llama-3.1-70B-Instruct-FP8-KV"
        tp = 8
        quant = "fp8"
        extra_args = "--cuda-graph-max-bs 1024 --mem-fraction-static 0.6"

    elif model == "LLAMA3.1-8B":
        env["RCCL_MSCCL_ENABLE"] = "1"
        model_path = "amd/Llama-3.1-8B-Instruct-FP8-KV"
        tp = 1
        quant = "fp8"
        extra_args = "--cuda-graph-max-bs 1024 --mem-fraction-static 0.6"

    # Add profiling argument if requested
    if enable_profiling:
        extra_args += " --enable-profiling"

    # Construct the command
    cmd = [
        "python3",
        "-m", "sglang.launch_server",
        "--model", model_path,
        "--tp", str(tp),
        "--quantization", quant,
        "--trust-remote-code",
        "--attention-backend", attn_backend,
    ]

    if tokenizer_path:
        cmd.extend(["--tokenizer-path", tokenizer_path])

    if extra_args:
        cmd.extend(extra_args.split())

    # Print environment and command
    print("Environment Variables:")
    for k in sorted(env.keys()):
        if k.startswith("SGLANG_") or k.startswith("RCCL_"):
            print(f"{k}={env[k]}")

    print("\nCommand:")
    print(" ".join(cmd))
    print(f"\nLog File: {logfile}")

    # If only print requested, stop here
    if args.print_only:
        print("\n--print-only mode: command not executed.")
        return

    # Execute the command
    with open(logfile, "w") as log_file:
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        process.wait()


if __name__ == "__main__":
    main()

'''
Usage:
# KV cache FP8:
python ~/SGLang-benchmarks/SERVER.py --model GROK1-INT4-KV_FP8
python ~/SGLang-benchmarks/CLIENT.py --model GROK1-INT4-KV_FP8

# KV cache Auto:
python ~/SGLang-benchmarks/SERVER.py --model GROK1-INT4-KV_AUTO
python ~/SGLang-benchmarks/CLIENT.py --model GROK1-INT4-KV_AUTO
'''