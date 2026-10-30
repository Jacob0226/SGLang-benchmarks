"""Prepare one SPEED-Bench qualitative category with NeMo Skills' prepare.py.

prepare.py resolves every category's external data, and some sources (cais/hle)
are gated. Filtering to the category first only touches the sources it needs.
Output matches prepare.py's qualitative.jsonl rows ("messages"), restricted to
the category.
"""

import argparse
import importlib.util
from pathlib import Path

from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--prepare", type=Path, required=True)
parser.add_argument("--category", default="coding")
parser.add_argument("--output_dir", type=Path, required=True)
args = parser.parse_args()

spec = importlib.util.spec_from_file_location("speed_prepare", args.prepare)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

config = "qualitative"
dataset = load_dataset("nvidia/SPEED-Bench", config, split="test")
dataset = dataset.filter(lambda row: row["category"] == args.category)
print(f"{args.category}: {len(dataset)} rows")
dataset = prep._resolve_external_data(dataset, config)
dataset = dataset.map(
    lambda example: {"messages": [{"role": "user", "content": turn} for turn in example["turns"]]},
    remove_columns=["turns"],
)
dataset.to_json(args.output_dir / f"{config}.jsonl")
