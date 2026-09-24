#!/usr/bin/env python3
"""Rewrite infmax_model_prefix in AgentX result JSONs.

The first GLM-5.3-Flash points were recorded as MODEL_PREFIX=glm5.3, which is
GLM-5.3 proper; upstream gives Flash variants their own prefix (dsv4 vs
dsv41flash). Nothing measured depends on the field -- both values glob to
glm5.3* and so selected the same unfiltered 1M-context corpus -- so this is a
metadata backfill, not a re-run.

    ./backfill_model_prefix.py --from glm5.3 --to glm5.3flash <result-dir>...
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--from", dest="src", default="glm5.3")
    parser.add_argument("--to", dest="dst", default="glm5.3flash")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    changed = skipped = 0
    for root in args.dirs:
        for dirpath, _, names in os.walk(root):
            if "aiperf_artifacts" in dirpath:
                continue
            for name in names:
                if not name.endswith(".json"):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    with open(path) as fh:
                        blob = json.load(fh)
                except (json.JSONDecodeError, OSError):
                    continue
                if blob.get("infmax_model_prefix") != args.src:
                    continue
                if args.dry_run:
                    print(f"would patch {path}")
                    skipped += 1
                    continue
                blob["infmax_model_prefix"] = args.dst
                # Same-directory temp plus rename: never leave a half-written
                # result behind if this dies mid-write.
                tmp = path + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(blob, fh, indent=2)
                os.replace(tmp, path)
                print(f"patched {path}")
                changed += 1
    print(f"{changed} patched, {skipped} pending")


if __name__ == "__main__":
    main()
