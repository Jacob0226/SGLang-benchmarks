#!/usr/bin/env bash
exec 200>/tmp/glm53_bench.lock
flock -n 200 || { echo "another run holds the lock; refusing"; exit 0; }
exec bash /home/jacchang/SGLang-benchmarks/tools/run_glm53_prof_routercfg.sh
