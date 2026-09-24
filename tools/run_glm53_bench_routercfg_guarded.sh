#!/usr/bin/env bash
# flock guard: nrun.sh retries a hung srun client, but the docker exec it
# already fired keeps running, so an unguarded launch starts the benchmark
# twice and the two bench_serving clients double the real concurrency.
exec 200>/tmp/glm53_bench.lock
flock -n 200 || { echo "another benchmark run holds the lock; refusing"; exit 0; }
exec bash /home/jacchang/SGLang-benchmarks/tools/run_glm53_bench_routercfg.sh
