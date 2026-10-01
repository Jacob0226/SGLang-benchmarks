#!/usr/bin/env bash
# One compact status line per interval for whichever AgentX point is live.
# Finds the run by most-recently-touched recipe.log rather than taking a path,
# so it keeps following a chained sweep across points without being restarted.
set -u
RESULTS="${RESULTS:-/mnt/home/jacchang/SGLang-benchmarks/results/nvidia_GLM-5.3-Flash-NVFP4/lmsysorg_sglang-v0.5.20-cu130}"
INTERVAL="${INTERVAL:-60}"

while true; do
    log=$(ls -t "$RESULTS"/bench-Agentic-*/*/recipe.log 2>/dev/null | head -1)
    if [ -z "$log" ]; then
        echo "PROGRESS $(date -u +%H:%M) 找不到任何 recipe.log"
        sleep "$INTERVAL"; continue
    fi
    dir=$(dirname "$log")
    run=$(basename "$(dirname "$dir")" | sed 's/^bench-Agentic-//')
    point=$(basename "$dir" | grep -oE 'conc[0-9]+')

    # Phase: profiling beats warmup beats server boot.
    target=$(grep -oE 'Phase profiling \(profiling\) started.*target: [0-9.]+s' "$log" 2>/dev/null | grep -oE '[0-9.]+s$' | tail -1)
    if [ -n "$target" ]; then
        t0=$(grep -E 'Phase profiling \(profiling\) started' "$log" | tail -1 | grep -oE '^[0-9]{2}:[0-9]{2}:[0-9]{2}')
        start_s=$(date -u -d "$t0" +%s 2>/dev/null)
        now_s=$(date -u +%s)
        # The log stamps are UTC wall clock; a run crossing midnight would wrap.
        [ "$now_s" -lt "$start_s" ] && start_s=$((start_s - 86400))
        elapsed=$((now_s - start_s))
        total=${target%s}; total=${total%.*}
        pct=$((elapsed * 100 / (total > 0 ? total : 1)))
        eta=$(date -u -d "@$((start_s + total))" +%H:%M)
        phase="profiling ${elapsed}/${total}s (${pct}%) ETA ${eta}"
    elif grep -q 'Phase warmup' "$log" 2>/dev/null; then
        phase="warmup"
    else
        phase="server 啟動中"
    fi

    srv=$(grep -oE 'prefix_cache_hit=[0-9.]+%? .*kv_usage=[0-9.]+%' "$log" 2>/dev/null | tail -1)
    hit=$(echo "$srv" | grep -oE 'prefix_cache_hit=[0-9.]+' | cut -d= -f2)
    kv=$(echo "$srv" | grep -oE 'kv_usage=[0-9.]+' | cut -d= -f2)
    # grep -c already prints 0 and exits 1 on no match; a || echo 0 would
    # append a second line and break the read below.
    oom=$(grep -c 'memory allocation failed with OOM' "$dir/server.log" 2>/dev/null)
    crash=$(grep -c 'Scheduler hit an exception' "$dir/server.log" 2>/dev/null)

    # AgentX replays have idle gaps between turns, so a single sample lands on
    # 0% often enough to look alarming. Average three.
    read -r util pw < <(for _ in 1 2 3; do
            nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits -i 0,1,2,3 2>/dev/null
            sleep 1
        done | awk -F', ' '{u+=$1; p+=$2; n++} END {printf "%d %d", u/n, p/n}')

    echo "PROGRESS $(date -u +%H:%M) ${run} ${point} | ${phase} | hit=${hit:-?}% kv=${kv:-?}% | GPU ${util:-?}% ${pw:-?}W | OOM警告 ${oom} 崩潰 ${crash}"
    sleep "$INTERVAL"
done
