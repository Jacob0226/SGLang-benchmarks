#!/usr/bin/env bash
# One compact status line per live AgentX point, per interval.
#
# A node fits two TP4 sweeps, so this reports every run whose recipe.log is
# still being written rather than just the most recent one, and reads each
# run's GPU set out of its own sweep.log instead of assuming 0-3.
set -u
# Model directory, not an image directory. Pinning the image meant the
# monitor went blind the moment the container moved to a new SGLang tag:
# it reported "no live sweep" while one was running one directory over.
RESULTS="${RESULTS:-/mnt/home/jacchang/SGLang-benchmarks/results/nvidia_GLM-5.3-Flash-NVFP4}"
INTERVAL="${INTERVAL:-60}"
STALE_MIN="${STALE_MIN:-5}"   # a recipe.log untouched this long is a finished run

gpu_avg() {
    # AgentX replays idle between turns, so a single sample lands on 0% often
    # enough to look like a hang. Average three.
    local sel="$1"
    for _ in 1 2 3; do
        nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits ${sel:+-i "$sel"} 2>/dev/null
        sleep 1
    done | awk -F', ' '{u+=$1; p+=$2; n++} END {if(n) printf "%d %d", u/n, p/n; else printf "? ?"}'
}

report_one() {
    local log="$1" dir run point sweep gpus target t0 start_s now_s elapsed total pct eta phase
    dir=$(dirname "$log")
    sweep=$(dirname "$dir")
    run=$(basename "$sweep" | sed 's/^bench-Agentic-//')
    point=$(basename "$dir" | grep -oE 'conc[0-9]+')
    gpus=$(grep -oE 'gpus=[0-9,]+' "$sweep/sweep.log" 2>/dev/null | tail -1 | cut -d= -f2)

    target=$(grep -oE 'Phase profiling \(profiling\) started.*target: [0-9.]+s' "$log" 2>/dev/null | grep -oE '[0-9.]+s$' | tail -1)
    if [ -n "$target" ]; then
        t0=$(grep -E 'Phase profiling \(profiling\) started' "$log" | tail -1 | grep -oE '^[0-9]{2}:[0-9]{2}:[0-9]{2}')
        start_s=$(date -u -d "$t0" +%s 2>/dev/null); now_s=$(date -u +%s)
        [ "$now_s" -lt "$start_s" ] && start_s=$((start_s - 86400))
        elapsed=$((now_s - start_s)); total=${target%s}; total=${total%.*}
        pct=$((elapsed * 100 / (total > 0 ? total : 1)))
        eta=$(date -u -d "@$((start_s + total))" +%H:%M)
        phase="profiling ${elapsed}/${total}s (${pct}%) ETA ${eta}"
    elif grep -q 'Phase warmup' "$log" 2>/dev/null; then
        phase="warmup"
    else
        phase="server 啟動中"
    fi

    local srv hit kv oom crash util pw
    srv=$(grep -oE 'prefix_cache_hit=[0-9.]+%? .*kv_usage=[0-9.]+%' "$log" 2>/dev/null | tail -1)
    hit=$(echo "$srv" | grep -oE 'prefix_cache_hit=[0-9.]+' | cut -d= -f2)
    kv=$(echo "$srv" | grep -oE 'kv_usage=[0-9.]+' | cut -d= -f2)
    # grep -c prints 0 and exits 1 on no match; a || echo 0 would add a line.
    oom=$(grep -c 'memory allocation failed with OOM' "$dir/server.log" 2>/dev/null)
    crash=$(grep -c 'Scheduler hit an exception' "$dir/server.log" 2>/dev/null)
    read -r util pw < <(gpu_avg "$gpus")

    echo "PROGRESS $(date -u +%H:%M) ${run} ${point} gpu[${gpus:-all}] | ${phase} | hit=${hit:-?}% kv=${kv:-?}% | GPU ${util}% ${pw}W | OOM警告 ${oom} 崩潰 ${crash}"
}

while true; do
    mapfile -t live < <(find "$RESULTS" -name recipe.log -mmin "-${STALE_MIN}" \
                        -path '*/bench-Agentic-*' 2>/dev/null | sort)
    if [ "${#live[@]}" -eq 0 ]; then
        echo "PROGRESS $(date -u +%H:%M) 沒有進行中的跑"
    else
        for log in "${live[@]}"; do report_one "$log"; done
    fi
    sleep "$INTERVAL"
done
