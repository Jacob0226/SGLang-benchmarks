#!/usr/bin/env bash
# Poll Agentic runs for crashes. Prints one line per new event:
#   CRASH <run>: <first error line>   server.log hit an error pattern
#   EXIT <driver line>                a run finished (non-zero exit is flagged as CRASH)
#   ALLDONE                           every lane printed "lane DONE"
# usage: agentic_crash_watch.sh <driver.log> <results_root_glob> <n_lanes> [interval_s]
DRIVER=$1 GLOB=$2 NLANES=$3 INTERVAL=${4:-60}
PAT='Traceback|Memory access fault|HSA_STATUS_ERROR|core dumped|out of memory|OutOfMemory|NCCL error|RCCL error|Watchdog caught|Segmentation fault|Killed'
declare -A seen
while :; do
    for log in $GLOB/*/server.log; do
        [ -f "$log" ] || continue
        [ -n "${seen[$log]:-}" ] && continue
        hit=$(grep -m1 -E "$PAT" "$log")
        if [ -n "$hit" ]; then
            seen[$log]=1
            run=$(basename "$(dirname "$(dirname "$log")")")
            echo "[$(date -Is)] CRASH $run: ${hit:0:200}"
        fi
    done
    while IFS= read -r line; do
        [ -n "${seen[$line]:-}" ] && continue
        seen[$line]=1
        case $line in
            *" exit 0") echo "[$(date -Is)] EXIT $line" ;;
            *" exit "*) echo "[$(date -Is)] CRASH $line" ;;
        esac
    done < <(grep -E ' exit [0-9]+$' "$DRIVER" 2>/dev/null)
    if [ "$(grep -c 'lane DONE' "$DRIVER" 2>/dev/null)" -ge "$NLANES" ]; then
        echo "[$(date -Is)] ALLDONE"
        exit 0
    fi
    sleep "$INTERVAL"
done
