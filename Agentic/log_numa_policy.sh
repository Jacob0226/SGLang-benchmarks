#!/usr/bin/env bash
# Record where an SGLang server's workers run and where their memory lives:
# CPU affinity, NUMA memory policy (bind:N under numactl --membind, default
# when nothing bound them) and resident memory per node, plus each node's
# free memory. Call after the server is ready, so the HiCache host pools are
# already allocated.
#
#   log_numa_policy.sh <server_pid> <out_file>
set -uo pipefail
root="$1" out="$2"

descendants() {
    local pid="$1" child
    for child in $(pgrep -P "$pid" 2>/dev/null); do
        echo "$child"
        descendants "$child"
    done
}

{
    echo "=== $(date -Is) numa_balancing=$(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo n/a)"
    for node in /sys/devices/system/node/node[0-9]*; do
        awk -v n="$(basename "$node")" '/MemTotal|MemFree/ {v[$3]=$4} END {printf "%s total=%.0fGB free=%.0fGB\n", n, v["MemTotal:"]/1048576, v["MemFree:"]/1048576}' "$node/meminfo"
    done
    for pid in $(descendants "$root"); do
        comm=$(cat "/proc/$pid/comm" 2>/dev/null) || continue
        case "$comm" in sglang::*) ;; *) continue ;; esac
        cpus=$(awk '/Cpus_allowed_list/ {print $2}' "/proc/$pid/status")
        mem=$(awk '
            { pol[$2]++; ps = 4
              for (i = 3; i <= NF; i++) if ($i ~ /^kernelpagesize_kB=/) { split($i, a, "="); ps = a[2] }
              for (i = 3; i <= NF; i++) if ($i ~ /^N[0-9]+=/) { split($i, a, "="); n[a[1]] += a[2] * ps } }
            END { for (k in pol) printf "%s:%d ", k, pol[k]
                  printf "| resident"
                  for (k in n) printf " %s=%.1fGB", k, n[k] / 1048576 }' "/proc/$pid/numa_maps" 2>/dev/null)
        echo "$pid $comm cpus=$cpus policy: $mem"
    done
} > "$out" 2>&1
cat "$out"
