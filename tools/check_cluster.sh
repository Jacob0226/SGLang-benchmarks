#!/bin/bash
# Is the cluster usable again? Checks the four things that have to line up
# before any GLM benchmark can run, in the order they fail.
#
#   ./check_cluster.sh          one-shot report
#   ./check_cluster.sh --watch  re-check every 60s until everything is READY
PART=${PART:-amd-spur}

check() {
    local ok=1

    # 1. spur token. Everything else errors out in the same confusing way when
    #    this is stale, so it has to be tested first.
    if ! out=$(timeout 30 squeue -u "$USER" 2>&1); then
        echo "AUTH    FAIL   run: spur token user"
        return 1
    fi
    if grep -q "authentication required" <<<"$out"; then
        echo "AUTH    FAIL   token expired -- run: spur token user"
        return 1
    fi
    echo "AUTH    ok"

    # 2. Partition must be AVAIL=up. Nodes can all be idle while the partition
    #    is administratively down, which is what happened on 2026-09-29.
    local avail
    avail=$(timeout 30 sinfo -h -p "$PART" -o "%a" 2>/dev/null | sort -u | tr '\n' ',' | sed 's/,$//')
    if [ "$avail" = "up" ]; then
        echo "PART    ok     $PART is up"
    else
        echo "PART    FAIL   $PART AVAIL=$avail (nodes may be idle but nothing will schedule)"
        ok=0
    fi

    # 3. A running job with a node allocated.
    local running
    running=$(timeout 30 squeue -h -u "$USER" -t RUNNING -o "%i %N" 2>/dev/null | head -1)
    if [ -n "$running" ]; then
        echo "JOB     ok     running: $running"
    else
        echo "JOB     FAIL   nothing RUNNING; queued:"
        timeout 30 squeue -h -u "$USER" -o "           %i %T %R" 2>/dev/null | head -5
        ok=0
    fi

    # 4. The node answers and still has our container. Distinguish the three
    #    ways this fails: the srun client hanging (normal, ~half the time, just
    #    retry), spurd on the node refusing the attach (admin issue, retrying
    #    never helps), and a reachable node that simply lost the container.
    if [ -n "$running" ]; then
        local jid=${running%% *}
        timeout 90 srun --jobid="$jid" --overlap bash -c \
            'docker ps --format "{{.Names}}" 2>/dev/null' >/tmp/cc_docker.txt 2>&1
        local rc=$?
        if grep -q "did not advertise a native auth audience" /tmp/cc_docker.txt; then
            echo "NODE    FAIL   spurd on ${running##* } is not running the spur auth plugin"
            echo "               -- admin issue, retrying will not help"
            ok=0
        elif [ $rc -eq 124 ]; then
            echo "NODE    ?      srun client hung (normal, ~half the time) -- retry"
            ok=0
        elif [ $rc -ne 0 ]; then
            echo "NODE    FAIL   srun failed: $(head -1 /tmp/cc_docker.txt)"
            ok=0
        elif grep -q jacchang_GLM53 /tmp/cc_docker.txt; then
            echo "NODE    ok     container up: $(grep jacchang_GLM53 /tmp/cc_docker.txt | tr '\n' ' ')"
        else
            echo "NODE    ok     reachable, but no GLM53 container"
            echo "               -- run tools/start_glm53_0928.sh (new node = cold JIT cache)"
            ok=0
        fi
    fi

    return $((1 - ok))
}

if [ "${1:-}" = "--watch" ]; then
    for i in $(seq 1 480); do
        echo "--- $(date '+%F %T') ---"
        if check; then
            echo "CLUSTER_READY"
            exit 0
        fi
        sleep 60
    done
    echo "gave up waiting"
    exit 1
fi

check
