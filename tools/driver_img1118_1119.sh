#!/usr/bin/env bash
# Node 152, 1118 image: stock (GPUs 0-3, FP32 KDA state, image monokernel) and stock + PR stack
# (GPUs 4-7, BF16 KDA state) AgentX side by side. The PR lane runs a GSM8K gate first and only
# starts its AgentX when the score is at least GSM8K_MIN.
# usage: driver_img1118_1119.sh <stock|pr>
set -uo pipefail
LANE=$1
D=/shared_nfs/jacchang/agentx_1119_img1118
T=/home/jacchang/SGLang-benchmarks/tools
GSM8K_MIN=${GSM8K_MIN:-0.955}
log() { echo "[$(date -Is)] [driver-$LANE] $*" | tee -a $D/driver.log; }
case $LANE in
    stock)
        for g in m1 m48 hi; do
            docker exec -e STATE=fp32 jacchang_stock1118 bash $T/agentx_img1118_1119.sh stock 0,1,2,3 30000 $g
        done ;;
    pr)
        docker exec jacchang_pr1118 bash $T/gsm8k_img1118_1119.sh 4,5,6,7 31900 $D/gsm8k_pr
        score=$(python3 - $D/gsm8k_pr/gsm8k <<'EOF'
import json, sys, glob
paths = sorted(glob.glob(sys.argv[1] + "/**/metrics.json", recursive=True))
print(json.load(open(paths[-1]))["aggregate"]["score"] if paths else "")
EOF
        )
        log "gsm8k score '${score}' (gate ${GSM8K_MIN})"
        if [ -z "$score" ] || ! python3 -c "import sys; sys.exit(0 if float('$score') >= $GSM8K_MIN else 1)"; then
            log "gsm8k gate failed; PR lane AgentX not started"
            exit 1
        fi
        for g in m1 m48 hi; do
            docker exec -e STATE=bf16 jacchang_pr1118 bash $T/agentx_img1118_1119.sh pr 4,5,6,7 31000 $g
        done ;;
esac
log "lane $LANE finished"
