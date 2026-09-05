#!/usr/bin/env bash
# Wait for a label's post-boot checks to finish cleanly, then run the soak.
cd "$(dirname "$0")"; LABEL="${1:?label}"; MIN="${2:-90}"; TOK="${3:-120000}"
until grep -q "checks done" "results/checks-$LABEL.console.log" 2>/dev/null; do sleep 20; done
grep -q "MEMORY GUARD\|rollout failed\|not the DCP stack\|generation FAILED" "results/checks-$LABEL.console.log" && { echo "checks flagged a problem; soak not started" > "results/soak-$LABEL.console.log"; exit 1; }
.venv/bin/python dcp_soak.py --base http://spark-06c4.local:8000 --label "$LABEL" --out "results/$LABEL" --minutes "$MIN" --tokens "$TOK" > "results/soak-$LABEL.console.log" 2>&1
echo "[$(date '+%H:%M:%S')] soak exit=$?" >> "results/soak-$LABEL.console.log"
