#!/usr/bin/env bash
# Wait for the soak to finish cleanly, then run the NVMe-tier deployment.
cd "$(dirname "$0")"; SOAK="${1:?soak label}"; LABEL="${2:?deploy label}"
until grep -q "soak exit=" "results/soak-$SOAK.console.log" 2>/dev/null; do sleep 30; done
if grep -q "GUARD\|Traceback\|flagged" "results/soak-$SOAK.console.log"; then
  echo "[$(date '+%H:%M:%S')] soak flagged a problem; deployment not started" > "results/deploy-$LABEL.log"; exit 1
fi
./deploy_kvtier.sh "$LABEL" 307200 2048 6000000000 100000
