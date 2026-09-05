#!/usr/bin/env bash
# NVMe tier deployment sequence (docs/NVME-DESIGN.md section 4), fully guarded:
#   1. rollout KVTIER=1 (auto-restores production on failure)
#   2. short post-boot checks (no long-context probe)
#   3. offload_checks: cold / warm / evict x3 / reload, files per node, guard
#   4. relaunch the same stack and re-send the prefix: durable across restarts
# Usage: ./deploy_kvtier.sh <label> [MAXLEN] [MAXBATCHED] [KVBYTES] [tokens]
set -uo pipefail
LABEL="${1:?label}"; MAXLEN="${2:-307200}"; MAXBATCHED="${3:-2048}"; KVBYTES="${4:-6000000000}"; TOK="${5:-100000}"
WS="$(cd "$(dirname "$0")" && pwd)"; cd "$WS"
LOG="results/deploy-$LABEL.log"; say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
say "== deploy NVMe tier: $LABEL MAXLEN=$MAXLEN KVBYTES=$KVBYTES tokens=$TOK"
./rollout_dcp.sh "$LABEL" "$MAXLEN" "$MAXBATCHED" "$KVBYTES" 1 > "results/rollout-$LABEL.console.log" 2>&1
R=$(ls -d results/rollout-$LABEL-*/ | tail -1)
grep -q "rollout '$LABEL' complete" "$R/rollout.log" || { say "rollout did not complete (see $R/rollout.log); stopping"; exit 1; }
say "stack up; tier startup lines:"
grep -h "MultiNodeTiering\|Worker fs tier\|Creating v1 connector\|offloading spec\|KV offloading" "$R"/up-*.log | sed 's/^(.*pid=[0-9]*) //' | cut -c1-160 | sort | uniq -c | head -12 | tee -a "$LOG"
./post_boot_checks.sh "$LABEL" results/baseline-dcp1-prod 0 > "results/checks-$LABEL.console.log" 2>&1
grep -q "checks done" "results/checks-$LABEL.console.log" || { say "post-boot checks did not finish; stopping"; exit 1; }
say "post-boot checks done; running offload checks"
./offload_checks.sh "$LABEL" "$TOK" > "results/offload-$LABEL.console.log" 2>&1
grep -q "offload checks done" "results/offload-$LABEL.console.log" || { say "offload checks did not finish; stopping"; exit 1; }
grep -q "MEMORY GUARD" "results/$LABEL/offload.log" && { say "offload checks tripped the memory guard; not restarting"; exit 1; }
say "offload checks done; relaunching the same stack for the durability test"
KVTIER_DISK_BYTES="${KVTIER_DISK_BYTES_RESTART:-${KVTIER_DISK_BYTES:-150000000000}}" ./rollout_dcp.sh "$LABEL-restart" "$MAXLEN" "$MAXBATCHED" "$KVBYTES" 1 > "results/rollout-$LABEL-restart.console.log" 2>&1
R2=$(ls -d results/rollout-$LABEL-restart-*/ | tail -1)
grep -q "rollout '$LABEL-restart' complete" "$R2/rollout.log" || { say "restart rollout did not complete; stopping"; exit 1; }
say "restarted; re-sending the cold prefix"
.venv/bin/python dcp_probe.py --base http://spark-06c4.local:8000 --label "$LABEL" --out "results/$LABEL" --phase reload --offload-tokens "$TOK" 2>&1 | tee -a "$LOG"
say "== deploy sequence done"
