#!/usr/bin/env bash
# Validation of the connector-scheduler fix (eagle trailing block revisited):
#   D. roll the fixed overlay out (slab store, real cap), then an offload
#      probe WITHOUT the warm step on fresh prefixes: cold -> 3 evictions ->
#      reload. The reload must HIT from a prefix stored exactly once.
#   E. restart: the same once-stored prefix must hit on a fresh engine, and so
#      must one of the eviction prefixes (stored once, never reloaded).
# Usage: ./deploy_slab_fix.sh <label> [MAXLEN] [MAXBATCHED] [KVBYTES] [tokens]
set -uo pipefail
LABEL="${1:?label}"; MAXLEN="${2:-307200}"; MAXBATCHED="${3:-2048}"; KVBYTES="${4:-6000000000}"; TOK="${5:-100000}"
CAP="${KVTIER_DISK_BYTES:-150000000000}"; SEED_BASE="${SEED_BASE:-1000}"
WS="$(cd "$(dirname "$0")" && pwd)"; cd "$WS"
LOG="results/deploy-$LABEL.log"; say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
HOSTS=(spark-06c4 spark-365c spark-ddbf spark-a218)
slabs() { for h in "${HOSTS[@]}"; do printf "  %s " "$h"; ssh -o BatchMode=yes "${SSH_USER:-$USER}@$h.local" 'd=$(ls -d /var/tmp/kvcache/*_r* 2>/dev/null | head -1); [ -n "$d" ] && { ls -l $d/g*.slab 2>/dev/null | awk "{s+=\$5} END{printf \"slab bytes=%d \", s}"; du -sh $d | cut -f1; } || echo "no slab dir"'; done | tee -a "$LOG"; }
run_rollout() { KVTIER_MODE=slab KVTIER_DISK_BYTES="$CAP" ./rollout_dcp.sh "$1" "$MAXLEN" "$MAXBATCHED" "$KVBYTES" 1 > "results/rollout-$1.console.log" 2>&1; R=$(ls -d results/rollout-$1-*/ | tail -1); grep -q "rollout '$1' complete" "$R/rollout.log" || { say "rollout $1 did not complete; stopping"; exit 1; }; grep -h "Slab store" "$R"/up-*.log | sed 's/^(.*pid=[0-9]*) //' | cut -c1-170 | sort -u | tee -a "$LOG"; }
probe() { .venv/bin/python dcp_probe.py --base http://spark-06c4.local:8000 --label "$1" --out "results/$1" --phase reload --offload-tokens "$TOK" --seed-base "$SEED_BASE" "${@:2}" 2>&1 | tee -a "$LOG"; }
say "== D: roll out the fixed connector scheduler (cap $CAP, fresh prefixes from seed base $SEED_BASE)"
run_rollout "$LABEL-d"
say "D: offload probe without the warm step (reload must HIT from a once-stored prefix)"
PROBE_EXTRA="--no-warm --seed-base $SEED_BASE" ./offload_checks.sh "$LABEL-d" "$TOK" > "results/offload-$LABEL-d.console.log" 2>&1
grep -q "offload checks done" "results/offload-$LABEL-d.console.log" || { say "offload checks (D) did not finish; stopping"; exit 1; }
grep -q "MEMORY GUARD" "results/$LABEL-d/offload.log" && { say "memory guard tripped; stopping"; exit 1; }
grep -h "cold \|evict\|reload" "results/$LABEL-d/offload.console.log" | tee -a "$LOG"
say "D: slab sizes:"; slabs
say "== E: restart (once-stored prefixes must hit on a fresh engine)"
run_rollout "$LABEL-e"
mkdir -p "results/$LABEL-e"
probe "$LABEL-e"                                                        # the cold prefix (stored once in D, reloaded once)
probe "$LABEL-e" --reload-seed $((5002 + SEED_BASE)) --reload-tag reload-evict2-after-restart   # stored once, never reloaded
say "E: slab sizes:"; slabs
say "== fix sequence done"
