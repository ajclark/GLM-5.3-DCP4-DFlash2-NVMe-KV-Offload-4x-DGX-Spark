#!/usr/bin/env bash
# Slab-store validation sequence:
#   A. boot with a small cap (KVTIER_DISK_BYTES_A): checks, offload probe (the cold
#      prefix is expected to be EVICTED by the time of the reload: a clean miss),
#      slab files never exceed the cap, engine healthy throughout;
#   B. restart into the real cap: offload probe again (reload must HIT);
#   C. restart again: the same prefix must hit on a fresh engine (durability,
#      index rebuilt from slot headers).
# Usage: ./deploy_slab.sh <label> [MAXLEN] [MAXBATCHED] [KVBYTES] [tokens]
set -uo pipefail
LABEL="${1:?label}"; MAXLEN="${2:-307200}"; MAXBATCHED="${3:-2048}"; KVBYTES="${4:-6000000000}"; TOK="${5:-100000}"
CAP_A="${KVTIER_DISK_BYTES_A:-4500000000}"; CAP="${KVTIER_DISK_BYTES:-150000000000}"
WS="$(cd "$(dirname "$0")" && pwd)"; cd "$WS"
LOG="results/deploy-$LABEL.log"; say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
HOSTS=(spark-06c4 spark-365c spark-ddbf spark-a218)
slabs() { for h in "${HOSTS[@]}"; do printf "  %s " "$h"; ssh -o BatchMode=yes "napta2k@$h.local" 'd=$(ls -d /var/tmp/kvcache/*_r* 2>/dev/null | head -1); [ -n "$d" ] && { ls -l $d/g*.slab 2>/dev/null | awk "{s+=\$5} END{printf \"slab bytes=%d \", s}"; du -sh $d | cut -f1; } || echo "no slab dir"'; done | tee -a "$LOG"; }
run_rollout() { KVTIER_MODE=slab KVTIER_DISK_BYTES="$2" ./rollout_dcp.sh "$1" "$MAXLEN" "$MAXBATCHED" "$KVBYTES" 1 > "results/rollout-$1.console.log" 2>&1; R=$(ls -d results/rollout-$1-*/ | tail -1); grep -q "rollout '$1' complete" "$R/rollout.log" || { say "rollout $1 did not complete; stopping"; exit 1; }; grep -h "Slab store" "$R"/up-*.log | sed 's/^(.*pid=[0-9]*) //' | cut -c1-170 | sort -u | tee -a "$LOG"; }
say "== A: boot with cap $CAP_A"
run_rollout "$LABEL-a" "$CAP_A"
./post_boot_checks.sh "$LABEL-a" results/baseline-dcp1-prod 0 > "results/checks-$LABEL-a.console.log" 2>&1
grep -q "checks done" "results/checks-$LABEL-a.console.log" || { say "checks did not finish; stopping"; exit 1; }
say "A: offload probe (reload expected to MISS: cap $CAP_A < 4 prefixes)"
./offload_checks.sh "$LABEL-a" "$TOK" > "results/offload-$LABEL-a.console.log" 2>&1
grep -q "offload checks done" "results/offload-$LABEL-a.console.log" || { say "offload checks did not finish; stopping"; exit 1; }
grep -q "MEMORY GUARD" "results/$LABEL-a/offload.log" && { say "memory guard tripped; stopping"; exit 1; }
say "A: slab sizes after 4 prefixes (cap $CAP_A):"; slabs
say "== B: restart into cap $CAP"
run_rollout "$LABEL-b" "$CAP"
./offload_checks.sh "$LABEL-b" "$TOK" > "results/offload-$LABEL-b.console.log" 2>&1
grep -q "offload checks done" "results/offload-$LABEL-b.console.log" || { say "offload checks (B) did not finish; stopping"; exit 1; }
say "== C: restart again (durability at cap $CAP)"
run_rollout "$LABEL-c" "$CAP"
.venv/bin/python dcp_probe.py --base http://spark-06c4.local:8000 --label "$LABEL-b" --out "results/$LABEL-b" --phase reload --offload-tokens "$TOK" 2>&1 | tee -a "$LOG"
say "C: slab sizes:"; slabs
say "== slab sequence done"
