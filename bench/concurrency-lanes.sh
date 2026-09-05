#!/usr/bin/env bash
# Concurrency sweep on the DCP=2 lane and the DCP=1 production launcher, then
# back to the DCP=4 default. Usage: ./bench/concurrency-lanes.sh
set -uo pipefail
WS="$(cd "$(dirname "$0")/.." && pwd)"; cd "$WS"
OUT="$WS/results"; LOG="$OUT/concurrency-lanes.log"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
URL=http://spark-06c4.local:8000
warm() { curl -sS -m 300 $URL/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"glm-5.3","messages":[{"role":"user","content":"Count from 1 to 100, separated by spaces. Output only the numbers."}],"max_tokens":300,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' >/dev/null; }
sweep() { "$WS/.venv/bin/python" "$HOME/spark-cluster-experiments/concurrency_cycle.py" --base-url $URL --concurrencies 1,2,4,8,12 --out "$OUT/concurrency-$1.json" 2>&1 | tail -7 | tee -a "$LOG"; }
say "== lane sweeps: DCP=2, DCP=1, then back to DCP=4"
say "-- DCP=2 rollout"
DCP_SIZE=2 ./rollout_dcp.sh dcp2-dflash-180k-sweep 180224 2048 6000000000 1 > "$OUT/rollout-dcp2-dflash-180k-sweep.console.log" 2>&1
R=$(ls -d "$OUT"/rollout-dcp2-dflash-180k-sweep-*/ | tail -1); grep -q "complete" "$R/rollout.log" || { say "DCP=2 rollout did not complete; stopping"; exit 1; }
warm; sweep dcp2
say "-- DCP=1 production launcher"
./restore_production.sh > "$OUT/restore-for-sweep.console.log" 2>&1
grep -q "production restored" "$OUT/restore-for-sweep.console.log" || { say "production restore did not verify; stopping"; exit 1; }
warm; sweep dcp1-prod
say "-- back to DCP=4"
./rollout_dcp.sh dcp4-dflash-300k-compact-prod6 > "$OUT/rollout-dcp4-dflash-300k-compact-prod6.console.log" 2>&1
R=$(ls -d "$OUT"/rollout-dcp4-dflash-300k-compact-prod6-*/ | tail -1); grep -q "complete" "$R/rollout.log" && say "DCP=4 back and generating" || say "DCP=4 ROLLOUT DID NOT COMPLETE: $(tail -2 "$R/rollout.log" | tr '\n' ' ')"
say "== lane sweeps done"
