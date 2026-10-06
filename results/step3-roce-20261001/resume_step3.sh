#!/usr/bin/env bash
# Resume the step-3 series (C2-C4 + full C1/agent probe) once the endpoint has been idle
# for IDLE_MIN minutes in a row (no running/waiting requests, no new generated tokens).
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$R/results/step3-roce-20261001
CAP=${CAP:?directory of probe and pi agent-turn captures (not published)}
IDLE_MIN=${IDLE_MIN:-10}
metric() { curl -s -m 5 http://spark-06c4.local:8000/metrics | awk -v k="$1" '$1 ~ "^vllm:"k"\\{" {s+=$2} END {print s+0}'; }
quiet=0; last=$(metric generation_tokens_total)
echo "$(date +%H:%M:%S) waiting for ${IDLE_MIN} idle minutes"
while [ $quiet -lt $((IDLE_MIN * 4)) ]; do
  sleep 15
  run=$(metric num_requests_running); wait_=$(metric num_requests_waiting); gen=$(metric generation_tokens_total)
  if [ "${run%.*}" = 0 ] && [ "${wait_%.*}" = 0 ] && [ "$gen" = "$last" ]; then quiet=$((quiet + 1)); else quiet=0; fi
  last=$gen
done
echo "$(date +%H:%M:%S) idle; resuming"
cd $R/runtime/vllm029
ssh spark-06c4.local "echo '{\"mode\": \"auto\", \"batch\": true, \"policy\": \"ratio\", \"costs\": \"costs-step3.json\"}' > ~/verify-cap-live/control.json"
sleep 3
python3 conc_bench.py --out $OUT/conc --n 2,3,4 --sets prose,code,mix --rounds 3 --captures $CAP --label step3 | grep -E "SUMMARY|CONTAM"
python3 spec_accept_probe.py --captures $CAP --out $OUT/c1-auto-full | grep SUMMARY
echo STEP3-DONE
