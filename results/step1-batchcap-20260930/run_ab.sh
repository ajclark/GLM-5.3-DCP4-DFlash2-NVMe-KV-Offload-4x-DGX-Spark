#!/usr/bin/env bash
# In-boot A/B of the batch verify cap (control file switches; no restart).
set -u
cd "$(dirname "$0")/../../runtime/vllm029"
OUT=../../results/step1-batchcap-20260930/ab
CAP=${CAP:?directory of probe and pi agent-turn captures (not published)}
declare -A CTL=(
  [base]='{"mode": "auto", "batch": false, "policy": "ratio", "prefill_cadence": 1}'
  [ratio]='{"mode": "auto", "batch": true, "policy": "ratio", "costs": "costs-batch.json", "prefill_cadence": 1}'
  [lambda]='{"mode": "auto", "batch": true, "policy": "lambda", "costs": "costs-batch.json", "prefill_cadence": 1}')
for n in ${NS:-2 3 4}; do
  for arm in ${ARMS:-base ratio lambda}; do
    ssh spark-06c4.local "echo '${CTL[$arm]}' > ~/verify-cap-live/control.json"
    sleep 3
    python3 conc_bench.py --out $OUT --n $n --sets ${SETS:-prose,code,mix} --rounds ${ROUNDS:-3} \
      --captures $CAP --label $arm | grep -E "SUMMARY.*\|n$n\||CONTAM"
  done
done
