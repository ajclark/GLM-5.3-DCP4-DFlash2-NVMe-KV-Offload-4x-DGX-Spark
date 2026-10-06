#!/usr/bin/env bash
# C1 policy check: ratio vs lambda on the probe's prose/code sets (agent replay skipped).
set -u
cd "$(dirname "$0")/../../runtime/vllm029"
for pol in ${POLS:-ratio lambda}; do
  ssh spark-06c4.local "echo '{\"mode\": \"auto\", \"batch\": true, \"policy\": \"$pol\", \"costs\": \"costs-batch.json\", \"prefill_cadence\": 1}' > ~/verify-cap-live/control.json"
  sleep 3
  python3 spec_accept_probe.py --captures "${CAP:?directory of probe captures (not published)}" --max-context-chars 0 \
    --out ../../results/step1-batchcap-20260930/c1-$pol | grep SUMMARY
done
