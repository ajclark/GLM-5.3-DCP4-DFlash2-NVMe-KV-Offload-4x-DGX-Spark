#!/usr/bin/env bash
# In-boot A/B arms on fixed-K live periods. Usage: bash arms.sh <tag> "<arm>=<l2windows>" ...
# Each arm: set the L2 prefetch windows on all four nodes, then for K in $KS: fixed K + a new
# periods epoch, one conc_bench round (n=$NS, prose+code), and a copy of periods.json.
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$R/results/step456-integration-20261001/$1; shift
KS=${KS:-"7 1"}; NS=${NS:-1,4}; COSTS=${COSTS:-costs-step7.json}
mkdir -p $OUT
cd $R/runtime/vllm029
epoch=$(date +%s)
for spec in "$@"; do
  arm=${spec%%=*}; win=${spec#*=}
  for h in spark-06c4 spark-365c spark-ddbf spark-a218; do
    ssh $h.local "echo '{\"windows\": \"$win\"}' > ~/verify-cap-live/glm_fast_l2pf.json"
  done
  for K in $KS; do
    epoch=$((epoch + 1))
    ssh spark-06c4.local "echo '{\"mode\": \"fixed\", \"fixed_k\": $K, \"batch\": true, \"policy\": \"ratio\", \"costs\": \"$COSTS\", \"periods_epoch\": $epoch}' > ~/verify-cap-live/control.json"
    sleep 4
    python3 conc_bench.py --out $OUT/conc-$arm-k$K --n $NS --sets prose,code --rounds 1 --label $arm-k$K | grep -E "SUMMARY|CONTAM"
    sleep 35
    scp -q spark-06c4.local:verify-cap-live/periods.json $OUT/periods-$arm-k$K.json
    python3 -c "
import json; P=json.load(open('$OUT/periods-$arm-k$K.json'))['periods']
print('$arm', 'K=$K', {k: (v['median_ms'], v['count']) for k, v in P.items() if k.endswith('ctx0')})"
  done
done
ssh spark-06c4.local "echo '{\"mode\": \"auto\", \"batch\": true, \"policy\": \"ratio\", \"costs\": \"$COSTS\"}' > ~/verify-cap-live/control.json"
echo ARMS-DONE
