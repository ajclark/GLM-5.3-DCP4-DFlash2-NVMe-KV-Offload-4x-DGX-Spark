#!/usr/bin/env bash
# Maintenance window 1 (model stopped): ring variants A/B on all four nodes, then the step-5 and
# step-6 single-GPU tests in parallel on spark-365c / spark-ddbf (+ step-6 Marlin bench on spark-a218).
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE=${IMAGE:-spark-vllm:0.29.0-nvme4-stridefix-kvtier-verifycap9-ring-20261001}
OV=$R/runtime/vllm029/verify_cap_overlay
if [ "${SKIP_AB:-0}" != 1 ]; then
echo "$(date +%H:%M:%S) stopping serving"
for h in spark-06c4 spark-365c spark-ddbf spark-a218; do ssh -o BatchMode=yes $h.local 'docker stop -t 10 vllm_glm53big >/dev/null 2>&1; true' & done; wait
echo "$(date +%H:%M:%S) ring A/B"
bash $R/results/step7-ring-allreduce-20261001/run_ring_ab.sh
fi
echo "$(date +%H:%M:%S) step 5/6 GPU tests"
DOCK="docker run --rm --gpus all --ipc=host --entrypoint python3 -e PYTHONDONTWRITEBYTECODE=1"
# step 5 on spark-365c
( ssh spark-365c.local "mkdir -p glm-fast-test/glm_fast" &&
  rsync -a --delete --exclude __pycache__ $OV/glm_fast/ spark-365c.local:glm-fast-test/glm_fast/ &&
  ssh spark-365c.local "$DOCK -v \$HOME/glm-fast-test:/w $IMAGE /w/glm_fast/gpu_test_step5.py" \
    > $R/results/step5-l2-argmax-20261001/gpu-test.log 2>&1; echo "step5 exit $?" ) &
# (step-6 kernel tests also ran in this window; that kernel was not adopted and is not published)
wait
echo "$(date +%H:%M:%S) WINDOW1-DONE"
