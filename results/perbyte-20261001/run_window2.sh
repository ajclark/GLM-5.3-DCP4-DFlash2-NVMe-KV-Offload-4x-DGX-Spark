#!/usr/bin/env bash
# Window part 2 (model still stopped): GPU pinned bench (fixed build) + ring with the pair edge on
# PCIe domain 0 (rocep1s0f1) instead of domain 2, unsplit and split; correctness run of the new link map.
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$R/results/perbyte-20261001
IMAGE=${IMAGE:-spark-vllm:0.29.0-nvme4-stridefix-kvtier-verifycap10-stack-20261001}
ROCE=$R/runtime/vllm029/roce
for h in spark-06c4 spark-365c spark-ddbf spark-a218; do rsync -a --delete --exclude tests "$ROCE/" "$h.local:ring-test/" || exit 1; done
ssh spark-06c4.local "docker run --rm --gpus all --ipc=host --entrypoint python3 -v \$HOME/ring-test/bench:/w $IMAGE /w/bench_gpu_pinned.py" > $OUT/gpu-pinned.log 2>&1 &
gpu=$!
# the GPU bench uses spark-06c4's GPU; run the ring variants after it
wait $gpu; echo "gpu bench exit $?"
for v in "dom0:-e GLM_ROCE_RING_EXCLUDE=roceP2p1s0f1" "dom0split:-e GLM_ROCE_RING_EXCLUDE=roceP2p1s0f1 -e GLM_ROCE_RING_SPLIT=1" "dom2split:-e GLM_ROCE_RING_SPLIT=1"; do
  tag=${v%%:*}; env=${v#*:}
  OUT=$OUT IMAGE=$IMAGE MOUNT=1 TAG=-bd-$tag PORT=$((29800 + RANDOM % 150)) EXTRA_ENV="-e GLM_ROCE_RING_TRACE=1 $env" \
    SCRIPT=/opt/glm-roce/bench/bench_ring_breakdown.py SCRIPT_ARGS="--pin none" \
    timeout 900 bash $ROCE/run_tp_ring_test.sh > $OUT/run-bd-$tag.txt 2>&1
  echo "breakdown $tag: $(tail -n 1 $OUT/run-bd-$tag.txt)"
done
for v in "dom0:-e GLM_ROCE_RING_EXCLUDE=roceP2p1s0f1" "dom0split:-e GLM_ROCE_RING_EXCLUDE=roceP2p1s0f1 -e GLM_ROCE_RING_SPLIT=1"; do
  tag=${v%%:*}; env=${v#*:}
  OUT=$OUT IMAGE=$IMAGE MOUNT=1 TAG=-test-$tag REPLAYS=300 PORT=$((29960 + RANDOM % 30)) EXTRA_ENV="$env" \
    timeout 900 bash $ROCE/run_tp_ring_test.sh > $OUT/run-test-$tag.txt 2>&1
  echo "exactness test $tag: $(tail -n 1 $OUT/run-test-$tag.txt)"
done
echo "WINDOW2-DONE"
