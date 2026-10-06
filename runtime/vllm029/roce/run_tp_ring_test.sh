#!/usr/bin/env bash
# Four-rank test of the TP ring all-reduce (glm_roce.ring), model stopped.
#   IMAGE=<tag> [MOUNT=1] [EXTRA_ENV="-e GLM_ROCE_RING_SPLIT=1"] [REPLAYS=300] [TAG=x] bash run_tp_ring_test.sh
# MOUNT=1 (default) runs this checkout's glm_roce/ and test_tp_ring.py inside an image
# built before the ring existed (rsynced to ~/ring-test on every node first).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${IMAGE:?set IMAGE}"
PORT="${PORT:-29691}"
OUT="${OUT:-$(pwd)}"
MOUNT="${MOUNT:-1}"
EXTRA_ENV="${EXTRA_ENV:-}"
REPLAYS="${REPLAYS:-300}"
TAG="${TAG:-}"
SCRIPT="${SCRIPT:-/opt/glm-roce/test_tp_ring.py}"   # or /opt/glm-roce/bench/bench_ring_breakdown.py
SCRIPT_ARGS="${SCRIPT_ARGS:---replays $REPLAYS}"
HOSTS=(spark-06c4 spark-365c spark-ddbf spark-a218)
IPS=(192.168.1.228 192.168.1.88 192.168.1.149 192.168.1.31)
mounts=""
if [ "$MOUNT" = 1 ]; then
  for h in "${HOSTS[@]}"; do rsync -a --delete --exclude tests "$HERE/" "$h.local:ring-test/" || exit 1; done
  mounts="-v \$HOME/ring-test/glm_roce:/opt/glm-roce/glm_roce:ro -v \$HOME/ring-test/test_tp_ring.py:/opt/glm-roce/test_tp_ring.py:ro -v \$HOME/ring-test/bench:/opt/glm-roce/bench:ro"
fi
run() {  # rank
  local h=${HOSTS[$1]} ip=${IPS[$1]}
  ssh -o BatchMode=yes "$h.local" "docker run --rm --name roce-ring-test --gpus all --network host --ipc host \
    --device /dev/infiniband:/dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1 --entrypoint python3 $mounts \
    -e GLM_ROCE_ALLREDUCE=1 -e GLM_ROCE_GROUPS=tp,dcp -e GLM_ROCE_REQUIRE=1 -e B12X_ROCE_HCA=rocep1s0f1 \
    -e GLM_ROCE_TP_RING=1 -e GLM_ROCE_TP_MAX_SIZE=1MiB -e GLM_ROCE_RING_CACHE_DIR=/tmp/ring-cache \
    -e B12X_ROCE_GID_INDEX=3 -e B12X_ROCE_SPIN_LIMIT=90000000 -e B12X_ROCE_IDLE_MAX_NAP_US=5000 \
    -e B12X_DISABLE_CUTLASS_RUNTIME_PATCHES=1 -e GLM_ROCE_MAX_SIZE=1MiB -e GLM_ROCE_GATHER_MAX_SIZE=4MiB \
    -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA='=roceP2p1s0f0,roceP2p1s0f1' \
    -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_SUBNET_PREFIX_LEN=24 -e NCCL_IB_MERGE_NICS=1 \
    -e NCCL_SOCKET_IFNAME=enP7s7 -e GLOO_SOCKET_IFNAME=enP7s7 -e NCCL_ALGO=Ring -e NCCL_MIN_CTAS=1 -e NCCL_MAX_CTAS=1 \
    -e NCCL_MAX_NCHANNELS=1 -e NCCL_MIN_NCHANNELS=1 -e NCCL_CUMEM_ENABLE=1 -e NCCL_RUNTIME_CONNECT=1 \
    -e NCCL_COLLNET_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_MNNVL_ENABLE=0 -e NCCL_PAT_ENABLE=0 -e NCCL_RMA_DISABLE=1 \
    -e NCCL_NUM_RMA_CTX=0 -e NCCL_RMA_EAGER_INIT=0 -e NCCL_GIN_ENABLE=0 -e NCCL_IB_ADAPTIVE_ROUTING=0 \
    -e NCCL_IB_QPS_PER_CONNECTION=1 -e NCCL_IGNORE_CPU_AFFINITY=1 -e VLLM_HOST_IP=$ip $EXTRA_ENV \
    $IMAGE $SCRIPT --rank $1 --master ${IPS[0]} --port $PORT $SCRIPT_ARGS" > "$OUT/ring${TAG}-$h.log" 2>&1
}
pids=()
for r in 3 2 1; do run $r & pids[$r]=$!; done
sleep 2
run 0 & pids[0]=$!
rc=0
for r in 0 1 2 3; do wait ${pids[$r]} || rc=1; done
for h in "${HOSTS[@]}"; do grep -h "^RESULT" "$OUT/ring${TAG}-$h.log" | cut -c1-600; done
echo "exit: $rc"
exit $rc
