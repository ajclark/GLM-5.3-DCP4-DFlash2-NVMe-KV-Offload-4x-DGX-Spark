#!/usr/bin/env bash
# Launch the multi-communicator NCCL microbenchmark on one node (runs inside
# the serving image with the serving launcher's NCCL environment).
# Usage: run-nccl-multicomm.sh <rank 0-3> <variant: default|ll>
# The serving stack must be down on every node (memory), and this container
# does not use --ipc host on purpose.
set -euo pipefail
NODE_RANK="${1:?rank}"; VARIANT="${2:?variant}"
IMAGE="vllm-glm52-b12x:dflash2-port2"
NAME="nccl_multicomm_${VARIANT}"
MASTER_ADDR="192.168.1.228"; MASTER_PORT="${NCCL_BENCH_MASTER_PORT:-29611}"
BENCH_PATH="$HOME/glm53big/nccl_multicomm.py"
case "$NODE_RANK" in
  0) HOST_IP=192.168.1.228 ;; 1) HOST_IP=192.168.1.88 ;; 2) HOST_IP=192.168.1.149 ;; 3) HOST_IP=192.168.1.31 ;;
  *) echo "rank must be 0-3" >&2; exit 2 ;;
esac
test -f "$BENCH_PATH"
ip -o -4 addr show enP7s7 | grep -q "$HOST_IP"
! docker inspect vllm_glm53big >/dev/null 2>&1 || { echo "serving container is still present on this node" >&2; exit 3; }
EXTRA_ENV=()
case "$VARIANT" in
  default) ;;
  ll) EXTRA_ENV+=( -e NCCL_PROTO=LL ) ;;
  ll128) EXTRA_ENV+=( -e NCCL_PROTO=LL128 ) ;;
  gm0) EXTRA_ENV+=( -e NCCL_GRAPH_MIXING_SUPPORT=0 ) ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart no \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  --network host --gpus all --shm-size 4g \
  --device /dev/infiniband:/dev/infiniband \
  -v "$BENCH_PATH:/bench/nccl_multicomm.py:ro" \
  -e MASTER_ADDR="$MASTER_ADDR" -e MASTER_PORT="$MASTER_PORT" \
  -e RANK="$NODE_RANK" -e WORLD_SIZE=4 -e LOCAL_RANK=0 \
  -e NCCL_BENCH_VARIANT="$VARIANT" -e NCCL_BENCH_NS="${NCCL_BENCH_NS:-1,2,4,8}" \
  -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA='=roceP2p1s0f0,roceP2p1s0f1' \
  -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_SUBNET_PREFIX_LEN=24 \
  -e NCCL_IB_MERGE_NICS=1 -e NCCL_IB_ADAPTIVE_ROUTING=0 \
  -e NCCL_SOCKET_IFNAME=enP7s7 -e GLOO_SOCKET_IFNAME=enP7s7 \
  -e NCCL_ALGO=Ring -e NCCL_RUNTIME_CONNECT=1 \
  -e NCCL_COLLNET_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_MNNVL_ENABLE=0 \
  -e NCCL_PAT_ENABLE=0 \
  -e NCCL_RMA_DISABLE=1 -e NCCL_NUM_RMA_CTX=0 -e NCCL_RMA_EAGER_INIT=0 \
  -e NCCL_GIN_ENABLE=0 \
  -e NCCL_MIN_CTAS=1 -e NCCL_MAX_CTAS=1 \
  -e NCCL_MAX_NCHANNELS=1 -e NCCL_MIN_NCHANNELS=1 \
  -e NCCL_IB_QPS_PER_CONNECTION=1 \
  -e NCCL_CUMEM_ENABLE=1 -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_DEBUG=WARN \
  "${EXTRA_ENV[@]}" \
  "$IMAGE" python3 /bench/nccl_multicomm.py
echo "$NAME launched rank=$NODE_RANK variant=$VARIANT"
