#!/usr/bin/env bash
# E3 benchmark launcher: runs bench/nccl_tp_dualring.py on all four Sparks (serving must be stopped).
#   IMAGE=<serving image> HCA='=roceP2p1s0f0,rocep1s0f1' MERGE=0 CHANNELS=1 CONFIGS=1fwd,2fwdrev LABEL=x \
#     bash run_nccl_tp_dualring.sh OUTDIR
# Same NCCL environment as runtime/vllm029/launch.sh except the overridable NIC list, merging and channels.
set -uo pipefail
OUT="${1:?outdir}"
IMAGE="${IMAGE:?set IMAGE}"
HCA="${HCA:-=roceP2p1s0f0,roceP2p1s0f1}"; MERGE="${MERGE:-1}"; CHANNELS="${CHANNELS:-1}"
CONFIGS="${CONFIGS:-1fwd,2fwdrev}"; LABEL="${LABEL:-run}"; PORT="${PORT:-29631}"
HERE="$(cd "$(dirname "$0")" && pwd)"
HOSTS=(spark-06c4 spark-365c spark-ddbf spark-a218)
mkdir -p "$OUT"
for r in 0 1 2 3; do
  h=${HOSTS[$r]}
  scp -q "$HERE/nccl_tp_dualring.py" "$h.local:/tmp/nccl_tp_dualring.py"
  ssh -o BatchMode=yes "$h.local" "docker inspect vllm_glm53big >/dev/null 2>&1 && docker ps -q --filter name=vllm_glm53big | grep -q . && { echo serving container running; exit 3; }
    docker rm -f e3bench >/dev/null 2>&1; docker run --rm --name e3bench --gpus all --network host --shm-size 4g \
    --cap-add IPC_LOCK --ulimit memlock=-1:-1 --device /dev/infiniband:/dev/infiniband --entrypoint python3 \
    -v /tmp/nccl_tp_dualring.py:/bench/nccl_tp_dualring.py:ro \
    -e MASTER_ADDR=192.168.1.228 -e MASTER_PORT=$PORT -e RANK=$r -e WORLD_SIZE=4 -e LOCAL_RANK=0 \
    -e NCCL_BENCH_CONFIGS=$CONFIGS -e NCCL_BENCH_LABEL=$LABEL -e NCCL_BENCH_STRESS=${STRESS:-0} \
    -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA='$HCA' -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_SUBNET_PREFIX_LEN=24 -e NCCL_IB_MERGE_NICS=$MERGE \
    -e NCCL_IB_ADAPTIVE_ROUTING=0 -e NCCL_SOCKET_IFNAME=enP7s7 -e GLOO_SOCKET_IFNAME=enP7s7 \
    -e NCCL_ALGO=Ring -e NCCL_RUNTIME_CONNECT=1 -e NCCL_COLLNET_ENABLE=0 -e NCCL_NVLS_ENABLE=0 \
    -e NCCL_MNNVL_ENABLE=0 -e NCCL_PAT_ENABLE=0 -e NCCL_RMA_DISABLE=1 -e NCCL_NUM_RMA_CTX=0 \
    -e NCCL_RMA_EAGER_INIT=0 -e NCCL_GIN_ENABLE=0 -e NCCL_MIN_CTAS=$CHANNELS -e NCCL_MAX_CTAS=$CHANNELS \
    -e NCCL_MAX_NCHANNELS=$CHANNELS -e NCCL_MIN_NCHANNELS=$CHANNELS -e NCCL_IB_QPS_PER_CONNECTION=1 \
    -e NCCL_CUMEM_ENABLE=1 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET \
    $IMAGE /bench/nccl_tp_dualring.py" > "$OUT/rank$r.log" 2>&1 &
done
wait
grep -h '^{' "$OUT/rank0.log"
grep -h -E "NET/IB : Using|via NET/IB|Channel 00/0" "$OUT/rank0.log" | head -12
