#!/usr/bin/env bash
# Node side of the NCCL hot-plug probe: run bench/nccl_hotplug_probe.py inside the serving
# image with the launcher's NCCL environment, the hot-plug net plugin bind-mounted, and the
# plugin control directory shared with the host (so spark-idle.sh can suspend/resume it).
# Usage: run-nccl-hotplug-probe.sh <rank 0-3> [cycles]     env PROBE_NET=builtin -> NCCL's builtin IB backend
# (no plugin; use with cycles=0 for a like-for-like performance baseline).
# The serving stack must be down on every node; this container does not use --ipc host.
set -euo pipefail
NODE_RANK="${1:?rank}"; CYCLES="${2:-1}"
IMAGE="${DCP_IMAGE:-vllm-glm52-b12x:dflash2-port2}"
NAME="nccl_hotplug_probe"
MASTER_ADDR="192.168.1.228"; MASTER_PORT="${NCCL_PROBE_MASTER_PORT:-29613}"
PROBE="$HOME/glm53big/nccl_hotplug_probe.py"
PLUGIN="$HOME/nccl-hotplug/libnccl-net-hotplug.so"
CTL_PORT="${HOTPLUG_PORT:-5711}"
case "$NODE_RANK" in
  0) HOST_IP=192.168.1.228 ;; 1) HOST_IP=192.168.1.88 ;; 2) HOST_IP=192.168.1.149 ;; 3) HOST_IP=192.168.1.31 ;;
  *) echo "rank must be 0-3" >&2; exit 2 ;;
esac
test -f "$PROBE"; test -f "$PLUGIN"
ip -o -4 addr show enP7s7 | grep -q "$HOST_IP"
! docker inspect vllm_glm53big >/dev/null 2>&1 || { echo "serving container is still present on this node" >&2; exit 3; }
if [ "${PROBE_NET:-hotplug}" = builtin ]; then NET_ENV=(-e NCCL_NET=IB -e NCCL_NET_PLUGIN=none); else NET_ENV=(-e NCCL_NET_PLUGIN=hotplug -e NCCL_HOTPLUG_CTL_PORT="$CTL_PORT"); fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart no \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  --network host --gpus all --shm-size 4g \
  -v /dev/infiniband:/dev/infiniband --device-cgroup-rule 'c 231:* rwm' --device-cgroup-rule 'c 10:* rwm' \
  -v "$PROBE:/bench/nccl_hotplug_probe.py:ro" \
  -v "$PLUGIN:/usr/lib/aarch64-linux-gnu/libnccl-net-hotplug.so:ro" \
  -e MASTER_ADDR="$MASTER_ADDR" -e MASTER_PORT="$MASTER_PORT" \
  -e RANK="$NODE_RANK" -e WORLD_SIZE=4 -e LOCAL_RANK=0 \
  -e NCCL_HOTPLUG_PROBE_CYCLES="$CYCLES" -e NCCL_HOTPLUG_PROBE_SETTLE_S="${PROBE_SETTLE_S:-0}" \
  "${NET_ENV[@]}" \
  -e NCCL_IB_DISABLE=0 \
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
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET \
  "$IMAGE" python3 /bench/nccl_hotplug_probe.py
echo "$NAME launched rank=$NODE_RANK cycles=$CYCLES net=${PROBE_NET:-hotplug} plugin=$PLUGIN ctl=127.0.0.1:$CTL_PORT"
