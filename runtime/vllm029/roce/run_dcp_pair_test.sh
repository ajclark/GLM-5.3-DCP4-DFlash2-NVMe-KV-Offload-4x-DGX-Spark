#!/usr/bin/env bash
# Two-rank RoCEnante test of the DCP collectives on one direct-cabled pair, model stopped.
#   IMAGE=<tag> bash run_dcp_pair_test.sh spark-06c4 192.168.1.228 spark-365c 192.168.1.88
# Overrides: GLUE (VLLM_DCP_GLUE, default 0), HCAS (B12X_ROCE_HCA, default rocep1s0f1), GATHER_MAX (default 4MiB), ARGS (extra test args,
# e.g. "--prefill-tokens 1024,2048 --skip-decode"), TEST_FILE (a local test_dcp_pair.py copied to the
# hosts and mounted over the image's copy).
set -uo pipefail
H0=$1 IP0=$2 H1=$3 IP1=$4
IMAGE="${IMAGE:?set IMAGE}"
PORT="${PORT:-29681}"
OUT="${OUT:-$(pwd)}"
HCAS="${HCAS:-rocep1s0f1}"
GATHER_MAX="${GATHER_MAX:-4MiB}"
ARGS="${ARGS:-}"
GLUE="${GLUE:-0}"
# E2: GLM_ROCE_LARGE_BLOCKS / GLM_ROCE_OWNCOPY_EARLY / GLM_ROCE_LARGE_MIN_BYTES pass through when set;
# ROCE_PKG=1 mounts this checkout's glm_roce over the image's copy (test new transport code without a rebuild).
E2ENV=""
for v in GLM_ROCE_LARGE_BLOCKS GLM_ROCE_OWNCOPY_EARLY GLM_ROCE_LARGE_MIN_BYTES GLM_ROCE_PIPE GLM_ROCE_PIPE_MIN_BYTES GLM_ROCE_PIPE_CHUNK_BYTES; do
  [ -n "${!v:-}" ] && E2ENV="$E2ENV -e $v=${!v}"
done
MOUNT=""
if [ -n "${TEST_FILE:-}" ]; then
  for h in "$H0" "$H1"; do scp -q "$TEST_FILE" "$h.local:/tmp/test_dcp_pair.py"; done
  MOUNT="-v /tmp/test_dcp_pair.py:/opt/glm-roce/test_dcp_pair.py:ro"
fi
if [ "${ROCE_PKG:-0}" = 1 ]; then
  HERE="$(cd "$(dirname "$0")" && pwd)"
  for h in "$H0" "$H1"; do rsync -a --delete --exclude __pycache__ "$HERE/glm_roce/" "$h.local:/tmp/glm_roce_pkg/"; done
  MOUNT="$MOUNT -v /tmp/glm_roce_pkg:/opt/glm-roce/glm_roce:ro"
fi
run() {  # host ip rank
  ssh -o BatchMode=yes "$1.local" "docker run --rm --name roce-pair-test --gpus all --network host --ipc host \
    --device /dev/infiniband:/dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1 --entrypoint python3 $MOUNT \
    -e VLLM_DCP_GLUE=$GLUE $E2ENV -e GLM_ROCE_ALLREDUCE=1 -e GLM_ROCE_GROUPS=dcp -e GLM_ROCE_REQUIRE=1 -e B12X_ROCE_HCA=$HCAS \
    -e B12X_ROCE_GID_INDEX=3 -e B12X_ROCE_SPIN_LIMIT=90000000 -e B12X_ROCE_IDLE_MAX_NAP_US=5000 \
    -e B12X_DISABLE_CUTLASS_RUNTIME_PATCHES=1 -e GLM_ROCE_MAX_SIZE=1MiB -e GLM_ROCE_GATHER_MAX_SIZE=$GATHER_MAX \
    -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA='=roceP2p1s0f0,roceP2p1s0f1' \
    -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_SUBNET_PREFIX_LEN=24 -e NCCL_IB_MERGE_NICS=1 \
    -e NCCL_SOCKET_IFNAME=enP7s7 -e GLOO_SOCKET_IFNAME=enP7s7 -e NCCL_ALGO=Ring -e NCCL_MIN_CTAS=1 -e NCCL_MAX_CTAS=1 \
    -e NCCL_MAX_NCHANNELS=1 -e NCCL_MIN_NCHANNELS=1 -e NCCL_CUMEM_ENABLE=1 -e VLLM_HOST_IP=$2 \
    $IMAGE /opt/glm-roce/test_dcp_pair.py --rank $3 --master $IP0 --port $PORT $ARGS" > "$OUT/pair-$1.log" 2>&1
}
run "$H1" "$IP1" 1 & p1=$!
sleep 2
run "$H0" "$IP0" 0 & p0=$!
wait $p0; r0=$?; wait $p1; r1=$?
grep -h "^RESULT" "$OUT/pair-$H0.log" "$OUT/pair-$H1.log" | cut -c1-400
echo "exit codes: $H0=$r0 $H1=$r1"
[ $r0 = 0 ] && [ $r1 = 0 ]
