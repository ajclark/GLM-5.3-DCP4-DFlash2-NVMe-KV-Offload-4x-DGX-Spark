#!/usr/bin/env bash
# vLLM 0.29.0 TP4/DCP2 + replicated DFlash2 + worker-local durable KV.
# Deploy through rollout.py, which retains exact original containers for rollback.
set -uo pipefail

NODE_RANK="${1:?usage: launch-glm53big-dcp.sh <0|1|2|3> [dflash|none]}"
SPEC_MODE="${2:-dflash}"

# These historical experiments require their matching engine and overlays.
if [ "${GLM_SPEC_POLICY:-off}" != off ] || [ "${GLM_SPEC_LOSSY:-0}" != 0 ] || \
   [ "${GLM_DCP_LSE_FOLD:-0}" != 0 ] || [ "${DCP_LSE_FOLD:-0}" != 0 ] || \
   [ "${DCP_RS_HEADMAJOR:-0}" != 0 ]; then
  echo "Experimental switches require VLLM_RUNTIME=legacy and the historical image" >&2
  exit 2
fi

IMAGE="${DCP_IMAGE:-spark-vllm:0.29.0-dcp1}"
NAME="vllm_glm53big"
PORT=8000
MASTER_PORT=29541
LAN_IF="enP7s7"
HEAD_IP="192.168.1.228"
WEIGHTS=/var/tmp/models/GLM-5.3-Int4-Int8Mix

DCP_SIZE="${DCP_SIZE:-2}"
MAXLEN="${MAXLEN:-180224}"
# DFlash draft tokens per cycle. 7 is trained in (block_size 8); not a knob.
DFLASH_K="${DFLASH_K:-7}"
MAXBATCHED="${MAXBATCHED:-2048}"
# Preserve the selected 6 GB/rank allocation and 180224-token serving window.
KVBYTES="${KVBYTES:-6000000000}"
# Worker-local durable NVMe slab, with bounded pinned transfer buffers.
KVTIER="${KVTIER:-1}"
KVTIER_DIR="${KVTIER_DIR:-/var/tmp/kvcache-vllm029}"
KVTIER_THREADS="${KVTIER_THREADS:-8}"
# Fixed-size ring buffer per rank; slot reuse bounds disk consumption.
KVTIER_MODE="${KVTIER_MODE:-slab}"
KVTIER_BOUNCE="${KVTIER_BOUNCE:-48}"
KVTIER_DISK_BYTES="${KVTIER_DISK_BYTES:-150000000000}"

# RANK ORDER MUST MATCH THE PHYSICAL RING (06c4 -> 365c -> ddbf -> a218 -> 06c4).
case "$NODE_RANK" in
  0) HOST_IP=192.168.1.228; HEADLESS=0 ;;   # spark-06c4
  1) HOST_IP=192.168.1.88;  HEADLESS=1 ;;   # spark-365c
  2) HOST_IP=192.168.1.149; HEADLESS=1 ;;   # spark-ddbf
  3) HOST_IP=192.168.1.31;  HEADLESS=1 ;;   # spark-a218
  *) echo "rank must be 0-3" >&2; exit 2 ;;
esac

test -f "$WEIGHTS/config.json" || { echo "weights not visible at $WEIGHTS" >&2; exit 3; }
ip -o -4 addr show "$LAN_IF" | grep -q "$HOST_IP" || {
  echo "$LAN_IF does not hold $HOST_IP on this node -- wrong rank?" >&2; exit 3; }

# Fail before stopping anything if the pinned runtime is not installed.
docker image inspect "$IMAGE" >/dev/null || exit 4
KVTIER_MOUNTS=(); KVTIER_ARGS=(); KVTIER_ENV=(); PROF_ARGS=()
if [ -n "${PROFILER_DIR:-}" ]; then
  PROF_ARGS=(--profiler-config.profiler=torch "--profiler-config.torch_profiler_dir=$PROFILER_DIR")
fi
if [ "$KVTIER" = 1 ]; then
  [ "$KVTIER_MODE" = slab ] || { echo "vLLM 0.29 supports KVTIER_MODE=slab" >&2; exit 2; }
  mkdir -p "$KVTIER_DIR" || exit 6
  SLAB_MANIFEST="${VLLM029_DIR:-$HOME/glm-vllm029-build}/manifest.json"
  test -f "$SLAB_MANIFEST" || { echo "missing $SLAB_MANIFEST" >&2; exit 4; }
  SLAB_SALT=$(sha256sum "$SLAB_MANIFEST" | cut -d' ' -f1)
  KVTIER_MOUNTS=(-v "$KVTIER_DIR:/kvcache")
  KVTIER_ENV=(-e "PYTHONHASHSEED=${KVTIER_HASHSEED:-0}")
  KVTIER_ARGS=(--kv-transfer-config '{"kv_connector":"MultiNodeSlabConnector","kv_connector_module_path":"vllm.v1.kv_offload.tiering.multinode","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"MultiNodeSlabOffloadingSpec","spec_module_path":"vllm.v1.kv_offload.tiering.multinode","root_dir":"/kvcache","disk_bytes_per_rank":'"$KVTIER_DISK_BYTES"',"bounce_blocks":'"$KVTIER_BOUNCE"',"n_read_threads":'"$KVTIER_THREADS"',"n_write_threads":'"$KVTIER_THREADS"',"slab_salt":"'"$SLAB_SALT"'"}}')
fi

DRAFT_MOUNT=()
case "$SPEC_MODE" in
  none)   SPEC=() ;;
  dflash)
    [ -d /var/tmp/models/GLM-5.3-DFlash2-draft ] || {
      echo "SPEC_MODE=dflash but draft weights are not staged at /var/tmp/models/GLM-5.3-DFlash2-draft" >&2; exit 7; }
    DRAFT_MOUNT=(-v /var/tmp/models/GLM-5.3-DFlash2-draft:/models/dflash2-draft:ro)
    SPEC=(--speculative-config '{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":'"$DFLASH_K"',"draft_tensor_parallel_size":1,"attention_backend":"FLASH_ATTN"}') ;;
  *) echo "spec must be dflash|none" >&2; exit 2 ;;
esac
# Same concurrency rule as the selected launcher: speculative modes batch
# more sequences to amortise the fixed per-step cost; plain decode does not.
if [ "$SPEC_MODE" = "none" ]; then MAXSEQS="${MAXSEQS:-4}"; else MAXSEQS="${MAXSEQS:-12}"; fi
# As in the selected launcher: the MLA target uses fp8_ds_mla either way; with
# DFlash the drafter's sliding-window layers must be excluded from fp8
# (fp8_ds_mla is MLA-only), which is what 'fp8' + the skip list does.
KVDTYPE=fp8_ds_mla; KVSKIP=""
if [ "$SPEC_MODE" = "dflash" ]; then KVDTYPE=fp8; KVSKIP="--kv-cache-dtype-skip-layers sliding_window"; fi

# NCCL_HOTPLUG=1: load the hot-plug-aware external NCCL net plugin (github.com/ajclark, fork of
# NVIDIA's nccl-rdma-sharp-plugins) instead of the builtin IB net. The plugin can suspend and
# resume every RDMA object under live communicators so the ConnectX-7 can be powered off at
# idle (spark-idle.sh --down/--up) with the serving stack resident. HOTPLUG_SO is the aarch64
# build staged by rollout_dcp.sh; HOTPLUG_CTL is the host directory the control files live in
# (bind-mounted at the same path; spark-idle.sh writes <gen> suspend|resume into it).
NCCL_HOTPLUG="${NCCL_HOTPLUG:-0}"
HOTPLUG_SO="${HOTPLUG_SO:-$HOME/nccl-hotplug/libnccl-net-hotplug.so}"
HOTPLUG_PORT="${HOTPLUG_PORT:-5711}"   # the plugin's control endpoint, 127.0.0.1:port inside the node (host network namespace)
NET_ENV=(-e NCCL_NET=IB -e NCCL_NET_PLUGIN=none)
HOTPLUG_MOUNTS=()
# /dev/infiniband: the default lane hands the container static copies of the device nodes present at
# launch (--device). Under the hot-plug lane the adapters are removed and re-added while the container
# lives, so the host directory is bind-mounted instead (udev's re-created nodes stay visible) and the
# device cgroup allows the whole uverbs/umad major (231) plus misc (rdma_cm), whatever minor the
# re-added devices get.
IB_DEV=(--device /dev/infiniband:/dev/infiniband)
if [ "$NCCL_HOTPLUG" = 1 ]; then
  IB_DEV=(-v /dev/infiniband:/dev/infiniband --device-cgroup-rule 'c 231:* rwm' --device-cgroup-rule 'c 10:* rwm')
  [ -f "$HOTPLUG_SO" ] || { echo "NCCL_HOTPLUG=1 but $HOTPLUG_SO is missing" >&2; exit 7; }
  # dlopen search: NCCL_NET_PLUGIN=hotplug -> libnccl-net-hotplug.so on the library path
  HOTPLUG_MOUNTS=(-v "$HOTPLUG_SO:/usr/lib/aarch64-linux-gnu/libnccl-net-hotplug.so:ro")
  NET_ENV=(-e NCCL_NET_PLUGIN=hotplug -e "NCCL_HOTPLUG_CTL_PORT=$HOTPLUG_PORT")
fi

[ "${DRYRUN:-0}" = 1 ] || docker rm -f "$NAME" 2>/dev/null   # never touch a running container in a dry run

# DRYRUN=1 prints the docker command (shell-quoted) instead of running it.
run_docker() { if [ "${DRYRUN:-0}" = 1 ]; then printf '%q ' docker "$@"; echo; exit 0; fi; docker "$@"; }
run_docker run -d --name "$NAME" \
  --label "spark.vllm.upgrade=${UPGRADE_LABEL:-vllm029}" \
  --restart no --entrypoint vllm \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  --ulimit nofile=1048576:1048576 \
  --network host --ipc host --shm-size 10gb --gpus all \
  "${IB_DEV[@]}" \
  -v /var/tmp/models:/cache/huggingface \
  -v "$WEIGHTS:/models/glm-5.3:ro" \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  "${KVTIER_MOUNTS[@]}" \
  "${DRAFT_MOUNT[@]}" \
  "${HOTPLUG_MOUNTS[@]}" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e HF_HOME=/cache/huggingface \
  -e TRITON_CACHE_DIR=/cache/huggingface/.tritoncache-vllm029 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e VLLM_DEBUG_WORKSPACE=1 \
  -e VLLM_KV_CACHE_LAYOUT=BLHNC \
  "${KVTIER_ENV[@]}" \
  -e VLLM_ALLREDUCE_USE_FLASHINFER=0 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  "${NET_ENV[@]}" -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA='=roceP2p1s0f0,roceP2p1s0f1' \
  -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_SUBNET_PREFIX_LEN=24 \
  -e NCCL_IB_MERGE_NICS=1 \
  -e NCCL_IB_ADAPTIVE_ROUTING=0 \
  -e NCCL_SOCKET_IFNAME=$LAN_IF -e GLOO_SOCKET_IFNAME=$LAN_IF \
  -e TP_SOCKET_IFNAME=$LAN_IF -e MN_IF_NAME=$LAN_IF \
  -e NCCL_ALGO=Ring -e NCCL_RUNTIME_CONNECT=1 \
  -e NCCL_COLLNET_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_MNNVL_ENABLE=0 \
  -e NCCL_PAT_ENABLE=0 \
  -e NCCL_RMA_DISABLE=1 -e NCCL_NUM_RMA_CTX=0 -e NCCL_RMA_EAGER_INIT=0 \
  -e NCCL_GIN_ENABLE=0 \
  -e NCCL_MIN_CTAS=1 -e NCCL_MAX_CTAS=1 \
  -e NCCL_MAX_NCHANNELS=1 -e NCCL_MIN_NCHANNELS=1 \
  -e "NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-1}" \
  -e NCCL_CUMEM_ENABLE=1 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e NODE_RANK="$NODE_RANK" -e MASTER_ADDR="$HEAD_IP" \
  -e VLLM_HOST_IP="$HOST_IP" \
  "$IMAGE" \
  serve /models/glm-5.3 \
    --served-model-name glm-5.3 --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
    --enable-prefix-caching \
    --async-scheduling \
    "${SPEC[@]}" \
    --tensor-parallel-size 4 --pipeline-parallel-size 1 \
    --decode-context-parallel-size "$DCP_SIZE" --dcp-comm-backend ag_rs --no-dcp-q-replicate \
    --attention-backend FLASHINFER_MLA_SPARSE_SM120 \
    --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" \
    --max-num-batched-tokens "$MAXBATCHED" \
    --long-prefill-token-threshold 2048 \
    --override-generation-config '{"temperature":0.0,"top_p":1.0}' \
    --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
    --gpu-memory-utilization 0.91 --kv-cache-memory-bytes "$KVBYTES" \
    --kv-cache-dtype "$KVDTYPE" $KVSKIP \
    "${KVTIER_ARGS[@]}" \
    "${PROF_ARGS[@]}" \
    --distributed-executor-backend mp --compilation-config '{"cudagraph_mode":"FULL"}' \
    --nnodes 4 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MASTER_PORT" \
    $( [ "$HEADLESS" = 1 ] && echo --headless )

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP nccl_hotplug=$NCCL_HOTPLUG spec=$SPEC_MODE $( [ "$SPEC_MODE" = dflash ] && echo "k=$DFLASH_K" ) tp4 dcp$DCP_SIZE maxlen=$MAXLEN maxseqs=$MAXSEQS kvtier=$KVTIER/$KVTIER_MODE image=$IMAGE"
sleep 3
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited immediately; docker logs $NAME" >&2; exit 1; }
