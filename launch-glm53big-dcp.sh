#!/usr/bin/env bash
#
# GLM-5.3 (743B) Int4-Int8Mix, TP=4 + DCP=4 over the 200G RoCE ring.
#
# The point of this launcher: with DCP the MLA latent cache and the DSA indexer
# k-cache are SHARDED across the four ranks (global token position p lives on
# rank p % 4) instead of replicated. The same 8 GB per-rank pool therefore
# holds ~4x the context.
#
# Derived from launch-glm53big-dflash.sh (the selected 120k/8GB launcher).
# Everything about the cluster -- rank order, the switchless-ring NCCL block,
# the ten sm12x kernel overlays, cache_flusher, gpu-memory-utilization 0.91 --
# is carried over unchanged and must stay that way. Deltas, and why:
#
#  1. --decode-context-parallel-size 4 --dcp-comm-backend ag_rs.
#     ag_rs is not a preference: 'a2a' needs NCCL all-to-all, which on a
#     switchless ring pairs non-adjacent (uncabled) ranks and dies with
#     ibv_modify_qp 110. The patched backend refuses a2a at startup.
#
#  2. Thirteen bind mounts from $DCP_DIR, the DCP patch set. Two of them
#     (flashmla_sparse.py, sparse_attn_indexer.py) REPLACE the mounts that
#     came from $KERNELS_DIR -- the DCP versions are derived from those, so
#     mounting both would silently drop the DCP changes. The other eight
#     sm12x overlays are mounted from $KERNELS_DIR exactly as before.
#     Five of the mounts make the sparse-MLA target DCP-aware; the other eight
#     let the DFlash drafter's sliding-window KV group stay REPLICATED across
#     the DCP ranks (every rank keeps every position for its TP shard of
#     heads, which is today's layout) while the target's cache is sharded.
#
#  3. SPEC_MODE defaults to dflash (K=7, the selected production drafter).
#     The drafter's 8 KV heads at TP4 cannot be DCP-sharded (vLLM's GQA DCP
#     needs tp > kv_heads), so its group ignores DCP: no extra memory, no extra
#     collectives, the target's verify pass pays the DCP cost. 'none' exists
#     only to bisect (compare token hashes against the DCP1 launcher).
#     See docs/DESIGN.md sections 3.2 and 6.
#
#  4. No --kv-transfer-config. The LMCache connector is not DCP-aware (it
#     would store each rank's shard under the same key). Separate workstream.
#
#  5. max-model-len 262144, up from 120000. Sizing, per rank:
#       KV per token   ~61.5 kB (MLA 656 B + indexer 132 B, x78 layers)
#       8 GB pool      ~131k tokens per rank, ~525k across the group
#       262144 context 65.5k tokens per rank of a single full-length request
#     so one max-length request uses half the local pool and concurrency is
#     ~2x. 400k also fits (100k/rank) -- raise MAXLEN once a boot at 262144
#     has shown the real headroom. Workspaces scale with max-model-len:
#     the indexer gather buffer is 40 * max_model_len * 132 B (1.4 GB here vs
#     0.63 GB at 120k), while the sparse bf16 prefill workspace (5 *
#     max_model_len * 576 * 2 B, 0.69 GB at 120k) is no longer allocated at
#     all under DCP -- the patch skips it because the path that used it is
#     unreachable. Net workspace change vs the 120k launcher: about +0.1 GB.
#
#  6. max-num-batched-tokens 2048 (validated; 4096 untested), down from 8192. Under DCP the attention
#     kernel runs on the DCP-gathered head count (16 -> 64 per rank), so the
#     gathered query and the fp32 accumulator are 4x bigger per token:
#     8192 tokens x 64 heads x 512 x 4 B is 1 GiB of accumulator alone.
#     4096 halves that. Raise it back only with a measured profile run.
#
# usage: launch-glm53big-dcp.sh <rank 0-3> [dflash|none]   (default dflash)
set -uo pipefail

NODE_RANK="${1:?usage: launch-glm53big-dcp.sh <0|1|2|3> [dflash|none]}"
SPEC_MODE="${2:-dflash}"

IMAGE="${DCP_IMAGE:-vllm-glm52-b12x:dflash2-port2}"
NAME="vllm_glm53big"
PORT=8000
MASTER_PORT=29541
LAN_IF="enP7s7"
HEAD_IP="192.168.1.228"
KERNELS_DIR="${KERNELS_DIR:-$HOME/glm-triton}"
DCP_DIR="${DCP_DIR:-$HOME/glm-dcp}"
WEIGHTS=/var/tmp/models/GLM-5.3-Int4-Int8Mix

DCP_SIZE="${DCP_SIZE:-4}"
MAXLEN="${MAXLEN:-307200}"
# DFlash draft tokens per cycle. 7 is trained in (block_size 8); not a knob.
DFLASH_K="${DFLASH_K:-7}"
MAXBATCHED="${MAXBATCHED:-2048}"
# KV pool bytes per rank. Production uses 8e9. 6e9 with max-model-len 307200
# (396,715 tokens, 1.29x) is the validated NVMe-tier configuration of
# 2026-09-04: it leaves ~2 GB per rank over the 512k boot for the tier's
# bounce buffer and headroom. 512k needs 8.2e9 and works with zero headroom.
KVBYTES="${KVBYTES:-6000000000}"
# NVMe-durable KV cache (docs/NVME-DESIGN.md): KVTIER=1 adds the native
# OffloadingConnector with the multi-node worker-executed filesystem tier.
# KVTIER_CPU_BYTES is the CPU tier total across the four workers (pinned
# host memory, i.e. GPU memory on a Spark: 4e9 = 1 GB per rank), KVTIER_DIR
# the per-node NVMe directory that is bind-mounted at /kvcache.
KVTIER="${KVTIER:-1}"
KVTIER_CPU_BYTES="${KVTIER_CPU_BYTES:-4000000000}"
KVTIER_DIR="${KVTIER_DIR:-/var/tmp/kvcache}"
KVTIER_THREADS="${KVTIER_THREADS:-8}"
# direct = the disk is the offload store (bounce buffer only, no size cap on a
# reload); tiered = CPU cache tier + worker fs tier (reload capped by the tier).
# slab = fixed-size ring buffer on NVMe (KVTIER_DISK_BYTES per rank, LRU slot
# reuse, no janitor); direct = one file per block, unbounded; tiered = CPU
# cache tier + worker fs tier (reload capped by the tier).
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

# --- kernel preflight: the sm12x overlays that DCP does NOT replace.
KERNEL_FILES=(sparse_mla_kernels.py sparse_mla_env.py sm12x_sparse_mla_attn.py
  patch_flashmla_ops.py sm12x_deep_gemm_fallbacks.py sm12x_mqa.py
  b12x_sparse_helpers.py deepseek_v2.py)
for f in "${KERNEL_FILES[@]}"; do
  [ -f "$KERNELS_DIR/$f" ] || { echo "kernel overlay missing: $KERNELS_DIR/$f" >&2; exit 4; }
done
# --- DCP preflight: the thirteen patched files.
DCP_FILES=(flashmla_sparse.py sparse_attn_indexer.py sparse_utils.py indexer.py
  mla_attention.py
  kv_cache_interface.py kv_cache_utils.py kv_cache_coordinator.py
  block_table.py gpu_input_batch.py gpu_model_runner.py cp_utils.py
  flash_attn.py
  scheduler.py)
for f in "${DCP_FILES[@]}"; do
  [ -f "$DCP_DIR/$f" ] || { echo "DCP overlay missing: $DCP_DIR/$f" >&2; exit 4; }
done
grep -q "triton_filter_and_convert_dcp_index" "$DCP_DIR/sparse_utils.py" || {
  echo "$DCP_DIR/sparse_utils.py is not the DCP version" >&2; exit 5; }
grep -q "_merge_dcp_topk_global" "$DCP_DIR/sparse_attn_indexer.py" || {
  echo "$DCP_DIR/sparse_attn_indexer.py is not the DCP version" >&2; exit 5; }
grep -q "DCP overlay: hybrid-aware" "$DCP_DIR/scheduler.py" || {
  echo "$DCP_DIR/scheduler.py is not the DCP version" >&2; exit 5; }
grep -q "cp_world_size_for_kv_cache_spec" "$DCP_DIR/kv_cache_interface.py" || {
  echo "$DCP_DIR/kv_cache_interface.py is not the DCP version" >&2; exit 5; }
KVTIER_MOUNTS=(); KVTIER_ARGS=(); KVTIER_ENV=()
if [ "$KVTIER" = 1 ]; then
  [ -f "$DCP_DIR/multinode.py" ] || { echo "KVTIER=1 but $DCP_DIR/multinode.py is missing" >&2; exit 4; }
  grep -q "class MultiNodeOffloadingConnector" "$DCP_DIR/multinode.py" || {
    echo "$DCP_DIR/multinode.py is not the multi-node tier" >&2; exit 5; }
  [ -f "$DCP_DIR/offloading_scheduler.py" ] || { echo "KVTIER=1 but $DCP_DIR/offloading_scheduler.py is missing" >&2; exit 4; }
  grep -q "DCP overlay: eagle trailing block is revisited" "$DCP_DIR/offloading_scheduler.py" || {
    echo "$DCP_DIR/offloading_scheduler.py is not the patched connector scheduler" >&2; exit 5; }
  mkdir -p "$KVTIER_DIR" || { echo "cannot create $KVTIER_DIR" >&2; exit 6; }
  # Block hashes are chained from NONE_HASH, which the engine seeds from
  # os.urandom() unless PYTHONHASHSEED is set. A durable cache needs the
  # same keys after a restart, so the tier lane pins the seed.
  KVTIER_ENV=(-e "PYTHONHASHSEED=${KVTIER_HASHSEED:-0}")
  # The connector scheduler carries one fix: an eagle group's deferred trailing
  # block is stored on the next step instead of being skipped for good.
  KVTIER_MOUNTS=(-v "$DCP_DIR/multinode.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/tiering/multinode.py:ro"
                 -v "$DCP_DIR/offloading_scheduler.py:/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:ro"
                 -v "$KVTIER_DIR:/kvcache")
  case "$KVTIER_MODE" in
    slab) KVTIER_ARGS=(--kv-transfer-config '{"kv_connector":"MultiNodeSlabConnector","kv_connector_module_path":"vllm.v1.kv_offload.tiering.multinode","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"MultiNodeSlabOffloadingSpec","spec_module_path":"vllm.v1.kv_offload.tiering.multinode","root_dir":"/kvcache","disk_bytes_per_rank":'"$KVTIER_DISK_BYTES"',"bounce_blocks":'"$KVTIER_BOUNCE"',"n_read_threads":'"$KVTIER_THREADS"',"n_write_threads":'"$KVTIER_THREADS"'}}') ;;
    direct) KVTIER_ARGS=(--kv-transfer-config '{"kv_connector":"MultiNodeDirectConnector","kv_connector_module_path":"vllm.v1.kv_offload.tiering.multinode","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"MultiNodeDirectFsOffloadingSpec","spec_module_path":"vllm.v1.kv_offload.tiering.multinode","root_dir":"/kvcache","bounce_blocks":'"$KVTIER_BOUNCE"',"n_read_threads":'"$KVTIER_THREADS"',"n_write_threads":'"$KVTIER_THREADS"'}}') ;;
    tiered) KVTIER_ARGS=(--kv-transfer-config '{"kv_connector":"MultiNodeOffloadingConnector","kv_connector_module_path":"vllm.v1.kv_offload.tiering.multinode","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"MultiNodeTieringOffloadingSpec","spec_module_path":"vllm.v1.kv_offload.tiering.multinode","cpu_bytes_to_use":'"$KVTIER_CPU_BYTES"',"secondary_tiers":[{"type":"fs_worker","root_dir":"/kvcache","n_read_threads":'"$KVTIER_THREADS"',"n_write_threads":'"$KVTIER_THREADS"'}]}}') ;;
    *) echo "KVTIER_MODE must be slab|direct|tiered" >&2; exit 2 ;;
  esac
fi
grep -q "GlmMoeDsaForCausalLM" "$KERNELS_DIR/deepseek_v2.py" || {
  echo "overlay deepseek_v2.py does not define GlmMoeDsaForCausalLM" >&2; exit 6; }

VLLM="/usr/local/lib/python3.12/dist-packages/vllm"
MLA="$VLLM/v1/attention/backends/mla"
OPS="$VLLM/v1/attention/ops/deepseek_v4_ops"
LAYERS="$VLLM/model_executor/layers"
MODELS="$VLLM/model_executor/models"

DRAFT_MOUNT=()
case "$SPEC_MODE" in
  none)   SPEC=() ;;
  dflash)
    [ -d /var/tmp/models/GLM-5.3-DFlash2-draft ] || {
      echo "SPEC_MODE=dflash but draft weights are not staged at /var/tmp/models/GLM-5.3-DFlash2-draft" >&2; exit 7; }
    DRAFT_MOUNT=(-v /var/tmp/models/GLM-5.3-DFlash2-draft:/models/dflash2-draft:ro)
    SPEC=(--speculative-config '{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":'"$DFLASH_K"',"draft_tensor_parallel_size":1}') ;;
  mtp)
    echo "MTP was dropped from this launcher (operator decision 2026-09-04); use dflash or none." >&2
    exit 2 ;;
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

[ "${DRYRUN:-0}" = 1 ] || docker rm -f "$NAME" 2>/dev/null   # never touch a running container in a dry run

# DRYRUN=1 prints the docker command (shell-quoted) instead of running it.
run_docker() { if [ "${DRYRUN:-0}" = 1 ]; then printf '%q ' docker "$@"; echo; exit 0; fi; docker "$@"; }
run_docker run -d --name "$NAME" \
  --restart no \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  --ulimit nofile=1048576:1048576 \
  --network host --ipc host --shm-size 10gb --gpus all \
  --device /dev/infiniband:/dev/infiniband \
  -v /var/tmp/models:/cache/huggingface \
  -v "$WEIGHTS:/models/glm-5.3:ro" \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  -v "$KERNELS_DIR/sparse_mla_kernels.py:$MLA/sparse_mla_kernels.py:ro" \
  -v "$KERNELS_DIR/sparse_mla_env.py:$MLA/sparse_mla_env.py:ro" \
  -v "$KERNELS_DIR/sm12x_sparse_mla_attn.py:$MLA/sm12x_sparse_mla_attn.py:ro" \
  -v "$KERNELS_DIR/patch_flashmla_ops.py:$MLA/patch_flashmla_ops.py:ro" \
  -v "$KERNELS_DIR/sm12x_deep_gemm_fallbacks.py:$MLA/sm12x_deep_gemm_fallbacks.py:ro" \
  -v "$KERNELS_DIR/sm12x_mqa.py:$OPS/sm12x_mqa.py:ro" \
  -v "$KERNELS_DIR/b12x_sparse_helpers.py:$OPS/b12x_sparse_helpers.py:ro" \
  -v "$KERNELS_DIR/deepseek_v2.py:$MODELS/deepseek_v2.py:ro" \
  -v "$DCP_DIR/flashmla_sparse.py:$MLA/flashmla_sparse.py:ro" \
  -v "$DCP_DIR/sparse_utils.py:$MLA/sparse_utils.py:ro" \
  -v "$DCP_DIR/indexer.py:$MLA/indexer.py:ro" \
  -v "$DCP_DIR/sparse_attn_indexer.py:$LAYERS/sparse_attn_indexer.py:ro" \
  -v "$DCP_DIR/mla_attention.py:$LAYERS/attention/mla_attention.py:ro" \
  -v "$DCP_DIR/kv_cache_interface.py:$VLLM/v1/kv_cache_interface.py:ro" \
  -v "$DCP_DIR/kv_cache_utils.py:$VLLM/v1/core/kv_cache_utils.py:ro" \
  -v "$DCP_DIR/kv_cache_coordinator.py:$VLLM/v1/core/kv_cache_coordinator.py:ro" \
  -v "$DCP_DIR/block_table.py:$VLLM/v1/worker/block_table.py:ro" \
  -v "$DCP_DIR/gpu_input_batch.py:$VLLM/v1/worker/gpu_input_batch.py:ro" \
  -v "$DCP_DIR/gpu_model_runner.py:$VLLM/v1/worker/gpu_model_runner.py:ro" \
  -v "$DCP_DIR/cp_utils.py:$VLLM/v1/worker/cp_utils.py:ro" \
  -v "$DCP_DIR/flash_attn.py:$VLLM/v1/attention/backends/flash_attn.py:ro" \
  -v "$DCP_DIR/scheduler.py:$VLLM/v1/core/sched/scheduler.py:ro" \
  "${KVTIER_MOUNTS[@]}" \
  "${DRAFT_MOUNT[@]}" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e HF_HOME=/cache/huggingface \
  -e TRITON_CACHE_DIR=/cache/huggingface/.tritoncache \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e VLLM_DEBUG_WORKSPACE=1 \
  "${KVTIER_ENV[@]}" \
  -e GLM52_BIND_HOST_TRITON=1 \
  -e GLM52_MQA_LOGITS_TRITON=1 \
  -e GLM52_PAGED_MQA_TRITON=1 \
  -e GLM52_PAGED_MQA_TOPK_CHUNK_SIZE=8192 \
  -e GLM52_B12X_MLA=1 -e VLLM_DISABLE_FLASHINFER_AUTOTUNE=1 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_IB_DISABLE=0 \
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
  -e NCCL_IB_QPS_PER_CONNECTION=1 \
  -e NCCL_CUMEM_ENABLE=1 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e NODE_RANK="$NODE_RANK" -e MASTER_ADDR="$HEAD_IP" \
  -e VLLM_HOST_IP="$HOST_IP" \
  "$IMAGE" \
  vllm serve /models/glm-5.3 \
    --served-model-name glm-5.3 --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
    --enable-prefix-caching \
    --async-scheduling \
    "${SPEC[@]}" \
    --tensor-parallel-size 4 --pipeline-parallel-size 1 \
    --decode-context-parallel-size "$DCP_SIZE" --dcp-comm-backend ag_rs \
    --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" \
    --max-num-batched-tokens "$MAXBATCHED" \
    --long-prefill-token-threshold 2048 \
    --override-generation-config '{"temperature":0.0,"top_p":1.0}' \
    --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
    --gpu-memory-utilization 0.91 --kv-cache-memory-bytes "$KVBYTES" \
    --kv-cache-dtype "$KVDTYPE" $KVSKIP \
    "${KVTIER_ARGS[@]}" \
    --distributed-executor-backend mp --compilation-config '{"cudagraph_mode":"FULL"}' \
    --nnodes 4 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MASTER_PORT" \
    $( [ "$HEADLESS" = 1 ] && echo --headless )

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP spec=$SPEC_MODE $( [ "$SPEC_MODE" = dflash ] && echo "k=$DFLASH_K" ) tp4 dcp$DCP_SIZE maxlen=$MAXLEN maxseqs=$MAXSEQS kvtier=$KVTIER/$KVTIER_MODE image=$IMAGE"
sleep 3
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited immediately; docker logs $NAME" >&2; exit 1; }
