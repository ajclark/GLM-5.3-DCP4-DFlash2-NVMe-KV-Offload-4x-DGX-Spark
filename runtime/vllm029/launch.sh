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

IMAGE="${DCP_IMAGE:-spark-vllm:0.29.0-nvme4-stridefix-kvtier-verifycap12-e2-20261006}"
NAME="vllm_glm53big"
PORT=8000
MASTER_PORT=29541
LAN_IF="enP7s7"
HEAD_IP="192.168.1.228"
WEIGHTS="${WEIGHTS:-/var/tmp/models/GLM-5.3-Int4-Int8Mix}"

DCP_SIZE="${DCP_SIZE:-2}"
MAXLEN="${MAXLEN:-180224}"
# The SM121 persistent_topk decode kernel raises (and kills the engine) past ~406K logits
# width when <= 4 indexer rows run, which the verify cap's K=1/3 steps do.
[ "$MAXLEN" -le 400000 ] || { echo "MAXLEN $MAXLEN > 400000: SM121 persistent_topk limit" >&2; exit 2; }
# DFlash draft tokens per cycle. 7 is trained in (block_size 8); not a knob.
DFLASH_K="${DFLASH_K:-7}"
MAXBATCHED="${MAXBATCHED:-2048}"
# Preserve the selected 6 GB/rank allocation and 180224-token serving window.
KVBYTES="${KVBYTES:-6000000000}"
# Worker-local durable NVMe slab, with bounded pinned transfer buffers.
KVTIER="${KVTIER:-1}"
# Durable KV belongs to the checkpoint that wrote it: one slab directory per checkpoint.
KVTIER_DIR="${KVTIER_DIR:-/var/tmp/kvcache-vllm029}"
KVTIER_THREADS="${KVTIER_THREADS:-8}"   # KVTIER_READ_THREADS defaults to 16 below
# Fixed-size ring buffer per rank; slot reuse bounds disk consumption.
KVTIER_MODE="${KVTIER_MODE:-slab}"
KVTIER_BOUNCE="${KVTIER_BOUNCE:-48}"
KVTIER_DISK_BYTES="${KVTIER_DISK_BYTES:-400000000000}"
# Per-direction slab I/O threads (default: KVTIER_THREADS for both) and the slab I/O
# engine ("threaded" = O_DIRECT + per-thread CUDA streams, "bounce" = step-driven
# bounce buffer; empty = the connector's default).
KVTIER_READ_THREADS="${KVTIER_READ_THREADS:-16}"
KVTIER_WRITE_THREADS="${KVTIER_WRITE_THREADS:-$KVTIER_THREADS}"
KVTIER_IO="${KVTIER_IO:-threaded}"
# 1 exposes vLLM's dev endpoints (/reset_prefix_cache for GPU-only cache resets in
# benchmarks). Off in production.
VLLM_DEV_MODE="${VLLM_DEV_MODE:-0}"
# Draft-aware verification cap (verify_cap_overlay image; docs/DFLASH2-VERIFY-CAP-RESULTS.md).
# On by default; VERIFY_CAP=0 disables it, VERIFY_CAP_FIXED=K forces a cap for A/B runs.
# VERIFY_CAP_BATCH_MAX (default 4) = largest batch that verifies one shorter K together
# (also sets which batch graphs are captured; 1 = single requests only). Runtime control:
# $VERIFY_CAP_LIVE_DIR/control.json on rank 0 (mode auto|fixed|off, fixed_k, batch, policy,
# costs), re-read on change without a restart; rank 0 writes periods.json there.
VERIFY_CAP_ENV=()
VERIFY_CAP_LIVE_DIR="${VERIFY_CAP_LIVE_DIR:-$HOME/verify-cap-live}"
if [ "${VERIFY_CAP:-1}" = 1 ] && [ "$SPEC_MODE" = dflash ]; then
  [ "${DRYRUN:-0}" = 1 ] || mkdir -p "$VERIFY_CAP_LIVE_DIR" || exit 6
  VERIFY_CAP_ENV=(-e VLLM_VERIFY_CAP=1 -e VLLM_VERIFY_CAP_CAL=/opt/verify-cap/cal.json
    -e VLLM_VERIFY_CAP_COSTS=/opt/verify-cap/costs.json
    -e "VLLM_VERIFY_CAP_BATCH_MAX=${VERIFY_CAP_BATCH_MAX:-4}"
    -e "VLLM_VERIFY_CAP_POLICY=${VERIFY_CAP_POLICY:-lambda}"
    -e VLLM_VERIFY_CAP_CONTROL=/opt/verify-cap-live/control.json
    -v "$VERIFY_CAP_LIVE_DIR:/opt/verify-cap-live")
  [ -n "${VERIFY_CAP_FIXED:-}" ] && VERIFY_CAP_ENV+=(-e "VLLM_VERIFY_CAP_FIXED=$VERIFY_CAP_FIXED")
  [ -n "${VERIFY_CAP_CAPS:-}" ] && VERIFY_CAP_ENV+=(-e "VLLM_VERIFY_CAP_CAPS=$VERIFY_CAP_CAPS")
  # Step 4 (verifycap10+, results/step4-hostgap-20261001): rank-local K decision with rank 0's
  # selector confidences broadcast on the GPU instead of the per-step gloo broadcast (default on:
  # GPU gap before each verify 1.4 -> 0.4 ms); early decision (never triggers under async
  # scheduling, off) and GPU idle-gap timing events (diagnostic, off). LOCAL_DECIDE and SYNC_EVERY
  # must match on every node.
  VERIFY_CAP_ENV+=(-e "VLLM_VERIFY_CAP_LOCAL_DECIDE=${VERIFY_CAP_LOCAL_DECIDE:-1}"
    -e "VLLM_VERIFY_CAP_EARLY=${VERIFY_CAP_EARLY:-0}"
    -e "VLLM_VERIFY_CAP_SYNC_EVERY=${VERIFY_CAP_SYNC_EVERY:-32}"
    -e "VLLM_VERIFY_CAP_GAP_EVENTS=${VERIFY_CAP_GAP_EVENTS:-0}")
fi
# Prefix-hit fix (verify_cap_overlay image): only the DFlash draft group takes the EAGLE
# last-block drop, and the sliding-window finder drops before aligning. PREFIX_HIT_FIX=0 = stock.
# Prefill cadence: with no voice request active, prefill chunks run on every Nth step while
# others decode (PREFILL_CADENCE=1 = off).
SCHED_TUNING_ENV=(-e "VLLM_PREFIX_HIT_FIX=${PREFIX_HIT_FIX:-1}" -e "VLLM_PREFILL_CADENCE=${PREFILL_CADENCE:-1}")
# Drafter diet (verify_cap_overlay image): DFLASH_FC_SPLIT=1 shards the drafter's fc over TP
# (one all-gather instead of every rank reading the whole matrix). An int8 drafter is chosen
# with DRAFT_DIR (default /var/tmp/models/GLM-5.3-DFlash2-draft-int8; the BF16 original is
# /var/tmp/models/GLM-5.3-DFlash2-draft). DFLASH_FC_SPLIT / DFLASH_HEAD_FP8 default on; 0 turns either off.
# DFLASH_HEAD_FP8=1 gives the drafter's candidate top-k an fp8 copy of the target lm_head.
SCHED_TUNING_ENV+=(-e "VLLM_DFLASH_FC_SPLIT=${DFLASH_FC_SPLIT:-1}" -e "VLLM_DFLASH_HEAD_FP8=${DFLASH_HEAD_FP8:-1}")
# DFLASH_DET_CONV=1 (default): the int8 drafter's replicated conv projections use Marlin's
# deterministic reduce instead of atomic-add split-K, so every TP rank drafts identical tokens.
SCHED_TUNING_ENV+=(-e "VLLM_DFLASH_DET_CONV=${DFLASH_DET_CONV:-1}")
# RoCEnante one-shot collectives for the DCP pairs (verifycap8+ image; docs/ROCENANTE-REVIEW.md).
# Each DCP pair (06c4-365c, ddbf-a218) is a direct cable; rocep1s0f1 faces the partner on all
# four Sparks and NCCL never opens it. Default on; ROCE_DCP=0 = NCCL.
# ROCE_TP=1 (default; ring image verifycap9+, results/step7-ring-allreduce-20261001): TP all-reduces up to
# ROCE_TP_MAX_SIZE (default 1MiB, i.e. decode and verify batches) take the ring transport
# (glm_roce/ring.py): RoCEnante's kernels, each rank writes both ring neighbours and forwards
# one, over roceP2p1s0f0/f1 (discovered from the GIDs; rocep1s0f1 stays the DCP pairs').
# Larger all-reduces (prefill) stay on NCCL.
# Prefill step 1c (results/prefill-item2-20261006/REPORT.md; default since
# 2026-10-06): the DCP pairs' RoCEnante gather limit is 36MiB, so prefill-size DCP gathers and reduce-scatters
# (<= 36 MiB per rank at 2048-token chunks) leave NCCL, and RoCEnante stripes every payload over both PCIe twins of
# the partner link (rocep1s0f1 on domain 0000, roceP2p1s0f1 on 0002). Byte-identical to NCCL (pair tests); prefill
# +10-11% at 4K-60K; decode unchanged. Pinned memory per rank = 6 x max(1MiB, ROCE_GATHER_MAX) = 216 MiB.
# Previous behaviour: ROCE_GATHER_MAX=4MiB ROCE_DCP_HCAS=rocep1s0f1. NCCL_CHANNELS / NCCL_CTAS set NCCL's channel and
# CTA counts (prefill TP all-reduces, TP all-gathers). Default 2 since 2026-10-06 (item 2 E3 bench: 25 MB all-reduce
# 3.17 -> 2.88 ms on the one x4 link; prefill +1.5-2.7% live; summation-order-level, inside the A/A envelope);
# previous behaviour NCCL_CHANNELS=1 NCCL_CTAS=1. NCCL_HCAS is NCCL's NIC list. Default since 2026-10-06 (item 2 E3,
# results/prefill-item2-20261006/REPORT.md): all four functions, merged into one virtual NIC per PCIe link
# (roceP2p1s0f0+roceP2p1s0f1 on domain 0002, rocep1s0f0+rocep1s0f1 on domain 0000), one NCCL channel on each: the
# prefill TP all-reduce uses both x4 links (25 MB: 2.88 -> 1.81 ms, bit-identical output; prefill +4.7-5.1%). Needs
# the persistent NetworkManager profile roce-p0-twin (static IPv4 on enp1s0f0np0, MTU 9000) on every node;
# start-glm53.sh checks it. Previous behaviour: NCCL_HCAS='=roceP2p1s0f0,roceP2p1s0f1'.
# Item 2 E2 (verifycap12+ image, glm_roce/gather_v2.py): ROCE_OWNCOPY_EARLY=1 copies a prefill-size gather's local
# shard into the output while the NIC moves the payload (bit-identical; -0.3 ms per 36 MiB collective in the pair
# test; prefill +2% on top of E1; default on since 2026-10-06); ROCE_LARGE_BLOCKS (8/16/32) is the grid for those
# gathers (no gain measured: keep 8). Shards < 4 MiB (decode) keep the vendored kernel either way.
# E2 pipelining (verifycap13+ image, glm_roce/pipe.py + _pipe_proxy.c): ROCE_PIPE=1 sends DCP gathers >= 4 MiB through
# a second, eager-only runtime whose kernel stages, sends and copies out in ROCE_PIPE_CHUNK-byte chunks (default 8 MiB)
# so the wire overlaps staging and copy-out; decode stays on the vendored runtime. Bit-identical output.
ROCE_ENV=()
roce_groups=()
[ "${ROCE_DCP:-1}" = 1 ] && roce_groups+=(dcp)
[ "${ROCE_TP:-1}" = 1 ] && roce_groups+=(tp)
if [ ${#roce_groups[@]} -gt 0 ]; then
  ROCE_ENV=(-e GLM_ROCE_ALLREDUCE=1 -e "GLM_ROCE_GROUPS=$(IFS=,; echo "${roce_groups[*]}")" -e GLM_ROCE_REQUIRE=1
    -e "B12X_ROCE_HCA=${ROCE_DCP_HCAS:-rocep1s0f1,roceP2p1s0f1}" -e B12X_ROCE_GID_INDEX=3
    -e "B12X_ROCE_SPIN_LIMIT=${ROCE_SPIN_LIMIT:-90000000}"
    -e "B12X_ROCE_IDLE_MAX_NAP_US=${ROCE_IDLE_MAX_NAP_US:-5000}"
    -e B12X_DISABLE_CUTLASS_RUNTIME_PATCHES=1
    -e GLM_ROCE_MAX_SIZE=1MiB -e "GLM_ROCE_GATHER_MAX_SIZE=${ROCE_GATHER_MAX:-36MiB}"
    -e "GLM_ROCE_OWNCOPY_EARLY=${ROCE_OWNCOPY_EARLY:-1}" -e "GLM_ROCE_LARGE_BLOCKS=${ROCE_LARGE_BLOCKS:-8}"
    -e "GLM_ROCE_PIPE=${ROCE_PIPE:-0}" -e "GLM_ROCE_PIPE_CHUNK_BYTES=${ROCE_PIPE_CHUNK:-8388608}"
    -e B12X_COMPILE_CACHE_DIR=/nvme-artifacts/b12x-compile)
  if [ "${ROCE_TP:-1}" = 1 ]; then
    # The NIC reaches the GB10 through two PCIe Gen5 x4 links (domains 0000 and 0002, ~13.6 GB/s
    # each way). roceP2p1s0f0/f1 share 0002, so a ring on both moves 3 payloads per op through
    # one x4. Excluding roceP2p1s0f1 puts the DCP-pair edge on rocep1s0f1 (0000): each node's cw
    # and ccw links then use different PCIe links; split mode balances the bytes over both
    # (docs/RDMA-COLLECTIVES-LATENCY.md: 49.6 -> 38.5 us at 8 tokens, 126.9 -> 81.9 at 32).
    ROCE_ENV+=(-e GLM_ROCE_TP_RING=1 -e "GLM_ROCE_TP_MAX_SIZE=${ROCE_TP_MAX_SIZE:-1MiB}"
      -e "GLM_ROCE_RING_EXCLUDE=${ROCE_TP_EXCLUDE:-roceP2p1s0f1,rocep1s0f0}" -e "GLM_ROCE_RING_SPLIT=${ROCE_TP_SPLIT:-1}")
  fi
fi
# Step 5 (glm_fast; verifycap10+ image, results/step5-l2-argmax-20261001), both default on: exact
# vocab-parallel target argmax (VOCAB_ARGMAX=0|1|check) and L2 prefetch of the next weights in
# collective windows (L2_PREFETCH=1; windows B,C,D switchable at runtime through
# ~/verify-cap-live/glm_fast_l2pf.json on every node; no file = all captured windows on).
FAST_ENV=(-e "VLLM_VOCAB_PARALLEL_ARGMAX=${VOCAB_ARGMAX:-1}" -e "VLLM_L2_PREFETCH=${L2_PREFETCH:-1}")
# Item 2 E1 (dcpglue image+, results/prefill-item2-20261006): DCP_GLUE=1 removes the layout copies around the
# DCP collectives, bit-identically: middle-dim gathers as 2-D last-dim gathers, the combine's correction writes
# head-major (no reduce-scatter pre-copy), the reduce-scatter add writes the output layout, and the backend's
# redundant full-output masked_fill_ is skipped. Must match on every rank; older images ignore it. Default on since
# 2026-10-06: prefill +6.0-6.4% on top of step 1c, byte-identical (pair + GPU tests), decode unchanged.
FAST_ENV+=(-e "VLLM_DCP_GLUE=${DCP_GLUE:-1}")
[ -n "${L2_PREFETCH_WINDOWS:-}" ] && FAST_ENV+=(-e "VLLM_L2_PREFETCH_WINDOWS=$L2_PREFETCH_WINDOWS")
for w in B C D; do v="L2_PREFETCH_MB_$w"; [ -n "${!v:-}" ] && FAST_ENV+=(-e "VLLM_L2_PREFETCH_MB_$w=${!v}"); done
[ "${L2_PREFETCH:-1}" = 1 ] && FAST_ENV+=(-e VLLM_L2_PREFETCH_CONTROL=/opt/verify-cap-live/glm_fast_l2pf.json)
# Voice-first scheduling (verify_cap_overlay/vllm/v1/core/sched/voice_first.py): requests sent
# with priority < 0 (e.g. a voice client) never wait behind other requests' prefills. On by default;
# VOICE_FIRST=0 restores the stock FCFS scheduler.
SCHED_ARGS=()
if [ "${VOICE_FIRST:-1}" = 1 ]; then
  SCHED_ARGS=(--scheduling-policy priority
    --scheduler-cls vllm.v1.core.sched.voice_first.VoiceFirstAsyncScheduler)
fi

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

# Coalesced original-checkpoint loading; see docs/stack/COALESCED-LOADER-IMPLEMENTATION.md.
# Persist compiler caches as well as the separate durable KV slabs.
NVME_CACHE_ROOT="${NVME_CACHE_ROOT:-/var/tmp/nvme-loader}"
NVME_MOUNTS=(-v "$NVME_CACHE_ROOT:/nvme-artifacts"
  -v "$NVME_CACHE_ROOT/vllm-cache:/root/.cache/vllm"
  -v "$NVME_CACHE_ROOT/torchinductor-cache:/tmp/torchinductor_root"
  -v "$NVME_CACHE_ROOT/flashinfer-cache:/root/.cache/flashinfer"
  -v "$NVME_CACHE_ROOT/cuda-cache:/root/.nv/ComputeCache")
NVME_ENV=(-e NVME_ARTIFACT_ROOT=/nvme-artifacts -e NVME_LOADER_MODE=stream
  -e NVME_STREAM_BACKEND=coalesced -e NVME_STREAM_DEVICE=cuda
  -e NVME_STREAM_BATCH_BYTES=134217728
  -e NVME_RUNTIME_ID=stridefix-20260920-coalesced-v1
  -e NVME_GENERATION=glm53-int4int8mix-fastboot-20260920
  -e CUDA_CACHE_PATH=/root/.nv/ComputeCache -e CUDA_CACHE_MAXSIZE=4294967296)

# Fail before stopping anything if the pinned runtime is not installed.
docker image inspect "$IMAGE" >/dev/null || exit 4
KVTIER_MOUNTS=(); KVTIER_ARGS=(); KVTIER_ENV=(); PROF_ARGS=()
if [ -n "${PROFILER_DIR:-}" ]; then
  # No stack capture and no stop-time CUDA-time table: on 2026-10-06 the table's key_averages pass on rank 0
  # (a 52 MB trace) pushed the node into a 13 GB swap storm and wedged the engine (results/prefill-item2-20261006).
  PROF_ARGS=(--profiler-config.profiler=torch "--profiler-config.torch_profiler_dir=$PROFILER_DIR"
    --profiler-config.torch_profiler_with_stack=false --profiler-config.torch_profiler_dump_cuda_time_total=false)
fi
if [ "$KVTIER" = 1 ]; then
  [ "$KVTIER_MODE" = slab ] || { echo "vLLM 0.29 supports KVTIER_MODE=slab" >&2; exit 2; }
  mkdir -p "$KVTIER_DIR" || exit 6
  SLAB_MANIFEST="${VLLM029_DIR:-$HOME/glm-vllm029-build}/manifest.json"
  test -f "$SLAB_MANIFEST" || { echo "missing $SLAB_MANIFEST" >&2; exit 4; }
  SLAB_SALT=$(sha256sum "$SLAB_MANIFEST" | cut -d' ' -f1)
  KVTIER_MOUNTS=(-v "$KVTIER_DIR:/kvcache")
  KVTIER_ENV=(-e "PYTHONHASHSEED=${KVTIER_HASHSEED:-0}")
  KVTIER_IO_JSON=""; [ -n "$KVTIER_IO" ] && KVTIER_IO_JSON="\"io_engine\":\"$KVTIER_IO\","
  KVTIER_ARGS=(--kv-transfer-config '{"kv_connector":"MultiNodeSlabConnector","kv_connector_module_path":"vllm.v1.kv_offload.tiering.multinode","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"MultiNodeSlabOffloadingSpec","spec_module_path":"vllm.v1.kv_offload.tiering.multinode","root_dir":"/kvcache","disk_bytes_per_rank":'"$KVTIER_DISK_BYTES"',"bounce_blocks":'"$KVTIER_BOUNCE"',"n_read_threads":'"$KVTIER_READ_THREADS"',"n_write_threads":'"$KVTIER_WRITE_THREADS"','"$KVTIER_IO_JSON"'"slab_salt":"'"$SLAB_SALT"'"}}')
fi

DRAFT_MOUNT=()
case "$SPEC_MODE" in
  none)   SPEC=() ;;
  dflash)
    DRAFT_DIR="${DRAFT_DIR:-/var/tmp/models/GLM-5.3-DFlash2-draft-int8}"
    [ -d "$DRAFT_DIR" ] || {
      echo "SPEC_MODE=dflash but draft weights are not staged at $DRAFT_DIR" >&2; exit 7; }
    DRAFT_MOUNT=(-v "$DRAFT_DIR:/models/dflash2-draft:ro")
    # DFLASH_EAGER=1 runs only the drafter eagerly (no compile/CUDA graphs) for debug hooks.
    DFLASH_EAGER_JSON=""; [ "${DFLASH_EAGER:-0}" = 1 ] && DFLASH_EAGER_JSON=',"enforce_eager":true'
    SPEC=(--speculative-config '{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":'"$DFLASH_K"',"draft_tensor_parallel_size":1,"attention_backend":"FLASH_ATTN"'"$DFLASH_EAGER_JSON"'}') ;;
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
  "${NVME_MOUNTS[@]}" \
  "${NVME_ENV[@]}" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e VLLM_SERVER_DEV_MODE="$VLLM_DEV_MODE" \
  "${VERIFY_CAP_ENV[@]}" \
  "${SCHED_TUNING_ENV[@]}" \
  "${ROCE_ENV[@]}" \
  "${FAST_ENV[@]}" \
  -e VLLM_NO_USAGE_STATS=1 -e VLLM_DO_NOT_TRACK=1 -e DO_NOT_TRACK=1 \
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
  -e "NCCL_IB_HCA=${NCCL_HCAS:-=roceP2p1s0f0,roceP2p1s0f1,rocep1s0f0,rocep1s0f1}" \
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
  -e "NCCL_MIN_CTAS=${NCCL_CTAS:-2}" -e "NCCL_MAX_CTAS=${NCCL_CTAS:-2}" \
  -e "NCCL_MAX_NCHANNELS=${NCCL_CHANNELS:-2}" -e "NCCL_MIN_NCHANNELS=${NCCL_CHANNELS:-2}" \
  -e "NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-1}" \
  -e NCCL_CUMEM_ENABLE=1 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e NODE_RANK="$NODE_RANK" -e MASTER_ADDR="$HEAD_IP" \
  -e VLLM_HOST_IP="$HOST_IP" \
  "$IMAGE" \
  serve /models/glm-5.3 \
    --load-format nvme \
    --served-model-name glm-5.3 --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
    --enable-prefix-caching \
    --async-scheduling \
    "${SCHED_ARGS[@]}" \
    "${SPEC[@]}" \
    --tensor-parallel-size 4 --pipeline-parallel-size 1 \
    --decode-context-parallel-size "$DCP_SIZE" --dcp-comm-backend ag_rs --no-dcp-q-replicate \
    --attention-backend FLASHINFER_MLA_SPARSE_SM120 \
    --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" \
    --max-num-batched-tokens "$MAXBATCHED" \
    --long-prefill-token-threshold 2048 \
    --override-generation-config '{"temperature":0.0,"top_p":1.0}' \
    --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
    --gpu-memory-utilization 0.90 --kv-cache-memory-bytes "$KVBYTES" \
    --kv-cache-dtype "$KVDTYPE" $KVSKIP \
    "${KVTIER_ARGS[@]}" \
    "${PROF_ARGS[@]}" \
    --distributed-executor-backend mp $( [ "${VLLM_ENFORCE_EAGER:-0}" = 1 ] && echo --enforce-eager || echo --compilation-config '{"cudagraph_mode":"FULL"}' ) \
    --nnodes 4 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MASTER_PORT" \
    $( [ "$HEADLESS" = 1 ] && echo --headless )

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP nccl_hotplug=$NCCL_HOTPLUG spec=$SPEC_MODE $( [ "$SPEC_MODE" = dflash ] && echo "k=$DFLASH_K" ) tp4 dcp$DCP_SIZE maxlen=$MAXLEN maxseqs=$MAXSEQS kvtier=$KVTIER/$KVTIER_MODE image=$IMAGE"
sleep 3
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited immediately; docker logs $NAME" >&2; exit 1; }
