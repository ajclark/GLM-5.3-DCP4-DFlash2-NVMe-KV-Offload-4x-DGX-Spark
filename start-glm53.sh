#!/usr/bin/env bash
# Coordinated four-Spark start of GLM-5.3 Int4-Int8Mix using the installed, documented fast-boot profile.
# --dry-run validates and prints commands without stopping serving containers.
set -euo pipefail
[[ $# == 0 || ( $# == 1 && $1 == --dry-run ) ]] || {
  echo 'usage: bash start-glm53.sh [--dry-run]' >&2; exit 2;
}
hosts=(spark-06c4.local spark-365c.local spark-ddbf.local spark-a218.local)
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)
# Overridable for rollouts/benchmarks; defaults are the production profile.
image="${DCP_IMAGE:-spark-vllm:0.29.0-nvme4-stridefix-kvtier-verifycap12-e2-20261006}"
kvtier="KVTIER_DISK_BYTES=${KVTIER_DISK_BYTES:-400000000000} KVTIER_BOUNCE=${KVTIER_BOUNCE:-48}"
kvtier+=" KVTIER_READ_THREADS=${KVTIER_READ_THREADS:-16} KVTIER_WRITE_THREADS=${KVTIER_WRITE_THREADS:-8}"
kvtier+=" KVTIER_IO=${KVTIER_IO:-threaded} VLLM_DEV_MODE=${VLLM_DEV_MODE:-0} DFLASH_EAGER=${DFLASH_EAGER:-0} VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-0}"
[ -n "${MAXSEQS:-}" ] && kvtier+=" MAXSEQS=$MAXSEQS"
[ -n "${DRAFT_DIR:-}" ] && kvtier+=" DRAFT_DIR=$DRAFT_DIR"
# Draft-aware verification cap (docs/DFLASH2-VERIFY-CAP-RESULTS.md): on by default;
# VERIFY_CAP=0 turns it off without changing the image.
verify_cap="${VERIFY_CAP:-1}"
kvtier+=" VERIFY_CAP=$verify_cap"
[ -n "${VERIFY_CAP_FIXED:-}" ] && kvtier+=" VERIFY_CAP_FIXED=$VERIFY_CAP_FIXED"
[ -n "${VERIFY_CAP_CAPS:-}" ] && kvtier+=" VERIFY_CAP_CAPS=$VERIFY_CAP_CAPS"
# Batch verify cap, prefix-hit fix and prefill cadence (verify_cap_overlay image, verifycap6+);
# older images ignore them.
for v in VERIFY_CAP_BATCH_MAX VERIFY_CAP_POLICY PREFIX_HIT_FIX PREFILL_CADENCE PROFILER_DIR DFLASH_FC_SPLIT DFLASH_HEAD_FP8 DFLASH_DET_CONV VERIFY_CAP_LOCAL_DECIDE VERIFY_CAP_EARLY VERIFY_CAP_SYNC_EVERY VERIFY_CAP_GAP_EVENTS ROCE_DCP ROCE_TP ROCE_TP_MAX_SIZE ROCE_TP_EXCLUDE ROCE_TP_SPLIT VOCAB_ARGMAX L2_PREFETCH L2_PREFETCH_WINDOWS L2_PREFETCH_MB_B L2_PREFETCH_MB_C L2_PREFETCH_MB_D ROCE_SPIN_LIMIT ROCE_IDLE_MAX_NAP_US ROCE_DCP_HCAS ROCE_GATHER_MAX NCCL_CHANNELS NCCL_CTAS DCP_GLUE ROCE_OWNCOPY_EARLY ROCE_LARGE_BLOCKS NCCL_HCAS ROCE_PIPE ROCE_PIPE_CHUNK; do
  [ -n "${!v:-}" ] && kvtier+=" $v=${!v}"
done
# Voice-first scheduling (priority < 0 requests skip other prefills); VOICE_FIRST=0 = stock FCFS.
voice_first="${VOICE_FIRST:-1}"
kvtier+=" VOICE_FIRST=$voice_first"
# SPEC_MODE=none (no drafter) and KVTIER=0 exist for bulk data generation runs only.
spec="${SPEC_MODE:-dflash}"
kvtier_on="${KVTIER:-1}"
profile="DCP_IMAGE=$image WEIGHTS=/var/tmp/models/GLM-5.3-Int4-Int8Mix DCP_SIZE=2 MAXLEN=180224 KVBYTES=6000000000 DFLASH_K=${DFLASH_K:-7} KVTIER=$kvtier_on KVTIER_MODE=slab $kvtier KVTIER_DIR=/var/tmp/kvcache-vllm029"

# Preflight every host before any interruption; fail if an old launcher would
# silently lose the fast loader, the checkpoint, or durable-KV isolation.
# NCCL's default NIC list includes rocep1s0f0 (item 2 E3): every node needs its IPv4 RoCE v2 GID at index 3
# (NetworkManager profile roce-p0-twin), or NCCL cannot connect and the engine never comes up.
if [[ "${NCCL_HCAS:-rocep1s0f0}" == *rocep1s0f0* ]]; then
  for host in "${hosts[@]}"; do
    gid=$(ssh "${ssh_opts[@]}" "$host" 'cat /sys/class/infiniband/rocep1s0f0/ports/1/gids/3 2>/dev/null; cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/3 2>/dev/null' | tr '\n' ' ')
    [[ $gid == *ffff:c0a8:* && $gid == *"RoCE v2"* ]] || {
      echo "$host: rocep1s0f0 has no IPv4 RoCE v2 GID at index 3 ($gid); bring up roce-p0-twin or set NCCL_HCAS='=roceP2p1s0f0,roceP2p1s0f1' ROCE_TP_EXCLUDE=roceP2p1s0f1" >&2; exit 1; }
  done
  echo "rocep1s0f0 IPv4 GIDs present on all four Sparks"
fi
for rank in 0 1 2 3; do
  command=$(ssh "${ssh_opts[@]}" "${hosts[$rank]}" \
    "DRYRUN=1 $profile bash ~/glm53big/launch-glm53big-dcp.sh $rank $spec")
  required_list=('--load-format nvme' 'NVME_STREAM_BACKEND=coalesced'
      'NVME_STREAM_DEVICE=cuda' 'NVME_STREAM_BATCH_BYTES=134217728'
      'GLM-5.3-Int4-Int8Mix:/models/glm-5.3:ro' '--decode-context-parallel-size 2')
  [[ $kvtier_on == 0 ]] || required_list+=('MultiNodeSlabConnector' '/var/tmp/kvcache-vllm029:/kvcache')
  [[ $verify_cap != 1 || $spec != dflash ]] || required_list+=('VLLM_VERIFY_CAP=1' 'VLLM_VERIFY_CAP_CAL=/opt/verify-cap/cal.json')
  [[ $voice_first != 1 ]] || required_list+=('--scheduling-policy priority' 'voice_first.VoiceFirstAsyncScheduler')
  [[ ${ROCE_DCP:-1} != 1 || $spec != dflash ]] || required_list+=('GLM_ROCE_GROUPS=dcp' 'B12X_ROCE_HCA=rocep1s0f1')
  [[ ${ROCE_TP:-1} != 1 || $spec != dflash ]] || required_list+=('GLM_ROCE_TP_RING=1')
  [[ $verify_cap != 1 || $spec != dflash || ${VERIFY_CAP_LOCAL_DECIDE:-1} != 1 ]] || required_list+=('VLLM_VERIFY_CAP_LOCAL_DECIDE=1')
  [[ ${L2_PREFETCH:-1} != 1 ]] || required_list+=('VLLM_L2_PREFETCH=1')
  [[ ${VOCAB_ARGMAX:-1} != 1 ]] || required_list+=('VLLM_VOCAB_PARALLEL_ARGMAX=1')
  [[ ${DCP_GLUE:-1} != 1 ]] || required_list+=('VLLM_DCP_GLUE=1' 'GLM_ROCE_GATHER_MAX_SIZE=36MiB')
  [[ ${ROCE_OWNCOPY_EARLY:-1} != 1 ]] || required_list+=('GLM_ROCE_OWNCOPY_EARLY=1')
  for required in "${required_list[@]}"; do
    [[ $command == *"$required"* ]] || {
      echo "${hosts[$rank]} launcher missing: $required" >&2; exit 1;
    }
  done
  echo "${hosts[$rank]}: profile verified"
  if [[ ${1:-} == --dry-run ]]; then printf '%s\n' "$command"; fi
done
[[ ${1:-} != --dry-run ]] || exit 0

pids=()
for host in "${hosts[@]}"; do
  ssh "${ssh_opts[@]}" "$host" \
    'if docker container inspect vllm_glm53big >/dev/null 2>&1; then docker stop -t 10 vllm_glm53big; fi' &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
for rank in 3 2 1 0; do
  ssh "${ssh_opts[@]}" "${hosts[$rank]}" \
    "bash ~/glm53big/start-flusher.sh && $profile bash ~/glm53big/launch-glm53big-dcp.sh $rank $spec"
done
deadline=$((SECONDS+600))
until ssh "${ssh_opts[@]}" "${hosts[0]}" \
    'curl -fsS --max-time 3 http://127.0.0.1:8000/health >/dev/null'; do
  if (( SECONDS >= deadline )); then
    echo 'Startup timed out; inspect docker logs vllm_glm53big on all four Sparks.' >&2
    exit 1
  fi
  sleep 2
done
for host in "${hosts[@]}"; do
  ssh "${ssh_opts[@]}" "$host" "pkill -f '[c]ache_flusher.sh' || true"
done
echo 'Ready: http://spark-06c4.local:8000/v1 — model glm-5.3'
