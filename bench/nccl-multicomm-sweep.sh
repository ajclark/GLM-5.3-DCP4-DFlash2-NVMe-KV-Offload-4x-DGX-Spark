#!/usr/bin/env bash
# Take the serving stack down, run the multi-communicator NCCL microbenchmark
# in the requested variants on all four nodes, collect rank 0's results, and
# roll the compaction stack back. Usage: ./bench/nccl-multicomm-sweep.sh [variants...]
set -uo pipefail
WS="$(cd "$(dirname "$0")/.." && pwd)"; cd "$WS"
VARIANTS=("${@:-default}"); [ $# -eq 0 ] && VARIANTS=(default ll)
OUT="$WS/results/nccl-multicomm"; mkdir -p "$OUT"
NS="${NCCL_BENCH_NS:-1,2,4,8}"
HOSTS=(spark-06c4 spark-365c spark-ddbf spark-a218)
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/sweep.log"; }
sshq() { ssh -o BatchMode=yes -o ConnectTimeout=8 "napta2k@$1.local" "$2"; }
say "== NCCL multi-communicator sweep: ${VARIANTS[*]}"
say "-- stopping the serving stack"
for h in "${HOSTS[@]}"; do sshq "$h" "docker rm -f vllm_glm53big >/dev/null 2>&1; pkill -f '[c]ache_flusher.sh' 2>/dev/null; true" & done; wait
sleep 5
for h in "${HOSTS[@]}"; do scp -q "$WS/bench/nccl_multicomm.py" "napta2k@$h.local:glm53big/nccl_multicomm.py"; scp -q "$WS/bench/run-nccl-multicomm.sh" "napta2k@$h.local:glm53big/run-nccl-multicomm.sh"; sshq "$h" "chmod +x glm53big/run-nccl-multicomm.sh"; done
for v in "${VARIANTS[@]}"; do
  say "-- variant $v"
  for i in 3 2 1 0; do sshq "${HOSTS[$i]}" "cd glm53big && NCCL_BENCH_NS=$NS ./run-nccl-multicomm.sh $i $v" 2>&1 | tail -1 | tee -a "$OUT/sweep.log"; sleep 2; done
  # wait for rank 0 to finish (containers exit when the script ends)
  for _ in $(seq 1 120); do st=$(sshq spark-06c4 "docker inspect -f '{{.State.Status}}' nccl_multicomm_$v 2>/dev/null"); [ "$st" = "exited" ] && break; sleep 5; done
  sshq spark-06c4 "docker logs nccl_multicomm_$v 2>&1" > "$OUT/$v-rank0.log"
  grep '^{' "$OUT/$v-rank0.log" > "$OUT/$v.jsonl" || true
  say "  $(grep -c '"type": "result"' "$OUT/$v.jsonl") results; $(grep -c 'NCCL WARN\|Error\|error' "$OUT/$v-rank0.log") warnings/errors in the log"
  for h in "${HOSTS[@]}"; do sshq "$h" "docker logs nccl_multicomm_$v 2>&1 | grep -i 'error\|warn' | head -3; docker rm -f nccl_multicomm_$v >/dev/null 2>&1; true" | sed "s/^/  $h: /" | tee -a "$OUT/sweep.log"; done
done
if [ "${NO_RESTORE:-0}" = 1 ]; then say "== sweep done (NO_RESTORE=1: serving stack left down)"; exit 0; fi
say "-- restoring the compaction stack"
SKIP_PREFLIGHT=1 ./rollout_dcp.sh dcp4-dflash-300k-compact-prod4 307200 2048 6000000000 1 > results/rollout-dcp4-dflash-300k-compact-prod4.console.log 2>&1
R=$(ls -d results/rollout-dcp4-dflash-300k-compact-prod4-*/ | tail -1)
grep -q "complete" "$R/rollout.log" && say "compaction stack back and generating" || say "ROLLOUT DID NOT COMPLETE: $(tail -2 "$R/rollout.log" | tr '\n' ' ')"
say "== sweep done"
