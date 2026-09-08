#!/usr/bin/env bash
# Sandbox-side driver for the on-cluster NCCL hot-plug probe (docs/NCCL-HOTPLUG-TEST.md).
# Requires the serving stack to be DOWN on all four nodes (it refuses otherwise).
#   1. stage the probe, the node runner and the aarch64 plugin to every node
#   2. launch the probe containers (rank 3..0), wait for rank 0 to print ready-for-cycle
#   3. spark-idle.sh --down (plugin suspend + adapters off), wait ~HOLD_S, spark-idle.sh --up
#   4. wait for the probe to finish, collect rank 0's JSON lines + every rank's log
# Usage: ./bench/nccl-hotplug-probe.sh [cycles] ; env HOLD_S (default 30), NO_CYCLE=1 (plugin + NCCL only, no adapter cycle),
#        PROBE_NET=builtin (NCCL's builtin IB backend, cycles must be 0: baseline timings only)
set -uo pipefail
WS="$(cd "$(dirname "$0")/.." && pwd)"; cd "$WS"
CYCLES="${1:-1}"; HOLD_S="${HOLD_S:-30}"
OUT="$WS/results/nccl-hotplug-probe/$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
HOSTS=(spark-06c4 spark-365c spark-ddbf spark-a218)
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
sshq() { ssh -o BatchMode=yes -o ConnectTimeout=8 "${SSH_USER:-$USER}@$1.local" "$2"; }
for h in "${HOSTS[@]}"; do
  st=$(sshq "$h" "docker inspect -f '{{.State.Status}}' vllm_glm53big 2>/dev/null | grep . || echo none")   # a missing container prints an empty line
  [ "$st" = none ] || { say "$h: serving container is '$st'; take the stack down first (docker rm -f vllm_glm53big)"; exit 1; }
done
say "== NCCL hot-plug probe: cycles=$CYCLES hold=${HOLD_S}s out=$OUT"
say "-- staging"
for h in "${HOSTS[@]}"; do
  scp -q "$WS/bench/nccl_hotplug_probe.py" "$WS/bench/run-nccl-hotplug-probe.sh" "${SSH_USER:-$USER}@$h.local:glm53big/" && \
  rsync -aq "$WS/stage/nccl-hotplug/" "${SSH_USER:-$USER}@$h.local:nccl-hotplug/" && \
  sshq "$h" "chmod +x glm53big/run-nccl-hotplug-probe.sh; sha256sum nccl-hotplug/libnccl-net-hotplug.so | cut -c1-16" | sed "s/^/  $h plugin sha /" | tee -a "$OUT/driver.log"
done
say "-- launching probe containers"
for i in 3 2 1 0; do sshq "${HOSTS[$i]}" "cd glm53big && PROBE_NET=${PROBE_NET:-hotplug} ./run-nccl-hotplug-probe.sh $i $CYCLES" 2>&1 | tail -1 | tee -a "$OUT/driver.log"; sleep 2; done
wait_for() {  # pattern, timeout seconds
  local pat=$1 t=$2 i; for i in $(seq 1 "$t"); do
    if sshq spark-06c4 "docker logs nccl_hotplug_probe 2>&1" | grep -q "$pat"; then return 0; fi
    st=$(sshq spark-06c4 "docker inspect -f '{{.State.Status}}' nccl_hotplug_probe 2>/dev/null"); [ "$st" = exited ] && return 1
    sleep 2; done; return 1; }
say "-- waiting for the baseline and ready-for-cycle"
READY='"phase": "ready-for-cycle"'; [ "$CYCLES" = 0 ] && READY='"phase": "done"'
if ! wait_for "$READY" 300; then say "probe did not reach $READY"; sshq spark-06c4 "docker logs nccl_hotplug_probe 2>&1 | tail -30" | tee -a "$OUT/driver.log"; exit 1; fi
sshq spark-06c4 "docker logs nccl_hotplug_probe 2>&1" | grep -o '{"phase"[^}]*}' | tee -a "$OUT/driver.log"
for c in $(seq 1 "$CYCLES"); do
  if [ "${NO_CYCLE:-0}" = 1 ]; then
    say "-- cycle $c: NO_CYCLE=1, plugin suspend/resume only"
    ./spark-idle.sh --suspend 2>&1 | tee -a "$OUT/driver.log"; sleep "$HOLD_S"; ./spark-idle.sh --resume 2>&1 | tee -a "$OUT/driver.log"
  else
    say "-- cycle $c: spark-idle.sh --down"
    ./spark-idle.sh --down --restore-after 0 2>&1 | tee -a "$OUT/driver.log"
    say "-- adapters off, holding ${HOLD_S}s (read the meter)"; sleep "$HOLD_S"
    say "-- cycle $c: spark-idle.sh --up"
    ./spark-idle.sh --up 2>&1 | tee -a "$OUT/driver.log"
  fi
  [ "$c" -lt "$CYCLES" ] && wait_for "\"phase\": \"ready-for-cycle\", \"cycle\": $((c+1))" $((300 + ${PROBE_SETTLE_S:-0}))
done
say "-- waiting for the probe to finish"
for i in $(seq 1 150); do st=$(sshq spark-06c4 "docker inspect -f '{{.State.Status}}' nccl_hotplug_probe 2>/dev/null"); [ "$st" = exited ] && break; sleep 2; done
for h in "${HOSTS[@]}"; do sshq "$h" "docker logs nccl_hotplug_probe 2>&1" > "$OUT/$h.log" 2>/dev/null; done
grep -o '{"phase"[^}]*}' "$OUT/spark-06c4.log" | tee "$OUT/rank0.jsonl" | tail -8
rc=$(sshq spark-06c4 "docker inspect -f '{{.State.ExitCode}}' nccl_hotplug_probe 2>/dev/null")
say "== probe exit code on rank 0: $rc ($( [ "$rc" = 0 ] && echo PASS || echo FAIL )); logs in $OUT"
for h in "${HOSTS[@]}"; do sshq "$h" "docker rm -f nccl_hotplug_probe >/dev/null 2>&1; true"; done
