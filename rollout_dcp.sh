#!/usr/bin/env bash
# Roll the DCP overlay + launcher onto the four Sparks, bring the stack up with
# a memory watchdog, verify a real generation, and on ANY failure restore the
# production launcher automatically. Never leaves the cluster down.
#
# Usage: ./rollout_dcp.sh <label> [MAXLEN] [MAXBATCHED] [KVBYTES] [KVTIER]
set -uo pipefail

LABEL="${1:?usage: rollout_dcp.sh <label> [MAXLEN] [MAXBATCHED] [KVBYTES] [KVTIER]}"
MAXLEN="${2:-180224}"
MAXBATCHED="${3:-2048}"
KVBYTES="${4:-6000000000}"
KVTIER="${5:-1}"
WS="$(cd "$(dirname "$0")" && pwd)"
HOSTS=(spark-06c4 spark-365c spark-ddbf spark-a218)   # index == rank
NAME=vllm_glm53big
PROD=/home/napta2k/glm53big/launch-glm53big-dflash.sh       # untouched production launcher
DCPL=/home/napta2k/glm53big/launch-glm53big-dcp.sh
OUT="$WS/results/rollout-$LABEL-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
LOG="$OUT/rollout.log"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
sshq() { ssh -o BatchMode=yes -o ConnectTimeout=8 "napta2k@$1.local" "$2"; }

health() { curl -fsS -m 5 http://spark-06c4.local:8000/health >/dev/null 2>&1; }
generate_ok() {
  curl -fsS -m 300 http://spark-06c4.local:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"glm-5.3","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c 'import sys,json; r=json.load(sys.stdin); c=r["choices"][0]["message"]["content"]; print("  content:",repr(c[:60]),"tokens:",r["usage"]["completion_tokens"]); sys.exit(0 if "OK" in c.upper() else 1)'
}
save_logs() {  # $1 = prefix
  for h in "${HOSTS[@]}"; do sshq "$h" "docker logs --tail 4000 $NAME 2>&1" > "$OUT/$1-$h.log" 2>/dev/null & done; wait
}
memline() {  # one line per host: MemAvailable(MiB) SwapFree(MiB) pswpout(pages)
  for h in "${HOSTS[@]}"; do
    printf "%s " "$h"; sshq "$h" 'awk "/MemAvailable/{a=\$2}/SwapFree/{s=\$2}END{printf \"avail=%dMiB swapfree=%dMiB \", a/1024, s/1024}" /proc/meminfo; awk "/^pswpout/{print \"pswpout=\"\$2}" /proc/vmstat' 2>/dev/null || echo "UNREACHABLE"
  done
}
containers_alive() {  # returns 1 (and prints) if any container is not running
  local bad=0
  for h in "${HOSTS[@]}"; do
    st=$(sshq "$h" "docker inspect -f '{{.State.Status}}' $NAME 2>/dev/null || echo missing")
    [ "$st" = "running" ] || { say "container on $h is '$st'"; bad=1; }
  done
  return $bad
}
teardown() {
  for h in "${HOSTS[@]}"; do sshq "$h" "docker rm -f $NAME >/dev/null 2>&1; pkill -f '[c]ache_flusher.sh' 2>/dev/null; true" & done; wait
}
start_flushers() {
  for h in "${HOSTS[@]}"; do
    r=$(sshq "$h" '$HOME/glm53big/start-flusher.sh' 2>&1 | tail -1)
    say "  flusher $h: $r"
    case "$r" in STARTED*) ;; *) return 1 ;; esac
  done
}
launch_ranks() {  # $1 = launcher path, $2 = extra env
  for i in 3 2 1 0; do
    say "  launching rank $i on ${HOSTS[$i]}"
    sshq "${HOSTS[$i]}" "cd /home/napta2k/glm53big && $2 $1 $i dflash" >> "$LOG" 2>&1 || { say "launch command failed on ${HOSTS[$i]}"; return 1; }
    sleep 3
  done
}
wait_healthy() {  # $1 = deadline seconds; watchdog inside
  local start=$SECONDS last_out=() t
  for h in "${HOSTS[@]}"; do last_out+=(0); done
  while [ $((SECONDS-start)) -lt "$1" ]; do
    if health; then say "HEALTHY after $((SECONDS-start))s"; return 0; fi
    containers_alive || return 2
    # memory watchdog: log, and abort on a swap-out storm after the load phase
    local i=0 storm=0
    while read -r h rest; do
      echo "[$(date '+%H:%M:%S')] $h $rest" >> "$OUT/mem.log"
      p=$(echo "$rest" | sed -n 's/.*pswpout=\([0-9]*\).*/\1/p')
      if [ -n "$p" ] && [ "${last_out[$i]}" != 0 ] && [ $((SECONDS-start)) -gt 600 ]; then
        d=$(( (p - last_out[$i]) / 256 ))   # MiB swapped out since last sample
        [ "$d" -gt 2048 ] && { say "SWAP STORM on $h: ${d}MiB out in one interval"; storm=1; }
      fi
      [ -n "$p" ] && last_out[$i]=$p
      i=$((i+1))
    done < <(memline)
    [ "$storm" = 1 ] && return 3
    sleep 20
  done
  say "TIMEOUT waiting for /health"; return 4
}
restore_production() {
  say "=== RESTORING PRODUCTION LAUNCHER ==="
  save_logs "failed"
  teardown; sleep 5
  start_flushers || say "warning: flushers did not start (continuing)"
  launch_ranks "$PROD" ""
  if wait_healthy 2400 && generate_ok | tee -a "$LOG"; then
    say "production restored and verified"
  else
    say "!!! production restore did not verify; retrying once"
    teardown; sleep 10; launch_ranks "$PROD" ""; wait_healthy 2400 && generate_ok | tee -a "$LOG" && say "production restored on retry" || say "!!! PRODUCTION RESTORE FAILED — needs a human"
  fi
  for h in "${HOSTS[@]}"; do sshq "$h" "pkill -f '[c]ache_flusher.sh' 2>/dev/null; true"; done
}

say "rollout '$LABEL' MAXLEN=$MAXLEN MAXBATCHED=$MAXBATCHED KVBYTES=$KVBYTES KVTIER=$KVTIER out=$OUT"

# 0. preflight: production healthy, save its logs and a memory baseline
if [ "${SKIP_PREFLIGHT:-0}" = 1 ]; then say "SKIP_PREFLIGHT=1: not requiring a healthy stack (recovery)"; else
health || { say "production is not healthy; refusing to start"; exit 1; }
fi
say "production healthy; saving pre-rollout logs + memory baseline"
save_logs "pre"
"$WS/../spark-cluster-experiments/node_snapshot.sh" > "$OUT/snap-pre.txt" 2>/dev/null || true
memline | tee -a "$LOG"

# 0b. ring GID sanity: the launchers pin NCCL_IB_GID_INDEX=3 (IPv4 RoCEv2). A ring
#     port that flapped (peer rebooted) can come back with that GID at index 4 and
#     index 3 empty -> ibv_modify_qp "No data available" on every rank. Re-adding the
#     IP (NM connection bounce) re-packs the table. Checked before anything is stopped.
ring_gid_check() {
  local h d bad=0
  for h in "${HOSTS[@]}"; do
    for d in roceP2p1s0f0 roceP2p1s0f1; do
      t=$(sshq "$h" "cat /sys/class/infiniband/$d/ports/1/gid_attrs/types/3 2>/dev/null; cat /sys/class/infiniband/$d/ports/1/gids/3 2>/dev/null" | paste -sd' ')
      case "$t" in *"RoCE v2"*ffff:*) ;; *)
        say "  $h $d: GID index 3 is '$t', expected IPv4 RoCE v2; bouncing its NM connection"
        sshq "$h" "n=\$(rdma link show $d/1 | awk '{print \$NF}'); c=\$(nmcli -t -f DEVICE,CONNECTION dev | awk -F: -v n="\$n" '\$1==n{print \$2}'); sudo -n nmcli con down "\$c" >/dev/null 2>&1; sleep 2; sudo -n nmcli con up "\$c" >/dev/null 2>&1; sleep 3" 
        t=$(sshq "$h" "cat /sys/class/infiniband/$d/ports/1/gid_attrs/types/3 2>/dev/null; cat /sys/class/infiniband/$d/ports/1/gids/3 2>/dev/null" | paste -sd' ')
        case "$t" in *"RoCE v2"*ffff:*) say "  $h $d: fixed ($t)" ;; *) say "  $h $d: STILL WRONG ($t)"; bad=1 ;; esac ;;
      esac
    done
  done
  return $bad
}
say "ring GID check (NCCL_IB_GID_INDEX=3 on both ring ports of every node)"
ring_gid_check || { say "ring GID table wrong on at least one node; refusing to start (nothing was stopped)"; exit 1; }

# 1. stage overlays + launcher (production launcher untouched)
WANT_L=$(sha256sum "$WS/launch-glm53big-dcp.sh" | cut -d' ' -f1)
for h in "${HOSTS[@]}"; do
  ( rsync -a --delete "$WS/stage/glm-dcp/" "napta2k@$h.local:glm-dcp/" && \
    rsync -a --delete "$WS/stage/glm-triton/" "napta2k@$h.local:glm-triton/" && \
    rsync -a "$WS/stage/nccl-hotplug/" "napta2k@$h.local:nccl-hotplug/" && \
    scp -q "$WS/launch-glm53big-dcp.sh" "napta2k@$h.local:$DCPL" && \
    sshq "$h" "chmod +x $DCPL" ) &
done; wait
for h in "${HOSTS[@]}"; do
  got=$(sshq "$h" "sha256sum $DCPL | cut -d' ' -f1"); [ "$got" = "$WANT_L" ] || { say "launcher sha mismatch on $h"; exit 1; }
  sshq "$h" "cd ~ && sha256sum -c --quiet -" < "$WS/stage/SHA256SUMS" >/dev/null 2>&1 || { say "overlay sha mismatch on $h"; exit 1; }
  say "  $h staged + verified"
done

# 2. teardown, 3. flushers, 4. launch  (teardown kills any flusher, so start
#    them after it; boots 1-2 ran their loads without flushers because of the
#    old order. Flushers are insurance here: the fixed KV pool skips the 0.91
#    admission check they exist for.)
say "stopping production ranks"
teardown; sleep 5
start_flushers || say "warning: flushers did not start (continuing; admission check is skipped with a fixed KV pool)"
say "launching DCP ranks (worker-first)"
say "adapters on and ring verified before launch (idempotent)"; ./spark-idle.sh --up --no-plugin >/dev/null 2>&1 || say "  note: spark-idle.sh --up reported a problem; the launch's own checks decide"
launch_ranks "$DCPL" "MAXLEN=$MAXLEN MAXBATCHED=$MAXBATCHED KVBYTES=$KVBYTES KVTIER=$KVTIER KVTIER_MODE=${KVTIER_MODE:-slab} KVTIER_BOUNCE=${KVTIER_BOUNCE:-48} KVTIER_DISK_BYTES=${KVTIER_DISK_BYTES:-150000000000} PROFILER_DIR=${PROFILER_DIR:-} DCP_Q_PREGATHER=${DCP_Q_PREGATHER:-0} DCP_COMPACT=${DCP_COMPACT:-1} DCP_SIZE=${DCP_SIZE:-2} NCCL_HOTPLUG=${NCCL_HOTPLUG:-0}" || { restore_production; exit 2; }

# 5. wait with watchdog
wait_healthy 1800; rc=$?
if [ $rc -ne 0 ]; then
  say "DCP stack did not come up cleanly (rc=$rc)"; restore_production; exit 3
fi
for h in "${HOSTS[@]}"; do sshq "$h" "pkill -f '[c]ache_flusher.sh' 2>/dev/null; true"; done

# 6. real generation (the /health-lies check)
if ! generate_ok | tee -a "$LOG"; then
  say "generation probe FAILED on the DCP stack"; restore_production; exit 4
fi
# 7. warm every request path while the process is fresh (Triton kernels compile and load now,
#    not hours later: a late CUDA module load on GB10 can kill a rank, see results/incident-20260907-1418-*)
say "warming the request paths (long prefill, odd lengths, a concurrent batch)"
if python3 "$WS/bench/warm_kernels.py" --base "http://${HOSTS[0]}.local:8000" 2>&1 | tee -a "$LOG" | tail -3 | grep -q '"ok": false'; then
  say "  warm-up had a failing request; the stack is up but check the logs"
fi
save_logs "up"
"$WS/../spark-cluster-experiments/node_snapshot.sh" > "$OUT/snap-up.txt" 2>/dev/null || true
memline | tee -a "$LOG"
say "DCP stack is up and generating. Startup facts:"
grep -h "Maximum concurrency\|KV cache groups\|GPU KV cache size\|kv_cache_groups\|Using.*DCP\|decode_context_parallel\|DCP" "$OUT/up-spark-06c4.log" | grep -v "NCCL INFO" | tail -8 | tee -a "$LOG"
grep -h "jit_monitor\|Traceback\|Error" "$OUT"/up-*.log | grep -v "NCCL INFO\|error_handling\|ERROR_HANDLING" | sort | uniq -c | sort -rn | head -10 | tee -a "$LOG"
say "rollout '$LABEL' complete; results in $OUT"
