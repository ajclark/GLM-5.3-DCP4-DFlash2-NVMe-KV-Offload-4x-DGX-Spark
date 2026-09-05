#!/usr/bin/env bash
# Standalone: tear down whatever vllm_glm53big is running on the four Sparks
# and bring the PRODUCTION launcher back, worker-first, with flushers and a
# real-generation verify. Same steps as rollout_dcp.sh's restore_production;
# use when something goes wrong outside a rollout. Usage: ./restore_production.sh
set -uo pipefail
HOSTS=(spark-06c4 spark-365c spark-ddbf spark-a218)   # index == rank
NAME=vllm_glm53big
PROD=/home/napta2k/glm53big/launch-glm53big-dflash.sh
WS="$(cd "$(dirname "$0")" && pwd)"
OUT="$WS/results/restore-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"; LOG="$OUT/restore.log"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
sshq() { ssh -o BatchMode=yes -o ConnectTimeout=8 "napta2k@$1.local" "$2"; }
health() { curl -fsS -m 5 http://spark-06c4.local:8000/health >/dev/null 2>&1; }
generate_ok() {
  curl -fsS -m 300 http://spark-06c4.local:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"glm-5.3","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c 'import sys,json; r=json.load(sys.stdin); c=r["choices"][0]["message"]["content"]; print("  content:",repr(c[:60])); sys.exit(0 if "OK" in c.upper() else 1)'
}
containers_alive() { local bad=0; for h in "${HOSTS[@]}"; do st=$(sshq "$h" "docker inspect -f '{{.State.Status}}' $NAME 2>/dev/null || echo missing"); [ "$st" = "running" ] || { say "container on $h is '$st'"; bad=1; }; done; return $bad; }
wait_healthy() { local start=$SECONDS; while [ $((SECONDS-start)) -lt "$1" ]; do health && { say "HEALTHY after $((SECONDS-start))s"; return 0; }; containers_alive || return 2; sleep 20; done; say "TIMEOUT"; return 4; }
say "saving logs of the current containers to $OUT"
for h in "${HOSTS[@]}"; do sshq "$h" "docker logs --tail 4000 $NAME 2>&1" > "$OUT/before-$h.log" 2>/dev/null & done; wait
say "teardown"
for h in "${HOSTS[@]}"; do sshq "$h" "docker rm -f $NAME >/dev/null 2>&1; pkill -f '[c]ache_flusher.sh' 2>/dev/null; true" & done; wait; sleep 5
for h in "${HOSTS[@]}"; do r=$(sshq "$h" '$HOME/glm53big/start-flusher.sh' 2>&1 | tail -1); say "  flusher $h: $r"; done
for i in 3 2 1 0; do say "  launching production rank $i on ${HOSTS[$i]}"; sshq "${HOSTS[$i]}" "cd /home/napta2k/glm53big && $PROD $i dflash" >> "$LOG" 2>&1 || say "launch command failed on ${HOSTS[$i]}"; sleep 3; done
if wait_healthy 2400 && generate_ok | tee -a "$LOG"; then say "production restored and verified"; else say "!!! production restore did not verify; retrying once"; for h in "${HOSTS[@]}"; do sshq "$h" "docker rm -f $NAME >/dev/null 2>&1; true" & done; wait; sleep 10; for i in 3 2 1 0; do sshq "${HOSTS[$i]}" "cd /home/napta2k/glm53big && $PROD $i dflash" >> "$LOG" 2>&1; sleep 3; done; wait_healthy 2400 && generate_ok | tee -a "$LOG" && say "production restored on retry" || say "!!! PRODUCTION RESTORE FAILED - needs a human"; fi
for h in "${HOSTS[@]}"; do sshq "$h" "pkill -f '[c]ache_flusher.sh' 2>/dev/null; true"; done
