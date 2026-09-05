#!/usr/bin/env bash
# Detached follow-on for a boot: wait for rollout_dcp.sh's verdict, or take over
# the verification if the rollout script itself is gone, then run the checks.
cd "$(dirname "$0")"
LABEL="${1:?label}"; LONGCTX="${2:-0}"
# Wait for the rollout to create its directory and launch rank 0 before
# anything else: started too early, an earlier version saw the previous stack
# healthy and no rollout process, and ran the checks against a stack that was
# about to be torn down.
for _ in $(seq 1 240); do L=$(ls -d results/rollout-$LABEL-*/ 2>/dev/null | tail -1); [ -n "$L" ] && grep -q "launching rank 0" "$L/rollout.log" 2>/dev/null && break; sleep 5; done
[ -n "${L:-}" ] && grep -q "launching rank 0" "$L/rollout.log" || { echo "no rollout '$LABEL' launched within 20 min; giving up" >&2; exit 1; }
LOG=results/checks-$LABEL.console.log
HOSTS=(spark-06c4 spark-365c spark-ddbf spark-a218)
deadline=$((SECONDS+2700)); mode=""
while [ $SECONDS -lt $deadline ]; do
  grep -q "rollout '$LABEL' complete" "$L/rollout.log" && { mode=complete; break; }
  grep -q "RESTORING PRODUCTION\|refusing\|aborting" "$L/rollout.log" && { echo "[$(date '+%H:%M:%S')] rollout failed; checks skipped" > "$LOG"; exit 1; }
  if ! pgrep -f "rollout_dcp.sh $LABEL" >/dev/null && curl -fsS -m 5 http://spark-06c4.local:8000/health >/dev/null 2>&1; then mode=orphan; break; fi
  sleep 15
done
[ -z "$mode" ] && { echo "[$(date '+%H:%M:%S')] timed out waiting for boot" > "$LOG"; exit 1; }
if [ "$mode" = orphan ]; then
  echo "[$(date '+%H:%M:%S')] rollout script gone without a verdict; verifying the stack directly" > "$LOG"
  ssh -o BatchMode=yes napta2k@spark-06c4.local 'docker inspect -f "{{.Config.Cmd}}" vllm_glm53big' | grep -q "decode-context-parallel-size" || { echo "not the DCP stack; stopping" >> "$LOG"; exit 1; }
  curl -fsS -m 300 http://spark-06c4.local:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"glm-5.3","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c 'import sys,json; r=json.load(sys.stdin); c=r["choices"][0]["message"]["content"]; print("  content:",repr(c[:60])); sys.exit(0 if "OK" in c.upper() else 1)' >> "$LOG" 2>&1 \
    || { echo "[$(date '+%H:%M:%S')] generation FAILED on boot; leaving it for a human decision" >> "$LOG"; exit 1; }
  for h in "${HOSTS[@]}"; do ssh -o BatchMode=yes "napta2k@$h.local" "pkill -f '[c]ache_flusher.sh' 2>/dev/null; true"; done
  for h in "${HOSTS[@]}"; do ssh -o BatchMode=yes "napta2k@$h.local" "docker logs --tail 4000 vllm_glm53big 2>&1" > "$L/up-$h.log" 2>/dev/null & done; wait
  echo "[$(date '+%H:%M:%S')] boot verified by the follow-on script" >> "$LOG"
fi
./post_boot_checks.sh "$LABEL" results/baseline-dcp1-prod "$LONGCTX" >> "$LOG" 2>&1
echo "[$(date '+%H:%M:%S')] checks exit=$?" >> "$LOG"
