#!/usr/bin/env bash
# After rollout_dcp.sh reports the DCP stack up: scan node logs, warm up,
# check determinism against the DCP1 baseline, benchmark, scan logs again.
# Usage: ./post_boot_checks.sh <label> <baseline_dir> [longctx_tokens]
set -uo pipefail
LABEL="${1:?label}"; BASE_DIR="${2:?baseline dir}"; LONGCTX="${3:-0}"
WS="$(cd "$(dirname "$0")" && pwd)"
HOSTS=(spark-06c4 spark-365c spark-ddbf spark-a218)
OUT="$WS/results/$LABEL"; mkdir -p "$OUT"
URL=http://spark-06c4.local:8000
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/checks.log"; }
sshq() { ssh -o BatchMode=yes -o ConnectTimeout=8 "napta2k@$1.local" "$2"; }
scan_logs() {  # $1 = tag
  for h in "${HOSTS[@]}"; do sshq "$h" "docker logs --since 30m vllm_glm53big 2>&1" > "$OUT/$1-$h.log" 2>/dev/null & done; wait
  say "log scan ($1):"
  grep -h "jit_monitor\|Traceback\|WorkerProc hit\|CUDA error\|Triton Error\|NotImplementedError\|AssertionError\|OutOfMemory" "$OUT"/$1-*.log \
    | grep -v "NCCL INFO" | sed 's/^(.*pid=[0-9]*) //' | cut -c1-180 | sort | uniq -c | sort -rn | head -12 | tee -a "$OUT/checks.log"
  [ -s /dev/null ]
}
mem() { for h in "${HOSTS[@]}"; do printf "  %s " "$h"; sshq "$h" 'awk "/MemAvailable/{a=\$2}/SwapFree/{s=\$2}END{printf \"avail=%dMiB swapfree=%dMiB\\n\", a/1024, s/1024}" /proc/meminfo'; done | tee -a "$OUT/checks.log"; }

say "== post-boot checks for $LABEL"
mem
scan_logs boot
say "-- warmup sweep (inference-time JITs happen here, under observation)"
"$WS/.venv/bin/python" "$WS/dcp_probe.py" --base $URL --label "$LABEL" --out "$OUT" --phase warmup 2>&1 | tee -a "$OUT/checks.log"
scan_logs warmup
mem
say "-- determinism vs baseline"
"$WS/.venv/bin/python" "$WS/dcp_probe.py" --base $URL --label "$LABEL" --out "$OUT" --phase determinism 2>&1 | tee -a "$OUT/checks.log"
say "-- bench"
"$WS/.venv/bin/python" "$WS/dcp_probe.py" --base $URL --label "$LABEL" --out "$OUT" --phase bench --reps 2 2>&1 | tee -a "$OUT/checks.log"
"$WS/.venv/bin/python" "$WS/compare_runs.py" "$BASE_DIR" "$OUT" 2>&1 | tee -a "$OUT/checks.log"
say "-- concurrent (mixed decode+prefill batches)"
"$WS/.venv/bin/python" "$WS/dcp_probe.py" --base $URL --label "$LABEL" --out "$OUT" --phase concurrent 2>&1 | tee -a "$OUT/checks.log"
scan_logs concurrent
mem
if [ "$LONGCTX" -gt 0 ]; then
  say "-- long context probe ($LONGCTX tokens)"
  mem
  # Memory guard: the probe is the largest transient we run. Abort it (kill the
  # client; the engine drops the request) if any node reports MemAvailable
  # under 300 MiB on two consecutive samples or swaps out more than 512 MiB in
  # one 20 s interval. Boot 3 (512k) hit exactly this at 23% of a 500k prompt.
  "$WS/.venv/bin/python" "$WS/dcp_probe.py" --base $URL --label "$LABEL" --out "$OUT" --phase longctx --longctx-tokens "$LONGCTX" > "$OUT/longctx.console.log" 2>&1 &
  probe_pid=$!
  declare -A low prevsw
  while kill -0 "$probe_pid" 2>/dev/null; do
    sleep 20
    for h in "${HOSTS[@]}"; do
      r=$(sshq "$h" 'awk "/MemAvailable/{printf \"%d \", \$2/1024}" /proc/meminfo; awk "/^pswpout/{print \$2}" /proc/vmstat' 2>/dev/null) || continue
      a=${r%% *}; sw=${r##* }; d=$(( (sw - ${prevsw[$h]:-$sw}) / 256 )); prevsw[$h]=$sw
      if [ "$a" -lt 300 ]; then low[$h]=$(( ${low[$h]:-0} + 1 )); else low[$h]=0; fi
      if [ "${low[$h]}" -ge 2 ] || [ "$d" -gt 512 ]; then
        say "MEMORY GUARD: $h avail=${a}MiB swapout=${d}MiB/20s -> aborting the long-context probe"
        pkill -P "$probe_pid" 2>/dev/null; kill "$probe_pid" 2>/dev/null
        echo "ABORTED_BY_MEMORY_GUARD" > "$OUT/longctx-$LONGCTX.aborted"
        break 2
      fi
    done
    echo "[$(date '+%H:%M:%S')] guard: $(for h in "${HOSTS[@]}"; do printf "%s=%sMiB " "${h#spark-}" "$(sshq "$h" 'awk "/MemAvailable/{printf \"%d\", \$2/1024}" /proc/meminfo' 2>/dev/null)"; done)" >> "$OUT/checks.log"
  done
  wait "$probe_pid" 2>/dev/null
  cat "$OUT/longctx.console.log" | tee -a "$OUT/checks.log"
  mem
fi
scan_logs final
"$WS/../spark-cluster-experiments/node_snapshot.sh" > "$OUT/snap-final.txt" 2>/dev/null || true
say "== checks done; results in $OUT"
