#!/usr/bin/env bash
# Validate the NVMe tier on a running KVTIER=1 stack: file counts per node
# before/after, the cold/warm/evict/reload probe under the memory guard,
# and a log scan. Usage: ./offload_checks.sh <label> [tokens]
set -uo pipefail
LABEL="${1:?label}"; TOK="${2:-100000}"
WS="$(cd "$(dirname "$0")" && pwd)"; OUT="$WS/results/$LABEL"; mkdir -p "$OUT"
HOSTS=(spark-06c4 spark-365c spark-ddbf spark-a218); URL=http://spark-06c4.local:8000
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/offload.log"; }
sshq() { ssh -o BatchMode=yes -o ConnectTimeout=8 "napta2k@$1.local" "$2"; }
files() { for h in "${HOSTS[@]}"; do printf "  %s " "$h"; sshq "$h" 'd=$(ls -d /var/tmp/kvcache/*_r* 2>/dev/null | head -1); [ -n "$d" ] && { printf "%s files=%s " "$(basename $d)" "$(find $d -name "*.bin" | wc -l)"; du -sh $d | cut -f1; } || echo "no tier dir"'; done | tee -a "$OUT/offload.log"; }
mem() { for h in "${HOSTS[@]}"; do printf "  %s " "$h"; sshq "$h" 'awk "/MemAvailable/{a=\$2}/SwapFree/{s=\$2}END{printf \"avail=%dMiB swapfree=%dMiB\\n\", a/1024, s/1024}" /proc/meminfo'; done | tee -a "$OUT/offload.log"; }
say "== offload checks for $LABEL ($TOK tokens)"; files; mem
# PROBE_EXTRA: extra dcp_probe.py flags (e.g. "--no-warm --seed-base 1000").
"$WS/.venv/bin/python" "$WS/dcp_probe.py" --base $URL --label "$LABEL" --out "$OUT" --phase offload --offload-tokens "$TOK" ${PROBE_EXTRA:-} > "$OUT/offload.console.log" 2>&1 &
pid=$!
declare -A low prevsw
while kill -0 "$pid" 2>/dev/null; do
  sleep 20
  for h in "${HOSTS[@]}"; do
    r=$(sshq "$h" 'awk "/MemAvailable/{printf \"%d \", \$2/1024}" /proc/meminfo; awk "/^pswpout/{print \$2}" /proc/vmstat' 2>/dev/null) || continue
    a=${r%% *}; sw=${r##* }; d=$(( (sw - ${prevsw[$h]:-$sw}) / 256 )); prevsw[$h]=$sw
    if [ "$a" -lt 300 ]; then low[$h]=$(( ${low[$h]:-0} + 1 )); else low[$h]=0; fi
    if [ "${low[$h]}" -ge 2 ] || [ "$d" -gt 512 ]; then say "MEMORY GUARD: $h avail=${a}MiB swapout=${d}MiB/20s -> aborting"; pkill -P "$pid"; kill "$pid"; break 2; fi
  done
  echo "[$(date '+%H:%M:%S')] guard: $(for h in "${HOSTS[@]}"; do printf "%s=%sMiB " "${h#spark-}" "$(sshq "$h" 'awk "/MemAvailable/{printf \"%d\", \$2/1024}" /proc/meminfo' 2>/dev/null)"; done)" >> "$OUT/offload.log"
done
wait "$pid" 2>/dev/null; cat "$OUT/offload.console.log" | tee -a "$OUT/offload.log"
say "after:"; files; mem
say "log scan:"; for h in "${HOSTS[@]}"; do sshq "$h" "docker logs --since 40m vllm_glm53big 2>&1" > "$OUT/offload-$h.log" 2>/dev/null & done; wait
grep -h "Traceback\|Error\|WARNING.*offload\|fs tier\|MultiNode\|tier\b" "$OUT"/offload-*.log | grep -v "NCCL INFO\|jit_monitor" | sed 's/^(.*pid=[0-9]*) //' | cut -c1-170 | sort | uniq -c | sort -rn | head -20 | tee -a "$OUT/offload.log"
say "== offload checks done"
