#!/usr/bin/env bash
# After a variant boot is up: trace one count100 (torch profiler must be armed),
# run the bench prompts once, compare decode/hashes with a baseline result dir,
# and record host memory. Usage: ./measure_variant.sh <label> <baseline_dir>
set -uo pipefail
LABEL="${1:?label}"; BASE="${2:?baseline dir}"
WS="$(cd "$(dirname "$0")" && pwd)"; OUT="$WS/results/$LABEL"; mkdir -p "$OUT"
URL=http://spark-06c4.local:8000
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/measure.log"; }
say "== measure $LABEL vs $BASE"
"$WS/.venv/bin/python" "$WS/dcp_profile.py" --out "$OUT" --max-tokens 100 2>&1 | grep -v "^W0905" | grep -E "tokens in|traced request|window" | tee -a "$OUT/measure.log"
f=$(ls "$OUT"/spark-06c4-*rank*.json.gz 2>/dev/null | head -1)
[ -n "$f" ] && "$WS/.venv/bin/python" "$WS/analyze_trace.py" "$f" 2>&1 | grep -v "^W0905" | tee -a "$OUT/measure.log"
say "-- bench (1 rep)"
"$WS/.venv/bin/python" "$WS/dcp_probe.py" --base $URL --label "$LABEL" --out "$OUT" --phase bench --reps 1 2>&1 | tee -a "$OUT/measure.log"
"$WS/.venv/bin/python" "$WS/compare_runs.py" "$BASE" "$OUT" 2>&1 | tee -a "$OUT/measure.log"
say "-- memory"
for h in spark-06c4 spark-365c spark-ddbf spark-a218; do printf "  %s " "$h"; ssh -o BatchMode=yes "napta2k@$h.local" 'awk "/MemAvailable/{printf \"avail=%dMiB\\n\", \$2/1024}" /proc/meminfo'; done | tee -a "$OUT/measure.log"
say "-- log scan (rank 0, 20m)"
ssh -o BatchMode=yes napta2k@spark-06c4.local "docker logs --since 20m vllm_glm53big 2>&1" | grep -E "Traceback|Error|B12x GLM sparse MLA failed|falling back" | grep -v "NCCL INFO\|error_handling" | sed 's/^(.*pid=[0-9]*) //' | cut -c1-160 | sort | uniq -c | sort -rn | head -6 | tee -a "$OUT/measure.log"
say "== done"
