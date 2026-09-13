#!/usr/bin/env bash
# Build from the immutable ARM64 release on the four Sparks; does not stop serving.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
"${PYTHON:-$ROOT/.venv/bin/python}" "$HERE/sync_overlay.py" --check
hosts=(spark-06c4.local spark-365c.local spark-ddbf.local spark-a218.local)
pids=()
for host in "${hosts[@]}"; do
  (
    rsync -a --delete --exclude __pycache__ "$HERE/" "$host:glm-vllm029-build/"
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
      'docker build -t spark-vllm:0.29.0-dcp1 "$HOME/glm-vllm029-build" > /tmp/vllm029-build.log 2>&1'
    echo "$host: built and verified"
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
