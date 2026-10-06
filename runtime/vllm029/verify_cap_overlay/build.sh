#!/usr/bin/env bash
# Build the verify-cap overlay image on all four Sparks (no serving interruption).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TAG="${TAG:?set TAG, e.g. spark-vllm:0.29.0-nvme4-stridefix-kvtier-verifycap6-batch-20261001}"
for f in cal.json costs.json; do [ -f "$HERE/$f" ] || { echo "missing $f" >&2; exit 1; }; done
for h in spark-06c4 spark-365c spark-ddbf spark-a218; do
  ( rsync -a --delete "$HERE/" "$h.local:verify-cap-build/" &&
    rsync -a --delete --exclude tests "$HERE/../roce/" "$h.local:verify-cap-build/roce/" &&
    ssh "$h.local" "cd verify-cap-build && docker build -t $TAG . > build.log 2>&1 && grep -E \"roce_proxy-|verify_targets\" build.log | tail -2 && echo $h built $TAG || { tail -30 build.log; echo $h BUILD FAILED; }" ) &
done
wait
