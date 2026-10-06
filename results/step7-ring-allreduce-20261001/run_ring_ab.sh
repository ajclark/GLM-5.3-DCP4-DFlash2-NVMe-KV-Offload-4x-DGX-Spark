#!/usr/bin/env bash
# Ring variants A/B (model stopped): v2 source, split on/off x kernel grid 8/16/32.
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$R/results/step7-ring-allreduce-20261001/ab
mkdir -p $OUT
cd $R/runtime/vllm029/roce
IMAGE=${IMAGE:-spark-vllm:0.29.0-nvme4-stridefix-kvtier-verifycap9-ring-20261001}
for v in "s0b8:0:8" "s0b16:0:16" "s1b8:1:8" "s1b16:1:16" "s0b32:0:32"; do
  IFS=: read tag split blocks <<< "$v"
  echo "== $tag split=$split blocks=$blocks"
  OUT=$OUT IMAGE=$IMAGE TAG=-$tag REPLAYS=60 PORT=$((29700 + RANDOM % 200)) \
    EXTRA_ENV="-e GLM_ROCE_RING_SPLIT=$split -e GLM_ROCE_RING_BLOCKS=$blocks" \
    timeout 900 bash run_tp_ring_test.sh > $OUT/run-$tag.txt 2>&1
  tail -1 $OUT/run-$tag.txt
  python3 - "$OUT" "$tag" <<'PY'
import json, sys, glob, statistics
out, tag = sys.argv[1:]
rows = []
for f in sorted(glob.glob(f"{out}/ring-{tag}-*.log")):
    for line in open(f):
        if line.startswith("RESULT "):
            rows.append(json.loads(line[7:]))
ok = all(r["ok"] for r in rows) and len(rows) == 4
lat = {}
for r in rows:
    for k, v in r.get("checks", {}).get("latency", {}).items():
        lat.setdefault(k, []).append(v["us_per_allreduce"])
print(tag, "ok" if ok else "FAIL", " ".join(f"{k}={statistics.mean(v):.1f}" for k, v in sorted(lat.items())))
if not ok:
    for r in rows:
        if not r["ok"]:
            print(r.get("error", "")[-800:])
PY
done
echo AB-DONE
