#!/usr/bin/env bash
# Step 3 measurement series (one boot): cost grid -> batch cost table -> fixed-7 acceptance
# -> C2-C4 auto -> full C1 probe with agents.
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$R/results/step3-roce-20261001
CAP=${CAP:?directory of probe and pi agent-turn captures (not published)}
cd $R/runtime/vllm029
ctl() { ssh spark-06c4.local "echo '$1' > ~/verify-cap-live/control.json"; sleep 3; }
ssh spark-06c4.local 'nohup bash -c "while true; do echo \$(date +%H:%M:%S) \$(awk \"/MemAvailable/{print int(\\\$2/1024)}\" /proc/meminfo); sleep 5; done" > ~/verify-cap-live/memlog-step3.txt 2>&1 &'
for K in 7 5 3 1; do
  ctl "{\"mode\": \"fixed\", \"fixed_k\": $K, \"batch\": true, \"policy\": \"ratio\"}"
  python3 conc_bench.py --out $OUT/costgrid --n 1,2,3,4 --sets prose,code --rounds 1 --label fixed$K | grep -E "SUMMARY.*fixed$K|CONTAM"
done
sleep 35
scp -q spark-06c4.local:verify-cap-live/periods.json $OUT/costgrid/periods.json
python3 - <<PY
import json
P=json.load(open("$OUT/costgrid/periods.json"))["periods"]
old=json.load(open("$R/results/step2-drafter-20260930/costs-step2.json"))
d1={str(k):P[f"n1_k{k}_ctx0"]["median_ms"]-old["batch"]["2"]["points"][0]["cycle_ms"][str(k)]*0 for k in (1,3,5,7)}
# C1: shift the historical context curve by the measured change at short context
step1={1:89.24,3:108.41,5:120.69,7:135.16}
pts=[]
for p in old["points"]:
    pts.append({"context":p["context"],"cycle_ms":{k:round(v+(P[f"n1_k{k}_ctx0"]["median_ms"]-step1[int(k)]),2) for k,v in p["cycle_ms"].items()}})
batch={str(n):{"points":[{"context":4096,"cycle_ms":{str(k):P[f"n{n}_k{k}_ctx0"]["median_ms"] for k in (1,3,5,7)}}]} for n in (2,3,4)}
json.dump({"source":"step3 live periods (step 2 + RoCEnante DCP); C1 = step-2 context curve shifted by the measured short-context change","points":pts,"batch":batch},open("$OUT/costs-step3.json","w"),indent=1)
print(json.dumps(batch)); print(json.dumps(pts))
PY
scp -q $OUT/costs-step3.json spark-06c4.local:verify-cap-live/costs-step3.json
ctl '{"mode": "fixed", "fixed_k": 7, "batch": true, "policy": "ratio", "costs": "costs-step3.json"}'
python3 spec_accept_probe.py --captures $CAP --max-context-chars 0 --out $OUT/c1-fixed7 | grep SUMMARY
ctl '{"mode": "auto", "batch": true, "policy": "ratio", "costs": "costs-step3.json"}'
python3 conc_bench.py --out $OUT/conc --n 2,3,4 --sets prose,code,mix --rounds 3 --captures $CAP --label step3 | grep -E "SUMMARY|CONTAM"
python3 spec_accept_probe.py --captures $CAP --out $OUT/c1-auto-full | grep SUMMARY
echo STEP3-DONE
