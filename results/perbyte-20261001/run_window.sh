#!/usr/bin/env bash
# What limits per-byte throughput of the RDMA collectives (model stopped):
#  1. raw RDMA write latency/bandwidth per size on two ring links (perftest, host memory)
#  2. GPU stage/reduce passes on pinned memory (bench_gpu_pinned.py, one GPU)
#  3. four-rank per-op proxy timeline of the TP ring (trace build), proxy unpinned / big / little core
#  4. NCCL protocol and channel variants in the same four-rank harness
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$R/results/perbyte-20261001
IMAGE=${IMAGE:-spark-vllm:0.29.0-nvme4-stridefix-kvtier-verifycap10-stack-20261001}
ROCE=$R/runtime/vllm029/roce
echo "$(date +%H:%M:%S) stopping serving"
for h in spark-06c4 spark-365c spark-ddbf spark-a218; do ssh -o BatchMode=yes $h.local 'docker stop -t 10 vllm_glm53big >/dev/null 2>&1; true' & done; wait

echo "$(date +%H:%M:%S) 1. perftest"
pt() {  # name server client server_lan_ip dev tool extra
  local name=$1 srv=$2 cli=$3 ip=$4 dev=$5 tool=$6; shift 6
  ssh $srv.local "timeout 120 $tool -d $dev -x 3 -F -p 18600 $* > /tmp/pt-$name-server.txt 2>&1" &
  sleep 2
  ssh $cli.local "timeout 120 $tool -d $dev -x 3 -F -p 18600 $* $ip" > $OUT/perftest-$name.txt 2>&1
  wait
}
pt lat-100-f1 spark-365c spark-06c4 192.168.1.88 roceP2p1s0f1 ib_write_lat -a -n 2000
pt bw-100-f1 spark-365c spark-06c4 192.168.1.88 roceP2p1s0f1 ib_write_bw -a -n 2000 --report_gbits
pt bw1-100-f1 spark-365c spark-06c4 192.168.1.88 roceP2p1s0f1 ib_write_bw -a -n 2000 -t 1 --report_gbits
pt lat-102-f0 spark-ddbf spark-365c 192.168.1.149 roceP2p1s0f0 ib_write_lat -a -n 2000
pt bw-102-f0 spark-ddbf spark-365c 192.168.1.149 roceP2p1s0f0 ib_write_bw -a -n 2000 --report_gbits
pt lat-101-dcp spark-365c spark-06c4 192.168.1.88 rocep1s0f1 ib_write_lat -a -n 2000

echo "$(date +%H:%M:%S) 2. GPU pinned-memory passes"
for h in spark-06c4 spark-365c spark-ddbf spark-a218; do rsync -a --delete --exclude tests "$ROCE/" "$h.local:ring-test/" || exit 1; done
ssh spark-06c4.local "docker run --rm --gpus all --ipc=host --entrypoint python3 -v \$HOME/ring-test/bench:/w $IMAGE /w/bench_gpu_pinned.py" > $OUT/gpu-pinned.log 2>&1
echo "gpu bench exit $?"

echo "$(date +%H:%M:%S) 3. ring timeline"
for pin in none big little; do
  OUT=$OUT IMAGE=$IMAGE MOUNT=1 TAG=-bd-$pin PORT=$((29800 + RANDOM % 150)) EXTRA_ENV="-e GLM_ROCE_RING_TRACE=1" \
    SCRIPT=/opt/glm-roce/bench/bench_ring_breakdown.py SCRIPT_ARGS="--pin $pin" \
    timeout 900 bash $ROCE/run_tp_ring_test.sh > $OUT/run-bd-$pin.txt 2>&1
  echo "breakdown pin=$pin: $(tail -n 1 $OUT/run-bd-$pin.txt)"
done

echo "$(date +%H:%M:%S) 4. NCCL variants"
for v in "ll:-e NCCL_PROTO=LL" "ll128:-e NCCL_PROTO=LL128" "simple:-e NCCL_PROTO=Simple" \
         "ch2:-e NCCL_MIN_NCHANNELS=2 -e NCCL_MAX_NCHANNELS=2 -e NCCL_MIN_CTAS=2 -e NCCL_MAX_CTAS=2" \
         "ch4:-e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_CTAS=4 -e NCCL_MAX_CTAS=4"; do
  tag=${v%%:*}; env=${v#*:}
  OUT=$OUT IMAGE=$IMAGE MOUNT=1 TAG=-nccl-$tag REPLAYS=30 PORT=$((29950 + RANDOM % 40)) EXTRA_ENV="$env" \
    timeout 900 bash $ROCE/run_tp_ring_test.sh > $OUT/run-nccl-$tag.txt 2>&1
  echo "nccl $tag: $(tail -n 1 $OUT/run-nccl-$tag.txt)"
done
echo "$(date +%H:%M:%S) WINDOW-DONE (serving still stopped)"
