#!/usr/bin/env bash
# Shared definitions for enter-low-power-idle-mode.sh and un-idle.sh.
# Sourced, not executed. Everything here is read-only against the cluster
# except run_node(), which honours DRY_RUN=1.
HOSTS=(spark-06c4 spark-365c spark-ddbf spark-a218)   # index == rank
declare -A MAC=( [spark-06c4]=4c:bb:47:2b:06:c4 [spark-365c]=4c:bb:47:2a:36:5c
                 [spark-ddbf]=4c:bb:47:2b:dd:bf [spark-a218]=4c:bb:47:2c:a2:18 )
declare -A IP=( [spark-06c4]=192.168.1.228 [spark-365c]=192.168.1.88
                [spark-ddbf]=192.168.1.149 [spark-a218]=192.168.1.31 )
MGMT_IF=enP7s7                                 # Realtek r8127 10GbE: SSH, API, WoL
RING_IFS=(enP2p1s0f0np0 enP2p1s0f1np1)         # the two cabled 200G ports (NCCL_IB_HCA roceP2p1s0f0/f1);
RING_RDMA=(roceP2p1s0f0 roceP2p1s0f1)          # enp1s0f0np0/f1np1 are the same ports via the 2nd PCIe path
KEEP_CPUS=3                                    # deep idle keeps cpu0..cpu3 online
NAME=vllm_glm53big
GPU_LOCK="${GPU_LOCK:-2000,2000}"              # the benchmark clock lock (docs/HANDOVER.md)
API=http://spark-06c4.local:8000
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$WS/results/idle-power"; STATE="$STATE_DIR/current.env"
mkdir -p "$STATE_DIR"
LOG="${LOG:-$STATE_DIR/$(date +%Y%m%d-%H%M%S)-$(basename "$0" .sh).log}"
DRY_RUN="${DRY_RUN:-0}"

say()   { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
sshq()  { ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "napta2k@$1.local" "$2"; }
reachable() { sshq "$1" true >/dev/null 2>&1; }
wait_ssh()  { local h=$1 t=${2:-90} i; for ((i=0; i<t; i+=5)); do reachable "$h" && return 0; sleep 5; done; return 1; }
run_node()  { # host, remote command; prints instead of running under DRY_RUN=1
  if [ "$DRY_RUN" = 1 ]; then echo "  [dry-run] $1: $2" | tee -a "$LOG"; return 0; fi
  sshq "$1" "$2" 2>&1 | sed "s/^/  $1: /" | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

# One SSH round trip per node: everything the two scripts decide on.
NODE_STATUS='
gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
mhz=$(( $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) / 1000 ))
online=$(cat /sys/devices/system/cpu/online)
eth=$(cat /sys/class/net/enP7s7/speed 2>/dev/null || echo -1)
ring=$(for i in enP2p1s0f0np0 enP2p1s0f1np1; do cat /sys/class/net/$i/operstate; done | paste -sd/)
rdma=$(rdma link show 2>/dev/null | awk "/roceP2p1s0f[01]\//{print \$4}" | paste -sd/)
ringip=$(for i in enP2p1s0f0np0 enP2p1s0f1np1; do ip -o -4 addr show dev $i 2>/dev/null | awk "{print \$4}" | head -1 | grep . || echo none; done | paste -sd/)
mtu=$(for i in enP2p1s0f0np0 enP2p1s0f1np1; do cat /sys/class/net/$i/mtu 2>/dev/null || echo 0; done | paste -sd/)
radios=$(rfkill list 2>/dev/null | grep -c "Soft blocked: yes")
gpu=$(nvidia-smi --query-gpu=power.draw,clocks.sm,persistence_mode --format=csv,noheader,nounits 2>/dev/null | tr -d " " | tr "," "/")
ctr=$(docker inspect -f "{{.State.Status}}" vllm_glm53big 2>/dev/null || echo none)
avail=$(awk "/MemAvailable/{printf \"%.1f\", \$2/1048576}" /proc/meminfo)
echo "gov=$gov mhz=$mhz online=$online eth=${eth}M ring=$ring rdma=$rdma ringip=$ringip mtu=$mtu radios_blocked=$radios/2 gpu_W/MHz/pm=$gpu ctr=$ctr availG=$avail"
'
node_status() { sshq "$1" "$NODE_STATUS" 2>/dev/null || echo "UNREACHABLE"; }
status_all()  { local h; for h in "${HOSTS[@]}"; do printf "%-11s %s\n" "$h" "$(node_status "$h")"; done; }
# field=value lookup in a status line
sfield() { echo "$1" | tr ' ' '\n' | awk -F= -v k="$2" '$1==k{print $2}'; }

health()          { curl -fsS -m 5 "$API/health" >/dev/null 2>&1; }
requests_active() { curl -fsS -m 5 "$API/metrics" 2>/dev/null | awk '/^vllm:num_requests_(running|waiting)/{s+=$2} END{print s+0}'; }
generate_ok() {   # a real generation, not /health (which is 200 on a wedged engine)
  curl -fsS -m 300 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"glm-5.3","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c 'import sys,json; r=json.load(sys.stdin); c=r["choices"][0]["message"]["content"]; print("  content:",repr(c[:60])); sys.exit(0 if "OK" in c.upper() else 1)'
}

state_set() { touch "$STATE"; { grep -v "^$1=" "$STATE" 2>/dev/null; echo "$1=$2"; } > "$STATE.tmp"; mv "$STATE.tmp" "$STATE"; }
state_get() { grep "^$1=" "$STATE" 2>/dev/null | tail -1 | cut -d= -f2-; }

wol() {   # host: 3 magic packets to the broadcast address and 3 to the node's last known IP
  python3 - "${MAC[$1]}" "${IP[$1]}" <<'PY'
import socket, sys
mac = bytes.fromhex(sys.argv[1].replace(":", "")); pkt = b"\xff" * 6 + mac * 16
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
for dst in ("255.255.255.255", sys.argv[2]):
    for _ in range(3): s.sendto(pkt, (dst, 9))
PY
}
