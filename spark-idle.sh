#!/usr/bin/env bash
# spark-idle.sh: cut a DGX Spark cluster's idle power by switching the ConnectX-7
# off with the cables still attached (~20 W per node; 202 -> 120 W measured on
# four idle nodes), and switch it back on with the ring verified.
# Uses NVIDIA's cx7-pcie-hotplug driver (DGX OS package dgx-spark-mlnx-hotplug).
#
# Usage: ./spark-idle.sh --down|--up|--status [--hosts a,b] [--restore-after SECONDS] [--dry-run]
#   --down     Unload mstflint_access, remove the four CX-7 PCIe functions, power the adapter
#              down. No checks: down means down, whatever is running. Stays off until --up;
#              --restore-after N arms a node-side dead-man timer that powers it back on itself.
#   --up       Power up, rescan, wait for the four functions, both ring ports at 200G with IPv4
#              and MTU 9000, RDMA ACTIVE, jumbo-ping each neighbour, reload mstflint_access.
#              Reports what came back; exit status non-zero if a node did not fully recover.
#   --status   One line per node (read-only). (--idle / --unidle are aliases of --down / --up.)
# Needs passwordless sudo on the nodes. Powering the adapter down kills every RDMA/NCCL
# connection on it; stop or expect to restart whatever uses the ring.
set -uo pipefail

HOSTS=(spark-06c4 spark-365c spark-ddbf spark-a218)   # edit for your cluster
SSH_USER="${SSH_USER:-$USER}"; SSH_SUFFIX="${SSH_SUFFIX:-.local}"
CONTAINER="${CONTAINER:-vllm_glm53big}"               # refuse --idle while this container runs
RING_IFS="enP2p1s0f0np0 enP2p1s0f1np1"                # the two cabled 200G ports
MGMT_IF="enP7s7"                                      # 10GbE management link (domain 0007)
CX7_BDFS="0000:01:00.0 0000:01:00.1 0002:01:00.0 0002:01:00.1"
SYS=/sys/devices/platform/MTKP0001:00/pcie_hotplug
HANDLER=/opt/nvidia/dgx-spark-mlnx-hotplug/mtk-hotplug-handler.sh

CMD=""; RESTORE_AFTER=0; DRY_RUN=0
while [ $# -gt 0 ]; do case "$1" in
  --down|--idle) CMD=down ;; --up|--unidle) CMD=up ;; --status) CMD=status ;;
  --hosts) IFS=, read -r -a HOSTS <<<"${2:?}"; shift ;;
  --restore-after) RESTORE_AFTER="${2:?}"; shift ;;
  --dry-run) DRY_RUN=1 ;;
  -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
esac; shift; done
[ -n "$CMD" ] || { echo "usage: $0 --down|--up|--status [--hosts a,b] [--restore-after SECONDS] [--dry-run]" >&2; exit 2; }

say()   { echo "[$(date '+%H:%M:%S')] $*"; }
sshq()  { ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "$SSH_USER@$1$SSH_SUFFIX" "$2"; }
reachable() { sshq "$1" true >/dev/null 2>&1; }
run_node()  { if [ "$DRY_RUN" = 1 ]; then echo "  [dry-run] $1: $2"; return 0; fi; sshq "$1" "$2" 2>&1 | sed "s/^/  $1: /"; return "${PIPESTATUS[0]}"; }

STATUS='
echo "cx7=$(cat '"$SYS"'/debug_state 2>/dev/null) hotplug_enabled=$(cat '"$SYS"'/hotplug_enabled 2>/dev/null) fns=$(lspci -D -d 15b3:1021 2>/dev/null | wc -l) ring=$(for d in '"$RING_IFS"'; do echo -n "$(cat /sys/class/net/$d/operstate 2>/dev/null || echo absent)/$(cat /sys/class/net/$d/speed 2>/dev/null || echo -)/mtu$(cat /sys/class/net/$d/mtu 2>/dev/null || echo -) "; done)rdma_active=$(rdma link show 2>/dev/null | grep -c " ACTIVE ") mgmt=$(cat /sys/class/net/'"$MGMT_IF"'/speed 2>/dev/null)M mstflint=$( [ -d /sys/module/mstflint_access ] && echo loaded || echo unloaded) timer=$(systemctl list-timers --no-legend cx7-restore.timer 2>/dev/null | awk "{print \$1, \$2, \$3}" | grep . || echo none) ctr=$(docker inspect -f "{{.State.Status}}" '"$CONTAINER"' 2>/dev/null || echo none)"
'

PREP='
if test -d /sys/module/mstflint_access; then sudo -n modprobe -r mstflint_access 2>/dev/null && echo "mstflint_access unloaded" || echo "mstflint_access busy, left loaded"; fi
echo "before: cx7=$(cat '"$SYS"'/debug_state 2>/dev/null) fns=$(lspci -D -d 15b3:1021 2>/dev/null | wc -l) ctr=$(docker inspect -f "{{.State.Status}}" '"$CONTAINER"' 2>/dev/null || echo none)"
'

OFF='
set -euo pipefail
if [ RESTORE_AFTER -gt 0 ]; then
  sudo -n systemctl stop cx7-restore.timer cx7-restore.service >/dev/null 2>&1 || true
  sudo -n systemctl reset-failed cx7-restore.service >/dev/null 2>&1 || true
  sudo -n systemd-run --quiet --unit=cx7-restore --on-active=RESTORE_AFTERs --timer-property=AccuracySec=1s '"$HANDLER"' plug-in
  echo "dead-man restore armed: RESTORE_AFTER s"
fi
for bdf in '"$CX7_BDFS"'; do echo 1 | sudo -n tee /sys/bus/pci/devices/$bdf/remove >/dev/null; done
for i in $(seq 1 40); do [ -z "$(lspci -D -d 15b3:1021)" ] && break; sleep 0.5; done
test -z "$(lspci -D -d 15b3:1021)" || { echo "FAIL: CX-7 functions still present, NOT powering down"; exit 1; }
echo 0 | sudo -n tee '"$SYS"'/debug_state >/dev/null
test "$(cat '"$SYS"'/debug_state)" = 0 || { echo "FAIL: debug_state did not go to 0"; exit 1; }
test "$(cat /sys/class/net/'"$MGMT_IF"'/carrier)" = 1 || { echo "FAIL: management link lost carrier"; exit 1; }
echo "CX7_OFF mgmt=$(cat /sys/class/net/'"$MGMT_IF"'/speed)M"
'

ON_A='
set -uo pipefail
sudo -n systemctl stop cx7-restore.timer >/dev/null 2>&1 || true
if [ "$(cat '"$SYS"'/debug_state)" != 1 ] || [ "$(lspci -D -d 15b3:1021 | wc -l)" != 4 ]; then
  echo 1 | sudo -n tee '"$SYS"'/debug_state >/dev/null; sleep 3
  echo 1 | sudo -n tee /sys/bus/pci/devices/0000:00:00.0/rescan >/dev/null
  echo 1 | sudo -n tee /sys/bus/pci/devices/0002:00:00.0/rescan >/dev/null
fi
for i in $(seq 1 60); do [ "$(lspci -D -d 15b3:1021 | wc -l)" = 4 ] && break; sleep 1; done
test "$(lspci -D -d 15b3:1021 | wc -l)" = 4 || { echo "FAIL: $(lspci -D -d 15b3:1021 | wc -l)/4 CX-7 functions after 60 s"; exit 1; }
linkok() { for d in '"$RING_IFS"'; do [ "$(cat /sys/class/net/$d/operstate 2>/dev/null)" = up ] && [ "$(cat /sys/class/net/$d/speed 2>/dev/null)" = 200000 ] && [ "$(cat /sys/class/net/$d/mtu 2>/dev/null)" = 9000 ] && ip -o -4 addr show dev $d 2>/dev/null | grep -q inet || return 1; done; }
for i in $(seq 1 45); do linkok && break; [ $i = 15 ] && for d in '"$RING_IFS"'; do st=$(nmcli -t -f DEVICE,STATE dev 2>/dev/null | awk -F: -v d=$d "\$1==d{print \$2}"); [ "$st" = connected ] || sudo -n nmcli dev connect $d >/dev/null 2>&1 || true; done; sleep 1; done
linkok || { echo "FAIL: ring ports not 200G/IPv4/MTU9000: $(for d in '"$RING_IFS"'; do echo -n "$d=$(cat /sys/class/net/$d/operstate 2>/dev/null)/$(cat /sys/class/net/$d/speed 2>/dev/null)/$(cat /sys/class/net/$d/mtu 2>/dev/null)/$(ip -o -4 addr show dev $d 2>/dev/null | awk "{print \$4}") "; done)"; exit 1; }
for i in $(seq 1 30); do [ "$(rdma link show 2>/dev/null | grep -c " ACTIVE ")" = 4 ] && break; sleep 1; done
test "$(rdma link show 2>/dev/null | grep -c " ACTIVE ")" = 4 || { echo "FAIL: RDMA links: $(rdma link show | awk "{print \$2, \$4}" | paste -sd,)"; exit 1; }
echo "CX7_ON fns=4 ring=200G/200G rdma=4 ACTIVE"
'

ON_B='
set -uo pipefail
fail=0
for d in '"$RING_IFS"'; do
  ip=$(ip -o -4 addr show dev $d | awk "{print \$4}" | cut -d/ -f1); n=${ip##*.}; peer=${ip%.*}.$((3-n))
  if ping -c 2 -W 2 -M do -s 8972 -I $d $peer >/dev/null 2>&1; then echo "$d $ip -> $peer jumbo ping ok"; else echo "FAIL: jumbo ping $d $ip -> $peer"; fail=1; fi
done
sudo -n modprobe mstflint_access 2>/dev/null && echo "mstflint_access reloaded" || echo "WARN: mstflint_access not reloaded"
exit $fail
'

status_all() { for h in "${HOSTS[@]}"; do printf "%-11s %s\n" "$h" "$(sshq "$h" "$STATUS" 2>/dev/null || echo UNREACHABLE)"; done; }
for h in "${HOSTS[@]}"; do reachable "$h" || { say "$h unreachable; aborting"; exit 1; }; done

case "$CMD" in
  status) status_all ;;
  down)
    say "== down: CX-7 off on ${HOSTS[*]} (restore-after=${RESTORE_AFTER}s dry-run=$DRY_RUN)"
    for h in "${HOSTS[@]}"; do run_node "$h" "$PREP"; done
    say "-- powering the adapters down"
    pids=(); for h in "${HOSTS[@]}"; do run_node "$h" "${OFF//RESTORE_AFTER/$RESTORE_AFTER}" & pids+=($!); done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
    [ "$DRY_RUN" = 1 ] && { say "== dry run complete"; exit 0; }
    say "-- state"; status_all
    [ "$rc" = 0 ] && say "== down: CX-7 off on ${#HOSTS[@]} node(s); bring it back with: $0 --up$( [ "$RESTORE_AFTER" -gt 0 ] && echo " (or wait ${RESTORE_AFTER}s for the dead-man timer)")" || say "== at least one node FAILED; see above"
    exit $rc ;;
  up)
    say "== up: CX-7 on for ${HOSTS[*]} (dry-run=$DRY_RUN)"
    say "-- power up, rescan, wait for links"
    pids=(); for h in "${HOSTS[@]}"; do run_node "$h" "$ON_A" & pids+=($!); done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
    [ "$DRY_RUN" = 1 ] && { say "== dry run complete"; exit 0; }
    if [ "$rc" = 0 ]; then
      say "-- ring pings and mstflint reload"
      pids=(); for h in "${HOSTS[@]}"; do run_node "$h" "$ON_B" & pids+=($!); done
      for p in "${pids[@]}"; do wait "$p" || rc=1; done
    fi
    say "-- state"; status_all
    [ "$rc" = 0 ] && say "== up: CX-7 on and ring verified on ${#HOSTS[@]} node(s); start your serving stack" || say "== at least one node FAILED verification; do not start the serving stack until fixed"
    exit $rc ;;
esac
