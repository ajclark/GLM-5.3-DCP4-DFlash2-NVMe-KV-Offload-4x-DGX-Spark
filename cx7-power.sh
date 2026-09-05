#!/usr/bin/env bash
# cx7-power.sh: power the DGX Spark ConnectX-7 off and back on with the cables
# attached, through NVIDIA's cx7-pcie-hotplug platform driver (package
# dgx-spark-mlnx-hotplug). Measured on a 4-node cluster at idle: 202 -> 120 W,
# about 20 W per node, full software recovery. Details: docs/CX7-POWER.md.
#
# Usage: ./cx7-power.sh off|on|status [--hosts h1,h2,...] [--restore-after SECONDS] [--dry-run]
#   off     Preflight every node (hotplug enabled, exactly the four CX-7 functions,
#           serving container not running, no RDMA/MST users, firmware manager idle),
#           unload mstflint_access, arm a node-side dead-man restore timer
#           (default 180 s; 0 = stay off until `on`), remove the four PCIe functions,
#           then power the adapter down. Nothing is powered down unless all four
#           functions are gone. The 10GbE management link is never touched.
#   on      Power up + rescan (NVIDIA's plug-in sequence), wait for the four functions,
#           both ring ports at 200G with IPv4 and MTU 9000, RDMA ACTIVE on all
#           devices, then jumbo-frame ping to each ring neighbour and reload
#           mstflint_access. Exit status is non-zero if any node fails a check.
#   status  One line per node.
# Runs from a management box over SSH; needs passwordless sudo on the nodes.
set -uo pipefail
source "$(dirname "$0")/idle-power-lib.sh"
CMD="${1:-}"; shift || true
RESTORE_AFTER=180; ONLY=""
while [ $# -gt 0 ]; do case "$1" in
  --hosts) ONLY="${2:?}"; shift ;; --restore-after) RESTORE_AFTER="${2:?}"; shift ;;
  --dry-run) DRY_RUN=1 ;; -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown argument: $1" >&2; exit 2 ;;
esac; shift; done
export DRY_RUN
[ -n "$ONLY" ] && IFS=, read -r -a HOSTS <<<"$ONLY"
CX7_BDFS="0000:01:00.0 0000:01:00.1 0002:01:00.0 0002:01:00.1"
SYS=/sys/devices/platform/MTKP0001:00/pcie_hotplug
HANDLER=/opt/nvidia/dgx-spark-mlnx-hotplug/mtk-hotplug-handler.sh

CX7_STATUS='
echo "cx7=$(cat '"$SYS"'/debug_state 2>/dev/null) hotplug_enabled=$(cat '"$SYS"'/hotplug_enabled 2>/dev/null) fns=$(lspci -D -d 15b3:1021 2>/dev/null | wc -l) ring=$(for d in enP2p1s0f0np0 enP2p1s0f1np1; do echo -n "$(cat /sys/class/net/$d/operstate 2>/dev/null || echo absent)/$(cat /sys/class/net/$d/speed 2>/dev/null || echo -)/mtu$(cat /sys/class/net/$d/mtu 2>/dev/null || echo -) "; done)rdma_active=$(rdma link show 2>/dev/null | grep -c " ACTIVE ") mgmt=$(cat /sys/class/net/enP7s7/speed 2>/dev/null)M mstflint=$( [ -d /sys/module/mstflint_access ] && echo loaded || echo unloaded) timer=$(systemctl list-timers --no-legend cx7-restore.timer 2>/dev/null | awk "{print \$1, \$2, \$3}" | grep . || echo none) ctr=$(docker inspect -f "{{.State.Status}}" vllm_glm53big 2>/dev/null || echo none)"
'

PREFLIGHT='
set -euo pipefail
test -f /etc/nvidia/cx7-hotplug-enabled || { echo "FAIL: /etc/nvidia/cx7-hotplug-enabled missing"; exit 1; }
test "$(cat '"$SYS"'/hotplug_enabled)" = 1 || { echo "FAIL: hotplug_enabled != 1"; exit 1; }
test "$(cat '"$SYS"'/debug_state)" = 1 || { echo "FAIL: CX-7 is not powered (debug_state != 1)"; exit 1; }
st=$(docker inspect -f "{{.State.Running}}" vllm_glm53big 2>/dev/null || echo false); test "$st" = false || { echo "FAIL: serving container is running"; exit 1; }
for bdf in '"$CX7_BDFS"'; do
  test "$(cat /sys/bus/pci/devices/$bdf/vendor)" = 0x15b3 && test "$(cat /sys/bus/pci/devices/$bdf/device)" = 0x1021 || { echo "FAIL: $bdf is not a ConnectX-7"; exit 1; }
done
test "$(lspci -D -d 15b3:1021 | wc -l)" = 4 || { echo "FAIL: expected exactly four CX-7 functions"; exit 1; }
test "$(lspci -D -s 0000:: | grep -vc "PCI bridge")" = 2 && test "$(lspci -D -s 0002:: | grep -vc "PCI bridge")" = 2 || { echo "FAIL: unexpected devices in the CX-7 PCIe domains"; exit 1; }
case "$(readlink -f /sys/class/net/enP7s7/device)" in *0007:01:00.0) ;; *) echo "FAIL: management NIC is not in domain 0007"; exit 1 ;; esac
test "$(systemctl show nvidia-spark-mlnx-firmware-manager.service -p ActiveState --value)" != active || { echo "FAIL: mlnx firmware manager active"; exit 1; }
if sudo -n fuser -s /dev/infiniband/uverbs* /dev/infiniband/rdma_cm /dev/*_mstconf 2>/dev/null; then echo "FAIL: something holds an RDMA/MST device open:"; sudo -n fuser -v /dev/infiniband/uverbs* /dev/infiniband/rdma_cm /dev/*_mstconf 2>&1 | head -8; exit 1; fi
if test -d /sys/module/mstflint_access; then sudo -n modprobe -r mstflint_access; fi
test ! -d /sys/module/mstflint_access || { echo "FAIL: mstflint_access still loaded"; exit 1; }
echo PREFLIGHT_OK
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
test -z "$(lspci -D -d 15b3:1021)" || { echo "FAIL: CX-7 functions still present, NOT powering down (dead-man timer will restore)"; exit 1; }
echo 0 | sudo -n tee '"$SYS"'/debug_state >/dev/null
test "$(cat '"$SYS"'/debug_state)" = 0 || { echo "FAIL: debug_state did not go to 0"; exit 1; }
test "$(cat /sys/class/net/enP7s7/carrier)" = 1 || { echo "FAIL: management link lost carrier"; exit 1; }
echo "CX7_OFF mgmt=$(cat /sys/class/net/enP7s7/speed)M"
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
linkok() { for d in enP2p1s0f0np0 enP2p1s0f1np1; do [ "$(cat /sys/class/net/$d/operstate 2>/dev/null)" = up ] && [ "$(cat /sys/class/net/$d/speed 2>/dev/null)" = 200000 ] && [ "$(cat /sys/class/net/$d/mtu 2>/dev/null)" = 9000 ] && ip -o -4 addr show dev $d 2>/dev/null | grep -q inet || return 1; done; }
for i in $(seq 1 45); do linkok && break; [ $i = 15 ] && for d in enP2p1s0f0np0 enP2p1s0f1np1 enp1s0f1np1; do st=$(nmcli -t -f DEVICE,STATE dev 2>/dev/null | awk -F: -v d=$d "\$1==d{print \$2}"); [ "$st" = connected ] || sudo -n nmcli dev connect $d >/dev/null 2>&1 || true; done; sleep 1; done
linkok || { echo "FAIL: ring ports not 200G/IPv4/MTU9000: $(for d in enP2p1s0f0np0 enP2p1s0f1np1; do echo -n "$d=$(cat /sys/class/net/$d/operstate 2>/dev/null)/$(cat /sys/class/net/$d/speed 2>/dev/null)/$(cat /sys/class/net/$d/mtu 2>/dev/null)/$(ip -o -4 addr show dev $d 2>/dev/null | awk "{print \$4}") "; done)"; exit 1; }
for i in $(seq 1 30); do [ "$(rdma link show 2>/dev/null | grep -c " ACTIVE ")" = 4 ] && break; sleep 1; done
test "$(rdma link show 2>/dev/null | grep -c " ACTIVE ")" = 4 || { echo "FAIL: RDMA links: $(rdma link show | awk "{print \$2, \$4}" | paste -sd,)"; exit 1; }
echo "CX7_ON fns=4 ring=200G/200G rdma=4 ACTIVE"
'

ON_B='
set -uo pipefail
fail=0
for d in enP2p1s0f0np0 enP2p1s0f1np1; do
  ip=$(ip -o -4 addr show dev $d | awk "{print \$4}" | cut -d/ -f1); n=${ip##*.}; peer=${ip%.*}.$((3-n))
  if ping -c 2 -W 2 -M do -s 8972 -I $d $peer >/dev/null 2>&1; then echo "$d $ip -> $peer jumbo ping ok"; else echo "FAIL: jumbo ping $d $ip -> $peer"; fail=1; fi
done
sudo -n modprobe mstflint_access 2>/dev/null && echo "mstflint_access reloaded" || echo "WARN: mstflint_access not reloaded"
exit $fail
'

case "$CMD" in
  status)
    for h in "${HOSTS[@]}"; do printf "%-11s %s\n" "$h" "$(sshq "$h" "$CX7_STATUS" 2>/dev/null || echo UNREACHABLE)"; done ;;
  off)
    say "== cx7 off: hosts=${HOSTS[*]} restore-after=${RESTORE_AFTER}s dry-run=$DRY_RUN"
    for h in "${HOSTS[@]}"; do reachable "$h" || { say "$h unreachable; aborting"; exit 1; }; done
    say "-- preflight"; ok=1
    for h in "${HOSTS[@]}"; do run_node "$h" "$PREFLIGHT" || ok=0; done
    [ "$ok" = 1 ] || { say "preflight failed on at least one node; nothing powered down (mstflint_access may have been unloaded: harmless)"; exit 1; }
    say "-- powering the adapters down"
    pids=(); for h in "${HOSTS[@]}"; do run_node "$h" "${OFF//RESTORE_AFTER/$RESTORE_AFTER}" & pids+=($!); done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
    [ "$DRY_RUN" = 1 ] && { say "== dry run complete"; exit 0; }
    say "-- state"; for h in "${HOSTS[@]}"; do say "  $h $(sshq "$h" "$CX7_STATUS" 2>/dev/null || echo UNREACHABLE)"; done
    [ "$rc" = 0 ] && say "== CX-7 off on ${#HOSTS[@]} node(s); restore with: $0 on$( [ "$RESTORE_AFTER" -gt 0 ] && echo " (or wait ${RESTORE_AFTER}s for the dead-man timer)")" || say "== at least one node FAILED; see above"
    exit $rc ;;
  on)
    say "== cx7 on: hosts=${HOSTS[*]} dry-run=$DRY_RUN"
    for h in "${HOSTS[@]}"; do reachable "$h" || { say "$h unreachable; aborting"; exit 1; }; done
    say "-- power up, rescan, wait for links"
    pids=(); for h in "${HOSTS[@]}"; do run_node "$h" "$ON_A" & pids+=($!); done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
    [ "$DRY_RUN" = 1 ] && { say "== dry run complete"; exit 0; }
    if [ "$rc" = 0 ]; then
      say "-- ring pings and mstflint reload"
      pids=(); for h in "${HOSTS[@]}"; do run_node "$h" "$ON_B" & pids+=($!); done
      for p in "${pids[@]}"; do wait "$p" || rc=1; done
    fi
    say "-- state"; for h in "${HOSTS[@]}"; do say "  $h $(sshq "$h" "$CX7_STATUS" 2>/dev/null || echo UNREACHABLE)"; done
    [ "$rc" = 0 ] && say "== CX-7 on and verified on ${#HOSTS[@]} node(s)" || say "== at least one node FAILED verification; do not relaunch the serving stack until fixed"
    exit $rc ;;
  *) echo "usage: $0 off|on|status [--hosts h1,h2] [--restore-after SECONDS] [--dry-run]" >&2; exit 2 ;;
esac
