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
#   --suspend / --resume   Only the plugin step (quiesce / re-connect its RDMA state), no adapter action.
# Needs passwordless sudo on the nodes. Powering the adapter down kills every RDMA/NCCL
# connection on it; stop or expect to restart whatever uses the ring, UNLESS the workload
# runs NCCL with the hot-plug-aware net plugin (github.com/ajclark/nccl-net-hotplug): if a
# node's plugin answers on 127.0.0.1:CTL_PORT (default 5711, NCCL_HOTPLUG_CTL_PORT), --down first
# quiesces every plugin process in two phases (prepare on every node: gate the data path,
# refused if a collective is in flight, in which case everything is aborted and nothing is
# powered down; then commit on every node: tear the RDMA state down) and --up asks them to
# resume after the ring is verified. Then the serving stack survives the cycle with its
# communicators and CUDA graphs intact.
#   --no-plugin  skip the plugin suspend/resume even if status files are present
set -uo pipefail

HOSTS=(spark-06c4 spark-365c spark-ddbf spark-a218)   # edit for your cluster
SSH_USER="${SSH_USER:-$USER}"; SSH_SUFFIX="${SSH_SUFFIX:-.local}"
CONTAINER="${CONTAINER:-vllm_glm53big}"               # refuse --idle while this container runs
RING_IFS="enP2p1s0f0np0 enP2p1s0f1np1"                # the two cabled 200G ports
MGMT_IF="enP7s7"                                      # 10GbE management link (domain 0007)
CTL_PORT="${CTL_PORT:-5711}"                                 # NCCL_HOTPLUG_CTL_PORT of the plugin (127.0.0.1 inside each node)
CX7_BDFS="0000:01:00.0 0000:01:00.1 0002:01:00.0 0002:01:00.1"
SYS=/sys/devices/platform/MTKP0001:00/pcie_hotplug
HANDLER=/opt/nvidia/dgx-spark-mlnx-hotplug/mtk-hotplug-handler.sh

CMD=""; RESTORE_AFTER=0; DRY_RUN=0; PLUGIN=1
while [ $# -gt 0 ]; do case "$1" in
  --down|--idle) CMD=down ;; --up|--unidle) CMD=up ;; --status) CMD=status ;;
  --suspend) CMD=suspend ;; --resume) CMD=resume ;;   # plugin only, no adapter action
  --hosts) IFS=, read -r -a HOSTS <<<"${2:?}"; shift ;;
  --restore-after) RESTORE_AFTER="${2:?}"; shift ;;
  --dry-run) DRY_RUN=1 ;;
  --no-plugin) PLUGIN=0 ;;
  -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
esac; shift; done
[ -n "$CMD" ] || { echo "usage: $0 --down|--up|--status [--hosts a,b] [--restore-after SECONDS] [--dry-run]" >&2; exit 2; }

say()   { echo "[$(date '+%H:%M:%S')] $*"; }
sshq()  { if [ "$1" = localhost ]; then bash -c "$2"; else ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "$SSH_USER@$1$SSH_SUFFIX" "$2"; fi; }
reachable() { sshq "$1" true >/dev/null 2>&1; }
run_node()  { if [ "$DRY_RUN" = 1 ]; then echo "  [dry-run] $1: $2"; return 0; fi; sshq "$1" "$2" 2>&1 | sed "s/^/  $1: /"; return "${PIPESTATUS[0]}"; }

# --- hot-plug plugin control: one NCCL process per node listens on 127.0.0.1:CTL_PORT inside the
# node (host network namespace); one verb line in (status|prepare|commit|resume), one reply line out
# ("<state> comms=N idle=<s since the last NCCL send/receive> wanted=<0|1> <detail>"); a verb runs to
# completion before it answers. Reached with ssh + bash's /dev/tcp: no files anywhere.
# Quiescing is two-phase across the nodes: every process must answer "prepared" (data path gated,
# nothing in flight) before any is told "commit" (RDMA state torn down). A busy process makes the
# whole operation back out (resume everywhere, which drops the gates) and nothing is powered off.
plugin_query() {   # host verb -> reply line on stdout; non-zero if unreachable or no answer
  sshq "$1" "exec 3<>/dev/tcp/127.0.0.1/$CTL_PORT || exit 1; printf '%s\\n' '$2' >&3; IFS= read -r -t ${PLUGIN_TIMEOUT:-400} line <&3; [ -n \"\$line\" ] && printf '%s\\n' \"\$line\""
}
plugin_present() { plugin_query "$1" status >/dev/null 2>&1; }
plugin_hosts() { PHOSTS=(); local h; for h in "${HOSTS[@]}"; do plugin_present "$h" && PHOSTS+=("$h"); done; }
plugin_all() {       # verb expected-state(s, a|b) -> 0 when every node answered with one of them; all nodes in parallel
  local verb=$1 want=$2 h pids=() rc=0
  for h in "${PHOSTS[@]}"; do plugin_cmd "$h" "$verb" "$want" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}
plugin_quiesce() {   # -> 0 when every plugin process on every node is suspended; 1 (and nothing torn down) otherwise
  plugin_hosts; [ "${#PHOSTS[@]}" -gt 0 ] || return 0
  [ "${#PHOSTS[@]}" -eq "${#HOSTS[@]}" ] || { say "-- plugin answers on ${PHOSTS[*]} but not on every node: not powering anything off"; return 1; }
  say "-- plugin prepare on ${PHOSTS[*]} (gate the data path; refused if anything is in flight)"
  if ! plugin_all prepare 'prepared|suspended'; then
    say "-- a process was busy or failed at prepare: dropping the gates again, nothing torn down"
    plugin_all resume 'active|suspended' || true
    return 1
  fi
  say "-- plugin commit on ${PHOSTS[*]} (tear the RDMA state down)"
  plugin_all commit suspended || { say "-- commit FAILED somewhere: that process is in its failed state, the serving stack must be relaunched; not powering anything off"; return 1; }
  return 0
}
plugin_resume() {    # -> 0 when every plugin process on every node is active again
  plugin_hosts; [ "${#PHOSTS[@]}" -gt 0 ] || return 0
  say "-- plugin resume on ${PHOSTS[*]} (re-open devices, re-connect over the retained sockets)"
  plugin_all resume active
}
plugin_cmd() {   # host verb expected-state(s, a|b case pattern) -> the verb's reply must carry one of them
  local h=$1 verb=$2 want=$3 line st
  if [ "$DRY_RUN" = 1 ]; then echo "  [dry-run] $h: $verb -> expect $want"; return 0; fi
  line=$(plugin_query "$h" "$verb") || { echo "  $h: PLUGIN_UNREACHABLE (127.0.0.1:$CTL_PORT on the node)"; return 1; }
  st=${line%% *}
  case "$st" in $want) echo "  $h: PLUGIN_OK $line"; return 0 ;; *) echo "  $h: PLUGIN_FAIL $line"; return 1 ;; esac
}

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
for h in "${HOSTS[@]}"; do [ "$h" = localhost ] || reachable "$h" || { say "$h unreachable; aborting"; exit 1; }; done

case "$CMD" in
  suspend|resume)
    plugin_hosts; [ "${#PHOSTS[@]}" -gt 0 ] || { say "no node answers on 127.0.0.1:$CTL_PORT"; exit 1; }
    if [ "$CMD" = suspend ]; then plugin_quiesce; else plugin_resume; fi
    exit $? ;;
  status) status_all ;;
  down)
    say "== down: CX-7 off on ${HOSTS[*]} (restore-after=${RESTORE_AFTER}s dry-run=$DRY_RUN)"
    if [ "$PLUGIN" = 1 ]; then
      plugin_quiesce || { say "== plugin could not be quiesced on every node; not powering anything down"; exit 1; }
    fi
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
    if [ "$rc" = 0 ] && [ "$PLUGIN" = 1 ]; then
      plugin_resume || rc=1
    fi
    say "-- state"; status_all
    [ "$rc" = 0 ] && say "== up: CX-7 on and ring verified on ${#HOSTS[@]} node(s)" || say "== at least one node FAILED verification/resume; do not trust the serving stack until fixed"
    exit $rc ;;
esac
