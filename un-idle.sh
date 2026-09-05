#!/usr/bin/env bash
# Bring the four Sparks back to the serving state after enter-low-power-idle-mode.sh.
# Status-driven and idempotent: every step looks at the node and fixes only what
# differs from the serving profile, so it is also safe to run with no idle journal.
# Run from the sandbox.
#
# Usage: ./un-idle.sh [--dry-run] [--status] [--no-relaunch] [--label NAME]
#
#   1. wake     Unreachable nodes get PLUG_ON_CMD (a smart-plug hook; %h is replaced by
#               the host, e.g. PLUG_ON_CMD='curl -s http://plug-%h.lan/relay/0?turn=on')
#               and Wake-on-LAN magic packets (NVIDIA says Spark ignores them), then up
#               to 5 min for SSH. A node that stays away needs the power button; the
#               script stops there with everything else untouched.
#   2. restore  Per node: cpus online, ring ports reconnected (NetworkManager), radios
#               back to the baseline, governor performance, GPU persistence on, clock
#               lock $GPU_LOCK, management link back to 10G (with a node-side revert
#               timer to the known-good 1G if 10G does not come up).
#   3. verify   governor, online mask, ring carrier + RDMA ACTIVE + IPv4 + MTU as in the
#               baseline, GPU clocks locked, eth 10G. Bounded waits, no resets: a failed
#               stage is reported and the script stops with SSH intact.
#   4. serving  Container running on all four and a real generation succeeds; otherwise
#               relaunch the default lane with ./rollout_dcp.sh <label> (~505 s, its own
#               auto-restore) unless --no-relaunch.
#   5. journal  Moved to results/idle-power/last.env; add-ons that recovered cleanly with
#               I_AM_PRESENT=1 get their <addon>-qualified marker.
set -uo pipefail
source "$(dirname "$0")/idle-power-lib.sh"
NORELAUNCH=0; LABEL="unidle-$(date +%m%d-%H%M)"
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRY_RUN=1 ;; --status) status_all; exit 0 ;; --no-relaunch) NORELAUNCH=1 ;;
  --label) LABEL="${2:?--label needs a name}"; shift ;;
  -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
esac; shift; done
export DRY_RUN
fail() { say "FAILED: $*"; say "nothing was reset or relaunched beyond this point; fix the node and rerun ./un-idle.sh"; exit 1; }

JOURNAL=0; [ -e "$STATE" ] && [ -n "$(state_get IDLE_TIER)" ] && JOURNAL=1
say "== un-idle: journal=$JOURNAL tier=$( [ $JOURNAL = 1 ] && state_get IDLE_TIER || echo none) dry-run=$DRY_RUN (log $LOG)"
if pgrep -f 'rollout_dcp.sh|restore_production.sh|deploy_kvtier.sh|deploy_slab' >/dev/null; then
  fail "a rollout/deploy is already running on this box"; fi

# ---- 1. wake
for h in "${HOSTS[@]}"; do
  reachable "$h" && continue
  say "-- $h unreachable"
  if [ -n "${PLUG_ON_CMD:-}" ]; then cmd="${PLUG_ON_CMD//%h/$h}"; say "  smart plug: $cmd"; [ "$DRY_RUN" = 1 ] || eval "$cmd" >/dev/null 2>&1; fi
  say "  sending Wake-on-LAN to ${MAC[$h]} (${IP[$h]})"; [ "$DRY_RUN" = 1 ] || { wol "$h"; sleep 5; wol "$h"; }
  [ "$DRY_RUN" = 1 ] && continue
  say "  waiting up to 300 s for SSH"; wait_ssh "$h" 300 || fail "$h did not come back; it needs the power button (or a PLUG_ON_CMD)"
  say "  $h is back"
done

# ---- 2. restore per node (parallel); radios follow the baseline when we have one
say "-- restoring the serving profile on every node"
for h in "${HOSTS[@]}"; do
  radios="sudo -n rfkill unblock wlan; sudo -n rfkill unblock bluetooth; echo 'radios unblocked'"
  if [ "$JOURNAL" = 1 ]; then
    rb=$(sfield "$(state_get "BASE_$h")" radios_blocked)
    case "$rb" in 0/*) ;; 1/*|2/*) radios="echo 'radios left as they were before idle ($rb blocked)'" ;; esac
  fi
  RESTORE='
for c in /sys/devices/system/cpu/cpu[0-9]*; do [ -e "$c/online" ] && [ "$(cat "$c/online")" = 0 ] && echo 1 | sudo -n tee "$c/online" >/dev/null; done
for d in enP2p1s0f0np0 enP2p1s0f1np1 enp1s0f1np1; do st=$(nmcli -t -f DEVICE,STATE dev 2>/dev/null | awk -F: -v d="$d" "\$1==d{print \$2}"); [ "$st" = connected ] || sudo -n nmcli dev connect "$d" >/dev/null 2>&1; done
for d in enP2p1s0f0np0 enP2p1s0f1np1 enp1s0f0np0 enp1s0f1np1; do sudo -n ip link set "$d" up 2>/dev/null; done
RADIOS_PLACEHOLDER
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo -n tee "$g" >/dev/null; done
sudo -n nvidia-smi -pm 1 >/dev/null 2>&1 || echo "WARN: nvidia-smi -pm 1 failed"
sudo -n nvidia-smi -lgc GPU_LOCK_PLACEHOLDER >/dev/null 2>&1 && echo "gpu clocks locked at GPU_LOCK_PLACEHOLDER" || echo "WARN: nvidia-smi -lgc failed"
echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor), cpus $(cat /sys/devices/system/cpu/online)"
'
  RESTORE=${RESTORE//RADIOS_PLACEHOLDER/$radios}; RESTORE=${RESTORE//GPU_LOCK_PLACEHOLDER/$GPU_LOCK}
  run_node "$h" "$RESTORE" &
done; wait

# ---- 2b. management link back to 10G, one node at a time
for h in "${HOSTS[@]}"; do
  spd=$( [ "$DRY_RUN" = 1 ] && echo 10000 || sshq "$h" "cat /sys/class/net/$MGMT_IF/speed" 2>/dev/null )
  [ "$DRY_RUN" = 1 ] && [ "$JOURNAL" = 1 ] && [ "$(state_get "ETH1G_$h")" = 1 ] && spd=1000
  [ "$spd" = 10000 ] && continue
  say "-- $h: management link at ${spd}M, renegotiating to 10G"
  run_node "$h" "sudo -n systemctl stop unidle-eth-revert.timer unidle-eth-revert.service >/dev/null 2>&1; sudo -n systemctl reset-failed unidle-eth-revert.service >/dev/null 2>&1; sudo -n systemd-run --quiet --unit unidle-eth-revert --on-active=150 /usr/sbin/ethtool -s $MGMT_IF autoneg on speed 1000 duplex full && echo 'revert-to-1G timer armed (150 s)'"
  run_node "$h" "sudo -n setsid sh -c 'sleep 1; /usr/sbin/ethtool -s $MGMT_IF autoneg on speed 10000 duplex full' >/dev/null 2>&1 </dev/null & echo 'renegotiating'"
  [ "$DRY_RUN" = 1 ] && continue
  sleep 10
  if wait_ssh "$h" 60 && [ "$(sshq "$h" "cat /sys/class/net/$MGMT_IF/speed" 2>/dev/null)" = 10000 ]; then
    sshq "$h" "sudo -n systemctl stop unidle-eth-revert.timer >/dev/null 2>&1; true"; say "  $h: back at 10G, revert timer cancelled"
  else
    say "  $h: 10G did not come up; waiting for the node's own revert to 1G"; sleep 130
    wait_ssh "$h" 90 && fail "$h is reachable at $(sshq "$h" "cat /sys/class/net/$MGMT_IF/speed")M but would not link at 10G" || fail "$h unreachable after the revert window"
  fi
done

# ---- 3. verify (bounded wait for the ring to retrain)
if [ "$DRY_RUN" = 1 ]; then say "== dry run complete; nothing was changed"; exit 0; fi
say "-- verifying"
ok=0
for attempt in $(seq 1 12); do
  bad=""
  for h in "${HOSTS[@]}"; do
    s=$(node_status "$h"); [ "$s" = UNREACHABLE ] && { bad="$bad $h:unreachable"; continue; }
    [ "$(sfield "$s" gov)" = performance ] || bad="$bad $h:gov=$(sfield "$s" gov)"
    [ "$(sfield "$s" online)" = 0-19 ] || bad="$bad $h:online=$(sfield "$s" online)"
    [ "$(sfield "$s" ring)" = up/up ] || bad="$bad $h:ring=$(sfield "$s" ring)"
    [ "$(sfield "$s" rdma)" = ACTIVE/ACTIVE ] || bad="$bad $h:rdma=$(sfield "$s" rdma)"
    [ "$(sfield "$s" eth)" = 10000M ] || bad="$bad $h:eth=$(sfield "$s" eth)"
    gpu=$(sfield "$s" gpu_W/MHz/pm); mhz=$(echo "$gpu" | cut -d/ -f2); pm=$(echo "$gpu" | cut -d/ -f3)
    [ "$pm" = Enabled ] || bad="$bad $h:pm=$pm"
    [ "${mhz:-0}" -ge 1900 ] 2>/dev/null || bad="$bad $h:sm=${mhz}MHz(lock?)"
    if [ "$JOURNAL" = 1 ]; then
      b=$(state_get "BASE_$h")
      [ "$(sfield "$s" ringip)" = "$(sfield "$b" ringip)" ] || bad="$bad $h:ringip=$(sfield "$s" ringip)"
      [ "$(sfield "$s" mtu)" = "$(sfield "$b" mtu)" ] || bad="$bad $h:mtu=$(sfield "$s" mtu)"
    else
      case "$(sfield "$s" ringip)" in *none*) bad="$bad $h:ringip=$(sfield "$s" ringip)" ;; esac
    fi
  done
  [ -z "$bad" ] && { ok=1; break; }
  say "  attempt $attempt: waiting on$bad"; sleep 5
done
for h in "${HOSTS[@]}"; do say "  $h $(node_status "$h")"; done
[ "$ok" = 1 ] || fail "node state does not match the serving profile:$bad"
say "  all four nodes match the serving profile"

# ---- 4. serving
running=1
for h in "${HOSTS[@]}"; do st=$(sshq "$h" "docker inspect -f '{{.State.Status}}' $NAME 2>/dev/null || echo missing"); [ "$st" = running ] || { say "  container on $h is '$st'"; running=0; }; done
gen_rc=1
if [ "$running" = 1 ] && health; then out=$(generate_ok 2>&1); gen_rc=$?; echo "$out" | tee -a "$LOG"; fi
if [ "$gen_rc" = 0 ]; then
  say "-- serving stack is up and generating"
elif [ "$NORELAUNCH" = 1 ]; then
  say "-- serving stack is not fully up; --no-relaunch given, leaving it"
else
  say "-- relaunching the default lane: ./rollout_dcp.sh $LABEL (~505 s)"
  ( cd "$WS" && ./rollout_dcp.sh "$LABEL" ) > "$STATE_DIR/$LABEL.console.log" 2>&1
  R=$(ls -d "$WS"/results/rollout-"$LABEL"-*/ 2>/dev/null | tail -1)
  if [ -n "$R" ] && grep -q "complete" "$R/rollout.log" 2>/dev/null; then say "  rollout complete: $R"
  else fail "rollout did not complete (see $STATE_DIR/$LABEL.console.log and ${R:-no result dir}); rollout_dcp.sh's own auto-restore applies"; fi
fi

# ---- 5. journal and qualification markers
if [ "$JOURNAL" = 1 ]; then
  if [ "$(state_get I_AM_PRESENT)" = 1 ]; then
    [ "$(state_get RING_DOWN)" = 1 ] && { touch "$STATE_DIR/ring-qualified"; say "  --ring-down qualified (marker dropped)"; }
    [ "$(state_get ETH1G)" = 1 ] && { touch "$STATE_DIR/eth1g-qualified"; say "  --eth-1g qualified (marker dropped)"; }
  fi
  mv "$STATE" "$STATE_DIR/last.env"
fi
say "== un-idle complete"
