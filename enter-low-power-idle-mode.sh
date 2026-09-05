#!/usr/bin/env bash
# Put the four Sparks into a lower-power idle state; ./un-idle.sh reverses it.
# Run from the sandbox. docs/IDLE-POWER.md says what each lever is worth and
# why the risky ones are gated.
#
# Usage: ./enter-low-power-idle-mode.sh [tier] [add-ons] [--yes] [--dry-run] [--status]
#
# Tiers
#   --light     (default) The serving stack stays up and keeps answering. Per node:
#               GPU clock lock released (nvidia-smi -rgc), CPU governor schedutil,
#               Wi-Fi + Bluetooth soft-blocked. Nothing touches memory, NCCL or the ring.
#   --deep      Drain, take the serving stack down (docker rm -f on all four, flushers
#               killed), then the light steps plus GPU persistence mode off. Discards
#               the in-memory KV pool (the NVMe slab tier survives); un-idle relaunches
#               the default lane, ~505 s. Needs --yes.
#   --shutdown  --deep, then `shutdown -h now` on every node: the only near-zero idle.
#               Power back on needs the power button or a smart plug (the BIOS boots on
#               AC restore); un-idle.sh runs PLUG_ON_CMD if you set one. Needs --yes.
#
# Add-ons. Each is first run with a human watching (I_AM_PRESENT=1); when un-idle
# recovers cleanly it drops results/idle-power/<addon>-qualified, which allows the
# add-on unattended from then on.
#   --cx7-off   (--deep only) Power the ConnectX-7 off with the cables attached via
#               NVIDIA's cx7-pcie-hotplug driver (./cx7-power.sh off): measured ~20 W per
#               node, 202 -> 120 W for four. CX7_RESTORE_AFTER=<s> arms a node-side
#               dead-man restore timer (default 0 = stay off until un-idle).
#   --eth-1g    Renegotiate the 10GbE management link to 1G. A node-side systemd timer
#               reverts to 10G after 150 s unless we cancel it once SSH is back.
#   --suspend   (--deep only) systemctl suspend (s2idle). NVIDIA states DGX Spark has
#               no Wake-on-LAN, so a suspended node may only come back via the power
#               button: an experiment with a finger on that button. Needs I_AM_PRESENT=1.
#
#   --yes       Acknowledge the KV-pool loss / relaunch cost of --deep and --shutdown,
#               or proceed while requests are still in flight.
#   --dry-run   Print every remote command instead of running it.
#   --status    One status line per node and exit (read-only).
set -uo pipefail
source "$(dirname "$0")/idle-power-lib.sh"
TIER=light; CX7=0; ETH1G=0; SUSPEND=0; YES=0
for a in "$@"; do case "$a" in
  --light) TIER=light ;; --deep) TIER=deep ;; --shutdown) TIER=shutdown ;;
  --cx7-off) CX7=1 ;; --eth-1g) ETH1G=1 ;; --suspend) SUSPEND=1 ;;
  --yes) YES=1 ;; --dry-run) DRY_RUN=1 ;; --status) status_all; exit 0 ;;
  -h|--help) sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown argument: $a (try --help)" >&2; exit 2 ;;
esac; done
export DRY_RUN
qualified() { [ -e "$STATE_DIR/$1-qualified" ] || [ "${I_AM_PRESENT:-0}" = 1 ]; }

say "== enter idle mode: tier=$TIER cx7-off=$CX7 eth-1g=$ETH1G suspend=$SUSPEND dry-run=$DRY_RUN (log $LOG)"

# ---- refuse the obviously wrong before touching anything
if [ -e "$STATE" ] && [ -n "$(state_get IDLE_TIER)" ] && [ "$DRY_RUN" != 1 ]; then
  say "already in idle mode since $(state_get ENTERED) (tier $(state_get IDLE_TIER)); run ./un-idle.sh first so the baseline journal is not overwritten"; exit 1; fi
[ "$CX7" = 1 ] && [ "$TIER" = light ] && { say "--cx7-off needs --deep: the serving stack runs on the ring"; exit 2; }
[ "$SUSPEND" = 1 ] && [ "$TIER" != deep ] && { say "--suspend needs --deep"; exit 2; }
if [ "$TIER" != light ] && [ "$YES" != 1 ]; then
  say "tier $TIER discards the in-memory KV pool and costs a ~505 s relaunch on un-idle: add --yes"; exit 2; fi
if [ "$TIER" = shutdown ] && [ -z "${PLUG_ON_CMD:-}" ] && [ "${I_AM_PRESENT:-0}" != 1 ]; then
  say "REFUSED: --shutdown needs a way back: set PLUG_ON_CMD (smart plug, see un-idle.sh) or I_AM_PRESENT=1 if you can press the power buttons."; exit 2; fi
if [ "$SUSPEND" = 1 ] && [ "${I_AM_PRESENT:-0}" != 1 ]; then
  say "REFUSED: NVIDIA states DGX Spark does not support Wake-on-LAN (forums.developer.nvidia.com/t/348168); a node that does not wake needs the power button. Set I_AM_PRESENT=1 only if you can press it."; exit 2; fi
[ "$CX7" = 1 ] && ! qualified cx7 && { say "REFUSED: --cx7-off is not yet qualified on this cluster; run it once with I_AM_PRESENT=1 while watching (un-idle drops the marker after a clean recovery)"; exit 2; }
[ "$ETH1G" = 1 ] && ! qualified eth1g && { say "REFUSED: --eth-1g is not yet qualified on this cluster; run it once with I_AM_PRESENT=1 while watching"; exit 2; }
if pgrep -f '(^|[ /])(rollout_dcp|restore_production|deploy_kvtier|deploy_slab[a-z_]*)\.sh( |$)' >/dev/null; then
  say "a rollout/deploy is running on this box; not touching the cluster"; exit 1; fi

# ---- preflight: reachability, serving state, drain
for h in "${HOSTS[@]}"; do reachable "$h" || { say "$h unreachable; aborting"; exit 1; }; done
STACK_UP=0; health && STACK_UP=1
if [ "$STACK_UP" = 1 ]; then
  n=$(requests_active)
  if [ "${n:-0}" != 0 ]; then
    say "$n request(s) running/waiting on the API"; [ "$YES" = 1 ] || { say "wait for them or pass --yes"; exit 1; }
  fi
fi
say "-- baseline"
declare -A BASE
for h in "${HOSTS[@]}"; do BASE[$h]=$(node_status "$h"); say "  $h ${BASE[$h]}"; done

# ---- journal (un-idle restores from it; never overwritten while idle)
if [ "$DRY_RUN" != 1 ]; then
  rm -f "$STATE"
  state_set ENTERED "$(date -Is)"; state_set IDLE_TIER "$TIER"; state_set STACK_WAS_UP "$STACK_UP"
  state_set CX7_OFF "$CX7"; state_set ETH1G "$ETH1G"; state_set SUSPENDED 0; state_set SHUTDOWN 0
  state_set I_AM_PRESENT "${I_AM_PRESENT:-0}"
  for h in "${HOSTS[@]}"; do state_set "BASE_$h" "${BASE[$h]}"; done
fi

# ---- deep / shutdown: take the serving stack down first
if [ "$TIER" != light ]; then
  say "-- taking the serving stack down"
  for h in "${HOSTS[@]}"; do
    run_node "$h" "docker rm -f $NAME >/dev/null 2>&1; pkill -f '[c]ache_flusher.sh' 2>/dev/null; echo 'container removed, flusher stopped'" &
  done; wait
  [ "$DRY_RUN" = 1 ] || sleep 5
fi

# ---- light steps on every node
LIGHT='
sudo -n nvidia-smi -rgc >/dev/null 2>&1 && echo "gpu clock lock released" || echo "WARN: nvidia-smi -rgc failed"
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo schedutil | sudo -n tee "$g" >/dev/null; done
echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
sudo -n rfkill block wlan; sudo -n rfkill block bluetooth
echo "wifi+bt soft-blocked ($(rfkill list | grep -c "Soft blocked: yes")/2)"
'
say "-- light steps (clock lock, governor, radios)"
for h in "${HOSTS[@]}"; do run_node "$h" "$LIGHT" & done; wait

if [ "$TIER" != light ]; then
  say "-- deep steps (persistence mode off; no CUDA context left)"
  for h in "${HOSTS[@]}"; do
    run_node "$h" 'sudo -n nvidia-smi -pm 0 >/dev/null 2>&1 && echo "gpu persistence mode off" || echo "WARN: nvidia-smi -pm 0 failed"' &
  done; wait
fi

if [ "$CX7" = 1 ]; then
  say "-- ConnectX-7 power off (cx7-power.sh off --restore-after ${CX7_RESTORE_AFTER:-0})"
  if [ "$DRY_RUN" = 1 ]; then echo "  [dry-run] $WS/cx7-power.sh off --restore-after ${CX7_RESTORE_AFTER:-0}" | tee -a "$LOG"
  else LOG="$LOG" "$WS/cx7-power.sh" off --restore-after "${CX7_RESTORE_AFTER:-0}" || { say "cx7-power.sh off failed on at least one node (state printed above); the other idle steps are applied; ./un-idle.sh recovers"; exit 1; }
  fi
fi

# ---- management link to 1G, one node at a time, with a node-side safety net
if [ "$ETH1G" = 1 ]; then
  say "-- management link 10G -> 1G (SSH drops for a few seconds per node)"
  for h in "${HOSTS[@]}"; do
    run_node "$h" "sudo -n systemctl stop idle-eth-revert.timer idle-eth-revert.service >/dev/null 2>&1; sudo -n systemctl reset-failed idle-eth-revert.service >/dev/null 2>&1; sudo -n systemd-run --quiet --unit idle-eth-revert --on-active=150 /usr/sbin/ethtool -s $MGMT_IF autoneg on speed 10000 duplex full && echo 'revert-to-10G timer armed (150 s)'"
    run_node "$h" "sudo -n setsid sh -c 'sleep 1; /usr/sbin/ethtool -s $MGMT_IF autoneg on speed 1000 duplex full' >/dev/null 2>&1 </dev/null & echo 'renegotiating'"
    [ "$DRY_RUN" = 1 ] && continue
    sleep 10
    if wait_ssh "$h" 60; then
      spd=$(sshq "$h" "cat /sys/class/net/$MGMT_IF/speed" 2>/dev/null)
      if [ "$spd" = 1000 ]; then
        sshq "$h" "sudo -n systemctl stop idle-eth-revert.timer >/dev/null 2>&1; true"
        say "  $h: management link at 1G, revert timer cancelled"; state_set "ETH1G_$h" 1
      else
        say "  $h: link came back at ${spd}M, not 1G; leaving the revert timer to restore 10G"; state_set "ETH1G_$h" 0
      fi
    else
      say "  $h: not reachable after renegotiation; the node reverts to 10G by itself in ~150 s"; sleep 130
      if wait_ssh "$h" 90; then say "  $h: back at $(sshq "$h" "cat /sys/class/net/$MGMT_IF/speed")M after self-revert"; state_set "ETH1G_$h" 0
      else say "  $h: STILL UNREACHABLE after the self-revert window; stopping here, no further nodes touched"; exit 1; fi
    fi
  done
fi

# ---- suspend / shutdown: the node goes away on purpose
if [ "$SUSPEND" = 1 ]; then
  say "-- suspending all four nodes (s2idle); un-idle will try Wake-on-LAN, expect to need the power button"
  for h in "${HOSTS[@]}"; do
    run_node "$h" "sudo -n ethtool -s $MGMT_IF wol g >/dev/null 2>&1; sudo -n setsid sh -c 'sleep 2; systemctl suspend' >/dev/null 2>&1 </dev/null & echo 'suspend scheduled'"
  done
  [ "$DRY_RUN" = 1 ] || { state_set SUSPENDED 1; sleep 20; }
elif [ "$TIER" = shutdown ]; then
  say "-- powering all four nodes off (shutdown -h now)"
  for h in "${HOSTS[@]}"; do
    run_node "$h" "sudo -n setsid sh -c 'sleep 2; shutdown -h now' >/dev/null 2>&1 </dev/null & echo 'shutdown scheduled'"
  done
  [ "$DRY_RUN" = 1 ] || { state_set SHUTDOWN 1; sleep 30; }
fi

# ---- report
if [ "$DRY_RUN" = 1 ]; then say "== dry run complete; nothing was changed"; exit 0; fi
say "-- state after"
for h in "${HOSTS[@]}"; do say "  $h $(node_status "$h")"; done
say "== idle mode entered (tier $TIER). Reverse with ./un-idle.sh; journal: $STATE"
