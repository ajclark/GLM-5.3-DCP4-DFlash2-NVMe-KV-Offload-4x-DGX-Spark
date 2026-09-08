#!/usr/bin/env bash
# Wait for spark-ddbf (10.99.1.149) / spark-a218 (10.99.1.31) to answer, then capture the
# previous boot's kernel log, errors, pstore and shutdown reason before anything else runs.
OUT="$(dirname "$0")"; declare -A IP=([spark-ddbf]=10.99.1.149 [spark-a218]=10.99.1.31)
declare -A DONE=()
S="ssh -o BatchMode=yes -o ConnectTimeout=6 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
for i in $(seq 1 720); do   # up to 2 h
  for h in spark-ddbf spark-a218; do
    [ -n "${DONE[$h]:-}" ] && continue
    ping -c 1 -W 1 "${IP[$h]}" >/dev/null 2>&1 || continue
    sleep 20   # let sshd come up
    $S "user@${IP[$h]}" 'echo "== $(hostname) $(date -u -Is) uptime: $(uptime -p)"; echo "== boots"; sudo -n journalctl --list-boots --no-pager 2>/dev/null | tail -3; echo "== previous boot: last 60 kernel lines"; sudo -n journalctl -k -b -1 --no-pager -o short-iso 2>/dev/null | tail -60; echo "== previous boot: errors/warnings (last 40)"; sudo -n journalctl -b -1 -p warning --no-pager -o short-iso 2>/dev/null | tail -40; echo "== previous boot: last 30 journal lines of any kind"; sudo -n journalctl -b -1 --no-pager -o short-iso 2>/dev/null | tail -30; echo "== pstore"; ls -la /sys/fs/pstore/ 2>/dev/null; for f in /sys/fs/pstore/*; do [ -f "$f" ] && { echo "-- $f"; sudo -n head -c 6000 "$f"; echo; }; done; echo "== /var/crash"; ls -la /var/crash/ 2>/dev/null | head; echo "== this boot: first 40 kernel lines"; sudo -n journalctl -k -b 0 --no-pager -o short-iso 2>/dev/null | head -40; echo "== cx7 now"; cat /sys/devices/platform/MTKP0001:00/pcie_hotplug/debug_state; lspci -D -d 15b3:1021 | wc -l; echo "== kernel/driver"; uname -r; head -1 /proc/driver/nvidia/version' > "$OUT/$h-capture.txt" 2>&1 && DONE[$h]=1 && echo "[$(date -u +%H:%M:%S)] captured $h -> $OUT/$h-capture.txt"
  done
  [ ${#DONE[@]} = 2 ] && { echo "both captured"; exit 0; }
  sleep 10
done
echo "timeout: captured ${!DONE[*]:-none}"
