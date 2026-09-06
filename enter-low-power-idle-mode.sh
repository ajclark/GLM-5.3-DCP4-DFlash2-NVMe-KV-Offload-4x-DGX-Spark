#!/usr/bin/env bash
# Idle mode = the ConnectX-7 powered off on every node, cables left attached
# (~20 W per DGX Spark, 202 -> 120 W measured on four). Nothing else is touched.
# Stop whatever uses the ring first (the serving stack); the preflight refuses
# while the serving container runs or anything holds an RDMA device open.
#
# Usage: ./enter-low-power-idle-mode.sh [--hosts a,b] [--restore-after SECONDS] [--dry-run] [--status]
#   default: stays off until ./un-idle.sh; --restore-after 180 arms a node-side
#   dead-man timer that powers the adapter back on by itself (good for a first try).
set -uo pipefail
for a in "$@"; do case "$a" in --status) exec "$(dirname "$0")/cx7-power.sh" status ;; -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac; done
exec "$(dirname "$0")/cx7-power.sh" off --restore-after 0 "$@"
