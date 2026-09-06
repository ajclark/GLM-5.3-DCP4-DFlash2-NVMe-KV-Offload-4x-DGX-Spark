#!/usr/bin/env bash
# Leave idle mode = power the ConnectX-7 back on, on every node where it is off,
# and verify the ring: four PCIe functions, both ports at 200G with IPv4 and
# MTU 9000, RDMA ACTIVE, jumbo pings to the neighbours, mstflint reloaded.
# Nothing else is touched; start your serving stack afterwards.
#
# Usage: ./un-idle.sh [--hosts a,b] [--dry-run] [--status]
set -uo pipefail
for a in "$@"; do case "$a" in --status) exec "$(dirname "$0")/cx7-power.sh" status ;; -h|--help) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac; done
exec "$(dirname "$0")/cx7-power.sh" on "$@"
