#!/usr/bin/env bash
# Persistent IPv4 for rocep1s0f0 (enp1s0f0np0, port 0's PCIe twin on domain 0000), needed by NCCL on both PCIe
# links (docs/NETWORK.md, step E3 of results/prefill-item2-20261006/REPORT.md). Same shape as the existing roce-p0 profile (manual IPv4, MTU 9000, IPv6 off,
# autoconnect); NetworkManager writes it to /etc/netplan/90-NM-<uuid>.yaml like the others.
#   bash node/roce-p0-twin.sh apply      create/activate roce-p0-twin on all four Sparks, then show state
#   bash node/roce-p0-twin.sh rollback   delete the profile (the device returns to NetworkManager's DHCP default)
#   bash node/roce-p0-twin.sh show
# Applied 2026-10-06 (survived a reboot of all four nodes the same day). Needs passwordless sudo on the Sparks.
set -u
declare -A IP=([spark-06c4]=192.168.106.2 [spark-a218]=192.168.106.1 [spark-365c]=192.168.107.1 [spark-ddbf]=192.168.107.2)
NAME=roce-p0-twin
case "${1:-show}" in
  apply) for h in spark-06c4 spark-a218 spark-365c spark-ddbf; do
           ssh -o BatchMode=yes "$h.local" "set -e
             nmcli -t -f NAME connection show | grep -qx $NAME || sudo nmcli connection add type ethernet con-name $NAME \
               ifname enp1s0f0np0 ipv4.method manual ipv4.addresses ${IP[$h]}/24 ipv6.method disabled \
               802-3-ethernet.mtu 9000 connection.autoconnect yes connection.autoconnect-priority 10
             sudo nmcli device set enp1s0f0np0 managed yes
             sudo nmcli connection up $NAME >/dev/null" && echo "$h applied ${IP[$h]}"
         done ;;
  rollback) for h in spark-06c4 spark-a218 spark-365c spark-ddbf; do
           ssh -o BatchMode=yes "$h.local" "sudo nmcli connection delete $NAME" && echo "$h rolled back"
         done ;;
  show) ;;
  *) echo "usage: $0 apply|rollback|show" >&2; exit 2 ;;
esac
for h in spark-06c4 spark-a218 spark-365c spark-ddbf; do
  ssh -o BatchMode=yes "$h.local" "echo \"$h: \$(ip -4 -br addr show enp1s0f0np0 | awk '{print \$3}') mtu \$(cat /sys/class/net/enp1s0f0np0/mtu) nm=\$(nmcli -t -f DEVICE,STATE,CONNECTION device status | grep '^enp1s0f0np0:' | cut -d: -f2-)\""
done
