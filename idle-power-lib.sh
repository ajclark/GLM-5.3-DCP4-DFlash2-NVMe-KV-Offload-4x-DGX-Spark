#!/usr/bin/env bash
# Shared definitions for cx7-power.sh (sourced, not executed).
# Edit HOSTS for your cluster; everything else is plumbing.
HOSTS=(spark-06c4 spark-365c spark-ddbf spark-a218)
SSH_USER="${SSH_USER:-napta2k}"
SSH_SUFFIX="${SSH_SUFFIX:-.local}"          # host -> ssh target: napta2k@spark-06c4.local
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$WS/results/idle-power"; mkdir -p "$LOG_DIR"
LOG="${LOG:-$LOG_DIR/$(date +%Y%m%d-%H%M%S)-$(basename "$0" .sh).log}"
DRY_RUN="${DRY_RUN:-0}"

say()   { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
sshq()  { ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "$SSH_USER@$1$SSH_SUFFIX" "$2"; }
reachable() { sshq "$1" true >/dev/null 2>&1; }
run_node()  { # host, remote command; prints instead of running under DRY_RUN=1
  if [ "$DRY_RUN" = 1 ]; then echo "  [dry-run] $1: $2" | tee -a "$LOG"; return 0; fi
  sshq "$1" "$2" 2>&1 | sed "s/^/  $1: /" | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
