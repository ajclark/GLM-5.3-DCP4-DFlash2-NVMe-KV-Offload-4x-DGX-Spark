#!/usr/bin/env bash
# Apply the DCP patch set to a checkout of the Spark fork's vLLM tree.
# Usage: apply.sh <path-to-vllm-package-dir>   (the dir containing v1/, model_executor/)
set -euo pipefail
TARGET="${1:?usage: apply.sh <vllm package dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
test -d "$TARGET/v1/attention/backends/mla" || { echo "not a vllm package dir: $TARGET" >&2; exit 2; }
for p in "$HERE"/*.patch; do
  echo "== $p"
  patch -p2 -d "$TARGET" --dry-run < "$p"
done
for p in "$HERE"/*.patch; do patch -p2 -d "$TARGET" < "$p"; done
# New files (no patch): the multi-node NVMe tier module.
cp "$HERE/../overlay/vllm/v1/kv_offload/tiering/multinode.py" "$TARGET/v1/kv_offload/tiering/multinode.py"
cp "$HERE/../overlay/vllm/v1/spec_decode/adaptive.py" "$TARGET/v1/spec_decode/adaptive.py"
cp "$HERE/../overlay/vllm/v1/spec_decode/confidence_trace.py" "$TARGET/v1/spec_decode/confidence_trace.py"
echo "applied"
