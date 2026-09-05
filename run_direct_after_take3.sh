#!/usr/bin/env bash
cd "$(dirname "$0")"
until grep -q "sequence done\|stopping" results/deploy-dcp4-dflash-300k-kvtier3.log 2>/dev/null; do sleep 20; done
for h in spark-06c4 spark-365c spark-ddbf spark-a218; do ssh -o BatchMode=yes "napta2k@$h.local" 'sudo -n rm -rf /var/tmp/kvcache/_models_glm-5.3_*'; done
KVTIER_MODE=direct KVTIER_BOUNCE=48 ./deploy_kvtier.sh dcp4-dflash-300k-direct 307200 2048 6000000000 100000
