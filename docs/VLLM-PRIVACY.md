# vLLM usage reporting is disabled

The managed Spark serving stack opts out of vLLM usage-statistics collection.
Both the release and NVMe-loader images, both serving launchers, and the NVMe
container controller set:

```text
VLLM_NO_USAGE_STATS=1
VLLM_DO_NOT_TRACK=1
DO_NOT_TRACK=1
```

Existing vLLM containers, including stopped rollback copies, also have the
supported `do_not_track` marker in their vLLM configuration directory. Host
markers cover root and the login user on each Spark. This protects retained
containers whose original Docker environment predates these defaults.

The installed vLLM caches its opt-out decision. Existing workers were stopped
before applying the setting and restarted with all three variables set; placing
a marker alone does not stop an already-running reporting thread.

The one-time marker migration is implemented in
[disable_usage_stats.py](../runtime/vllm029/disable_usage_stats.py). It runs on
each host with sudo, refuses to proceed while any detected vLLM container is
running, writes markers through Docker's archive API, and verifies them by
reading them back. It preserves the existing containers and does not print
their full environment or credentials.

```bash
# On a Spark, after stopping its vLLM containers:
sudo python3 ~/glm-vllm029-build/disable_usage_stats.py
```

The older host production launch scripts used by legacy recovery also pass
the three opt-out variables into Docker. The helper's `--launchers-only` mode
applies this change without restarting serving, keeps private host backups under
`/var/tmp/vllm-privacy-backups`, and validates shell syntax.

The completed deployment audit verified all four installed vLLM checks returned
`False`, all nine detected live vLLM processes inherited the opt-out variables,
and all 48 retained/current vLLM containers had environment or marker protection.
Serving health and count-to-100 generation passed after the restart. See the
[runtime audit](../results/vllm-privacy/verified.json) and
[marker audit](../results/vllm-privacy/opt-out-markers.json).

For a running managed container, this verifies the installed vLLM opt-out check:

```bash
docker exec vllm_glm53big python3 -c \
  'from vllm.usage.usage_lib import is_usage_stats_enabled; assert not is_usage_stats_enabled(); print("usage reporting disabled")'
```

These settings disable vLLM's outbound usage reporting. Local serving logs,
health checks, and Prometheus metrics remain available for operating the cluster.
The supported opt-out mechanisms are documented in
[vLLM's usage-statistics documentation](https://docs.vllm.ai/en/v0.15.1/usage/usage_stats.html).
