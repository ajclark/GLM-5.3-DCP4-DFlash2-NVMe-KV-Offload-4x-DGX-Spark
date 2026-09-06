# Incident 2026-09-06: spark-a218 and spark-ddbf went offline

All times UTC. Cluster: 4x DGX Spark; a218 and ddbf run kernel 6.17.0-1031-nvidia /
driver 580.173.02, 06c4 and 365c run 1026 / 580.159.03.

- 23:16-23:17 (05 Sep) Codex stops the serving containers for the CX-7 power experiment.
- 23:19:25 canary: spark-06c4 CX-7 off; 23:22:57 restored.
- 23:25:44 all four CX-7 off; 23:28:44 timers restore all four; 23:29:11 all verified
  (4 functions, 200G, IPs, MTU, RDMA, pings); ~23:32 mstflint_access reloaded.
  Survivors' kernel logs at re-probe: `mlx5_pcie_event: PCIe slot power capability was
  not advertised` and `Detected insufficient power on the PCIe slot (27W)` on every function.
- 00:09-00:19 (06 Sep) SKIP_PREFLIGHT=1 rollout dcp2-dflash-180k-prod3: all four ranks up,
  generation OK. 365c logs one NVRM NV_ERR_NO_MEMORY during model load (seen before, benign).
- 00:23-00:25 spark-idle.sh --status and --dry-run runs from the sandbox (read-only SSH).
- **00:40:34 spark-a218 drops off**: 06c4 logs `enP2p1s0f0np0: Link down` (its link to a218).
- **00:42:11 spark-ddbf drops off**: 365c logs `enP2p1s0f0np0: Link down` (its link to ddbf).
- Wall power falls to ~100 W (two idle Sparks); UniFi shows both 10GbE ports offline;
  no ping, no ARP for 192.168.1.31 / .149.
- 01:39 a pi request hangs: rank 0's EngineCore logs "No available shared memory
  broadcast block found in 60 seconds" every minute (workers on the dead nodes).
- Survivors: no periodic job touches the adapter (mlnx firmware manager inactive, no
  timers), no MST users, temps normal, uptime 2d23h.

Open: cause on a218/ddbf (need their previous-boot journal + pstore:
`watch-and-capture.sh` waits for them). Correlation only: both dead nodes are the two on
the newer kernel/driver build and both went through the hot-plug cycle ~70 min earlier.
Sequential (97 s apart), so not a shared power event.

## Resolution (01:46-01:55 UTC)

Both nodes were powered on by the human at ~01:45 UTC. Their previous-boot journals
(`spark-a218-capture.txt`, `spark-ddbf-capture.txt`, plus a targeted pull) settle the cause:

```
spark-a218  2026-09-05T17:40:21-07:00  systemd-logind: Power key pressed short.  -> Powering off...
spark-ddbf  2026-09-05T17:41:58-07:00  systemd-logind: Power key pressed short.  -> Powering off...
```

Each was an **orderly power-off started by a short press of the physical power button**,
98 s apart. No kernel error, no pstore entry, no kdump; `last -x` records a clean
"shutdown system down". The CX-7 hot-plug cycle, the newer kernel build, the slot-power
warnings and every script are exonerated: the correlation noted above was coincidence.

Both nodes came back healthy (CX-7 powered, four functions, ring at 200G with the right IPs,
RDMA ACTIVE, governor performance, 115 GB free). The 2000 MHz GPU clock lock was re-applied
by hand (it does not survive a reboot); the slab caches on those two nodes were wiped by
design (boot id changed). Serving relaunched with `SKIP_PREFLIGHT=1 ./rollout_dcp.sh
dcp2-dflash-180k-prod4`.

Follow-up worth considering: logind's default `HandlePowerKey=poweroff` means one short
press shuts a Spark down. `HandlePowerKey=ignore` (or requiring a long press via
`HandlePowerKeyLongPress=poweroff`) in `/etc/systemd/logind.conf.d/` on each node would make
an accidental touch harmless; power-on still needs the button either way.
