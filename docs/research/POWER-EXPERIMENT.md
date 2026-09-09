# Bounded frequency and governor experiments

The active four-node deployment uses the 2000 MHz GPU lock documented in
[HANDOVER.md](../HANDOVER.md), reading approximately 1995 MHz. All 20 CPU
policies per node use `performance`; `schedutil` is available. The September 9
read-only inventory found NVIDIA device power around 8–10 W per idle GPU,
no exposed GPU power-limit values, no module/CPU power and no energy hwmon.
The existing ConnectX-7 wall-power result remains a separate established
mechanism; these tests do not change network adapters or communicators.

NVIDIA documents clock locking and reset through
[`nvidia-smi`](https://docs.nvidia.com/deploy/nvidia-smi/index.html). The local
experiment tests these supported controls rather than assuming a working
watts-based cap on this hardware. No new model or larger KV pool is allocated.

`bench/spec_power_screen.py` admits only the explicitly named, already-held
experimental deployment and an idle endpoint. It saves original per-policy
CPU governors and checks that current graphics clocks agree with the known
2000 MHz lock before changing anything. The configured GPU lock cannot be
queried directly through the available interface; its saved restoration
value is therefore explicitly tied to the handover record plus this check.
Each node starts its own 45-second heartbeat watchdog, with a hard one-hour
limit, and restores the original settings if the controller disappears.
Failure on one node still restores every initialized node.

| Profile | GPU | CPU | Purpose |
|---|---|---|---|
| baseline | 2000 MHz lock | Original governors | Bracket comparisons and restore |
| gpu1800 | 1800 MHz lock | Original governors | Moderate frequency reduction |
| gpu1600 | 1600 MHz lock | Original governors | Larger reduction |
| schedutil | 2000 MHz lock | `schedutil` | CPU response and idle policy |
| idle_auto | Lock reset | `schedutil` | Loaded-idle screen; stock GPU clocks may rise on work |

The active screen uses fixed-seven verification to isolate hardware policy.
Compare the same coding/prose prompts and output limits, with baseline phases
on both sides of changed profiles. Record actual graphics clocks, endpoint
activity, per-node memory pressure and device power throughout. Every phase
has a bounded idle window followed by fixed work unless `--idle-only` is set.
The model remains resident. Separate later tests are required to combine a
winning frequency with recalibrated adaptive cap costs.

Device joules per token require all four sensors, bracketing samples and no
gaps exceeding five seconds, using the existing energy integrator. Report
active energy, decode rate and latency together: lower instantaneous watts
can lose if execution becomes longer. The idle comparison uses settled
windows after at least 20 seconds. It cannot measure CPU or whole-node savings;
there is no discovered remotely queryable wall meter in this workspace.

The existing idle-power notes estimate only a small gain from these controls.
That estimate is not new measured evidence. The output of this experiment is
a device-energy/latency tradeoff and a measured loaded-idle GPU floor, with
CPU governor changes reported as configuration rather than inferred watts.

Local tests verify per-core restoration, failure before mutation when a
governor is unavailable, and watchdog retention after a failed restore.
No frequency/governor experiment has run yet.
