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
| gpu2200 | 2200 MHz lock | Original governors | Higher-frequency throughput/energy tradeoff |
| gpu1800 | 1800 MHz lock | Original governors | Moderate frequency reduction |
| gpu1600 | 1600 MHz lock | Original governors | Larger reduction |
| schedutil | 2000 MHz lock | `schedutil` | CPU response and idle policy |
| idle_auto | Lock reset | `schedutil` | Loaded-idle screen; stock GPU clocks may rise on work |
| idle600 | 600 MHz lock | Original governors | Explicit low GPU clock, admitted only with `--idle-only` |
| idle300 | 300 MHz lock | Original governors | Lower idle floor, admitted only with `--idle-only` |

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
CPU governor changes, reported CPU frequency and exported idle-residency
counters treated as proxies rather than inferred watts. Missing counters stay
missing. The report divides each treatment by the geometric mean of its nearest
same-prompt baseline before and after; it retains absent energy measurements and
output hashes. These are small development screens, with no promotion claim.

Local tests verify per-core restoration, failure before mutation when a
governor is unavailable, and watchdog retention after a failed restore.

## Completed Spark screen, September 9

The repaired `8f684a4` runtime completed these controls with the original
model, KV capacity and network configuration. Four development prompts each
generated up to 256 tokens. Treatment ratios use the nearest same-prompt
baseline before and after. All profiles restored the 2000 MHz lock and each
CPU's original governor; the original serving containers were also restored.

| Active profile, fixed K7 | Decode tok/s ratio | Device decode J/token ratio |
|---|---:|---:|
| GPU 2200 MHz | 0.9893 | 1.1919 |
| GPU 1800 MHz | 0.9483 | 0.9080 |
| GPU 1600 MHz | 0.9537 | 0.8200 |
| CPU schedutil, GPU 2000 MHz | 0.9386 | 1.0804 |

The [active report](../../results/adaptive-next/cache-width-r1/power-active/power-profile-report.json)
retains all 36 requests and changing output hashes. This is a small screen,
without a confidence interval. In a separate 12-request baseline/1600/baseline
[adaptive screen](../../results/adaptive-next/cache-width-r1/power-adaptive/power-profile-report.json),
the ratios were 0.9290 tok/s and 0.8355 device J/token. That control retained
the same frozen 2000 MHz adaptive cost curve at both settings; it does not
claim to have calibrated or optimized the policy at 1600 MHz.

The 120-second loaded-idle brackets, excluding the first 20 seconds, measured
32.84/32.27/32.19 W for successive baselines, 32.42 W for schedutil, and
**47.08 W for GPU auto plus schedutil**. In the auto window the GPUs actually
rose to approximately 2405–2411 MHz. Releasing the clock lock therefore failed
to save idle device power on this setup. CPU reported frequency fell with
schedutil, but CPU watts were not measured; the GPU result alone cannot
determine whether whole-node CPU energy improved.

Separate 90-second idle brackets measured 32.01/31.90/31.81 W baseline,
**22.14 W at 600 MHz**, and **21.56 W at 300 MHz**. Recorded GPU clocks confirm
the requested low states. The 600 MHz result is approximately a 9.8 W saving
across four devices; going to 300 MHz adds only about 0.5 W after accounting
for the adjacent baselines. See the
[low-idle report](../../results/adaptive-next/cache-width-r1/power-idle-low/power-profile-report.json).

After clocks were restored and the final 90-second baseline elapsed, a
generation probe produced the correct count from 1 to 30 at 52.05 tok/s and
0.274 s TTFT, with no model reload. This proves resident state survived; it
does **not** measure immediate low-clock wake latency. No automatic idle
policy was installed. [IDLE-POWER.md](../IDLE-POWER.md) records the next
integration contract: authoritative server quiescence, restore before work,
and a restoration watchdog. A single Pi pane being idle is insufficient to
declare the shared server idle.
