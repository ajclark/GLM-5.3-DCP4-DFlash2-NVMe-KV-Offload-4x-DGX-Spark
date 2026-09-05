# Idle power on the four Sparks: what is controllable, what it is worth, how the scripts work

Written 2026-09-05 from a read-only survey of all four nodes (DGX OS 7.2.3, OTA
7.5.0, kernel 6.17.0-1026-nvidia) with the serving stack up but idle, plus a
Codex second opinion (`results/codex-review-idle-power.md`). No lever was
exercised on the cluster while writing this; every wattage below is an
estimate until a wattmeter is on the plugs (section 5).

## 1. What a Spark exposes at idle (measured)

| item | state on all four nodes | control |
|---|---|---|
| CPU | 20 cores online, `cppc_cpufreq`, governor `performance`, 2808 MHz reported, min 338 MHz; `acpi_idle` with LPI-0..3, LPI-3 (433 µs) carries almost all idle time | governor per cpu, min/max, online mask |
| GPU | `power.draw` 7.1-8.7 W at the 2000 MHz lock, P0, persistence on, 0 % util; one CUDA context (the vLLM worker, 108 GB) | `nvidia-smi -rgc` / `-lgc`, `-pm` |
| ConnectX-7 | one adapter, two cabled QSFP28 ports at 200G to the ring neighbours; each port appears as **two** netdevs/RDMA devices (PCIe 0000:01 and 0002:01 paths), NCCL uses `roceP2p1s0f0/f1`; root ports report `HotPlug- PwrCtrl-`, ASPM not supported; runtime PM `control=on`, D0 | `nmcli dev disconnect` / `ip link set down` only; no slot power control |
| 10GbE mgmt (Realtek r8127, `enP7s7`) | 10000M, WoL `g` advertised, ASPM L0s/L1 capable (disabled), on hot-plug slot 7 | speed renegotiation (1G/2.5G/5G/10G), WoL, slot power |
| Wi-Fi (mt7925e) + Bluetooth | Wi-Fi link down but not rfkill'd; BT up; spark-06c4 has one radio already soft-blocked | `rfkill`, or UEFI disable (Jan 2026 DGX OS) |
| NVMe (Samsung) | APST enabled, `default_ps_max_latency_us` 100000, ASPM L1 capable (disabled), slot 4 | nothing worth touching |
| PCIe ASPM | policy `default`; **every link shows `ASPM Disabled`**; CX-7 endpoints do not support it at all | `pcie_aspm` policy, would only affect NVMe, r8127, Wi-Fi, GPU link |
| Memory | 121 GB unified LPDDR5x, 117 GB used by the idle stack | none: self-refresh is controller-managed and independent of allocation |
| Suspend | `/sys/power/state` = `freeze mem`, `mem_sleep` = `s2idle` only | `systemctl suspend`; **NVIDIA states Spark does not support Wake-on-LAN** (forums.developer.nvidia.com/t/348168, NVES 2025-10-18) and a GB10 s2idle failure is on record (thread 380263) |
| Desktop | `multi-user.target`, no display manager, no DRM connectors active | nothing to save |
| Sensors | temperatures only (acpitz, nvme, mlx5, wifi); no power/energy hwmon, no BMC | GPU `power.draw` is the only on-box wattage |
| Idle CPU load | 0.07; vLLM container 3 % of one core | nothing to save |

Two things we hoped for are not there: the "up to 18 W when the ConnectX-7 is
not in use" from NVIDIA's January 2026 DGX OS release notes is delivered through
hot-plug power management, and on this firmware the CX-7 root ports expose no
hot-plug slot or power controller, so software cannot trigger it; and the Realtek
NIC's WoL bit is advertised but NVIDIA says the platform does not honour it.

## 2. Levers, ranked by watts per unit of risk (estimates)

| lever | est. per node | risk | tier |
|---|---|---|---|
| Release the GPU clock lock (`nvidia-smi -rgc`) | 0-5 W (the 8 W idle floor already limits it; may be ~0 if the live context pins clocks) | none: clocks only, the context survives; benchmarks need the lock back | light |
| Governor `schedutil` (keep min/max) | 0-2 W (LPI residency already does most of the work) | none | light |
| rfkill Wi-Fi + Bluetooth | 0-1 W | none (management is wired) | light |
| Persistence mode off after the stack is down | 1-3 W (lets the GPU fully power-gate with no context) | none once no client remains | deep |
| Serving stack down | frees 115 GB but **saves ~0 W by itself** (DRAM refresh is capacity-based; the idle engine uses 3 % of a core); it is only the enabler for the levers below | 505 s relaunch, KV pool lost (slab tier survives) | deep |
| Ring ports admin-down (all four CX-7 functions) | unknown, 0-18 W; NVIDIA's 18 W needs the slot power-off we do not have | ports must retrain and NM must reapply IP/MTU; node stays reachable via 10GbE | deep add-on, qualify with a meter |
| 10GbE mgmt link 10G → 1G | 1-3 W (10GBASE-T PHY) plus the same on the switch port | renegotiating the only SSH path; node-side revert timer covers it | add-on, qualify once |
| PCIe ASPM `powersave` | 0-3 W (NVMe, Realtek, Wi-Fi, GPU link only) | Realtek + ASPM is a classic hang; not automated | human-present experiment |
| Offline 16 of 20 cores | 0-1 W over good idle residency | IRQ affinity, restoration | dropped |
| Fixed low CPU frequency, NVMe tuning, "free RAM" | placebo | | dropped |
| s2idle suspend | most of the node | **no supported wake**; a stuck node needs the button | gated experiment |
| Shutdown + smart plug | everything (a few W for the plug) | needs a plug (or a walk); ~2 min boot + 505 s relaunch | shutdown tier |

Whole-node idle is unknown without a meter. Light tier probably recovers a
few watts per node; deep with the ring down could be 10-20 W per node if
admin-down really idles the SerDes; shutdown is the only way to a near-zero
cluster, and the smart plug that makes it recoverable also gives this cluster
the remote power-cycle it does not have today (the forum thread's author does
exactly this: BIOS "auto boot" on AC + a phone-controlled plug).

## 3. The scripts

`./enter-low-power-idle-mode.sh` and `./un-idle.sh` share `idle-power-lib.sh`
(hosts, MACs, IPs, one-SSH status line, journal, WoL sender). Both run from
the sandbox; both have `--dry-run` (prints every remote command) and
`--status` (read-only, one line per node: governor, MHz, online mask, mgmt
link speed, ring operstate, RDMA state, ring IPv4/MTU, radios, GPU W/MHz/pm,
container, MemAvailable).

Enter, tiers: `--light` (default, serving stays up: `-rgc`, schedutil,
rfkill), `--deep` (drain, `docker rm -f` on all four + flushers, light steps,
`-pm 0`; needs `--yes`), `--shutdown` (deep + `shutdown -h now`; needs
`--yes` and either `PLUG_ON_CMD` or `I_AM_PRESENT=1`). Add-ons: `--ring-down`
(deep only), `--eth-1g`, `--suspend` (deep only). Add-ons refuse to run
unattended until qualified: run once with `I_AM_PRESENT=1` while watching;
when un-idle recovers cleanly it drops `results/idle-power/<addon>-qualified`.
`--suspend` always needs `I_AM_PRESENT=1`.

Safety in enter: refuses if already idle (the baseline journal is never
overwritten), if a rollout/deploy is running on the sandbox, if a node is
unreachable, or if requests are in flight (override `--yes`). The 1G
renegotiation arms a node-side `systemd-run --on-active=150` timer that
reverts to 10G unless the sandbox confirms SSH at 1G and cancels it; nodes
are done one at a time and the script stops at the first one that does not
come back. Journal: `results/idle-power/current.env` with the tier, add-ons,
whether the stack was up, `I_AM_PRESENT`, and the full baseline status line
per node.

Un-idle is status-driven and idempotent (fine with no journal): (1) wake
unreachable nodes via `PLUG_ON_CMD` (`%h` = host) and WoL, wait ≤300 s, stop
if one stays away; (2) per node: cpus online, `nmcli dev connect` on the ring
connections (`roce-p0`, `roce-p2p`, `roce-p2p-unused`) + `ip link set up`,
radios back to the baseline, governor performance, `-pm 1`, `-lgc 2000,2000`;
(2b) mgmt link back to 10G with a revert-to-1G timer; (3) verify for ≤60 s:
governor, `0-19` online, ring `up/up`, RDMA `ACTIVE/ACTIVE`, ring IPv4 and MTU
equal to the baseline, `eth=10000M`, pm Enabled, SM clock ≥1900 MHz; a mismatch
is reported and the script stops, it never resets a NIC or GPU; (4) container
running on all four **and a real generation** (not `/health`), else
`./rollout_dcp.sh <label>` with the daily defaults unless `--no-relaunch`;
(5) journal → `last.env`, qualification markers.

Exercised so far: `--status` and every tier's `--dry-run` against the live
cluster, and the gates (`--deep` without `--yes`, unqualified add-ons,
`--suspend` without presence, `--ring-down` without `--deep`). Not yet
exercised: a real light entry and un-idle (the stack was in use), any
add-on, any measurement.

## 4. Qualification protocol (with a human, meter on the plugs)

1. Light: `./enter-low-power-idle-mode.sh` during a quiet hour, read the
   meter after 10 min, `./un-idle.sh`, confirm the count100 hash and cycle
   time are unchanged. Decide whether light is worth automating at all.
2. `--eth-1g` once with `I_AM_PRESENT=1`, one node at a time is built in;
   watch the 150 s timer do nothing.
3. Deep + `--ring-down` with `I_AM_PRESENT=1`: this is the only lever that
   can be worth double digits; the meter decides. Un-idle then relaunches
   (505 s) and checks the ring came back with the right IPs/MTU.
4. ASPM `powersave` on one node, manually, meter in hand; revert on any
   Realtek hiccup.
5. Suspend: only if steps 1-4 leave you wanting more, with a finger on the
   button, one node, short interval first.

## 5. Measuring

There is no on-box wattage except GPU `power.draw`. The cheapest truth is an
AC wattmeter per node (a metered smart plug doubles as the remote power
switch for the shutdown tier; disable its schedules). A USB-C inline meter
must be rated for 48 V / 5 A EPR (240 W) or it will not pass the Spark's
supply. Compare thermally settled 10-minute windows, A/B twice. On-box
proxies worth logging alongside: `nvidia-smi --query-gpu=power.draw,clocks.sm`,
per-CPU LPI residency deltas from `/sys/devices/system/cpu/cpu*/cpuidle/state*/time`,
and the acpitz/mlx5/nvme temperatures.

## 6. Cold-boot housekeeping this surfaced

- The 2000 MHz clock lock and `performance` governor do not survive a
  reboot; un-idle reapplies both, and a systemd unit on each node would make
  the serving profile boot-persistent.
- The slab NVMe cache wipes itself when the boot id changes; a shutdown tier
  used routinely would benefit from the clean-shutdown marker discussed in
  `docs/BOOT-TIME.md`.
- Wi-Fi and Bluetooth can be disabled in UEFI since the January 2026 DGX OS
  release if the radios are never used.
