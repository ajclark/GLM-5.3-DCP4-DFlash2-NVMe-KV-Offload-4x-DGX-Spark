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

**Correction, same evening:** the "up to 18 W when the ConnectX-7 is not in
use" from NVIDIA's January 2026 DGX OS release notes *is* software-reachable.
It is not PCIe slot power (the CX-7 root ports report `HotPlug- PwrCtrl-`,
which is what the survey looked for) but a platform driver, `cx7-pcie-hotplug`
(`/sys/devices/platform/MTKP0001:00/pcie_hotplug/debug_state`) with NVIDIA's
handler `/opt/nvidia/dgx-spark-mlnx-hotplug/mtk-hotplug-handler.sh removal|plug-in`.
The human and Codex measured it the same evening with the stack stopped:
**202 → 120 W for the four nodes, ~20 W per node, cables attached, full
software recovery.** Report and exact sequence: `docs/CX7-POWER.md`. The
other disappointment stands: the Realtek NIC's WoL bit is advertised but
NVIDIA says the platform does not honour it.

## 2. Levers, ranked by watts per unit of risk (estimates)

| lever | est. per node | risk | tier |
|---|---|---|---|
| Release the GPU clock lock (`nvidia-smi -rgc`) | 0-5 W (the 8 W idle floor already limits it; may be ~0 if the live context pins clocks) | none: clocks only, the context survives; benchmarks need the lock back | light |
| Governor `schedutil` (keep min/max) | 0-2 W (LPI residency already does most of the work) | none | light |
| rfkill Wi-Fi + Bluetooth | 0-1 W | none (management is wired) | light |
| Persistence mode off after the stack is down | 1-3 W (lets the GPU fully power-gate with no context) | none once no client remains | deep |
| Serving stack down | frees 115 GB but **saves ~0 W by itself** (DRAM refresh is capacity-based; the idle engine uses 3 % of a core); it is only the enabler for the levers below | 505 s relaunch, KV pool lost (slab tier survives) | deep |
| **ConnectX-7 powered off** via NVIDIA's cx7-pcie-hotplug handler (`docs/CX7-POWER.md`) | **~20 W measured** (202 → 120 W for four nodes) | PCIe functions removed and re-enumerated; NM reapplies IP/MTU; node stays reachable via 10GbE; unload `mstflint_access` first; dead-man restore timer | deep add-on (to replace `--ring-down`) |
| Ring ports admin-down only (`nmcli dev disconnect`) | superseded by the row above | | dropped |
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

## 3. The script (decision 2026-09-06: one script, ConnectX-7 only)

After the CX-7 measurement the human cut the tooling to one script for the one
lever worth double digits: `spark-idle.sh --down` runs the preflight (hotplug
enabled, exactly the four CX-7 functions, serving container not running, no
RDMA/MST users, firmware manager idle), unloads `mstflint_access`, removes the
four PCIe functions and powers the adapter down (optionally with a node-side
dead-man restore timer, `--restore-after SECONDS`); `spark-idle.sh --up`
powers it on and verifies four functions, 200G, IPv4, MTU 9000, RDMA ACTIVE,
jumbo pings, mstflint reloaded; `--status` is read-only. It does not touch the
GPU clock lock, governor, radios, management link or the serving stack; the
operator stops and starts the stack around it. The light-tier levers in
section 2 stay documented for reference only: none was measured and the
estimates are single-digit watts.

## 4. First run (with a meter on the plugs)

1. Stop the serving stack.
2. `./spark-idle.sh --down --hosts <one-node> --restore-after 180`; watch
   the meter drop and the node come back by itself.
3. `./spark-idle.sh --down` (all nodes, stays off), read the settled watts,
   `./spark-idle.sh --up`, confirm every node reports the ring verified.
4. Start the serving stack and run a real generation.

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
  reboot; a systemd unit on each node would make
  the serving profile boot-persistent.
- The slab NVMe cache wipes itself when the boot id changes; a shutdown tier
  used routinely would benefit from the clean-shutdown marker discussed in
  `docs/BOOT-TIME.md`.
- Wi-Fi and Bluetooth can be disabled in UEFI since the January 2026 DGX OS
  release if the radios are never used.
