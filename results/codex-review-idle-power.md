# Codex second opinion: lowering the Sparks' idle power (2026-09-05)

Prompt: the human asked for a brainstorm on cutting the idle draw of the
4x DGX Spark cluster (ASPM, the mlx NIC, RAM/CPU sleep) and for two scripts,
`enter-low-power-idle-mode.sh` and `un-idle.sh`. Codex (gpt-6-astra, xhigh)
got the read-only survey of one node (docs/IDLE-POWER.md §1) and the hard
constraint that the human cannot power-cycle a node. It ran nothing. Its
substance, condensed; two source claims were re-checked by the Claude
session and hold (NVIDIA release notes, forum thread 348168).

**Framing.** "Make warm idle the default, targeting modest savings without
sacrificing the 505-second startup. Keep suspend and device power-gating
outside unattended operation." Two corrections up front: the four mlx5
netdevs are two physical QSFP ports seen through two PCIe paths, so there
are no spare 200G PHYs to power off; and NVIDIA has stated Spark Wake-on-LAN
is unsupported despite the NIC advertising it. Estimates are per node,
hypotheses, not additive.

1. **GPU clock lock release + schedutil: the first warm-idle experiments.**
   `nvidia-smi -rgc` resets clocks, not the context. Budget 0-5 W, possibly
   zero or a regression if the live CUDA context keeps clocks high; the
   reported 8.71 W already limits the opportunity, and P0 alone does not
   mean high consumption. Keep persistence enabled while vLLM is alive.
   Governor schedutil preserving min/max: 0-2 W. Heavy LPI-3 residency means
   cpuidle already does the important work. Forcing 338 MHz mainly penalises
   wake-up responsiveness. Leave idle states alone; LPI-2 unused is not
   suspicious.
2. **Unused peripherals: low risk, small gains.** rfkill Wi-Fi/BT 0-1 W
   (interface-down is not radio-off; preserve previous block states).
   Display blanking 1-3 W only if a display is active (none here). USB
   autosuspend 0-1 W with an allowlist, not `powertop --auto-tune`. These plus
   item 1 form warm idle: no NIC, PCIe, allocation, cache or container changes.
3. **Cold idle: the plausible big lever is the ConnectX-7, not freeing RAM.**
   NVIDIA documents up to 18 W saved when the CX-7 is unused through its
   hot-plug power-management support; that does not establish that
   `ip link set down` with cables attached achieves it. Qualify link-down and
   recovery with the human present; do not automate driver unload, PCIe
   removal or forced D3. Freeing 115 GB does not power off LPDDR capacity;
   self-refresh is controller-managed, allocation is not activity. GPU clock
   release and optional persistence-off after all clients exit may add a few
   watts; measure before accepting an eight-minute restart.
4. **Human-present only.** 10G → 1G on the management link 1-3 W but it is the
   only SSH path. ASPM 0-3 W, touches the management NIC, CX-7 and root NVMe:
   leave default. NVMe APST already on, further tuning 0-1 W with timeout
   risk. Offlining 16 cores 0-1 W and complicates IRQ affinity: skip.
   Placebos: "free RAM", fixed minimum CPU frequency, persistence-off with
   live clients.
5. **s2idle + WoL: no unattended use.** Advertised `g` does not prove platform
   wake support; test only with someone able to recover a canary, arm an RTC
   alarm if supported, expect RDMA/NCCL reconstruction, verify GPU health,
   Docker and time sync. A GB10 s2idle failure report exists.
6. **Measure at the AC outlet before optimising further.** Plug-in wattmeter
   or logging metering plug (schedules off). USB-C measurement needs genuine
   48 V / 5 A EPR compatibility. Compare repeated, thermally settled A/B
   windows; log GPU power/clocks, LPI residency deltas, wake-ups and
   temperatures sparsely; none substitutes for wall watts.
7. **Make un-idle a verified transaction.** Idempotent restoration journal
   outside containers; repeated entry must not overwrite the original
   baseline. Restore online masks, every policy's governor/min/max,
   persistence, `-lgc 2000,2000`, confirm ~1995 MHz under work. Require all
   four SSH endpoints, 200G carrier, expected MTUs/IPs and active RDMA
   mappings before cold start; recreate the identical configuration, allow
   505 s plus margin, then a canary exercising all ranks, not HTTP health.
   Bounded retries; failures leave SSH intact and report the stage, never
   escalate to NIC/GPU resets.

## What the Claude session did with it

- Verified the two NVIDIA claims from source. The first read of the firmware
  (CX-7 root ports `HotPlug- PwrCtrl-`) concluded the 18 W path was not
  software-reachable; Codex then found NVIDIA's `dgx-spark-mlnx-hotplug`
  handler and platform driver, and the human and Codex measured 202 → 120 W
  for the four nodes the same evening (`docs/CX7-POWER.md`).
- Dropped core offlining and all placebo levers; kept ASPM as a manual
  experiment only.
- Built the scripts around item 7: baseline journal never overwritten,
  status-driven idempotent un-idle, IP/MTU/RDMA/clock verification before a
  relaunch, real generation as the canary, stop-and-report on any mismatch.
- Added the forum thread's practical route to near-zero idle: shutdown plus a
  metered smart plug with BIOS auto-boot, which also gives the cluster the
  remote power-cycle it lacks (`--shutdown` tier, `PLUG_ON_CMD` hook).
