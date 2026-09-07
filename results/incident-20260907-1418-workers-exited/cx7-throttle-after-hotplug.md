# The hot-plug cycle throttles the ConnectX-7 to ~13 Gb/s until a cold reboot

Found while explaining a decode regression on the DCP=2 lane (count100 144 -> 181 ms/cycle,
54.5 -> 43.3 tok/s) that appeared after the idle-power hot-plug work began.

## Evidence

1. **Profiler trace** (`results/dcp2-builtin-prof2` vs `results/dcp2-dflash-180k-prof`, per rank,
   one decode window): only the NCCL collectives slowed; compute is unchanged.

   | kernel | 2026-09-05 | 2026-09-07 |
   |---|---:|---:|
   | AllReduce_bf16_RING (2550) | 99 us | 249 us |
   | AllGather_RING (2700) | 40 us | 90 us |
   | ReduceScatter (1170) | 45 us | 128 us |
   | marlin_moe expert GEMM (2250) | 447 us | 439 us |
   | marlin / cutlass dense GEMM | 63 / 64 us | 63 / 64 us |

2. **RDMA bandwidth** on a ring link: `ib_write_bw -q 16 -s 1M` = **12.7 Gb/s** both directions,
   both ring links; latency (`ib_write_lat`) normal at ~2 us. The PCIe link is full (`126 Gb/s
   available, 32 GT/s x4`). So the cap is NIC-internal, not PCIe and not link integrity.

3. **dmesg**: after every hot-plug re-add (`spark-idle.sh --up`, and the earlier probe cycles)
   the driver logs, per function:
   `mlx5_pcie_event: PCIe slot power capability was not advertised.`
   `mlx5_pcie_event: Detected insufficient power on the PCIe slot (27W).`
   The first line appears ONLY after a re-add, never at the 12:59 cold boot. The mlx5 firmware,
   unable to confirm slot power, caps port throughput.

4. **NVIDIA's handler** (`/opt/nvidia/dgx-spark-mlnx-hotplug/mtk-hotplug-handler.sh`) plug-in path
   only does `echo 1 > /sys/bus/pci/devices/<root port>/rescan` on the two CX-7 root ports, exactly
   what `spark-idle.sh --up` does. Neither re-runs the firmware/ACPI step that advertises the slot
   power limit, so the throttle is not specific to our script.

## Consequence

The idle-power feature works (adapters off ~80 W saved, ring back in ~15 s), but every spin-up
leaves the ring at ~13 Gb/s until a cold reboot, which costs ~26% decode on this lane (more on
bandwidth-bound phases: prefill, long-context DCP exchange). Not cumulative: one cycle reaches the
capped state, more cycles do not worsen it, and a reboot clears it.

## Open / next

- Confirm cold boot restores full bandwidth: reboot one node, `ib_write_bw` before relaunch. Needs
  a downtime window.
- Candidate un-throttle without a full reboot, to add to `spark-idle.sh --up` (test in a window,
  serving down on that node): `mlxfwreset -d <bdf> reset` (PCI-level firmware reset, re-reads slot
  power); or a remove+rescan from the root complex that re-runs ACPI; or an mstconfig knob to
  disable the slot-power throttle if one exists.
- Until then: the idle-power feature trades ~26% decode for the idle-power saving. The watcher does
  not make it worse (the state is already capped), so it is safe to leave running or to stop,
  the user's call.
