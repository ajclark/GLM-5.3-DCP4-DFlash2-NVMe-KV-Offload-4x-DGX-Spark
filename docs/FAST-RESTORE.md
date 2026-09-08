# Fast restore: power-off to serving in under a minute — feasibility (2026-09-08)

Research summary (sub-agent, grounded in docs/BOOT-TIME.md, CX7-POWER.md, IDLE-POWER.md,
NCCL-HOTPLUG-TEST.md + web). Verdict first.

## Under 60 s from TRUE power-off (0 W / S5) is not physically achievable here

Two fixed floors no snapshot removes:
1. Firmware POST + kernel boot: tens of seconds, uncontrollable on this platform.
2. 95 GB of weights must re-enter unified LPDDR5x from NVMe (the weights ARE system RAM, so a
   0 W state loses them). Pure-I/O floor ~10-19 s/node; today's page-fault loader takes 283 s
   at ~0.6 GB/s (the loader, not NVMe, is the bottleneck).

Add GPU re-init + CUDA-graph capture (14-24 s) + NCCL ring formation. Realistic floor from S5
is ~2-3 minutes. Sub-minute is only reachable if RAM stays powered (never truly 0 W).

## Ranked options

| technique | keeps weights | keeps GPU ctx/graphs | keeps RDMA | 0 W -> serving | risk |
|---|---|---|---|---|---|
| Resident stack + CX-7 hot-plug off (~30 W/node, NOT 0 W) | yes | yes | no, rebuilt | ~30-60 s from idle | proven on-cluster |
| cuda-checkpoint + CRIU | yes | yes (incl JIT) | no, rebuilt | blocked on driver 580 | needs 595+ aarch64 |
| Hibernation (S4) | yes (image) | partly, GPU re-init | no | ~90-150 s + POST | unproven on GB10/aarch64 |
| kexec | no | no | no | trims POST only | accelerator only |
| Optimized reload (no snapshot) | reloads fast | recaptures | rebuilt | ~2-3 min from S5 | in progress |
| S3 suspend-to-RAM | yes | yes | no | not available on GB10 (s2idle broken) | - |

## Recommendation

The platform's real sub-minute answer is the feature we already built and just paused: keep the
stack resident, power down only the ConnectX-7, rebuild the ring on demand with the NCCL
hot-plug plugin (~30 W/node standby, dominated by 128 GB DRAM self-refresh). Its only blocker is
the CX-7 bandwidth throttle (results/incident-20260907-1418-workers-exited/cx7-throttle-after-hotplug.md),
not the resume time. cuda-checkpoint + CRIU is the most promising true snapshot (preserves CUDA
graphs and JIT without recapture) but needs a 595+ aarch64 GB10 driver; revisit then. Hibernation
and S3 are doubtful/unavailable on this platform. For genuine near-0 W, accept ~2-3 min via the
disk-image loader + shutdown + a smart plug; kexec can trim POST on planned cycles.

## Next experiments (a downtime window, ordered)

1. End-to-end resume timing of the resident-stack path: drop CUDA graphs, destroy the process
   groups, spark-idle.sh --down; then --up, rebuild groups, recapture graphs, first token.
   Measure --down -> first token, target < 60 s. (stack up; needs the engine suspend/resume RPC)
2. True cold floor: docs/BOOT-TIME.md experiment 1 (AOT cache on NVMe, autotune off, direct-I/O
   loader). Measure docker run -> serving. (stack down)
3. Cheap S4 viability probe: cat /sys/power/state /sys/power/disk; size a ~120 GB resume device
   before committing. Drop hibernation if S4 is not cleanly offered. (no stack)
4. cuda-checkpoint smoke test — only after a 595+ aarch64 driver: --toggle on one idle
   single-GPU worker, measure ~95 GB checkpoint/restore to NVMe, audit for UVM/IPC. Not on 580.

Full report with sources: results/fast-restore-feasibility.md.
