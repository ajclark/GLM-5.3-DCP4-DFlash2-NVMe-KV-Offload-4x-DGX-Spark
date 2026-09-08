# CX-7 pre-power trick: single-cycle validation PASSED, then HALTED (2026-09-08, spark-a218)

**SUPERSEDED — READ `INCIDENT-MTKID-corrected-error.md` FIRST.** The single manual cycle below
worked, but the automated validation battery that followed produced a firmware hardware-error
record (MTKID, corrected) and three reboots on spark-a218. The experiment is halted and the
script change reverted. The result below is a true single-cycle measurement, NOT a validated fix.

Validated the userspace GPIO pre-power fix for the hot-plug throughput throttle. One node
(spark-a218), peer spark-ddbf/spark-06c4 for bandwidth; serving stack down for the test.
`ib_write_bw -q 16 -s 1M --report_gbits` (16 QP, 1 MB) on the ring ports.

## Clean A/B on the same node

| state | bw a218->ddbf (f1) | bw a218->06c4 (f0) | dmesg mlx5_pcie_event signature |
|---|---:|---:|---|
| baseline (cold boot) | 111.86 Gb/s | - | "insufficient (27W)" only |
| CONTROL: normal `--down`/`--up` (no trick) | **12.68 Gb/s** | - | "not advertised" appeared (throttled) |
| TRICK: pre-power `--up` | **111.83 Gb/s** | 111.86 Gb/s | "insufficient (27W)" only, no "not advertised" |
| TRICK repeat (stability) | 111.86 Gb/s | | |

The identical `--up` throttles to 13 Gb/s without the pre-power and stays at full 112 Gb/s with it.
Both NIC ports recover (throttle is NIC-global). a218 after the trick: 4 functions, rdma ACTIVE x4,
both ports up/200000.

## The trick (exact steps run on a218, serving down)
1. `./spark-idle.sh --down --hosts spark-a218`  (adapters off; debug_state=0, EN low)
2. `echo 658 | sudo tee /sys/class/gpio/export; echo high | sudo tee /sys/class/gpio/gpio658/direction`
   (EN = chip base 512 + line 146 = 658; value read back 1; debug_state still 0; 0 CX-7 functions
   present; no stray plugin uevent) then `sleep 3` (CX-7 main firmware image up at ~1.3 s)
3. `./spark-idle.sh --up --hosts spark-a218`  (debug_state=1 + rescan against the booted firmware)
4. `echo 658 | sudo tee /sys/class/gpio/unexport`  (release our owner; EN stays high from the driver)

## Confirmed facts
- GPIO lines 94 (PERST#) / 146 (EN) are NOT claimed by the driver (/sys/kernel/debug/gpio); legacy
  sysfs GPIO present, chip base 512, root-writable. EN=658, PERST#=606 (never touched).
- The pre-power did not fire a plugin uevent; the driver never power-cycled EN on `--up`.
- Cleanup left gpio658 unexported; EN held high by the driver's own write.
- MPEIN pwr_status reads 2 in BOTH throttled and healthy states at query time (it holds the last of
  the 0-then-2 event pair), so the reliable signature is the "not advertised" dmesg line + bandwidth,
  NOT the register snapshot.

## Caveat for integration
`spark-idle.sh --up` reported "FAILED verification" in the trick cycle (its ring-ping/resume check),
while the adapter was fully healthy (112 Gb/s both ports). The pre-power path changes bring-up
timing; integrating the trick into `--up` needs the ping verification to retry/settle. Not an
adapter fault.

## Verdict (revised)
One manual cycle restored full 112 Gb/s. It was NOT reproduced: the automated battery's first
cycle (which deviated from the manual sequence: rapid debugfs polling and unexport-before-up)
triggered a corrected MTKID firmware error and a reboot, and each following cycle rebooted the
node again. "Blast radius = one warm reboot" was wrong in practice: it was three reboots and a
firmware error record. HALTED; not to be folded into spark-idle.sh.
