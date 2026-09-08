# INCIDENT: MTKID firmware hardware-error record + repeated reboots on spark-a218 (2026-09-08)

**Status: the GPIO pre-power experiment is HALTED (user decision). Do not resume it.**

## What happened (verified from logs)
During an automated validation battery of the "pre-power" CX-7 hot-plug fix, spark-a218
rebooted THREE times in ~4 minutes (boot records 19:12, 19:14, 19:16 UTC), once per battery
cycle, beginning with the very first cycle. Each boot replayed a firmware error record:

```
ACPI: BERT 0x0000000087065D18 000030 (v01 MTKID  MTKTABLE 00000001 CREA 00000001)
BERT: Error records from previous boot:
[Hardware Error]: It has been corrected by h/w and requires no further action
[Hardware Error]: event severity: corrected
[Hardware Error]:  Error 0, type: corrected
[Hardware Error]:   section type: unknown, 3c1e3f4b-1e1a-43df-af28-59820e958e3c
[Hardware Error]:   section length: 0x3e
[Hardware Error]:   00000000: 000d0000 544d0000 0044494b 00000000  ......MTKID.....
```
- The section is a MediaTek vendor-specific type (unknown GUID, "MTKID" payload): the exact
  subsystem that faulted CANNOT be decoded from Linux.
- No kernel panic, oops, watchdog, or SError was logged in the boot that died; its last kernel
  lines were an ordinary mlx5 teardown (a `--down` in progress). The resets were therefore
  firmware/hardware-level, not a Linux crash.
- Only the cycled node had the record; an un-cycled node (spark-ddbf) had zero "Hardware Error"
  lines.

## What triggered it (verified timeline; cause partly inferred)
Battery timeline (UTC): 19:08:17 baseline 111.86 Gb/s; 19:08:17 the first cycle = the
"BOOT-gpio probe"; 19:09:23 a218 already unreachable (rebooting); then cycles C1 (19:10),
C2 (19:12), C3 (19:13) each ran against a node that had just come back, and the node rebooted
again after each (19:14, 19:16).

The first cycle was NOT the validated manual sequence. The validated manual cycle (earlier the
same day, no reboot, 111.83 Gb/s) was: `--down`; export EN gpio high; sleep 3; `--up`; THEN
unexport. The battery's first cycle did: `--down`; export EN gpio high; poll
`/sys/kernel/debug/gpio` 40 times in 4 s with sudo; **unexport BEFORE `--up`**; `--up`.
Hypothesis (not proven, and we are not going to test it): unexporting the line while the
driver still held EN low (debug_state=0) returned the pin to its default/input state and
dropped or floated EN, cutting power to a card that had been powered for ~4 s and was mid-boot;
`--up` then re-powered it: a power glitch during firmware boot. The subsequent normal cycles
then hit a platform already in a faulted state and reset it twice more.

Independently, Codex's review found the integrated `--up` had a bug: its writability guard
`[ -w /sys/class/gpio/export ]` ran as the ssh user while the write needed sudo, so the
pre-power was SILENTLY SKIPPED in every battery cycle (C1/C2/C3 were plain re-adds). The
battery therefore never tested the fix it was written to validate.

## Recovery (verified)
a218 came back on its own from the last reboot (cold-boot state, un-throttled) and, 38 minutes
later, showed: no new hardware errors this boot, PCIe AER counters 0 on both CX-7 functions and
both root ports, ConnectX-7 firmware health "healthy, error 0, recover 0", ring bandwidth
111.86 Gb/s, serving healthy. A corrected error is by definition handled without data loss.
This platform exposes no memory-ECC counters, so RAM is not independently verifiable.

## Decision
- The GPIO pre-power route is STOPPED. `spark-idle.sh` was reverted to the pre-GPIO version.
- The idle-power feature stays paused; the hot-plug throughput throttle remains unresolved.
- Watch item: if the MTKID BERT record ever reappears on a normal boot with no hot-plug
  cycling, that changes the picture and should go to NVIDIA immediately.
- The right path for the throttle is the NVIDIA bug report (root cause: their hot-plug driver
  releases PERST#/trains the link before the CX-7 main firmware image is up); not further
  userspace manipulation of the card's power line on this platform.
