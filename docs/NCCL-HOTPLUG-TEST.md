# NCCL hot-plug plugin: the on-cluster test plan (one downtime window, ~1 h)

Everything up to here was built and tested on the sandbox against soft-RoCE
(github.com/ajclark/nccl-net-hotplug, `README.md`). What only the Sparks can
answer: NCCL 2.31.2 core loading the v11 plugin, the real ConnectX-7 hot-plug
under it, performance parity with the builtin IB backend, and the serving
stack surviving a cycle. Steps are ordered so each one can stop the plan
early with a clear verdict. All commands run on the sandbox from
`~/spark-cluster-mla-dcp`.

## 0. Pre-flight (no downtime)

- The aarch64 build is staged: `stage/nccl-hotplug/libnccl-net-hotplug.so`
  (built by `~/nccl-net-hotplug/plugin/build-arm64.sh`; depends only on
  libmlx5, libibverbs, libc). `rollout_dcp.sh` and the probe driver rsync it
  to `~/nccl-hotplug/` on every node.
- `spark-idle.sh --status` shows the ring healthy; the meter is visible.

## 1. Stack down

```
for h in spark-06c4 spark-365c spark-ddbf spark-a218; do ssh napta2k@$h.local 'docker rm -f vllm_glm53big; pkill -f "[c]ache_flusher.sh"; true'; done
```

## 2. Probe A: NCCL core + plugin + real adapter cycle (≈10 min)

```
./bench/nccl-hotplug-probe.sh 2          # two cycles; HOLD_S=30 between --down and --up
```

What it does: launches `bench/nccl_hotplug_probe.py` in the serving image on
all four ranks with the launcher's NCCL environment plus
`NCCL_NET_PLUGIN=hotplug` and the control directory. Each rank builds one
NCCL communicator over the ring (vLLM's PyNccl wrapper, real libnccl 2.31.2),
all-reduces and all-gathers with checks and timing, then waits. The driver
runs `spark-idle.sh --down` (plugin suspend on every node through the control
files, then adapters off) and `--up` (adapters on, ring verified, plugin
resume). The **same** communicator then all-reduces again, checked and timed.

Verdicts, from rank 0's JSON lines (`results/nccl-hotplug-probe/<ts>/rank0.jsonl`):

| line | what it proves |
|---|---|
| `"phase": "init", "plugin": "hotplug"` and NCCL INFO `NET/Plugin: Loaded net plugin` in the logs | NCCL 2.31.2 core accepted the v11 plugin (if it fell back to the builtin IB net the plugin was rejected: stop here) |
| `"phase": "baseline", "correct": true, "allreduce_us": …` | plugin data path works on the CX-7 ring; compare `allreduce_us`/`allgather_us` with `results/nccl-multicomm/RESULTS.md` (n_comms=1, ~83 and ~105 µs): parity expected, same code lineage |
| `spark-idle.sh --down` prints `PLUGIN_OK 1 process(es) prepared` for all four nodes, then `PLUGIN_OK 1 process(es) suspended` for all four, before the adapters go off | two-phase quiesce under a live NCCL communicator: every rank gated with nothing in flight, then torn down |
| `spark-idle.sh --up` prints `PLUGIN_OK 1 process(es) active` after the ring verification | resume re-connected through the retained sockets on real hardware |
| `"phase": "after-cycle-1", "correct": true` with unchanged µs | the communicator NCCL core holds is fully usable again; nothing above the plugin noticed |
| `"phase": "done", "ok": true` and exit 0 | pass |

If `--down` reports `busy` at prepare, the probe had a collective in flight:
it should not (it idles at the barrier); investigate before anything else.
`--down` then aborts on every node (gates dropped, nothing torn down) and
powers nothing off. A failure at commit or resume puts that process into the
plugin's `failed` state, in which the data path errors out loudly instead of
hanging: expect the probe (or vLLM) to die, and relaunch. If resume
reports `error: devices did not come back`, check `/dev/infiniband` inside the
container after the cycle. The probe runner and the `NCCL_HOTPLUG=1` lane
already bind-mount the host directory (`-v /dev/infiniband:/dev/infiniband`)
and allow the whole uverbs major in the device cgroup (`c 231:* rwm`) instead
of `--device`, so udev's re-created nodes are visible whatever minor numbers
the re-added adapters get (the kernel hands out the lowest free ones, 192-195
today).

## 3. Probe B: same, without the adapter cycle (control, ≈3 min)

```
NO_CYCLE=1 ./bench/nccl-hotplug-probe.sh 1
```

Suspend and resume only, adapters stay on. Separates plugin logic from
hot-plug effects if probe A fails.

## 4. The serving stack under the plugin (≈15 min)

```
NCCL_HOTPLUG=1 SKIP_PREFLIGHT=1 ./rollout_dcp.sh dcp2-hotplug-1
```

Same lane as production (DCP=2, 180224, 6 GB pool, slab tier) with the plugin
loaded. The rollout's own checks (health, real generation) apply. Then the
standard post-boot check and a count100 against the builtin-backend numbers
(`README.md` lane table: 54.5 tok/s, 144 ms cycle at DCP=2): parity expected.
A regression here is the plugin's data path, not hot-plug.

## 5. The whole point: idle cycle with the model resident (≈5 min)

With the stack up and idle (no requests), meter visible:

```
./spark-idle.sh --down          # plugin suspend on 4 nodes (refuses if a collective is in flight), adapters off
# meter: expect ~125 W for the four nodes
./spark-idle.sh --up            # adapters on, ring verified, plugin resume
curl … count100                 # first request after the cycle: must answer, cycle time unchanged
```

Then `./spark-idle.sh --down`, wait 10 minutes, `--up`, another generation,
and a quick concurrency run to shake out anything the first collective after
resume did not touch (DCP groups, the drafter's group, the EP group).

## 6. Leave the cluster in a known state

Either keep the plugin lane serving (it is the new default candidate) or
relaunch the builtin-backend lane: `SKIP_PREFLIGHT=1 ./rollout_dcp.sh <label>`.
Record the results in `docs/JIT-PROXY.md` and the plugin README's Status.

## Results (2026-09-07, first window)

**Probe A: PASS** (`results/nccl-hotplug-probe/20260907-052859`). NCCL
2.31.2 loaded the v11 plugin (`Using network NCCL RDMA Plugin v11`), one
four-rank communicator over the ring, two full adapter cycles under it
(`--down`: prepared then suspended on all four nodes, adapters off, 30 s
hold; `--up`: adapters on, ring verified, resume on all four in parallel), and
the same communicator all-reduced and all-gathered correctly after each cycle.
Wall time from `--down` to "active" again: ~80 s including the 30 s hold.

| measurement | allreduce µs | allgather µs |
|---|---:|---:|
| builtin IB backend, same probe (`PROBE_NET=builtin`, `20260907-053406`) | 346.8 | 371.8 |
| plugin, before any cycle | 338.7 | 384.6 |
| plugin, right after cycle 1 | 577.9 | 409.4 |
| plugin, right after cycle 2 | 570.5 | 419.3 |

Parity before the cycle. The all-reduce measured right after a cycle is
slower (1.7x) while the all-gather is nearly unchanged; the re-added
adapters are at full PCIe capability (32 GT/s x4, ASPM off), so this is not
the link. Open: whether it settles (`PROBE_SETTLE_S=N` re-measures N seconds
later) and what the serving stack's decode cycle shows.

Three things the first attempts taught, all fixed in the staged build and
scripts: the cross build must not pull glibc symbols newer than the image
(`__inet_pton_chk@GLIBC_2.42` made dlopen fail silently, NCCL fell back to
the builtin backend); the plugin needed NCCL's subnet-aware routing for the
switchless ring (first RTR on the wrong port timed out); and the operator
must send `resume` to every node before waiting on any (a resume
re-handshakes with its peers).

## What is deliberately not in this window

- The proxy. It is route-agnostic and only needs the hooks that already
  exist: `spark-idle.sh --down` and `--up`.
- `NCCL_IB_GID_INDEX` unset (subnet-aware selection): a separate relaunch.
- Long idle soak: overnight, after the window, once step 5 passes.
