# ConnectX-7 power-off with cables attached: 82 W (41 %) off the idle cluster

Experiment run 2026-09-05 16:15-16:33 PDT by the human with Codex in the
sibling pane, serving stack stopped, wall power read from the human's Grafana
(all four Sparks). Codex hit its usage limit before writing this report; the
Claude session reconstructed it from the pane scrollback
(`results/idle-power/cx7-experiment/codex-pane.txt`) and Codex's session log
(`results/idle-power/cx7-experiment/codex-commands.txt`, the exact command
blocks), then re-verified the cluster state afterwards (section 5).

## 1. Result

| state | wall power, 4 nodes | per node |
|---|---:|---:|
| idle, serving stack stopped, CX-7 on (baseline) | ~202-207 W | ~51 W |
| canary: spark-06c4's CX-7 off, other three on | 177 W | -25 to -30 W on that node |
| all four CX-7 off | 168 W after ~30 s, **120 W settled** | **-20.5 W** |
| all four restored | ~202 W | back to baseline |

The two ring cables stayed plugged in throughout. Every node stayed reachable
over the 10GbE management link at 10G the whole time. Restoration was fully in
software: all four PCIe functions per node came back, both ring ports at 200G,
the original IPv4 addresses and MTU 9000 (NetworkManager re-applied them),
RDMA links ACTIVE, jumbo-frame pings across the ring OK, no kernel errors.
This is the single biggest idle lever on the box, and NVIDIA's "up to 18 W"
release-note figure is conservative for this cluster.

## 2. Mechanism: NVIDIA ships it

DGX OS package `dgx-spark-mlnx-hotplug 26.01-1` provides a platform driver
(`cx7-pcie-hotplug`, ACPI device `MTKP0001:00`) with two sysfs knobs:

```
/sys/devices/platform/MTKP0001:00/pcie_hotplug/hotplug_enabled   # 1 on these nodes
/sys/devices/platform/MTKP0001:00/pcie_hotplug/debug_state       # 1 = powered ("plug-in"), 0 = off ("plug-out")
```

and a handler, `/opt/nvidia/dgx-spark-mlnx-hotplug/mtk-hotplug-handler.sh`,
driven by `/lib/udev/rules.d/90-mtk-hotplug.rules` on physical cable events
and callable by hand with `removal` / `plug-in`. It is armed by the presence
of `/etc/nvidia/cx7-hotplug-enabled` (present on all four nodes). This is not
PCIe slot power (the CX-7 root ports report `HotPlug- PwrCtrl-`), which is
why the earlier survey in `docs/IDLE-POWER.md` missed it.

The handler's `remove()` deletes every PCI device under the two CX-7 root
ports (`0000:00:00.0`, `0002:00:00.0`, i.e. functions `0000:01:00.0/.1` and
`0002:01:00.0/.1`) via `/sys/bus/pci/devices/<bdf>/remove`, then writes 0 to
`debug_state`, which powers the adapter down. `plugin()` writes 1, sleeps 3 s,
and rescans both root ports; mlx5_core re-probes, udev restores the netdev
names, NetworkManager re-activates `roce-p0`, `roce-p2p`, `roce-p2p-unused`.

Codex's review of the shipped script, worth keeping in mind before automating:

- `remove()` powers down if *either* domain's removal succeeded (line 108) and
  always returns 0; `plugin()` ignores rescan failures; the presence test
  accepts any single CX-7 function, so a partial restore could make later
  `plug-in` calls skip recovery. Our own wrapper should require all four
  functions gone before writing 0, and all four back (plus 200G, IPs, MTU,
  RDMA) after writing 1.
- `mstflint_access` can hold stale PCI device pointers across a hotplug
  removal and a later firmware query can panic the kernel
  (github.com/Mellanox/mstflint/issues/1786, an earlier 6.17 kernel; not
  established whether 6.17.0-1026 has the fix). Codex unloaded the module
  before every removal and reloaded it after restoration. Do not run
  MST-based diagnostics between the two.
- Do not write 0 to `debug_state` directly without the orderly device
  removal first.
- A first-hand report of the same trick with cables attached:
  forums.developer.nvidia.com/t/disable-connectx-7-even-when-qsfp-is-connected-to-save-power/371302.

## 3. Exact sequence used (per node, all four in parallel over SSH)

Preflight, each node (abort on any failure):

```
test "$(docker inspect -f '{{.State.Running}}' vllm_glm53big)" = false      # stack stopped (docker stop, containers kept)
test "$(cat /sys/devices/platform/MTKP0001:00/pcie_hotplug/debug_state)" = 1
test -f /etc/nvidia/cx7-hotplug-enabled
for bdf in 0000:01:00.0 0000:01:00.1 0002:01:00.0 0002:01:00.1; do             # exactly the CX-7, nothing else in those domains
  test "$(cat /sys/bus/pci/devices/$bdf/vendor)" = 0x15b3; test "$(cat /sys/bus/pci/devices/$bdf/device)" = 0x1021; done
readlink -f /sys/class/net/enP7s7/device                                       # management NIC is in domain 0007, untouched
test "$(systemctl show nvidia-spark-mlnx-firmware-manager.service -p ActiveState --value)" = inactive
sudo fuser /dev/infiniband/uverbs* /dev/infiniband/rdma_cm /dev/*_mstconf     # no RDMA / MST users
if test -d /sys/module/mstflint_access; then sudo modprobe -r mstflint_access; fi
```

Power off with a dead-man restore (the timer fires whether or not the sandbox
is still connected):

```
sudo systemd-run --unit=cx7-all-restore --on-active=180s --timer-property=AccuracySec=1s \
     /opt/nvidia/dgx-spark-mlnx-hotplug/mtk-hotplug-handler.sh plug-in
for bdf in 0000:01:00.0 0000:01:00.1 0002:01:00.0 0002:01:00.1; do
  echo 1 | sudo tee /sys/bus/pci/devices/$bdf/remove >/dev/null; test ! -e /sys/bus/pci/devices/$bdf; done
echo 0 | sudo tee /sys/devices/platform/MTKP0001:00/pcie_hotplug/debug_state >/dev/null
test -z "$(lspci -D -d 15b3:1021)"                                             # all four functions gone
test "$(cat /sys/class/net/enP7s7/carrier)" = 1 && test "$(cat /sys/class/net/enP7s7/speed)" = 10000
```

Restore is the timer running `mtk-hotplug-handler.sh plug-in` (or run it by
hand earlier), then verify: four functions present, `debug_state` = 1, both
ring ports 200G with their IPv4 and MTU 9000, `rdma link show` ACTIVE on all
four devices, jumbo pings to both ring neighbours, then
`sudo modprobe mstflint_access`. The single-node canary used the same steps
with unit `cx7-canary-restore`.

Timeline (PDT): stack stopped 16:16-16:17; canary off 16:19:25, restore fired
16:22:57, recovered by 16:23:16; all four off 16:25:44, fully off 16:26:12,
timers fired 16:28:44, all recovered by 16:29:11; ring pings and MST reload
16:32; Codex out of budget 16:33 while gathering journals.

## 4. What it means for the idle design

- Deep idle (`enter-low-power-idle-mode.sh --deep`) should gain a `--cx7-off`
  add-on built on this sequence, replacing the speculative `--ring-down`
  admin-down: ~20 W per node measured versus unknown. Preflight, dead-man
  timer, all-four-functions checks and the mstflint unload/reload go into the
  script; un-idle's verification (IPs, MTU, RDMA, 200G) already matches what
  the restore needs before relaunching the stack.
- It only works with the serving stack down: removing the PCIe functions
  destroys NCCL/RDMA state, so the cost is the 505 s relaunch (or the
  disk-image boot in `docs/BOOT-TIME.md` once that exists).
- Cluster idle with the stack stopped is ~51 W per node with the CX-7 on and
  ~30 W with it off. The remaining 30 W is CPU/SoC, DRAM, NVMe, the 10GbE PHY
  and the GPU's ~8 W; the light-tier levers chip at that.

## 5. State check after Codex stopped (Claude session, 2026-09-05 23:35-23:45 UTC)

All four nodes: `debug_state` = 1, four CX-7 PCI functions, all RDMA links
ACTIVE, ring ports up at 200G, MTU 9000, IPs as before, `mstflint_access`
loaded, governor performance, GPU clocks locked (1995 MHz), management 10G,
no `cx7-*` systemd units, timers, transient files or unit files left, no
leftover scripts, zero kernel errors since the experiment. The serving
containers exist but are **stopped** (`docker stop` at 23:16-23:17 UTC), the
cache flushers are gone, ~115 GB free per node. Relaunch is
`./rollout_dcp.sh <label>` (daily DCP=2 lane) or `./un-idle.sh`.

## 6. Roadmap: idle with the model resident ("just-in-time inference")

Goal state: the vLLM stack stays up with weights and KV cache resident, the
CX-7s are off (~30 W per node), and the first request brings the ring back
and serves within a minute instead of the 505 s reload. What that takes in the
engine, per worker, on `suspend`:

1. Quiesce (no running requests; the API queues new ones).
2. Drop captured CUDA graphs (they embed NCCL kernels and communicator
   buffers). vLLM's elastic EP already has this: `_release_cuda_graphs()`.
3. Destroy every NCCL communicator (torch process groups for TP/DCP/world and
   the drafter's replicated group, plus vLLM's PyNccl communicators) while
   the adapter is still up, so the teardown is clean. Gloo/TCP-store traffic
   rides the 10GbE and survives.
4. `cx7-power.sh off`.

On `resume` (triggered by the first request or by hand): `cx7-power.sh on`,
re-create the process groups with fresh NCCL ids through the surviving TCP
store (elastic EP's `StatelessGroupCoordinator` / `_replace_active_groups`),
`compile_or_warm_up_model()` to re-capture graphs with the block tables saved
and restored (elastic EP does exactly this), then unpause. Expected: ~20 s
adapter, ~5 s communicators, ~15-20 s graph capture, so 40-60 s to first
token from cold idle. The weights, KV pool and slab tier never move.

The one unknown that decides feasibility: NCCL opens its IB device contexts
once per process and caches them. After a hot-remove those contexts are dead
(uverbs disassociation), and the re-created communicators may fail to build
queue pairs even though the adapter is back. Crux experiment, stack down, no
meter needed: a small script in the serving image on all four nodes that
builds a NCCL communicator over the ring, all-reduces, destroys it, waits
while the host runs `cx7-power.sh off` then `on`, builds a new communicator
and all-reduces again. If that passes, the engine-side work is a
`suspend_network` / `resume_network` RPC pair built from elastic EP's parts
(days, in the fork). If it fails, the fallback is to restart the worker
processes rather than the communicators, which only becomes cheap once the
disk-image boot (`BOOT-TIME.md` in the DCP repo) exists, or to find a lower
CX-7 power state that keeps the device contexts alive (ports admin-down; its
savings are unmeasured).
