# Just-in-time inference: the energy-saving proxy, layer by layer

Goal (owner, 2026-09-06): an LLM proxy in front of vLLM. No request for 90
minutes: power the ConnectX-7s off but leave the serving stack resident
(measured 208 W → 125 W for four nodes). A request arrives: power the
adapters on and pay a first-message penalty. Reviewed with Codex
(`results/codex-review-proxy-jit.md`) and with GLM-5.3 running on the cluster
itself (`results/glm-review-proxy-jit.md`); the NCCL facts below were read
from the 2.31.2 source by Codex and Claude independently.

## 0. The one fact that shapes everything

Removing the adapter kills the RDMA state inside the live processes, and NCCL
2.31.2 cannot recover from that in-process:

- `ncclIbFinalizeDevices()` only decrements a reference count; device
  contexts stay open and device discovery runs only while `ncclNIbDevs == -1`,
  which finalize never restores (`src/transport/net_ib/init.cc` :281, :298,
  :590). Destroying every communicator and building new ones reuses contexts
  that died with the device.
- A device-fatal async event bumps a per-device fatal counter that is never
  cleared, so later operations fail with `ncclSystemError`.
- Port down/up events, by contrast, are logged and ignored unless
  `NCCL_IB_RESILIENCY_PORT_FAILOVER=1` (default 0). An idle communicator's
  queue pairs are left untouched by a link flap.

So there are two fundamentally different routes, and the choice depends on a
measurement we have not made yet:

| route | what goes down | wake cost | engine work | open question |
|---|---|---|---|---|
| **L: links down** | ports admin-down, adapter stays powered, contexts and QPs intact | link retrain, a few s | **none** if QPs survive | watts saved (0-20 W/node, unmeasured); GID stability (needs NetworkManager `ignore-carrier`) |
| **A: adapter off + restart** ("accept the reload": the model is *not* resident, weights reload and the in-memory KV pool is lost; the slab tier survives) | PCIe functions removed, 20 W/node saved | 505 s today, 80-120 s with the disk-image loader | none (orchestration only) | none |
| **B: adapter off + in-process rebuild** | same 20 W/node | 40-60 s target, unproven | 700-1300 lines across worker, groups, graphs, API, proxy, plus an NCCL transport fix | NCCL cannot reset its IB backend; needs a patched NCCL or a hot-plug-aware external plugin, and fresh uverbs nodes inside the container |

Route L is the cheapest experiment with the largest payoff and is tested
first. Route A is the guaranteed fallback. Route B is only worth its cost if
L saves too little.

## 1. Experiments, in order

1. **Links-down power and survival (route L).** Stack up and idle. GLM's
   correction: `ignore-carrier` only covers carrier loss; an admin-down of an
   NM-managed device makes NetworkManager deactivate the connection, which
   removes the IPv4 address and with it the RoCEv2 GID the surviving queue
   pairs address. The kernel itself keeps IP-derived GIDs across an
   admin-down (`roce_gid_mgmt` reacts to address removal, not to
   NETDEV_DOWN), so the addresses must simply stay: on all four nodes
   `nmcli device set <port> managed no` for both ring ports first, then
   `ip link set <port> down`; verify `ip -4 addr` and the GID at index 3 are
   still there before reading the meter. After a few minutes `ip link set
   ... up`, wait for 200G and RDMA ACTIVE, re-check address and GID, set the
   devices managed again, send a real generation. Outcomes: (a) big saving and
   generation works: route L, no engine work; (b) generation hangs: route L
   dead as a transparent trick; (c) small saving: ignore L.
2. **Same-process re-init after hot-plug (route B gate).** Stack down. Four
   persistent processes in the serving image, tiny GPU buffers, Gloo/store
   over the management LAN: allreduce, synchronize, destroy all comms, all
   ranks acknowledge, host cycles the adapter, fresh id, re-init, checked
   allreduce. Log library paths, uverbs opens/closes, context pointers, GIDs,
   buffer checksums. Expected to fail on stock NCCL; also shows whether the
   container's `/dev/infiniband` nodes come back usable.
3. **Disk-image boot** (`docs/BOOT-TIME.md`) if route A is the one shipped:
   it sets the wake cost.

## 2. Proxy layer (all routes)

- One serialized state machine: READY → DRAINING → OFF → WAKING → READY /
  FAILED. Timer starts at the last request *completion*; health polls are not
  activity; require zero active requests before DRAINING.
- Single-flight wake with a generation counter so a stale idle timer cannot
  power down during a wake. Bounded request queue, propagate client
  cancellations, never replay an already-forwarded request.
- Streaming clients: SSE comment lines every ~10 s with buffering disabled;
  once 200 is committed, a failure is a stream error and close.
  Non-streaming JSON cannot carry keepalives: hold with an explicit deadline
  or answer 503 with Retry-After.
- Liveness and readiness are different: `/health` from vLLM is 200 while the
  engine is wedged (seen twice this weekend); readiness after a wake is a
  real generation probe.
- Hooks are scripts, so the proxy is the same for L, A and B; only the hooks
  differ. Never let the proxy touch the engine's memory or scheduler directly.
- ~200-400 lines of async Python; vLLM moves to 8001, the proxy takes 8000.

## 3. Client layer

Check pi's and the SDKs' first-byte, read-idle, total-request and
intermediary timeouts; keepalives defeat only some of them. Route A's 9-minute
wake needs a client that tolerates it or a 503; routes L/B fit under typical
first-byte timeouts.

## 4. OS layer (`spark-idle.sh` and around it)

- Route L needs `ignore-carrier` and a `--links-down/--links-up` pair (no
  PCIe removal), plus the same verification (200G, IPs, MTU, GID value and
  type at index 3, RDMA ACTIVE, jumbo pings).
- Route A/B keep today's `--down/--up`; the resume path must also confirm the
  `/dev/infiniband` nodes and permissions inside the surviving container.
- Hardware restoration must never by itself unpause scheduling.

## 5. Engine layers (route B only)

- **API / EngineCore**: `suspend_network` / `resume_network` through the
  engine client → EngineCore → worker `collective_rpc`. Pause with
  `pause_scheduler(mode="wait", clear_cache=False)` (the `/sleep?level=0`
  path is scheduling-only; level 1 offloads weights and discards KV: never).
  Drain scheduler output and slab/bounce transfers first; serialize control
  RPCs during the rebuild; keep the message queues.
- **Worker suspend**, links still up: drain and synchronize; release target
  and drafter CUDA graphs including retained references; destroy device
  communicators in a fixed order (auxiliary/DCP/TP before world, PyNccl
  wrappers before their torch groups); wait for a real teardown-complete
  acknowledgement from all four ranks before the adapter goes down.
  `PyNcclCommunicator.destroy` starts a daemon abort thread and waits five
  seconds; that is not an acknowledgement. Do not call the generic
  `destroy_model_parallel` / `destroy_distributed_environment`: they also
  drop Gloo and the broadcaster. Keep a persistent Gloo control world and the
  TCP store.
- **Worker resume**: fresh NCCL ids with epoch-qualified rendezvous keys,
  replace every cached communicator reference, test collectives, re-capture
  graphs with the block-table save/restore pattern from elastic EP
  (`elastic_execute.py` :498) extended to the drafter; preserve RNG and make
  sure dummy runs cannot write into resident KV. Elastic EP is a template,
  not a drop-in: it swaps DP/EP groups, not TP/DCP, and its graph-release
  helper resets compilation.
- **Transport**: the unfixable part on stock NCCL 2.31.2. Options are a
  resettable IB backend (patch), a hot-plug-aware external net plugin
  selected at startup that preserves Spark's subnet-aware routing, or worker
  restart (which is route A). Codex advised against socket-first, lazy-init
  or plugin-reload tricks.
- Sizes (Codex): transport spike 150-250 lines; worker/group/graph overlay
  400-800; API/core 100-200; proxy 200-400.

## 5b. Risks GLM added (reviewed on the cluster itself)

- **Split-brain restore.** The dead-man timers are per node. If one node loses
  its transient unit or reboots while the other three restore, the ring is
  half up, per-node checks pass, collectives fail. The proxy must gate every
  resume on a cluster-level "all four verified" (what `spark-idle.sh --up`'s
  non-zero exit already encodes), never on per-node state.
- **The proxy is a single point of failure that is idle exactly when it
  matters.** systemd-supervised with a watchdog, and a wake path triggerable
  out of band over the management LAN.
- **Cycle fatigue and hysteresis.** Bursty traffic (a request every 95 min)
  turns the scheme into dozens of PCIe hot-plug cycles a day; every removal
  already logs correctable PCIe RxErr on a218's root ports. Add a minimum
  off time, a daily cycle budget, and track AER counters. `debug_state` is a
  debug knob NVIDIA can rename in any DGX OS update: pin the package version.
- **Pinned NCCL environment versus re-enumeration.** `NCCL_IB_GID_INDEX=3`
  already bit us once; after a cycle verify the GID's value and type, and
  test the launcher with the index unset (NCCL's automatic RoCEv2/IPv4
  selection) at the next relaunch.
- **Other RDMA users.** Inventory every ibverbs consumer in the workers
  before sizing route B. Here the answer is known: DCP KV exchange rides the
  NCCL collectives and the slab tier moves KV through CPU and NVMe with no
  RDMA, so NCCL is the only RDMA state, but the inventory must be repeated
  after any connector change.
- **Suspend must confirm NCCL's threads are gone.** `ncclCommAbort`, not
  `ncclCommDestroy`, for any communicator that might have work outstanding,
  and verify the proxy threads exit, or each cycle leaks a spinning thread.
- **Route A is not resident idle.** It is "accept the reload"; say so in any
  status page or README so nobody expects the 40-60 s wake from it.

## 5c. Route B evaluated in depth (sub-agent, 2026-09-07)

`results/subagent-nccl-rebuild-vs-shim.md`: option 2 (patch NCCL so the IB
backend resets at refcount zero, rebuild communicators in vLLM) is **feasible**,
150-250 lines in three NCCL files plus 500-800 lines in the fork, two to three
weeks; option 3 (a verbs interposer that keeps NCCL's handles alive) is
**feasible but brittle and not recommended**: LD_PRELOAD cannot reach NCCL's
verbs calls (it dlopens libibverbs and resolves versioned symbols itself, and
the data path is inlined through `context->ops`), so the shim must virtualize
every verbs object and rewrite rkeys and QPNs in flight. Two facts neither
earlier review had: NCCL's core caches the virtual-NIC list once per process
(`topo.cc`), so a backend-only reset still fails on the merged `f0+f1` device,
and vLLM creates an `_EP` group for every MoE model, so there are three PyNccl
communicators per worker, not two. Experiments E0-E4 are listed in the report.

## 5d. Route P: a hot-plug-aware NCCL network plugin (decision 2026-09-07)

The owner's requirements: model-agnostic, framework-agnostic (vLLM today,
maybe SGLang later), and a plugin to NCCL if such a path exists. It does, and
it is a better seam than patching NCCL (option 2) or interposing verbs
(option 3). Facts verified in NCCL v2.31.2-1 and NVIDIA's out-of-tree plugin:

- NCCL loads external network plugins (`NCCL_NET_PLUGIN=<name>` →
  `libnccl-net-<name>.so`, API `ncclNet_v12` at 2.31.2 with v6-v11 accepted,
  `src/include/plugin/net/net_v*.h`), and unloads/reloads them at refcount
  zero (`src/plugin/net.cc:78-97, 314-328`), so plugin static state resets
  by construction.
- NCCL core keeps only opaque handles from the plugin: `netSendComm`,
  `netRecvComm` and per-protocol `mhandles` (`src/transport/net.cc:90-148`).
  It never sees queue pair numbers, keys or device contexts. Everything RDMA
  is the plugin's private state, so it can be rebuilt underneath a live
  communicator without NCCL core, torch, the engine or the captured CUDA
  graphs noticing (the graphs reference NCCL core's own FIFO/flag buffers,
  not plugin objects).
- The IB transport keeps its TCP control socket to the peer for the whole
  life of a connection (`ncclIbNetCommBase.sock`, closed only in
  `ncclIbCloseSend/Recv`, `connect.cc:1718-1760`); it is bound to
  `NCCL_SOCKET_IFNAME` = the 10GbE, which is exactly the out-of-band channel
  a re-handshake needs after both adapters were cycled. Receiver-side keys
  and addresses already travel per request through the FIFO protocol, so
  re-registered memory picks up new keys on the next receive; only the FIFO
  base itself needs re-exchange.
- NVIDIA's out-of-tree IB plugin (github.com/Mellanox/nccl-rdma-sharp-plugins,
  active, pushed 2026-08) is a standalone C port of NCCL's IB transport:
  plugin API v6-v11, subnet-aware routing (which is also upstream in 2.31.2's
  `connect.cc`), merged NICs, ECE, relaxed ordering; no async-event thread. It
  is the natural base to fork.

Design: `libnccl-net-hotplug.so` = the out-of-tree IB plugin plus a
suspend/resume layer driven by a local control channel (Unix socket or file
under `/run`) that the proxy or `spark-idle.sh` toggles on every node:

1. `suspend`: assert no outstanding requests (idle by construction; otherwise
   refuse), destroy queue pairs, completion queues, registrations, protection
   domains, close the device contexts, keep every comm's TCP socket and every
   `mhandle` struct (now marked stale). The adapter can then be removed with
   no RDMA users at all.
2. `resume`: re-enumerate devices by name, re-open, re-create PD/CQ, re-register
   every `mhandle`'s VA range (new keys behind the same handle), re-create
   queue pairs, re-exchange QP numbers, GIDs and the FIFO base with the peer
   plugin over the retained socket (the same handshake as connect), bring
   them to RTS, mark comms active. A request arriving while suspended is held
   (`test` returns not-done) or failed fast, configurable.
3. Fallback detection: an async-event thread that treats `DEVICE_FATAL` /
   disassociation as an implicit suspend, so an unplanned cycle degrades to a
   clean error instead of a silent spin.

What this removes from the earlier plan: the NCCL patch, every vLLM change
(no RPCs, no communicator teardown, no graph recapture, no RNG concerns),
and any dependence on the model or the serving framework. The proxy stays,
as policy and as the thing that calls `spark-idle.sh` and toggles the plugin.
Wake becomes adapter (12 s) + plugin resume (~1 s) + normal first token.

Sandbox-buildable: the plugin is plain C against libibverbs, no CUDA. Develop
and test it on this VM with soft-RoCE (`rdma_rxe` on enp1s0, verified working)
and a NCCL-free harness that drives the `ncclNet_v11` entry points in two
processes, including `rdma link delete/add` as the hot-plug stand-in. Then
build for aarch64 (plain C, trivial cross or CI) and test on the Sparks with
`NCCL_NET_PLUGIN=hotplug` under the existing NCCL bench before the stack.

Open questions for the first Spark run: performance parity with the builtin
IB net on this ring (same code lineage, expect none); plugin API v11 on a
v12 core (NCCL's compatibility path); whether `NCCL_IB_GID_INDEX` should be
unset in favour of subnet-aware selection; behaviour of torch's NCCL
watchdog if a collective is ever issued during a suspension (the proxy's job
is to make sure none is). Effort: 600-1000 lines of C on the plugin, two to
three weeks, almost all of it on the sandbox.

## 5e. Route P status (2026-09-07): built, reviewed, sandbox-tested

The plugin exists: github.com/ajclark/nccl-plugin-low-power-dgx-sparks (local
`~/nccl-net-hotplug/plugin`), a fork of NVIDIA's out-of-tree IB plugin with
a two-phase quiesce layer (`prepare` gates the data path and refuses if
anything is in flight, `commit` tears the RDMA state down, `abort` drops the
gate, `resume` rebuilds and re-handshakes over the retained sockets), a file
control channel and exported `ncclHotplug{Prepare,Commit,Abort,Suspend,Resume}`.
Codex reviewed the first version against NCCL 2.31.2's sources
(`results/codex-review-plugin.md`): the two-phase protocol, the fail-closed
`failed` state (data path errors out instead of hanging), gating of
connect/accept with pending-handshake tracking, deferred buffer registration
and flush while gated, DMA-BUF descriptor duplication, one event thread per
shared verbs context, GUID check on re-open, and thread/device teardown at
the last finalize all come from that review. On the sandbox against
soft-RoCE it passes: busy refusal at prepare and at suspend, gate held for
`isend` and `connect`, abort, registration while suspended, device delete/re-add
cycles (50 in a row with flat memory and descriptor counts), the file channel,
and device removal after finalize. Cross-compiled for the Sparks
(`stage/nccl-hotplug/libnccl-net-hotplug.so`, depends only on libmlx5,
libibverbs, libc). Cluster wiring is in place behind `NCCL_HOTPLUG=1`
(launcher, rollout; the hot-plug lane bind-mounts `/dev/infiniband` with a
major-wide device-cgroup rule so re-created device nodes stay reachable) and
`spark-idle.sh --down/--up` runs prepare-everywhere-then-commit and resume
around the adapter cycle. Not exercised anywhere yet: the real NCCL core
loading the plugin, the real adapter, GPU Direct RDMA paths (disabled on the
Sparks: `GDR 0` in the serving logs). The on-cluster plan is
`docs/NCCL-HOTPLUG-TEST.md` (one downtime window); the probe that runs first is
`bench/nccl-hotplug-probe.sh`.

## 5f. On the cluster (2026-09-07 window): it works, and without a proxy

Probe A passed (two adapter cycles under a live NCCL 2.31.2 communicator,
data path at parity with the builtin backend: `docs/NCCL-HOTPLUG-TEST.md`,
Results). The serving lane was then rolled out on the plugin
(`dcp2-hotplug-1`, healthy, count100 200 tokens in 4.9 s wall) and step 5
run with the model resident: `spark-idle.sh --down` found every rank idle
(prepared, then suspended; the non-zero ranks do not sit in a blocking
receive while idle, so the pending-receive question in 5e is closed),
adapters off; a request sent while off was held by the plugin and raised
`wanted=1` on all four ranks within 3 s; `spark-idle.sh --up` took 14 s and
the held request completed at normal speed (TTFT 32 s including the
operator's delay, decode unchanged); count100 after the cycle equals the
baseline.

The proxy of sections 2 and 3 is therefore not needed. The plugin's status
line carries the idle clock (`idle=` seconds since the last send/receive)
and the wake request (`wanted=`), and `spark-idle-watch.sh` in the sibling
repo turns them into `--down` after `IDLE_MIN` idle minutes on every node and
`--up` on the first `wanted=1`. Nothing sits on the request path; the first
request after a spin-down pays the wake and nothing else changes.

## 6. Decision

Route P (the hot-plug-aware NCCL net plugin) is the design: it meets the
owner's constraints (model- and framework-agnostic, a plugin to NCCL) and
needs no engine work. Route L is dropped on the owner's judgement that a
link bounce saves too little. Route A (proxy + stop/relaunch) remains the
interim and the fallback. Options 2 and 3 are superseded by P. Order: fork
the out-of-tree IB plugin and build it on the sandbox; add suspend/resume
and test it against soft-RoCE with device delete/add; build the proxy with
route A hooks in parallel; then one downtime window on the Sparks for the
NCCL bench with `NCCL_NET_PLUGIN=hotplug`, an adapter cycle under it, and
finally the serving stack. In no case promise a 12-second wake: that is the
adapter, not the first token.

Update 2026-09-07 (5f): done and measured. Route P is in service as the
`dcp2-hotplug-1` lane, the operator is `spark-idle.sh` plus
`spark-idle-watch.sh`, and no proxy is built or needed. Measured wake: 14 s
for `--up` (adapters, ring verification, RDMA rebuild) plus the request's own
first token.
