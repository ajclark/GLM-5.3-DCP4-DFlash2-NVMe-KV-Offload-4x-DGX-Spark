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

## 6. Decision

Run experiment 1 next (with the unmanaged-before-down fix). If the
links-down route saves most of the 20 W per node and the stack answers
afterwards, build the proxy with links-down hooks and stop there. GLM's two
additions to the order stand: run a *new-process* NCCL all-reduce inside the
serving container after a full adapter cycle before any same-process gate
(it validates device nodes, permissions, GIDs and fresh uverbs cheaply and
every route needs it), and build the proxy with route A hooks in parallel
because it is route-agnostic and needed in every outcome. Otherwise build the proxy with stop/relaunch hooks (route A),
push the disk-image boot to cut the wake to ~100 s, and treat route B as a
separate NCCL project. In no case should the wake be promised as 12 seconds:
that is the adapter, not inference.
