# GLM review: just-in-time inference proxy (2026-09-06)

Read: CX7-POWER.md §7, JIT-PROXY.md, codex-review-proxy-jit.md. I ran on this
cluster, so I care about the parts that brick it silently.

## Where I agree

The core diagnosis is right and §7 proves it: the processes survive, the
RDMA state does not, and the failure is silent (`/health` 200 while rank 0
logs shm-broadcast timeouts and workers busy-poll dead queue pairs). Both
documents correctly identify NCCL 2.31.2's IB backend as the blocker:
`ncclIbFinalizeDevices()` only drops a refcount, device discovery is gated on
`ncclNIbDevs == -1`, and nothing resets it. I also agree that route A
(stop/relaunch) is the only shippable fallback today, that the proxy must be
a serialized state machine with single-flight wake and a generation counter,
that liveness ≠ readiness (a real generation probe after every wake), and
that nobody should promise a 12 s wake from a 12 s adapter restore.

## Where I disagree

**Route L as specified in JIT-PROXY §1 will not work as written.**
`ignore-carrier=yes` handles *carrier loss*, not *admin-down*. If the proxy
runs `ip link set <port> down` on an NM-managed device, NetworkManager sees
the device go unavailable (admin state, not carrier) and deactivates the
connection — which drops the IPv4 address and flushes the RoCEv2 GID at
index 3 that the surviving QPs' address handles depend on. The correct
sequence is: `nmcli device set <p0> <p2p> managed no` (or a `unmanaged`
`[device]` rule), *then* `ignore-carrier` becomes irrelevant, then admin-down.
This must be fixed before experiment 1 or you'll measure a false negative.

**Codex's "process survival did not prove CUDA state integrity" is backwards
emphasis.** The CUDA contexts are fine — the GPU was never touched. The real
residue is what §7 actually observed: NCCL proxy threads wedged at ~50 % CPU
in kernels that will never complete. For route B this matters concretely:
`ncclCommDestroy` on a comm with an in-flight (or permanently stalled)
collective blocks; you need `ncclCommAbort` in the suspend path, and you must
verify those threads actually exit, or every suspend/resume cycle leaks a
spinning thread and eventually the SMs.

**Route A quietly abandons the stated goal.** "Keep the model resident" is
the whole point; a 505 s (or 100 s) full relaunch reloads weights and drops
KV. It's the right fallback, but the decision section should say plainly that
route A is "accept the reload," not a variant of resident idle.

## Risks neither review mentions

1. **The NVMe KV / slab transfer path has its own RDMA state.** DCP moves KV
   between ranks; if that goes through anything other than the NCCL
   communicators (an ibverbs-based transfer engine, LMC-style), those
   contexts die with the adapter too. Route B's rebuild inventory must cover
   them, and "drain slab/bounce transfers" is not enough — they need teardown
   and re-init alongside the process groups. Nobody has inventoried this.
2. **Split-brain restore.** The dead-man timer is per-node systemd. If one
   node reboots (or its transient unit is lost) while the other three
   restore, you get a half-up ring that passes per-node checks and fails
   collectives. Restore needs cluster-level reconciliation: all four nodes
   report `--up` verified before any resume proceeds.
3. **The proxy is a new SPOF that is idle exactly when it matters.** If it
   dies while the adapters are off, requests arrive with nothing to wake the
   cluster. It must be systemd-supervised with a watchdog, and the wake hook
   should be triggerable out-of-band (management LAN).
4. **Cycle fatigue and hysteresis.** Sparse-bursty traffic (one request every
   95 minutes) turns this into dozens of PCIe hot-plug cycles per day. §7
   already logged correctable RxErr on both root ports per removal. Add a
   minimum-off time, a max-cycles-per-day budget, and track AER counters;
   `debug_state` is a debug knob NVIDIA can renumber in any DGX OS update.
5. **Pinned env vs. re-enumeration.** If the serving image sets
   `NCCL_IB_GID_INDEX` or `NCCL_IB_HCA` at startup, a post-cycle GID index
   shift silently breaks re-init even with a patched transport. Verify the
   *value and type* of the GID, and prefer `NCCL_IB_GID_INDEX` unset with
   subnet-aware routing.

## Stated-wrongly corner

- "NetworkManager re-applied them" (§1) is true for the *unmanaged* hot-plug
  path but does not carry over to admin-down, per above.
- The shm-broadcast timeout in §7 is vLLM's own IPC, not NCCL — fine as
  reported, but don't cite it as NCCL evidence.
- Codex's point that the drafter's "replicated group" is a KV-layout term is
  correct and worth repeating: inventory real handles before sizing route B.

## Recommended order

1. Fix and run route L with correct NM handling; measure watts. Half a day,
   and it may end the project.
2. New-process NCCL allreduce after a full adapter cycle, inside the serving
   container — validates devnodes, permissions, GIDs, fresh uverbs. Cheap,
   required by every route, not yet done.
3. Build the proxy with route A hooks *now*, in parallel: it is
   route-agnostic and needed in every outcome.
4. Same-process re-init gate (route B), with `ncclCommAbort` semantics and
   the KV-transfer inventory included.
5. Only on a passed gate: the 700–1300-line engine work, transport patch
   last.
