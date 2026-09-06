# Codex second opinion: the energy-saving vLLM proxy / just-in-time inference (2026-09-06)

Prompt: the human's idea (no request for 90 min -> CX-7 off with the serving stack resident; request -> CX-7 on, pay the first-message penalty) with last night's measurements (docs/CX7-POWER.md §7) and the fork's elastic-EP machinery; asked for a layered work breakdown. Codex (gpt-6-astra, xhigh) read the fork and NCCL 2.31.2's IB transport source. Verbatim answer:

• The resident-stack design is worth pursuing, but communicator teardown alone is insufficient in upstream NCCL 2.31.2. Make transport recovery the first
  gate, before proxy integration.

  ### 1. NCCL/ibverbs: the critical blocker

  In NCCL’s init.cc:281 (https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/transport/net_ib/init.cc#L281), ncclIbFinalizeDevices() only decrements
  netRefCount. Initialization enumerates devices only when ncclNIbDevs == -1 (:298), which finalization does not restore. ncclIbFinalize() (:590) frees
  per-communicator configuration—not the cached ibv_contexts.

  Thus, even successful destruction of every communicator does not imply fresh device contexts after hotplug. Merged-device tables and GID information are
  also cached; detached IB event threads complicate resetting them.

  I would not pursue socket-first initialization, lazy initialization, or dynamic plugin reload as workarounds: none establishes safe teardown of the built-
  in IB state. Credible options are a properly resettable NCCL IB backend, a hotplug-aware external plugin selected from startup, or worker restart. Any
  replacement must preserve Spark’s subnet-aware routing behavior.

  Decisive experiment: stack down, exact serving image/library, four persistent processes, tiny GPU buffers, independent management-LAN Gloo/store:

  allreduce → synchronize → destroy all comms → all-rank acknowledgement → NIC cycle → fresh ID → re-init → checked allreduce

  Use an external watchdog. Log library paths/builds, timestamps, NCCL INIT/NET diagnostics, uverbs opens/closes, context pointers, GIDs, and persistent-
  buffer checksums. Repeat with overlapping PyTorch/PyNccl groups, then graphs. Expect the unpatched backend to fail; the installed build is authoritative.

  ### 2. Proxy and clients

  Use one serialized state machine:

  READY → DRAINING → OFF → WAKING → READY/FAILED

  Start the 90-minute timer after the last request completes, not merely the last arrival. Require no active requests; exclude health polling from activity.
  Bound queued requests, propagate cancellations, and make wake single-flight. A generation counter prevents an old idle timer powering down during wake.

  For streaming clients, send SSE comments every approximately 10 seconds with buffering disabled. Once HTTP 200 is committed, failure requires a stream
  error and closure—not a later HTTP 503. Non-streaming JSON cannot contain SSE comments: hold with explicit deadlines or return 503 Retry-After.

  Check pi/SDK first-byte, read-idle, total-request, and intermediary timeouts; keepalives defeat only some. Never fabricate tokens or automatically replay
  an already-forwarded request.

  Separate liveness from readiness. /health = 200 is insufficient; readiness requires completed recovery and a real generation probe.

  ### 3. API/EngineCore: pause without sleeping memory

  Add authenticated suspend_network/resume_network orchestration through the API’s engine client, EngineCore, and worker collective_rpc.

  Reuse pause_scheduler(mode="wait", clear_cache=False) or /sleep?level=0&mode=wait: /home/napta2k/lmcache-mg/spark-src/vllm/v1/engine/core.py:761 explicitly
  makes level zero scheduling-only. Never use default sleep level one: it offloads weights and discards KV.

  Drain scheduler/output activity and slab/bounce transfers before worker teardown. Keep scheduling paused until all workers pass recovery.

  Serialize control RPCs during rebuilding. Workers cannot consume another RPC while blocked rebuilding; /home/napta2k/lmcache-mg/spark-src/vllm/v1/executor/
  multiproc_executor.py:374 enqueues without passing the response timeout. That timeout does not cancel stuck GPU work. Preserve the message queues; do not
  treat the broadcast warning as evidence they need rebuilding.

  ### 4. Workers: suspend sequence

  While links remain up:

  1. Drain transfers; synchronize GPU work.
  2. Release target and drafter graphs, including retained references.
  3. Destroy every actual device communicator in a deterministic, identical group order—auxiliary/DCP/TP before world; PyNccl/device wrappers before their
     backing torch groups.

  4. Complete transport cleanup; acknowledge all four ranks before power-off.

  The current PyNccl destroy:148 (/home/napta2k/lmcache-mg/spark-src/vllm/distributed/device_communicators/pynccl.py:148) starts a daemon abort thread and
  waits only five seconds. Its return is not a teardown-complete acknowledgement.

  Do not call generic group/environment destruction: /home/napta2k/lmcache-mg/spark-src/vllm/distributed/parallel_state.py:1170 also destroys Gloo and drops
  broadcaster state. For fixed membership, establish a persistent Gloo default/control world and explicitly NCCL-backed device groups at startup; retain
  store ownership.

  Also, the drafter’s “replicated group” describes KV layout (tests/test_dcp_replicated_group.py:2), not necessarily another communicator. Inventory actual
  handles.

  ### 5. Workers: resume and elastic-code reuse

  After hardware validation, rebuild device groups with fresh IDs and epoch-qualified rendezvous keys, replace every cached communicator reference, test
  collectives, then recapture graphs.

  Elastic EP is a template, not a drop-in:

  - _replace_active_groups() replaces DP/EP/world/EPLB—not TP/DCP.
  - _release_cuda_graphs():331 (/home/napta2k/lmcache-mg/spark-src/vllm/distributed/elastic_ep/elastic_execute.py:331) resets compilation and handles the
    target wrapper.

  - Reuse the block-table save/clear/restore pattern at :498, adding try/finally and drafter coverage.
  - Full warmup also autotunes and resets RNG: /home/napta2k/lmcache-mg/spark-src/vllm/v1/worker/gpu_worker.py:674. Preserve RNG and ensure dummy runs cannot
    overwrite resident KV.

  Keep weights, KV allocation, slab index and cache epoch unchanged. Validate retained-prefix hits and generation afterward. The documented 40–60-second wake
  remains an estimate; process survival did not prove CUDA state integrity.

  ### 6. OS recovery

  Extend existing checks to validate the GID value/type, not index alone, and fresh uverbs nodes/permissions inside the surviving container. Retain
  NetworkManager checks, mstflint precautions, management connectivity and node-local restore timers. Hardware restoration must not automatically unpause
  scheduling.

  Admin-down avoids PCIe disassociation, but does not guarantee QPs survive port events. Test separately; its power savings remain unmeasured.

  ### 7. Work order and fallback

  Estimated implementation sizes, excluding tests:

  1. Transport spike: 150–250 lines.
  2. Worker/group/graph overlay changes: 400–800 lines across gpu_worker.py, parallel_state.py, pynccl.py and graph helpers.
  3. API/core orchestration: 100–200 lines.
  4. Proxy: 200–400 lines.

  Transport reset is separate, higher-risk work. If that gate fails, ship proxy-controlled stop/relaunch: approximately 505 seconds today, potentially 80–120
  seconds after the unbuilt disk-image loader. Do not promise a 12-second inference wake from a 12-second adapter restoration.

