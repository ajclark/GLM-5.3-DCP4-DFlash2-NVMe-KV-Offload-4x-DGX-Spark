      1322    static_assert(NCCL_STEPS <= NCCL_NET_MAX_REQUESTS, "Not enough net requests to cover for steps");
      1323
• Searching the web
• Searched the web for https://raw.githubusercontent.com/NVIDIA/nccl/v2.31.2-1/src/plugin/plugin_open.cc
• Ran git diff --stat && git status --short && nl -ba src/ibvwrap.c | sed -n '380,405p' && rg -n 'NODELETE|nodelete|destructor|constructor' src include
  └ (no output)
⚠ Heads up, you have less than 25% of your weekly limit left. Run /status for a breakdown.
✔ You approved codex to run curl -fsSL --max-time 25 https://raw.githubusercontent.com/NVIDIA/nccl/v2.31.... this time
• Ran curl -fsSL --max-time 25 https://raw.githubusercontent.com/NVIDIA/nccl/v2.31.2-1/src/plugin/plugin_open.cc -o /tmp/hpreview-plugin-open.cc
  │ https://raw.githubusercontent.com/NVIDIA/nccl/v2.31.2-1/src/os/linux.cc -o /tmp/hpreview-os-linux.cc
  └ (no output)
• Explored
  └ Search RTLD_|Dlclose|Dlopen|dlclose|closePlugin|ClosePlugin|refCount in hpreview-plugin-open.cc
    Read hotplug.inc.c
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
• I’ve completed the source trace. NULL from isend/irecv is retried safely by NCCL, and I found no logger re-entry or core lock inversion. I also confirmed a shutdown
  defect: finalization leaves the control and async threads running, while NCCL 2.31.2 can unload the library after the last communicator closes. I’m consolidating the
  ranked findings and test blockers now.
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
• I found several correctness defects beyond the passing harness. This was a read-only review of nc46c29e against NCCL’s v2.31.2-1 sources; I did not rerun hardware
  tests.
  1. P1 — Shared verbs contexts can cause double-close and an uninterruptible async thread. ~/nccl-net-hotplug/plugin/src/p2p_plugin.c:667,
     ~/nccl-net-hotplug/plugin/src/hotplug.inc.c:139. Initialization shares one context across its ports and Data Direct aliases, but creates a thread per
     entry; suspend closes each entry separately. Two threads polling the same async fd can both wake, then one blocks inside ibv_get_async_event after the other
     consumes the event. Its eventfd cannot interrupt that read. Use one context owner, event reader, and close per physical context; handle nonblocking EAGAIN.
     MERGE_NICS=0 alone does not eliminate context aliasing. Polling public async_fd is supported. Verbs documentation
     (https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_get_async_event.3)
  2. P1 — Local quiescence does not establish distributed quiescence. ~/nccl-net-hotplug/plugin/src/hotplug.inc.c:125. A sender can have no requests while
     its peer has posted—or is about to post—an unmatched receive. Suspending that sender destroys the destination of the peer’s CTS; the peer then reports busy or
     encounters a transport error. Waiting for every rank’s suspend acknowledgment happens too late. Require application admission stopped and work drained across ranks,
     followed by a coordinated prepare/commit barrier before destroying anything.
  3. P1 — Failed resume is not safely retryable. ~/nccl-net-hotplug/plugin/src/hotplug.inc.c:189, ~/nccl-net-hotplug/plugin/src/
     hotplug.inc.c:448. A second attempt overwrites contexts, thread handles, CQs, QPs and MRs without releasing the previous attempt’s objects. Discarding ctxs also
     loses TCP offsets and protocol state: unread metadata or a ready flag can become the next attempt’s metadata. Some peers may already be active. Preserve resumable
     state or implement coordinated rollback with protocol generations; otherwise enter an explicit terminal failure state. Suspend also needs fail-closed handling:
     device-close failures are swallowed, and repeated suspend returns success after partial teardown.
  4. P1 — Connect/accept and registration remain unsafe during suspension. ~/nccl-net-hotplug/plugin/src/ib_plugin.c:1159, ~/nccl-net-hotplug/
     plugin/src/ib_plugin.c:1469, ~/nccl-net-hotplug/plugin/src/ib_plugin.c:1586. Connections enter the registry only after handshake completion, although
     intermediate stages already own RDMA objects. The rwlock protects individual calls, not intervals between calls. Suspend can therefore close their context
     underneath them. Subsequent connect/accept/regMr calls also lack a suspended gate. Track pending connections and refuse suspension while any exist; return NULL
     comms while suspended. Registration can create a tracked logical handle and defer verbs registration until resume.
  5. P1 when DMA-BUF is used — Saved descriptors are already closed. ~/nccl-net-hotplug/plugin/src/ib_plugin.c:1585,
     ~/nccl-net-hotplug/plugin/src/hotplug.inc.c:231. NCCL closes the supplied DMA-BUF fd immediately after registration. Retaining its integer gives resume
     an invalid or reused descriptor. Duplicate it with F_DUPFD_CLOEXEC, retain that duplicate through suspension, and close it on deregistration and failure cleanup.
     NCCL registration paths (https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/transport/net.cc#L1008)
  6. P1 when flushing is required — iflush has a suspension race. ~/nccl-net-hotplug/plugin/src/ib_plugin.c:1971. Contrary to the request’s description, this
     implementation has no suspended gate. Suspend can intervene between receive test(done=1) and iflush, leaving NULL MRs/QPs. Adding a NULL-request gate would
     incorrectly declare flushing complete: NCCL does not retry iflush. Support a deferred flush request, or guarantee required flushing before receive completion
     permits suspension. With core GDR disabled this path is avoided; plugin flushEnabled itself depends on detected capabilities. NCCL flush progression
     (https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/transport/net.cc#L1663)
  7. P1 — Finalization leaves threads executing plugin code. ~/nccl-net-hotplug/plugin/src/ib_plugin.c:710,
     ~/nccl-net-hotplug/plugin/src/hotplug.inc.c:523. Finalization only decrements a reference count; neither async workers nor the detached control thread
     stop. NCCL can unload the library after its last communicator, causing execution of unmapped code. Stop/join workers and release device resources before
     finalization returns at the last reference. The ordinary serialized stop path does not double-join. NCCL unload path
     (https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/plugin/net.cc#L308)
  8. P2 — Heterogeneous rails expose protocol defects. ~/nccl-net-hotplug/plugin/src/hotplug.inc.c:346 omits the receiver-side MTU minimum; apply
     min(local,remote) on both endpoints. Separately, inherited ~/nccl-net-hotplug/plugin/src/ib_plugin.c:1736 indexes remote sizes-FIFO keys using local
     devIndex; use qp->remDevIdx. Unequal merged-device counts can otherwise break grouped receives.
  9. P2 — Reopen assumes device identity and capabilities survive unchanged. ~/nccl-net-hotplug/plugin/src/hotplug.inc.c:171. Synthetic _dma names cannot
     match real verbs device names. Store physical identity separately and validate GUID/PCI identity, provider, link layer and required capabilities. Speed/width may
     renegotiate, affecting tuning. Preserve existing vProps numbering because NCCL retains topology; reject incompatible changes instead of silently rebuilding
     membership.
  For the specific quiescence question: UNUSED requests do not prove an empty SQ, but completed receives establish that their CTS was consumed. After both endpoints
  drain, unreaped unsignaled CTS work does not require replay. Preserving fifoHead, fifoTail, qpIndex and slot indices is correct. ECE ordering, FIFO address/key
  directions, TC/SL reuse and refreshed SGEs otherwise match establishment.
  NULL isend/irecv requests are retried without advancing proxy counters or testing NULL. I found no core/logger lock inversion. Keeping ready=1 is acceptable behind the
  gates. The v11 adapter (https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/plugin/net/net_v11.cc#L54) copies entry points after initialization; older shims reach the
  wrappers. getDeviceMr/irecvConsumed are NULL; makeVDevice does not post RDMA.
  Before power cycling, address the applicable P1s, ensure retained TCP uses a surviving interface, and verify device opening inside the container after
  re-enumeration—static device mappings may miss new minors. Docker device guidance (https://docs.docker.com/reference/cli/docker/container/run/#device-cgroup-rule) Then
  test actual NCCL, graph replay, grouped receives, FIFO wraparound and multiple QPs. Performance parity remains unmeasured; include small-message latency because the
  process-wide rwlock adds contention.
─ Worked for 5m 43s ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
› Design review, answer in under 400 words, ranked recommendation first. Context: four DGX Sparks in a switchless RoCE ring run one vLLM process per node (NCCL 2.31.2)
  with our hot-plug-aware NCCL net plugin (~/nccl-net-hotplug/plugin, src/hotplug.inc.c). Proven on the cluster today: an operator issues prepare on all four ranks
  (plugin gates isend/irecv/connect/accept, refuses if anything is in flight), then commit on all four (QPs/MRs/CQs/PDs destroyed, devices closed), powers the ConnectX-
  7 off via sysfs on each host, later powers it on and issues resume on all four (devices reopened, QPs re-connected over the plugin's retained TCP sockets). The plugin
  exposes a status line: state, idle=<seconds since the last NCCL send/receive>, wanted=<1 once NCCL called into the gated plugin>. Today the control channel is a file
  per node (NCCL_HOTPLUG_CTL_DIR, cmd/status files polled every 250 ms) and a shell watcher over SSH decides: every rank idle >= 90 min -> down; any wanted=1 -> up.
  That works but the owner rejects temp files and wants "the watcher to consult a metric", fewest lines of code and moving parts; SSH from a head node is acceptable; a
  watcher on the head Spark is acceptable.
  Option A (my proposal, ~100 LOC): replace the file channel with a tiny TCP endpoint in the plugin's existing control thread, bound to the NCCL_SOCKET_IFNAME address
  on NCCL_HOTPLUG_CTL_PORT: one line in (prepare|commit|abort|resume|status), one status line out, close. Watcher on the head Spark polls the four ports (bash /dev/tcp,
  no ssh) and runs the existing spark-idle.sh (ssh + sysfs) for the power actions. Everything else unchanged.
  Option B (~700 LOC, rejected as too complex by the owner): fully autonomous plugin with a peer agreement protocol over the retained sockets (PREPARE/PREPARED/BUSY,
  COMMITTING/COMMIT_OK, WAKE, META) and in-container sysfs power writes.
  Option C: no plugin status at all; the watcher reads vLLM's /metrics. Concern: the wake signal. A request arriving while suspended is held inside the engine step
  (NCCL call gated), and vLLM V1 emits scheduler stats only after a step completes, so num_requests_running would not move; is there any vLLM metric or endpoint that
  reflects an accepted-but-not-yet-completed request from the API server process itself?
  Questions: 1) Is A the right minimum, and any pitfalls (port binding per process, multiple NCCL-using processes per node, security on a LAN, blocking the control
  thread, what the watcher should do if a port stops answering)? 2) Anything simpler than A that still gives a reliable wake signal? 3) Is there a standard NCCL/vLLM-
  side metric we should expose instead of our own status line (e.g. Prometheus text format on that port so any scraper works)? Read-only review; do not modify files.
  When done, end your reply with the words "codex done".
• I’ll check the current control code and vLLM’s request metrics, then recommend the smallest design with a reliable wake signal.
• Searching the web
• Ran rg -n 'hpState|prepare|commit|wanted|idle|Ctl|Control|Resume|LOCK' src/hotplug.inc.c && rg -n 'vllm.*(version|image)|VLLM_IMAGE|vllm/vllm' ~/
  │ spark-cluster-mla-dcp/*.sh ~/spark-cluster-mla-dcp/*.md
  └ 9: *   prepare: refuse (busy) if any request or connection handshake is in
    13: *            of the job reports prepared, nothing is in flight anywhere and
    … +104 lines (ctrl + t to view transcript)
    ~/spark-cluster-mla-dcp/README.md:93:| `baseline/vllm/…` | sixteen files exactly as the running image has them |
    ~/spark-cluster-mla-dcp/README.md:116:`~/lmcache-mg/spark-src/vllm` (the image's vLLM at commit ab666069); point
• Searched the web for site.docs.vllm.ai metrics http_requests_in_progress API server prometheus middleware ...
• Searching the web
• Explored
  └ Search server_load|/load|inprogress|should_instrument_requests_inprogress|Instrumentator|enable_server_load_tracking in entrypoints
    Read hotplug.inc.c
    Read hotplug.inc.c
• Searched the web for site.docs.vllm.ai "server_load" "/load" ...
• Searching the web
• Searched the web for https://docs.vllm.ai/en/latest/api/vllm/entrypoints/utils/
• Explored
  └ Read hotplug.inc.c
    Read hotplug.inc.c
    List napta2k
    Search /(api_server|utils|metrics|cli_args)\.py$
• Searching the web
• Searched the web for https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/entrypoints/serve/instrumentator/server_load.py
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
• vLLM does have a relevant frontend signal: /load with server-load tracking enabled counts requests before awaiting the handler. That could wake a request stalled in
  NCCL, subject to checking the deployed version and API worker layout. The TCP design also needs concurrent resume commands; waiting for one rank’s reply before
  commanding the next can deadlock.
• Searching the web
• Searched the web for site:docs.vllm.ai/en/latest "server_load_metrics" "load_aware_call" ...
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
• Ranking: 1) A for the complete solution; 2) vLLM /load as an optional wake input; 3) B offers insufficient benefit for its complexity.
  A is the smallest complete replacement for files while preserving plugin quiescence checks. Since SSH is acceptable, loopback TCP queried over SSH is a simpler
  security choice than remotely accessible command ports. No additional daemon is needed.
  For A, address these points:
  - Surviving interface: bind to management networking that remains available with ConnectX-7 removed. Do not assume NCCL_SOCKET_IFNAME necessarily selects that
    interface.
  - Process identity: one listener per process, initialized once across communicators. Multiple processes need explicit distinct ports and an expected-process inventory;
    never use SO_REUSEPORT to hide collisions. Every NIC user must participate.
  - Concurrent resume: send resume to all four before awaiting replies. Sequential command/reply execution can deadlock on peer handshakes.
  - Bounded handling: limit command length and socket read/write time. Resume blocks the existing control thread; either keep status independently responsive or
    explicitly accommodate that interval in watcher deadlines.
  - Failure policy: missing status means unknown, never idle. Inhibit power-off; abort preparation where possible. If already down, restore power through SSH and
    reconcile plugin states. Preserve command generations and process incarnation identifiers; a lost reply does not mean the command failed.
  - Security: for direct LAN access, restrict the port to the watcher host. Binding an address alone provides no authorization.
  C has a better wake signal than scheduler metrics: enable --enable-server-load-tracking and query GET /load
  (https://docs.vllm.ai/en/v0.16.0/api/vllm/entrypoints/openai/basic/api_router/), returning {"server_load": N}. Its request decorator
  (https://docs.vllm.ai/en/v0.10.2/api/vllm/entrypoints/utils.html#vllm.entrypoints.utils.load_aware_call) increments before awaiting the handler, so an NCCL-stalled
  request can remain visible. Validate the deployed version, streaming/cancellation behavior, and every API worker: this counter is process-local. It supplies
  application demand, but does not replace plugin commands or quiescence acknowledgments. Plugin wanted remains the strongest transport wake signal.
  For scraper compatibility, expose your own gauges—nccl_hotplug_idle_seconds, nccl_hotplug_wanted, and labeled state—through HTTP GET /metrics, with controls using
  POST. Prometheus text exposition (https://prometheus.io/docs/instrumenting/exposition_formats/) requires HTTP, not merely metric-shaped text over raw TCP. Use atomic
  snapshots and keep wanted sticky until abort/resume. No Prometheus server is necessary; the watcher can use curl.
  codex done
─ Worked for 1m 53s ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
› Root-cause analysis, then a fix. Read-only review; do not modify files. Evidence is in /tmp/claude-1000/-home-napta2k-spark-cluster-experiments/070fd9a0-7c72-4917-
  bd94-b1701ab77623/scratchpad/triton-crash/ (incident-2026-09-04.md, incident-2026-09-07.md, rank0-traceback.txt, rank0-crash-window.log, and the code involved:
  sm12x_mqa.py (overlay Triton kernels, the crash launches _fp8_mqa_logits_kernel at line ~156), sm12x_deep_gemm_fallbacks.py, sparse_attn_indexer.py (vLLM, line 547
  call site), sparse_mla_kernels.py, model_runner.py (vLLM V1 GPU model runner: warmup/dummy runs/cudagraph capture), deep_gemm_and_monitor.txt (vllm/utils/
  deep_gemm.py, the jit_monitor that logs "Triton kernel JIT compilation during inference", versions). Web search is welcome; you can read files, please do not run the
  cluster.
  System: 4x NVIDIA DGX Spark (GB10, sm_121, unified memory 128 GB per node, no discrete VRAM), driver 580.159.03, CUDA 13.0, torch 2.11.0+cu130, triton 3.6.0, vLLM
  0.23.1rc1.dev190. GLM-5.3 (DeepSeek-V3-style MoE with DSA sparse attention indexer), TP=4 across the nodes over RoCE, DCP=2, DFlash speculative decoding, an NVMe KV
  tier (vLLM's own MultiNodeSlabConnector writing slab files on local NVMe), CUDA graphs PIECEWISE + FULL captured at startup. TRITON_CACHE_DIR is a host volume that
  persists across launches (/cache/huggingface/.tritoncache). Env: GLM52_MQA_LOGITS_TRITON=1 (use the Triton fallback for the indexer's fp8 MQA logits). Nodes run hot
  on memory: after launch ~4.5-5.9 GB MemAvailable, swap in use (worker VmSwap ~0.7 GB 35 min after launch; pswpout cumulative in the millions of pages),
  vm.max_map_count=1048576, worker fds ~220 of 1M, no OOM kills, no NVRM/Xid in the kernel log at either crash.
  The two crashes (identical): rank 1 on 2026-09-04 02:29 PDT and rank 0 on 2026-09-07 13:59 PDT. Traceback: sparse_attn_indexer.sparse_attn_indexer ->
  deep_gemm.fp8_fp4_mqa_logits -> sm12x_deep_gemm_fallbacks._fp8_mqa_logits_sm12x -> sm12x_mqa.fp8_mqa_logits_triton -> _fp8_mqa_logits_kernel[grid](...) -> triton/
  runtime/jit.py:743 run -> compiler.py:482 launch_metadata -> compiler.py:465 _init_handles -> driver.active.utils.load_binary -> RuntimeError: Triton Error [CUDA]:
  operation not permitted (CUDA_ERROR_NOT_PERMITTED, 800; not 900 "when stream is capturing"). The worker dies, the three peers wait in a collective (today:
  _ALLGATHER_BASE of 8,093,696 elements, a prefill-sized DCP all-gather), the torch NCCL watchdog fires after 600 s and terminates everything 8 minutes later. Both
  crashes happened hours into the process (4.5 h and 3 h), under a long prompt (prefill path), with MemAvailable well under 1 GB (09-04) and ~2.5 GB (09-07). The
  jit_monitor prints a warning whenever Triton COMPILES a kernel at inference; today the last such warning was 3 hours before the crash (the same kernel compiled fine
  at the first request, and a DFlash kernel later); the crashing launch produced NO compile warning, so the cubin was a Triton cache hit and the thing that failed is
  the lazy first LOAD of that (new specialization's) cached cubin into the CUDA context of a process that had been running for hours. On a fresh launch the same loads
  always succeed (1042 "JIT during inference" warnings in our logs, 2 failures).
  Questions:
  1. What returns CUDA_ERROR_NOT_PERMITTED from cuModuleLoadData in this situation? Candidates to confirm or kill with evidence: (a) memory exhaustion on a unified-
  memory system (the driver pinning/mapping the module's code+data fails with EPERM/ENOMEM and surfaces as 800 rather than OUT_OF_MEMORY); (b) something that makes
  module loading illegal in this process state: a stream capture in progress on another thread in global mode (does vLLM capture anything lazily at runtime here:
  DFlash, PIECEWISE compilation, torch.compile aot?), a host-func callback context, CUDA lazy loading (CUDA_MODULE_LOADING) semantics; (c) a stale/foreign cubin in the
  shared-across-launches Triton cache (different arch/driver/triton build) - would that give 800 or 209/222?; (d) the process's mapped-memory or descriptor limits; (e)
  anything GB10/driver-580 specific you can find (Jetson/Grace/Spark reports of "operation not permitted" on module load). Say which one the evidence supports and what
  single measurement would settle it.
  2. Why is this kernel loaded lazily at inference at all, hours after start, when vLLM warmup ran? Look at what specializations _fp8_mqa_logits_kernel gets (integer-
  arg divisibility/==1 specialization on num_q, seq_len_kv, strides, etc. in sm12x_mqa.py) and what the model runner's dummy/warmup runs cover for the sparse indexer
  prefill path. Is do_not_specialize on the runtime-varying int args a correct fix (one binary, loaded once at warmup)? Any performance cost?
  3. The fix, ranked, with the smallest change first: e.g. (i) warm the kernel at startup so the load happens before serving (how, concretely, in this stack: a warmup
  request of the right shapes after boot in the launcher/rollout, or a hook in the overlay), (ii) do_not_specialize + one warm launch, (iii) eager module loading
  (CUDA_MODULE_LOADING=EAGER) or pre-loading every cubin in the Triton cache at startup, (iv) memory headroom (which knob: KV pool, the slab tier's page-cache pressure,
  swappiness), (v) making a failed load non-fatal (retry once in the overlay after freeing memory) so a rank does not die. For each: does it remove the failure or only
  move it, and what is the blast radius if it is wrong.
  Under 700 words. End with the words "codex done".
• The confirmed defect is excessive specialization plus incomplete warmup. The reason CUDA returns 800 remains unproven. Memory pressure is the strongest observed
  correlate, but the evidence does not establish an ENOMEM→800 driver mapping.
  Two corrections matter:
  - /tmp/claude-1000/-home-napta2k-spark-cluster-experiments/070fd9a0-7c72-4917-bd94-b1701ab77623/scratchpad/triton-crash/deep_gemm_and_monitor.txt:773 uses warning_once
    per kernel name. No warning does not prove a disk-cache hit. The 1,042 warnings are not a count of successful loads.
  - Triton’s load_binary calls context APIs, cuModuleLoadData, cuModuleGetFunction, and attribute APIs, using indistinguishable error messages. The traceback does not
    identify which CUDA call failed. Triton backend (https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/backend/driver.c#L93).
  CUDA documents OUT_OF_MEMORY for module loading and distinct errors for incompatible binaries/PTX. Thus memory exhaustion and foreign-cache explanations remain
  hypotheses, not diagnoses. Module APIs can also surface earlier asynchronous errors. CUDA module API
  (https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__MODULE.html).
  Calling CUDA from a CUDA callback can produce NOT_PERMITTED, but this traceback shows the worker execution loop, not a callback. CUDA stream API
  (https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html). model_runner.py:1293 is explicitly the eager/NONE branch; supplied DFlash/graph capture calls
  occur during startup. An AOT wrapper frame does not establish capture. Another thread’s global capture remains unexcluded. Descriptor exhaustion looks unlikely;
  configured max_map_count alone does not establish actual mapping usage.
  There is an unresolved report on GB10 with driver 580.159.03 (https://github.com/vllm-project/vllm/issues/52877) showing the same Triton load error after extended
  uptime. It corroborates the symptom, not its cause.
  The next measurement should be one instrumented failure trace: record the exact failing CUDA API and numeric result inside the installed load_binary, cubin hash,
  thread/context, all-thread capture transitions, and contemporaneous memory. That distinguishes the currently conflated failure sites. Proving the proposed pinning/
  allocation mechanism additionally requires a driver allocation-failure trace; MemAvailable alone cannot settle it.
  Warmup misses this because /tmp/claude-1000/-home-napta2k-spark-cluster-experiments/070fd9a0-7c72-4917-bd94-b1701ab77623/scratchpad/triton-crash/sm12x_mqa.py:64
  declares dimensions and every stride tl.constexpr: specialization uses exact values, not merely divisibility/==1. In particular, output row stride equals seq_len_kv.
  The indexer’s profiling path (sparse_attn_indexer.py:396–414) returns without launching MQA; actual prefill chunks introduce new query/context lengths. Finite graph
  warmup cannot cover that space.
  Smallest interventions first; item 3 is my preferred durable correction:
  1. Emergency switch: GLM52_MQA_LOGITS_TRITON=0 selects the existing Torch fallback. This removes this kernel’s Triton-load failure opportunity, but increases latency
     and potentially memory: the fallback materializes float32 K. Validate before adopting.
  2. Startup request: send an uncached long prefill before admitting traffic; verify every worker actually executes nonempty indexer chunks. This moves covered loads
     earlier, but another exact length can still introduce a binary. Cost: startup time and transient memory.
  3. Reduce specialization, then launch during startup. Remove tl.constexpr from num_q, seq_len_kv, stride_lm, and other varying strides; add those arguments to
     do_not_specialize. Keep structural dimensions/tiles constexpr. The decorator alone is insufficient. Launch and synchronize a small representative case in each
     worker context, outside capture, with production dtypes/layout and clean_logits=False. Triton .warmup() only compiles. Cover remaining alignment/configuration
     variants. This removes shape-driven late loads; benchmark masking/addressing and vectorization costs, and verify logits/top-k correctness.
  4. Restore headroom: reduce explicit KV-cache bytes and prefill batch size; bound slab staging/dirty-page pressure. Clean page cache is reclaimable; swappiness is not
     a CUDA allocation remedy. This mitigates pressure at a capacity/throughput cost.
  5. EAGER/cache preloading: EAGER cannot load Triton binaries that Triton has not submitted yet. Loading the entire cache wastes memory and does not populate Triton’s
     handle bookkeeping. CUDA loading semantics (https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html).
  6. Do not blindly retry 800. Partial initialization may require cleanup. Permit bounded retry/fallback only after identifying a recoverable pre-launch failure and
     usable context; otherwise it risks another failure or collective hang.
  codex done
