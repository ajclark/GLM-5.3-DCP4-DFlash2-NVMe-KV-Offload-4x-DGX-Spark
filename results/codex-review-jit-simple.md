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
