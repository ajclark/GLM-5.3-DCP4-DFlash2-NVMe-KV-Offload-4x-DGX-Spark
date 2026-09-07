 Recommendation: ship Option A — and note it's already implemented.
 src/hotplug.inc.c already contains the TCP control endpoint (one verb line in,
 one reply out, per-connection detached threads), plus a metrics verb that
 answers both bare TCP and plain HTTP GET with Prometheus text format
 (text/plain; version=0.0.4). So A costs you only the watcher script; there is
 nothing to add in the plugin.
 Q1 — pitfalls:
 - Port per process: handled — EADDRINUSE walks the next 15 ports and logs the
   winner. The watcher must therefore probe the range (or parse the log line)
   rather than assume the base port. With one vLLM per node the first port
   usually wins; don't rely on it.
 - Multiple NCCL processes per node: each binds its own port. The watcher should
   scrape the whole range and aggregate: power down only if every answering
   endpoint reports idle >= 90m and wanted=0; resume if any reports wanted=1.
 - Security: commit/resume are unauthenticated job-killers on a LAN-reachable
   port. Binding to the NCCL_SOCKET_IFNAME management NIC helps; add an iptables
   rule allowing only the head node, or at minimum a shared-token prefix on
   mutating verbs. Cheap and worth it.
 - Blocking: already correct — status/metrics answer while a resume runs
   (separate thread per connection); verbs serialize on the state lock by design.
 - Port stops answering: treat as "unknown", never as "idle". If any of the four
   endpoints is unreachable or reports failed/error, hold the current power state
   and alert. Powering off on a missing answer is the one way this design can
   corrupt a live job. Also distinguish connection-refused (process gone → node
   is down anyway) from timeout (hung — alert).
 Q2 — anything simpler? No. Option C's wake signal does not exist: vLLM V1's
 API-server /metrics is populated from engine-core state updates emitted after a
 completed scheduler step, so a request accepted while the engine is gated inside
 a step never increments num_requests_running or num_requests_waiting — the front
 end simply hasn't been told. There is no standard vLLM metric or endpoint
 reflecting accepted-but-not-yet-stepped requests. Your wanted=1 (NCCL called
 into the gated plugin) is the only reliable wake signal, and it's already
 produced.
 Q3 — standard metric: done. Expose the watcher to GET /metrics (Prometheus
 format) on the control port; any scraper works, and your bash watcher can curl
 it unchanged. Optionally add a state="active|prepared|suspended|failed" label
 for alerting.
 glm done
