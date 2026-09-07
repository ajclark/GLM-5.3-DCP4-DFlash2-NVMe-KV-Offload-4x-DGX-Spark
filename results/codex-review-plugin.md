# Codex review of the NCCL hot-plug plugin (2026-09-07) and what was done

Reviewer: Codex (gpt-6-astra, xhigh), read-only review of
`~/nccl-net-hotplug/plugin` at c46c29e against NCCL v2.31.2-1 sources
(`src/transport/net.cc`, `src/plugin/net/net_v11.cc`, `src/plugin/net.cc`,
`src/plugin/plugin_open.cc`, `src/proxy.cc`). Its verdict on the parts it
confirmed: the v11 compatibility path is sound (NCCL calls `init` before
copying the plugin's entry points, so the locked wrappers are what the core
calls); `NULL` from `isend`/`irecv` is retried without advancing proxy
counters; no logger re-entry or core lock inversion; preserving
`fifoHead`/`fifoTail`/`qpIndex` across a rebuild is correct once both ends
are drained; ECE ordering, FIFO key directions and TC/SL reuse match
connection establishment.

| # | finding (Codex's severity) | applies on the Sparks? | done |
|---|---|---|---|
| 1 | P1: several `ncclIbDevs` entries can share one verbs context (ports of one HCA, the Data Direct alias); one async thread per entry means two readers on one async fd, one can block uninterruptibly; suspend closed each entry separately (double close) | not with `NCCL_IB_HCA=roceP2p1s0f0,roceP2p1s0f1` (one port per device, CX-7 has no Data Direct), but cheap to get right | one event thread per context at init and at re-open; close each distinct context once; `ibvName` kept per entry for re-open |
| 2 | P1: local quiescence is not distributed quiescence; a peer's CTS write can target a rank that already suspended | yes, whenever a request lands during `--down` | two-phase protocol: `prepare` gates the data path (isend/irecv/connect/accept return "call again") and refuses if anything is in flight; only when every rank reports `prepared` does the operator `commit`; otherwise `abort` everywhere. With all ranks gated and idle nothing new can start, so commit is safe |
| 3 | P1: a failed resume was not retryable (second attempt overwrote live objects; socket protocol state lost) | yes | resume failure before the comm rebuild (devices not back, GUID mismatch) closes what reopened and stays `suspended` (retryable); failure after that releases the half-built objects and enters `failed`, where the data path returns `ncclSystemError` (the job dies loudly instead of hanging) and every verb reports `error`. Teardown failures also go to `failed` |
| 4 | P1: connect/accept mid-handshake own RDMA objects but are not in the registry; regMr while suspended | yes at init/warm-up, not while idle | pending-handshake counter (allocation to registration or failure), prepare refuses while > 0; connect/accept gated at entry; regMr while suspended returns a tracked handle and registers at resume |
| 5 | P1 if DMA-BUF: NCCL closes the DMA-BUF fd right after `regMrDmaBuf`, resume reused the number | no (GDR disabled on the Sparks, host buffers) | fd duplicated (`F_DUPFD_CLOEXEC`) at registration, used at re-registration, closed at deregistration |
| 6 | P1 if flushing: `iflush` had no gate; NULL would mean "flushed" | no (GDR 0) | while gated, `iflush` hands out a deferred request that `test` keeps pending and posts once devices are back; it does not count as busy |
| 7 | P1: the last `finalize` left the control and async threads running while NCCL may `dlclose` | yes at shutdown | last finalize joins the control thread, stops the event threads, closes devices, removes the status file, resets discovery (verified: device removable right after the harness peers finalize) |
| 8 | P2: receiver did not apply the MTU minimum; inherited bug indexes the remote sizes-FIFO key with the local device index | symmetric hardware today | both fixed (accept path, resume path, `qp->remDevIdx`) |
| 9 | P2: re-open assumed device identity; `_dma` names cannot match verbs names | identity: yes | re-open by the verbs name, `sys_image_guid` compared, port speed/width/MTU changes logged |

Also from the review: static `--device /dev/infiniband` nodes may miss
re-enumerated minors. The hot-plug lane and the probe runner now bind-mount
`/dev/infiniband` and allow the whole uverbs major in the device cgroup.
Serving logs confirm the retained sockets use the LAN (`OOB enP7s7`), not the
adapter being cycled.

Sandbox verification after the changes (`test/hp_test.c`, soft-RoCE): busy
refusal at suspend and at prepare, gate held for isend and connect, abort then
transfer, prepare/commit, registration while suspended, device delete/re-add,
resume, transfers with checksums, 4 cycles direct plus 3 cycles through the
control files, device removal after finalize; 50-cycle soak flat on RSS and
descriptors. Still untested anywhere: the real NCCL core loading the plugin,
the ConnectX-7, GDR/flush/DMA-BUF paths.
