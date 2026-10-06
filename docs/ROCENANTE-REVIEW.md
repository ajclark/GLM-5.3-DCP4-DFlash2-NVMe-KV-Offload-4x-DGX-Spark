# Review: b12x RoCEnante (RDMA proxy and kernels) before integration (2026-10-01)

**Subject:** the RoCE one-shot collectives that knapcio's stack vendors (`knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4@770d115`,
`roce/`), proposed for step 3 of `docs/DECODE-SPEEDUPS.md`. Step 3 routes only our DCP-pair collectives through
it; each pair is a direct ring link.

**What was reviewed, line by line:**
- `b12x/comm/roce/_roce_proxy.c` (614 lines), `_proxy.py` (build and ctypes binding), `roce_oneshot.py` (runtime).
- `_oneshot_cute.py`, `_cute_intrinsics.py` (GPU kernels and PTX).
- The parts of `b12x/__init__.py` and `b12x/_lib/` that load with them (`compiler.py` environment and cache handling,
  `runtime_patches.py`).
- knapcio's vLLM shim `glm_roce/` (`adapter.py`, `install.py`, `boot.py`, the `.pth` hook).

Nothing was copied into this repository or onto the Sparks during the review. The copy that step 3 then vendored, with
the local patches listed below, is `runtime/vllm029/roce/b12x/` (shim: `runtime/vllm029/roce/glm_roce/`).

## Verdict

- **Provenance:** the code is what it claims to be.
- **Intent:** nothing in it reaches outside its stated purpose.
- **Risks:** they are operational and have mitigations, not trust issues.
- **Conditions:** integrate with the changes under "Conditions", behind a switch that is off by default until
  measured.

## Provenance

- **Byte-identical to upstream:** all 30 vendored b12x files match `github.com/local-inference-lab/b12x` at the pinned
  commit `b58f34ea` (2026-09-05, "Merge pull request #315 … roce/gather-geometry"). They also match the sha256 list in
  knapcio's `PROVENANCE.json`, and there are no unlisted files.
- **Upstream changes since the pin:** only one touches the proxy. It is a configurable IP traffic class for switched
  fabrics (ABI 3→4), not needed on point-to-point cables.
- **The other fix (#383):** it concerns a startup-preparation module that does not exist at the pinned commit, so no
  safety fix is being missed.
- **Licences:** b12x is Apache-2.0 (LICENSE included). knapcio's shim is MIT, derived from local-inference-lab/vllm#597
  (Apache-2.0).

## What the code does

**The proxy:**
- **Region:** each rank pins one host region, `torch.zeros(pin_memory=True)`:
  - receive slots, two per peer;
  - per-stripe flags;
  - two send slots;
  - a 128-byte control record.
  - At world size 2 with 4 MiB slots it is about 24 MiB.
- **Registration:** the region is registered with `IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE` only: no remote
  read and no atomics. One RC queue pair is created per peer per HCA.
- **Connection setup:** QP numbers, rkeys, GIDs and the region address are exchanged once, through
  `dist.all_gather_object` over the existing gloo group. vLLM already pickles objects over that same channel.
- **Proxy thread:** a plain C pthread spins on the control word the GPU kernel writes. For each new sequence it posts
  the payload as RDMA WRITEs to every peer, each stripe followed on the same RC QP by a 4-byte inline flag write.
  RC ordering makes the flag visible only after its payload.
- **Interface:** the thread does nothing else. There are no sockets, files or processes, and no other device access.

**The kernels:**
- **Stage:** copy the input into the pinned send slot.
- **Doorbell:** `fence.sc.sys`, then the last arriving block writes the per-slot byte count, another
  `fence.sc.sys`, then the sequence number.
- **Wait:** one thread per peer and HCA spins with `ld.acquire.sys` on its flag, with a poll limit.
- **Reduce:** in fixed rank order, in fp32. Every rank gets bit-identical output. At world size 2 it equals NCCL's
  single add, and a reduce-scatter built as "exchange halves, add" stays exact.
- **Graph capture:** the sequence number lives in a device-side epoch, so CUDA-graph replays stay in step. All
  collectives must run on one stream; ours do.
- **On a timeout:** the kernel writes error words, poisons the runtime (later launches do nothing), and the host
  health check raises. The failure is fail-stop and never a silent fallback.

## Findings

| # | Severity | Finding | Mitigation |
|---|---|---|---|
| 1 | Medium (integration) | With `B12X_ROCE_HCA` unset, HCA discovery falls back to `NCCL_IB_HCA`. Ours (`=roceP2p1s0f0,roceP2p1s0f1`) puts `roceP2p1s0f0` first, and on 06c4 that faces a218, not the DCP partner 365c. | Set `B12X_ROCE_HCA=rocep1s0f1`. On all four Sparks that device faces the DCP partner (subnets .101 and .105), and NCCL never opens it. |
| 2 | Medium (power) | The proxy thread busy-spins one CPU core for as long as collectives are flowing, which is the whole time we serve. After ~20M idle polls (about 70 ms on a Spark) it naps 20 µs per poll, indefinitely: tens of thousands of wakeups a second, so one core never reaches deep idle. | **Done.** Local patch 1: hot for 50 ms after the last doorbell, then naps doubling from 50 µs to `B12X_ROCE_IDLE_MAX_NAP_US` (2 ms by default; the launcher sets 5 ms). Measured on a Spark with `runtime/vllm029/roce/tests/test_proxy_idle.c`: upstream idles at 1.7–2.4% of a core; patched at 0.2–1% (noisy at this level), with at most a few hundred wakeups a second. The first doorbell after an idle stretch is picked up within one nap (measured 0.2–1.9 ms). Node power with RoCEnante on and off is to be checked by the operator. |
| 3 | Low–medium (robustness) | Kernel poll limit: the default is 20M polls (~20 s). knapcio raises it to 300M (~5 min) so one rank's cold-boot JIT cannot time out its peer. The cost is that a dead peer takes that long to surface; NCCL would hang too. | ~60–120 s, plus the shim's barrier after `prepare()` (already in knapcio's adapter). |
| 4 | Low (defence in depth) | `post_op` checks that the byte count is a positive multiple of 16 but not that it fits a slot. An oversized count would fault locally (outside the memory region, so a completion error), but could also write past the peer's receive slot into the rest of the peer's own registered region. The Python side caps every message at a slot (`should_allreduce` ≤ `max_size`, `should_all_gather` ≤ `max_gather_bytes`, slot = max of both), so this is unreachable today. | One-line check in `post_op`: `nbytes > c->slot_bytes` → error. |
| 5 | Low (supply chain) | At first use, `_proxy.py` compiles the C file with `$CC`/gcc into a cache directory keyed by the source hash. If a `.so` of that name already exists, it loads it without verifying it. | Compile at image build with `-O2 -Wall -Wextra -fstack-protector-strong -D_FORTIFY_SOURCE=2` into a root-owned image path, point `B12X_ROCE_CACHE_DIR` at it, and record the `.so` hash. |
| 6 | Low (process-wide side effects) | Importing b12x's compiler monkeypatches the CUTLASS DSL for the whole process: it silences one warning, turns a memory-debug hook into a no-op, and wraps `inspect` for faster source locations. Benign, but it would also apply to any other CuTe DSL code in our worker. It also writes compiled kernels to `~/.cache/b12x/compile`, which is ephemeral in the container. | `B12X_DISABLE_CUTLASS_RUNTIME_PATCHES=1`; the RoCE kernels are tiny, so the faster-compile patch is not needed. |
| 7 | Info (exposure) | **Remote access:** the region is remote-writable only by the one connected peer QP per HCA, over a point-to-point cable between two of our Sparks, and nothing can read it remotely. **Startup hook:** the `.pth` line runs in every Python process in the image, but returns immediately unless `GLM_ROCE_ALLREDUCE=1`. | None needed. |
| 8 | Info | **Stale comment:** the C header's comment on the control-record layout is out of date. The code and the kernel agree: per-slot byte counts are words 4 and 5, the missing HCA is word 6. **Catch-up limit:** the doorbell holds only the newest sequence, and the proxy can catch up at most 2 missed ops; more is a fail-stop, which is consistent with "one collective in flight". | None needed. |
| 9 | Info (resources) | Per rank: about 24 MiB pinned host memory and 20–40 MiB device scratch at the planned limits (1 MiB all-reduce, 4 MiB gather). One proxy thread, and 1 QP per peer per HCA. Rank 0 currently has ~1.8 GB free after step 2. | None needed. |
| 10 | Info (environment) | No idle power-down job touches the ConnectX-7 on the Sparks (no `spark-idle` timer or cron, and NCCL hot-plug is off), so nothing will tear down the QPs underneath the runtime. | Keep it that way while RoCEnante is enabled. |

## Conditions for integrating (step 3)

1. **Vendoring:** vendor exactly the 30 upstream files at `b58f34ea` with LICENSE and PROVENANCE. Add a test that
   re-checks the hashes, plus knapcio's shim adapted to route the `dcp` groups instead of `tp` (MIT notice kept). The
   adaptation:
   - reduce-scatter as a halves exchange plus add;
   - non-edge-dim gathers via a dim-0 gather plus `movedim`, exactly like vLLM's own all-gather;
   - entering the DCP communicator's capture context from `graph_capture()`.
2. **Local patches,** kept as a separate, reviewed diff: finding 4 (slot bound), and finding 2 (idle backoff), only if
   the idle power measurement shows a cost.
3. **Build:** build the `.so` at image build (finding 5).
4. **Environment:**
   - `B12X_ROCE_HCA=rocep1s0f1`
   - `B12X_ROCE_GID_INDEX=3`
   - `B12X_ROCE_SPIN_LIMIT` ≈ 60–120 s worth of polls
   - `B12X_DISABLE_CUTLASS_RUNTIME_PATCHES=1`
   - `GLM_ROCE_MAX_SIZE=1MiB`
   - `GLM_ROCE_GATHER_MAX_SIZE=4MiB`
   - a launcher switch (`ROCE_DCP`), default off until measured.
5. **Validation before serving:**
   - a 2-rank value-checked collective test on one DCP pair with the model stopped;
   - count100 byte parity against NCCL;
   - forced-K cycle timing (the expected gain is −4.5 to −5.5 ms/cycle);
   - a 2-hour mixed-traffic soak;
   - a cold boot with empty JIT caches;
   - node power at idle, with the runtime enabled and disabled.
