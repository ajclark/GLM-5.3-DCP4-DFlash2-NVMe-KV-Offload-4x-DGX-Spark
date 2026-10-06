# NVMe KV tier optimization — results (2026-09-28/29)

Follow-up to the durable KV tier described in [NVME-DESIGN.md](NVME-DESIGN.md).
Deployed on the live 4× DGX Spark GLM-5.3 endpoint on 2026-09-29 00:12 (production
restart without dev endpoints at 00:58). Evidence: `results/nvme-kvtier-20260928/`.

Measured on an abliterated graft of GLM-5.3 with the same quantization envelope as GLM-5.3
Int4-Int8Mix (Int4 group-128 routed experts, Int8 group-128 elsewhere, identical shapes).

## What was wrong

1. **The slab held ~222k tokens, not ~1M.** `DRAFTER_PER_TARGET = 4` was a DCP=4
   assumption (256-token target block / 64-token drafter block). At DCP=2 the
   target block spans 128 tokens, so the drafter group needs 2 slots per target
   row. The 30 GB budget gave 1,737 target slots (6 GB) and 6,948 drafter slots
   (24 GB). A hit needs both groups, so the tier held about as many tokens as
   GPU KV (208k). The sibling pairing that refreshes drafter blocks on a target
   hit used the same wrong ratio.
2. **Reloads were clocked by engine steps, not by the disk.** The bounce-buffer
   state machine advanced only in `pump()`, once per engine step. A chunk of up
   to 48 blocks had to finish reading, wait a step for the host→GPU copy, and
   wait another step for its slots. Stores shared the same 48 slots through
   `fdatasync`. Result: 0.4 GB/s under load and ≤1.8 GB/s idle, on a drive that
   reads 12–14 GB/s with O_DIRECT.
3. Any resize, and any change to `multinode.py` (which changes the manifest
   hash, i.e. `slab_salt`), starts a new slab. The old one stays on disk
   untouched, which keeps rollback warm.

## What changed

`runtime/vllm029/overlay/vllm/v1/kv_offload/tiering/multinode.py`:

- **`ThreadedSlabController`** (new default, `io_engine: "threaded"`):
  - Each I/O thread owns 4 KB-aligned pinned rows and its own CUDA stream.
  - A load is one O_DIRECT `pread` of the whole slot (header + payload + pad),
    verified by key, length, epoch and payload CRC. The payload segments then
    go straight to the GPU block with `swap_blocks_batch` on the thread's
    stream, ordered after an event recorded on the compute stream at submit.
  - Rows are double-buffered, so disk reads overlap host→GPU copies.
  - A store is GPU→row, CRC, header, then one O_DIRECT `pwrite` (a torn slot
    fails its CRC, as before).
  - Stores yield up to 50 ms to queued loads.
  - Nothing waits for an engine step; the main thread only drains finished
    jobs.
  - Pinned memory is (16+8)×2×3.45 MB ≈ 166 MB, the same as the old
    48-slot bounce buffer.
- **Geometry:** `group_row_ratios()` derives slots per target row from the
  groups' block token spans ([1, 2] at DCP=2). The sibling pairing uses the
  same ratio.
- **Slab files are preallocated** with `fallocate`, so aligned direct writes
  never extend the file and run in parallel on ext4.
- **Scheduler:** the header index rebuild reads in parallel (16 threads), and
  the eviction preflight stops counting once it has enough slots (it was
  O(slots) per store).
- The old engine remains selectable with `io_engine: "bounce"`.

Launcher (`runtime/vllm029/launch.sh`, `start-glm53.sh`)
new defaults:

| Setting | Value |
| --- | --- |
| image | `spark-vllm:0.29.0-nvme4-stridefix-kvtier-20260928` (thin: production image + overlay; `runtime/vllm029/Dockerfile.kvtier`) |
| `KVTIER_DISK_BYTES` | 400e9 per rank (1 rank/node): 38,614 target rows ≈ **4.94M tokens** |
| `KVTIER_IO` | `threaded` |
| `KVTIER_READ_THREADS` / `KVTIER_WRITE_THREADS` | 16 / 8 |
| `VLLM_DEV_MODE` | 0; set to 1 only for benchmarks that need GPU-only `/reset_prefix_cache` |

The slab lives at `<KVTIER_DIR>/_models_glm-5.3_<digest>_r<rank>/` and is
preallocated, so it takes 400 GB on each node immediately.

Client timeouts: a queued request waits silently until its first token, so a client's idle timeout must exceed the
worst queue. Our agent clients use a 15-minute idle timeout (a 5-minute one killed queued requests) and 6 retries
with a 5 s base delay. Streamed tokens reset the timeout, and 15 min covers the worst queue measured: 613–627 s
TTFT for five concurrent cold ~80k prefills. A timed-out request loses little, because its already-prefilled blocks
are on NVMe and a retry reloads them.

## Results

All numbers come from `runtime/vllm029/kvtier_bench.py` with the same seed before and after,
so the prompts are identical.

### Engine microbenchmark, GPU, vLLM stopped (`results/nvme-kvtier-20260928/engine-bench/*.log`)

| per node | 06c4 | 365c | ddbf | a218 |
| --- | --- | --- | --- | --- |
| raw O_DIRECT slot reads, 16 thr | 13.9 GB/s | 14.1 | 14.2 | 14.2 |
| engine load NVMe→GPU, 16 thr × 2 rows (CRC + content verified) | **11.1 GB/s** | 11.2 | 11.3 | 11.3 |
| engine store GPU→NVMe, 8 thr | 10.6 GB/s | 10.9 | 10.5 | 10.2 |

### Isolated restore: GPU prefix cache cleared, KV restored from NVMe (`iso`)

The request floor is ~0.7–0.96 s: that is the TTFT of the same request on a
GPU cache hit.

| context | before TTFT | after TTFT | disk read during restore, after (peak 0.5 s, per node) |
| --- | --- | --- | --- |
| 20k (0.65 GB/rank) | 0.76 s | 0.77 s | at the floor either way |
| 80k (2.26 GB/rank) | 1.57 s | **0.95 s** | ~0.23 s above the floor ≈ 10 GB/s |
| 160k (4.42 GB/rank) | 2.5–4.5 s | **1.21–1.24 s** | **9.5–11.0 GB/s** |
| 80k while another stream decodes | 9.14 s | **2.81 s** | |

The NVMe is saturated during restores: 9.5–11 GB/s against a 12–14 GB/s raw
ceiling. What is left of the TTFT is request overhead, not I/O.

### Five concurrent agents (`agents`)

Each agent has a 12k shared prefix plus a 60–80k private history, 4 rounds,
+1–3k tokens per turn and 200 generated tokens.

| | before | after |
| --- | --- | --- |
| wall time | 2,425 s | **782 s** (3.1×) |
| warm-turn TTFT p50 / mean | 481 s / 477 s | **30 s / 76 s** |
| warm-turn TTFT p95 | 565 s | 294 s (round-1 turns queued behind the other agents' cold prefills) |
| rounds 2–3 TTFT | 370–590 s | 20–33 s |
| external (NVMe) hit rate | 12% | **76%** |
| KV loaded from NVMe (all ranks) | 25 GB | 141 GB |
| client errors | 0 | 0 |

Before, the five contexts (~400k tokens) did not fit in the 222k-token slab,
so every warm turn was a full re-prefill. After, they stay resident. The
remaining ~30 s per warm turn is queueing for GPU KV (only two ~90k contexts
fit in 208k tokens) plus prefill of the new tokens.

### Correctness (`after-needles`, `after-verify`)

- Ten codes were spread across 60k- and 150k-token contexts. Recall was 10/10
  from computed KV, 10/10 from a GPU cache hit, and 10/10 from NVMe-restored KV
  (59,648 / 149,632 tokens restored).
- Largest change in logprob over the first 16 generated tokens, compared with
  the cold run: GPU hit 0.038 / 0.0003, NVMe restore 0.017 / 0.0004.
- Greedy text is not bit-identical across cache paths on this stack (a GPU
  cache hit also differs from the cold run), so exact-match is not the test.
- No slab load/store failures in any node's log.

### Decode (no regression beyond run-to-run variance)

| tok/s | before | after | after re-run (5 reps) |
| --- | --- | --- | --- |
| C1 count | 58.2 | 58.0–58.2 | 58.0–58.3 |
| C1 prose | 24.4–25.1 | 22.7–24.8 | 23.1–25.1 |
| C1 code | 43.1–44.3 | 41.3–41.7 | 41.0–42.7 (historical baseline 41.5–42.7) |
| C4 count / prose / code (aggregate) | 150 / 48.5 / 96 | 155 / 48.8 / 92 | 141 / 51.0 / 91 |

### Restart with a populated slab

The production restart (dev mode off) took 165 s. The scheduler re-indexed
25,189 stored blocks from slot headers at attach, so the tier survives
restarts warm.

## Operations

- **Restart:** `bash start-glm53.sh`. The defaults are
  the production profile.
- **Benchmark:**
  1. Restart with `VLLM_DEV_MODE=1`.
  2. Run `python3 runtime/vllm029/kvtier_bench.py <label> --phases needles,iso,agents,decode`.
  3. Restart with the defaults.
  - Never call `/reset_prefix_cache?reset_connector=true`: it bumps the slab
    epoch, which logically wipes the whole tier.
- **Engine test/benchmark on a node** (vLLM stopped): see the headers of
  `runtime/vllm029/test_slab_threaded.py` and `runtime/vllm029/kvtier_engine_bench.py`.
- **Rollback to the pre-change tier.** Old slabs were deleted on 2026-09-29 (only the live
  400 GB slab remains per node; ~118 GB freed per node), so a rollback starts with an empty
  30 GB slab:
  1. On every node: `cp ~/glm-vllm029-build/manifest.json.pre-kvtier-20260928 ~/glm-vllm029-build/manifest.json`
     (this restores the old salt).
  2. `DCP_IMAGE=spark-vllm:0.29.0-nvme4-stridefix-20260920 KVTIER_DISK_BYTES=30000000000 KVTIER_IO= KVTIER_READ_THREADS=8 bash start-glm53.sh`
  - A same-slab A/B of just the engine: `KVTIER_IO=bounce`, without changing
    the manifest.
- **Growing the budget** (e.g. to 512 GB) changes the slot counts, so the slab
  starts empty. Preallocation means the full budget is reserved on disk
  immediately.
