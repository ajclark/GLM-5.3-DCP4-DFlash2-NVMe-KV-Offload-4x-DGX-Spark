# NVMe-durable KV cache for GLM-5.3 on the Sparks (TP4 + DCP4 + DFlash)

Goal: a prefix that was computed once should not be prefilled again, not after
a harness compaction, not after an engine restart. Prefill runs at ~250-300
tok/s here (15 min at 250k, 32 min at 500k), so this is the dominant cost of
long sessions now that DCP holds them. Everything below is against the
fork's vLLM (`~/lmcache-mg/spark-src/vllm`, the image's copy) as of
2026-09-04, read in full; nothing on the cluster is touched by this document.

## 1. What the fork ships, and why it does not work here as-is

`OffloadingConnector` (`kv_connector/v1/offloading_connector.py`,
`offloading/{scheduler,worker,common}.py`) with:

- **CPU tier** (`v1/kv_offload/cpu/`): completed GPU blocks are copied,
  whole physical block at a time, into pinned host memory owned by each
  worker (`CpuGpuOffloadingHandlers`, `swap_blocks_batch`), keyed by
  `(block hash, KV-group index)`. The scheduler side keeps the metadata
  (LRU/ARC), the worker side moves bytes. Multi-node is fine: each worker
  keeps its own shard in its own pinned buffer, block ids are the same on
  every rank, and a job is only complete when all `num_workers` report
  (`OffloadingWorkerMetadata.completed_jobs`, `pending_count`).
- **Hybrid-model support** (`SchedulerOffloadConfig.from_spec`): per-group
  block sizes, sliding-window groups (only the last window's blocks matter,
  and hits are aligned to the full-attention group's block), MLA + SWA
  explicitly (DeepSeek V4), draft groups. This covers our layout: target
  group at 256 tokens per block, DFlash drafter group (window 2048) at 64.
- **DCP is transparent** to it by construction: under DCP the manager's block
  is `kernel_block x dcp` = 256 global tokens, and on every rank that block
  is one physical 64-slot block holding the 64 positions the rank owns. The
  scheduler hashes 256-token chunks; each rank offloads its own physical
  block under that hash. No slot mapping, no -1 padding, nothing to patch.
  (`FileMapper` even records `dcp_size` and the global rank.)
- **TieringOffloadingSpec** (`v1/kv_offload/tiering/`): CPU primary tier plus
  secondary tiers; the filesystem tier (`tiering/fs/`) writes one file per
  block with a thread pool, `O_DIRECT`, temp-file-then-rename, and looks
  blocks up by file existence. This is the "native NVMe" feature.

The blocker: tiering is **single-host by construction**.

- `SharedOffloadRegion` (`cpu/shared_offload_region.py`) is one
  `/dev/shm/vllm_offload_<instance>.mmap` file laid out `[block][worker]`,
  "shared across all workers for a vLLM instance". The scheduler process
  mmaps and `MADV_POPULATE_WRITE`s the *entire* region (all workers' pages);
  each worker mmaps its slice, with `rank = torch.accelerator.current_device_index()`,
  which is 0 on every Spark.
- Secondary tiers run **inside the scheduler process** over a memoryview of
  that region (`TieringOffloadingManager`, `FileSystemTierManager.submit_*`).

On four nodes this means: the scheduler's region on node 0 holds rank 0's
pages only; ranks 1-3 write into private files on their own nodes that no
tier ever reads; the fs tier persists rank 0's shard and, on a hit, ranks
1-3 load whatever stale bytes sit in their local pages. Silent corruption,
plus the scheduler would populate the whole `cpu_bytes_to_use` on node 0.
The single-tier `CPUOffloadingSpec` avoids all of this (no shm, per-worker
pinned buffers) but has no durability.

Alternatives considered: the fork's `example_connector.py` (per-request
safetensors on shared storage, reference quality, whole-prompt granularity),
`hf3fs` (needs a 3FS deployment), and LMCache, where the other session has
already built multi-group engine-driven transfer to a remote host
(`~/lmcache-mg/docs/DESIGN.md`, results 2026-09-03: warm TTFT +0.8-0.9 s
overhead, reload ~130 MB/s client-bound). LMCache's adapter works on token
chunks through slot mappings, so DCP's per-rank position ownership would be
new work there too, and the user asked for the native feature. The native
connector's block-id design is the one that is DCP-transparent for free.

## 2. Design: worker-executed filesystem tier

Keep the scheduler-side tiering *bookkeeping* (which keys live in which
tier, promotions, ref counts) and move the *bytes* to where they already
are: each worker's pinned CPU tier and each node's own NVMe.

### 2.1 Worker side (new, ~250 lines)

- `FSLoadStoreSpec(LoadStoreSpec)`: `keys: list[OffloadKey]` (medium "FS").
- Two `OffloadingHandler`s registered by the spec's `get_handlers` next to
  the existing GPU<->CPU pair: `CPULoadStoreSpec -> FSLoadStoreSpec`
  (store) and `FSLoadStoreSpec -> CPULoadStoreSpec` (load). Each runs the
  fork's `store_block` / `load_block` (`tiering/fs/io.py`, `O_DIRECT`,
  atomic rename) on a `DualQueueThreadPool` over the worker's own pinned
  CPU-tier tensors, one file per `(hash, group)` under
  `<root>/<model>_<digest>_r<global rank>/`, exactly the documented layout,
  with `FileMapper(parallel_agnostic=False, rank=parallel_config.rank)`.
- The per-block file holds this worker's canonical tensors for the block
  concatenated, same as one worker cell of the shm layout; page-aligned
  because the pinned buffers are allocated per canonical tensor with
  `BLOCK_SIZE_ALIGNMENT = PAGESIZE` (needed by `O_DIRECT`).
- `get_finished()` reports `TransferResult(job_id, success)`; a load that
  finds a missing or short file reports `success=False` instead of the
  current `assert transfer_result.success`, and the worker adds the
  affected GPU block ids to `KVConnectorOutput.invalid_block_ids` so the
  scheduler recomputes them (the protocol vLLM already has for failed
  external loads).

### 2.2 Scheduler side (new tier class + two small hooks, ~250 lines)

- `WorkerFileSystemTier(SecondaryTierManager)`: `submit_store/submit_load`
  do no I/O; they append `(job_id, TransferSpec(CPU block ids <-> FS keys))`
  to a pending list. `lookup(key)` answers from node 0's own files (same
  path scheme, rank 0), which is correct because a store job is only
  complete when all workers have written their file, and because
  `lookup` is already async/batched in the fork (`FsAsyncLookupManager`).
- `OffloadingConnectorScheduler.build_connector_meta` drains the tier's
  pending transfers into `store_jobs` / `load_jobs` with `TransferJobStatus`
  entries flagged `is_tier=True`; `update_connector_output` routes their
  completion (after `num_workers` reports) to the tier's finished queue,
  which `TieringOffloadingManager._process_finished_jobs` already consumes.
  Job ids come from the connector's counter so the worker sees one id space.
- `MultiNodeTieringOffloadingSpec(TieringOffloadingSpec)`: builds the primary
  tier without a `SharedOffloadRegion` (per-worker pinned tensors, as the
  single-tier spec does) and instantiates `WorkerFileSystemTier` for
  `type: "fs"`. Selected with `spec_name`. Nothing else in the connector
  changes; the single-host tiers keep working where they work.

### 2.3 What is deliberately not changed

- Hashing, prefix-cache composition (256/64 block hashes) and lookups are
  the connector's own; `PYTHONHASHSEED` must be fixed on all nodes so
  filenames match across restarts (documented requirement of the fs tier).
- `use_eagle()` is true for DFlash and no group is flagged `is_eagle_group`
  (that flag is only set for DeepSeek V4), so the connector treats every
  group's trailing block as volatile and never offloads it. Cost: the last
  256 tokens of each request are recomputed on a hit. Acceptable; a later
  patch can flag only the drafter group.

## 3. Memory budget (the constraint that sizes everything)

On the Sparks "pinned host memory" is the same physical pool as the GPU, so
the CPU tier comes out of context. `cpu_bytes_to_use` is a cluster total;
each worker pins `cpu_bytes_to_use / 4`. Per rank a 256-token block is
64 slots x 78 layers x (656 + 132) B = 3.94 MB for the target group plus
the drafter's 64-token blocks (2 heads x 128 x 2 x 2 B x 6 layers = 6 KB
each). So 1 GB per rank of tier holds ~250 blocks = 64k tokens of global
context in flight to disk, which is plenty for streaming: NVMe writes at
GB/s absorb a 300k prefill (4.7 GB per rank) in seconds, and the tier is
also a small extra LRU prefix cache. File I/O buffers are the tier pages
themselves (`O_DIRECT`), so no page-cache growth from the writes.

Serving configuration for this phase, chosen 2026-09-04 with the user:
max-model-len 307,200, KV pool 6 GB per rank (~396k tokens, 1.29x),
boot-4 overlay (local workspace 405 MB at 300k, global logits budget),
leaving about 2 GB per rank over the 512k configuration for a 1 GB tier
plus margin on rank 0. To be confirmed by the soak before the tier goes on.

## 4. Behaviour to validate on the cluster

1. Store path: after a 200k cold prefill, files appear under each node's
   `_r<rank>` directory, count and size match `blocks x 3.94 MB`, memory
   flat (guard), decode unaffected.
2. Warm path in-process: same prefix again is a GPU prefix hit (control).
3. Evicted path: fill the pool with other contexts until the first prefix
   is evicted from GPU (and from the CPU tier), re-send it: TTFT must be
   NVMe reload time, not prefill. Target: well under a minute for 200k
   (4.7 GB per rank at 1-2 GB/s), against 12 minutes of prefill.
4. Durable path: restart the engine (rollout), re-send: same as 3.
5. Failure path: delete one rank's file for a block, re-send: the block is
   recomputed (invalid_block_ids), the answer is still correct.
6. Concurrency and the drafter: mixed batches during stores/loads; DFlash
   acceptance unchanged on reloaded contexts.

## 5. Order of work

1. Serve the 300k configuration; run the check sequence and the soak
   (`dcp_soak.py`): cold TTFT, warm TTFT, concurrency, decode, memory per
   iteration. That is the baseline the tier must beat and the headroom it
   may use.
2. Implement 2.1-2.2 as overlay files (new modules plus the two hooks),
   with CPU tests: a fake worker pool exercising store/load/lookup through
   real `FileMapper` paths on a temp dir, the connector hooks with a fake
   scheduler output, and failure injection.
3. Stage, deploy with `rollout_dcp.sh` (new `--kv-transfer-config` lane in
   the launcher, `KVTIER=1`), run 4.1-4.6 with the memory guard.

## 6. Implementation notes (2026-09-04, `overlay/vllm/v1/kv_offload/tiering/multinode.py`)

Everything lives in one new module selected through the connector and spec
registries' out-of-tree hooks (`kv_connector_module_path`, `spec_module_path`);
no existing connector file is patched. What differed from section 2 once the
code was written and tested (`tests/test_nvme_multinode.py`, 12 tests running
the fork's real tiering manager, CPU manager, thread pool, file mapper and
worker dispatch under `tests/nvme_harness.py`; `tests/test_nvme_invariants.py`
pins the fork surfaces relied on):

- **Per-group block size.** The base `OffloadingSpec` scales every group's
  block size by the DCP factor. The drafter's sliding-window group is
  replicated (manager block 64), so the spec recomputes `gpu_block_size`
  with the overlay's `cp_world_size_for_kv_cache_spec`: (256, 64). Without
  this the drafter's keys would be taken from every 4th request hash while
  its block ids are 64-token blocks.
- **One thread pool per direction.** A pool shared by the store and load
  handlers hands one handler the other's completions (the test caught it).
- **Buffered I/O, `fdatasync`, `FADV_DONTNEED`.** The indexer page (8448 B)
  is not 512-byte aligned, so `O_DIRECT` is out; `BLOCK_SIZE_ALIGNMENT = 1`.
- **Failures.** A worker reports a failed tier job in `failed_jobs`
  (metadata subclass, aggregated as a union); the scheduler completes the
  tier job with `success=False` once all workers reported; a failed
  promotion leaves the block out of the CPU tier (a miss, recomputed).
  Transfer stats count successful bytes only.
- **Rank-independent run digest.** The first deployment (19:44) showed the
  scheduler looking up under `_1c3a6cf092fe` while the workers wrote under
  `_b58237a2c54f`: `FileMapper.from_offloading_spec` hashes each group's
  layer-name list, and the per-worker KV-cache config lists layers
  differently from the scheduler's. `make_file_mapper` builds the mapper
  from fields that are identical everywhere (model, hash block size, tp/pp/
  pcp/dcp, dtype, per-group block sizes); the rank stays outside the digest.
  Even then the digests differed (take 2): the workers' `cache_dtype` reads
  `fp8_ds_mla` (resolved for the MLA layers) while the scheduler's reads the
  CLI value `fp8`. So the scheduler no longer trusts its own digest for
  lookups: `discover_rank0_base_path` finds the `<model>_<digest>_r0`
  directory rank 0's worker created on this node (newest first) and the
  lookup manager switches to it, logging the difference once; workers write
  `.run_config.json` beside their files so the difference stays visible.
  Files are written by the container as root; clean-up needs `sudo`.
- **Trailing block.** `use_eagle()` is true for DFlash; no group is flagged,
  so every group's last block is never offloaded (256 tokens recomputed on
  a hit). Left as is.

Wiring: `KVTIER=1` in `launch-glm53big-dcp.sh` (mounts `multinode.py`, binds
`/var/tmp/kvcache` at `/kvcache`, adds `--kv-transfer-config`; `KVTIER_CPU_BYTES`
default 4e9 = 1 GB pinned per rank, `KVTIER_THREADS` 8); `rollout_dcp.sh`'s 5th
argument; `offload_checks.sh <label>` runs `dcp_probe.py --phase offload`
(cold / warm / evict x3 / reload) under the memory guard with per-node file
counts before and after. `prefix_caching_hash_algo` defaults to sha256, whose
seed is fixed, so filenames are stable across restarts without
`PYTHONHASHSEED`.

## 7. Why the tiered design cannot serve long reloads here, and the direct tier

Take 3 (20:06-20:52) had everything working mechanically: stores on all four
ranks (8,455 files, 7.6 GB per rank after 400k tokens), lookups resolving
rank 0's directory, no I/O errors. The reload was still a full prefill (335 s,
zero external hits). The cause is in the connector's lookup contract, not
in the plumbing: `_maximal_prefix_lookup` walks every block of the prefix
and a block only counts as hit when the tiering manager has it *in the CPU
tier*; a block on disk is promoted by allocating a CPU-tier block, and
promotions fail once the tier is full, at which point the walk stops. The
drafter's sliding-window lookup then needs 33 more CPU blocks for its
trailing window, gets none, and the whole lookup returns 0. With a 1 GB
tier (260 blocks) the theoretical maximum is ~58k tokens, and a 300k
context would need ~4.7 GB of pinned memory per rank, which is the memory
the tier was supposed to save. `results/dcp4-dflash-300k-kvtier3/`.

**Direct tier (`MultiNodeDirectFsOffloadingSpec`, `MultiNodeDirectConnector`,
`KVTIER_MODE=direct`, the default).** The disk is the offload store:

- Scheduler: `FSOffloadingManager` answers `lookup` by file existence under
  rank 0's directory (async, batched, discovered as before), returns
  `FSLoadStoreSpec(keys)` from `prepare_load` and `prepare_store` directly
  (no CPU tier, no promotion, no capacity), keeps an LRU of known-on-disk
  keys and the in-flight sets. The base `OffloadingConnectorScheduler` is
  used unchanged: FS specs travel as ordinary load/store jobs.
- Worker: `BounceController` owns `bounce_blocks` (default 48, ~4 MB each
  here, ~190 MB pinned per rank) and drives two-hop transfers: store =
  GPU -> bounce (the fork's `SingleDirectionOffloadingHandler`, CUDA copy
  kernels) -> file (write threads); load = file (read threads) -> bounce ->
  GPU. `pump()` runs on the main thread from the handlers' `get_finished`,
  which the connector calls every engine step, so the CUDA copies keep the
  fork's stream semantics; jobs are chunked through the free slots, loads
  before stores. A missing file fails only its chunk: the job completes with
  `success=False`, the request is released, and the affected GPU block ids
  are reported through `get_block_ids_with_load_errors` so the scheduler
  recomputes them (`MultiNodeDirectConnectorWorker`).
- Memory: bounce slots only; the KV pool can stay at 6 GB or grow back.
- Tests: `tests/test_nvme_direct.py` (round trip through 3 bounce slots,
  chunking, re-store no-op, partial failure, worker release + invalid ids,
  fresh-scheduler lookup after a restart, connector wiring).

The tiered variant stays in the module (`KVTIER_MODE=tiered`) for a host
with enough memory for a real CPU cache tier.

## 8. Durability across restarts needs a pinned hash seed

The direct tier's first run (21:07-21:52) reloaded an evicted 100k prefix
from NVMe in 3.4 s, but after an engine restart the same prefix missed
entirely (335 s, zero external hits) with all 8,502 files per rank still on
disk. Cause: `init_none_hash` seeds the block-hash chain with
`os.urandom(32)` unless `PYTHONHASHSEED` is set (the fork only warns about
this for the CBOR hash functions), so every process computes different
block hashes and different file names. The tier lane now sets
`PYTHONHASHSEED=0` (`KVTIER_HASHSEED`) in the container; files written by
an unseeded process are dead and were removed. Anything that changes the
hash inputs (tokenizer, chat template, `--prefix-caching-hash-algo`, the
seed) invalidates the on-disk cache the same way; the run-config digest
does not cover those, so a cache directory should be wiped when they change.
No size or age management exists yet: `/var/tmp/kvcache` grows by ~7.7 GB
per rank per 400k tokens of distinct prefill (652 GB free on the NVMe at
deployment); add a janitor before it matters.

## 9. Slab store (fixed-size ring buffer), the shipped design

The per-block-file tier grows without bound; the user wanted a hard cap
enforced by the store itself. `docs/SLAB-DESIGN.md` has the proposal, two
prior reviews' findings, the lean revision, and the Codex review that led
to the final fixes. What ships (`KVTIER_MODE=slab`, the default):

- Two slab files per rank (`g0.slab` target rows, `g1.slab` drafter rows),
  fixed 4 KB-aligned slots, counts from `disk_bytes_per_rank` (default
  150e9: ~27k target rows, ~6.9M tokens) with 4 drafter rows per target row;
  128-byte slot header (magic, version, group, epoch, length, write
  sequence, key). Workers own the byte math and write `slab-meta.json`
  (geometry, boot id); the scheduler attaches lazily to rank 0's directory,
  reads the counts, rebuilds its LRU index from the slot headers in write
  order, and keeps the epoch beside the slabs.
- Stores: LRU slot reuse, transactional per call (preflight capacity,
  never evict a key of the same call, dedupe; `None` = retry next step),
  epoch and sequence carried in the job spec so every rank writes the same
  header; write order blank header, payload, header, all writes complete.
- Loads: the header's key and length are verified on every rank; a failure
  drops the key from the index at once but keeps the slot out of reuse
  until every in-flight reader of that key has completed (Codex finding 1),
  and reports the GPU blocks as invalid so the engine recomputes them. The
  engine scheduler's recovery assumed a single KV-cache group and would
  have crashed with our two (Codex finding 5); `overlay/vllm/v1/core/sched/scheduler.py`
  walks every group with its own tokens-per-block and truncates at the
  earliest invalid position (14th overlay file).
- Identity: the cache directory digest includes the draft model (Codex
  finding 4). Shrinking the budget truncates the slab; a torn meta file or
  a different boot id starts that rank empty; `wait()` never returns with
  work in flight.
- Drafter siblings: the four drafter blocks stored in the same call as a
  target block are touched with it on a hit (bounded, approximate).
- Not done, by choice: payload checksums, cross-rank validation at startup,
  persisted LRU order across restarts, sequence-based validity.

## 10. A store-progress bug in the fork's connector scheduler (found 2026-09-05)

The slab validation (`deploy_slab.sh dcp4-dflash-300k-slab2`, boots A/B/C)
passed its headline checks: eviction held the slabs at the 4.5 GB cap, the
index rebuilt from slot headers, and a prefix stored in boot B reloaded on
boot C in 5.39 s. But boot B showed something the direct-tier runs never
had: two prefixes that boot A had stored completely (its cold prefix and
its third eviction prompt) hit exactly 1,280 tokens, five 256-token blocks,
and were then prefilled from scratch. The other two eviction prompts,
which A's LRU had evicted, hit nothing, as expected. So the rows were
indexed, and the prefix lookup itself stopped at block five.

**Evidence.** A dump of every slot header on rank 0 (`results/dcp4-dflash-300k-slab2-b/rank0-headers.json`,
made with a read-only script while boot C served) settles what was on disk:

| what | rows | note |
|---|---|---|
| boot A's surviving target rows | 892 | store sequences 157-291, one 6-7 row batch per prefill step |
| target rows per 100k prompt as first stored | 339 of 390 | prompts are delimited by the 4-row final batch |
| boot B's stores while prefilling A's cold prefix | 50 single rows, one per step | none of these keys existed on disk before |
| same for A's third eviction prompt in B | 50 single rows | |
| boot C's stores while prefilling B's second eviction prompt | 50 single rows | the same shape one restart later |

Every first-time store of a prompt leaves about one block per prefill
step unstored, in both KV groups, and the scheduler's in-memory index
agrees with the disk (the singles are keys the connector never presented
to `prepare_store`). A later request for the same prompt whose prefix is a
GPU cache hit presents the whole prompt to the connector in one step and
fills exactly those holes. That is why every in-process reload in the
direct and slab runs looked complete: the probe's "warm" step (same
prefix, new question) came right after the cold prefill and refilled it.
A prompt that is seen once and then evicted, or seen once before a
restart, hits only up to its first hole.

**Root cause** (`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`,
`_build_store_jobs`). For an eagle group the code holds back the trailing
block, whose drafter KV is still volatile:

```
num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
if group_config.is_eagle_group:
    num_blocks = max(0, num_blocks - 1)
offload_keys = group_state.offload_keys[start_block_idx:num_blocks]
```

but the store progress index is then advanced past it, in the job-building
loop and in `advance_stored_idx`:

```
num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
...
group_state.next_stored_block_idx = num_blocks
```

so the held-back block is never revisited. With DFlash, `use_eagle()` is
true and no KV-cache group carries the `is_eagle_group` flag (the fork sets
it only for DeepSeek V4's MTP layer), so `SchedulerOffloadConfig.from_spec`
treats every group as eagle and both groups lose one block per step.
Upstream vLLM has since restructured this code (chunk-based progress,
eagle handling in `storable_chunks`); the fork's copy is the version above.

**Fix** (15th overlay file, `overlay/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`,
patch `patches/distributed_kv_transfer_kv_connector_v1_offloading_scheduler.patch`,
staged as `stage/glm-dcp/offloading_scheduler.py`, mounted by the tier lane
of the launcher next to `multinode.py`): one helper, `storable_blocks(group_config, num_tokens)`,
returns the eagle-adjusted count and is used for the key slice and for
every assignment of `next_stored_block_idx`. The held-back block is stored
on the next step, once it is no longer trailing; the request's final block
stays unstored, as designed (99,328 of 99,975 tokens hit, as before).
Non-eagle behaviour is byte-for-byte unchanged.

**Test** (`tests/test_offloading_store_progress.py`): loads the real
overlay and baseline modules under the test harness's stub package, builds
the connector's own config and request-state classes for our two-group
geometry (256/64-token offloaded blocks, factor 4), and drives
`_build_store_jobs` over a 99,975-token prefill in 1,700-token steps. The
overlay stores every block but the volatile tail with eagle groups, and
exactly what the baseline stores without them; the baseline witness test
pins the bug: `steps - 1` blocks missing per group.

**Validation on the cluster** (`deploy_slab_fix.sh dcp4-dflash-300k-slab3`):
boot D with the fix, then the offload probe with `--no-warm --seed-base 1000`
(fresh prefixes, no GPU-hit step): cold, three evictions, reload; boot E
(restart) and the reload of the once-stored cold prefix and of the third
eviction prompt, which was stored once and never reloaded. Results
(2026-09-05 02:28): once-stored reload 3.39 s in-process, 8.25 s after the
restart, the never-reloaded eviction prompt 3.85 s, all 99,328 tokens hit;
on disk 389 of 390 target rows per prompt instead of 339. Full table in
`docs/DESIGN.md` section 7.

**Rule for future probes:** a tier probe that puts a warm step between the
cold prefill and the reload does not test the first-time store. Use
`--no-warm` (and fresh seeds with `--seed-base`) whenever the store path
changes.
