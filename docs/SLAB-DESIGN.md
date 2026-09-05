# Fixed-size slab store for the direct NVMe tier (design for review)

Status: proposal, 2026-09-04 23:00. Replaces the per-block-file layer of the
direct tier (`docs/NVME-DESIGN.md` §7-8, `overlay/vllm/v1/kv_offload/tiering/multinode.py`:
`FSOffloadingManager`, `BounceController`, `WorkerFsHandler`) with a
bounded, preallocated slab per KV-cache group per rank and slot reuse.
Goal: the on-disk cache never exceeds a configured size, no external
janitor, no per-key files, no directory discovery.

## Context the reviewer needs

- Cluster: 4 DGX Spark nodes, TP4 + DCP4 (decode context parallel) +
  DFlash speculative decoding. The target model's MLA KV is token-sharded:
  the KV manager's block is 256 global tokens, and on every rank that block
  is one physical 64-slot block holding the 64 positions the rank owns
  (3.94 MB per rank: 78 layers x 64 x (656 + 132) B). The DFlash drafter's
  sliding-window group is replicated (block = 64 tokens, ~393 KB per rank:
  6 layers x 64 x 2 heads x 128 x 2 x 2 B; window 2048 = 32 blocks). So a
  256-token stretch of context costs 1 target block + 4 drafter blocks
  (~5.5 MB per rank).
- The connector (`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`)
  derives an offload key per group from the request's block hashes
  (every 4th hash for the target group, every hash for the drafter) and
  calls the manager: `lookup(key) -> True/False/None(defer)`,
  `prepare_store(keys) -> PrepareStoreOutput(keys_to_store, store_spec, evicted_keys)`,
  `prepare_load(keys) -> LoadStoreSpec`, `complete_store/complete_load(keys)`,
  `on_new_request/on_request_finished/on_schedule_end/has_pending_work/reset_cache`.
  Keys are group-major and parallel to the GPU block ids the connector
  puts in `GPULoadStoreSpec(block_ids, group_sizes, block_indices)`. A job
  is complete when all `num_workers` report it. The connector counts a hit
  only if every block of the prefix (and the drafter's last 33 blocks
  before the boundary) answers True, and re-queries on None.
- The trailing block of every request is never offloaded (eagle rule).
- Workers execute transfer jobs by `(src medium, dst medium)`; the direct
  tier's `BounceController` moves blocks GPU <-> bounce (48 pinned slots,
  fork's `SingleDirectionOffloadingHandler` copy kernels, driven from the
  main thread in `pump()` each engine step) <-> disk (thread pools).
- All ranks execute the same job with the same keys and block ids; each
  rank's bytes are its own shard. Block hashes are stable across restarts
  only with `PYTHONHASHSEED` pinned (done).
- Validated today with per-key files: reload of an evicted 100k prefix
  3.05 s, after an engine restart 4.61 s, vs 335 s cold prefill; 8,502
  files / 7.7 GB per rank after 400k tokens.

## Proposed design

### On-disk layout (per rank, per group)

`<root>/<model>_<digest>_r<rank>/g<group>.slab`: a file of `N_g` fixed
slots. Slot = 4 KB header + payload padded to 4 KB. Header: magic (8 B),
format version (4 B), group index (4 B), payload length (4 B), write
sequence number (8 B, monotonically increasing per scheduler lifetime,
restart continues from the max seen), block hash (32 B), CRC32 of the
header (4 B). Payload = the rank's canonical tensor pages for the block
(same bytes as today's file). Slot sizes: target 4 KB + 3.94 MB -> 3.95 MB;
drafter 4 KB + 393 KB -> 397 KB. Files are created sparse and grow to their
full size as slots are first used; the cap is the slot count.

Sizing from one knob `disk_bytes_per_rank` (default 150e9): with
`R = 4` drafter blocks per target block, `N_target = disk_bytes / (S_target + R * S_drafter)`,
`N_drafter = R * N_target`. 150 GB -> ~27k target slots ~ 6.9M tokens of
distinct context per cluster. If a group's block bytes differ across
ranks (they do not here) the slab size differs per rank but slot numbers
still match.

### Scheduler-side index (`SlabOffloadingManager`)

- Per group: `index: OrderedDict[key -> slot]` (LRU order), `free: deque[slot]`,
  `seq` counter. Plus `inflight_store: dict[key -> slot]`, `inflight_load: set[key]`.
- `lookup(key)`: `True` if in `index` (and touch = move_to_end), `None` if
  in `inflight_store`, else `False`. No filesystem access on the lookup path.
- `prepare_store(keys)`: for each key not in `index`/`inflight_store`, take a
  free slot; if none, pop the LRU entry whose key is not in `inflight_load`
  and reuse its slot (record it in `evicted_keys`). Reserve the slot in
  `inflight_store`. Return `SlabLoadStoreSpec(keys, slots, seqs)`.
- `complete_store(keys, success)`: on success move the key into `index`
  (most recent); on failure return the slot to `free` (the slot's header
  on disk may be partial; the sequence number rule below makes it unusable).
- `prepare_load(keys)`: keys must be in `index`; return
  `SlabLoadStoreSpec(keys, slots, seqs)`; add to `inflight_load`;
  `complete_load` removes them.
- Eviction never touches disk: reuse overwrites the slot on the next store.
- `reset_cache`: clear index/free, truncate slabs on next use? (proposal:
  rebuild `free` = all slots, leave data; sequence numbers make stale
  headers irrelevant).
- Persistence: none needed beyond the slabs themselves (see restart).

### Worker side

`BounceController` keeps its GPU<->bounce path; the disk layer becomes:
- store: write header + payload with `pwritev` at `slot * slot_size`; one
  `fdatasync` per chunk (up to 48 blocks), then report the chunk done.
- load: `preadv` header + payload; verify magic, group, hash == key,
  seq == expected seq from the spec, length; a mismatch fails the block
  (job reports `success=False`, GPU block ids reported invalid, recomputed).
- Per-group slab files opened once at startup; `posix_fadvise(DONTNEED)`
  after each chunk as today.

### Restart

The scheduler rebuilds its index by reading every slot header of its own
node's slabs (rank 0): 4 KB per slot, ~50k+110k slots at 150 GB, a few
seconds. Valid headers (magic, CRC, group, length) populate `index` ordered
by sequence number (oldest first = LRU order), `seq` resumes from the max.
Other ranks hold the same slot assignments because the scheduler assigned
them and every rank executed the same jobs; a rank that missed a write
(crash mid-job) has a header whose hash/seq does not match the spec at
load time and fails that chunk only.

### Failure and concurrency rules

1. A slot is in exactly one of: `free`, `inflight_store`, `index`.
2. Eviction skips keys in `inflight_load`; if every candidate is in
   flight, `prepare_store` returns the keys it could place and leaves the
   rest (the connector treats un-stored keys as not offloaded).
3. Loads check the header hash and seq, so a slot reused between
   `prepare_load` and the worker's read (impossible by rule 2, but cheap to
   verify) is detected rather than served.
4. A store whose job fails on any rank returns the slot to `free`; the
   scheduler's index never contains a key that not all ranks reported.
5. Chunk-level `fdatasync` bounds the loss on a power cut to the last
   chunk; the header seq/CRC makes a torn slot detectable.

### Not in scope

Cross-node replication, compression, per-request pinning, sharing between
model configurations (a different digest = a different slab directory).

## Lean revision (what is being built, 2026-09-04 23:30)

After two reviews (an independent Claude agent reading the connector code,
and GLM-5.3; findings in `results/glm-review-slab-design.md` and the
session log), the user chose the slab store but asked for the simplest
version that keeps one invariant: **a hit never serves wrong bytes on any
rank; every doubt is a miss.** Changes from the proposal above:

Kept, each a few lines:
1. Failure channel: a store that fails on any rank never enters the index;
   a load that fails on any rank drops the key (workers report
   `failed_jobs`; a scheduler subclass marks the manager before the base
   completion path runs).
2. All-or-nothing placement per `prepare_store` call: if any key cannot get a
   slot, return `None` (the connector retries next step, without advancing
   its stored index).
3. Key in every slot header, verified on load (also group and length). Any
   cross-rank divergence fails that chunk, the key is dropped, the next
   request re-stores it. No superblock, no cross-rank handshake.
4. Slot write order: blank header, payload, header. A process crash leaves a
   blank or a complete slot. A node reboot (boot_id in `slab-meta.json`
   differs) wipes that rank's slabs at startup.
5. Drafter siblings: the drafter keys stored in the same call as a target key
   are remembered and touched with it on a hit.

Dropped, and why it is safe for a cache: payload CRC (NVMe/RAM are ECC; the
power-loss case is covered by rule 4's reboot wipe); sequence-number
validity (key + slot is the identity; a write counter remains only to order
the LRU after a restart); cross-rank validation at startup (rule 3 heals
lazily at one wasted read per divergent block); in-flight-load deadlines
(loads take seconds; an all-in-flight LRU tail returns `None` = retry);
`fdatasync` on the main thread (stores are one pool task per chunk that
writes its slots sequentially and syncs once, off the engine path).

Geometry: two slab files per rank (`g0.slab` target rows of 4 KB header +
3.94 MB, `g1.slab` drafter rows of 4 KB + 393 KB, both 4 KB aligned), slot
counts from `disk_bytes_per_rank` with 4 drafter rows per target row
(150 GB -> ~27k target rows -> ~6.9M tokens). Files grow on write (sparse);
the cap is the slot count. Workers own the byte math and write
`slab-meta.json`; the scheduler attaches lazily to rank 0's directory,
reads the counts from that file and rebuilds its index from the slot
headers (~135k x 64 B reads). `reset_cache` bumps an epoch stored beside the
slabs; headers with an older epoch are ignored at rebuild.

Validation plan: unit tests on the fork's real connector pieces (slot I/O
write order and key check, LRU/all-or-nothing/siblings, failure marks,
rebuild, epoch); then one cluster sequence: boot with a deliberately small
cap (3 GB per rank ~ 138k tokens), store a 100k prefix, push two more 100k
prompts through, confirm the oldest misses and the newest hits and the
slab files never exceed the cap; restart into the 150 GB cap and confirm
the durable reload; leave it serving.
