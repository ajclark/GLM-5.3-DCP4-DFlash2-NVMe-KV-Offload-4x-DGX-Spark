# Review: Fixed-size slab store for the direct NVMe tier

## Blockers

**B1. Restart rebuild from rank 0's headers only is unsound (C).**
The design asserts "other ranks hold the same slot assignments because every rank executed the same jobs." But rule 4 says a store that fails on *any* rank returns the slot to free — meaning ranks can legitimately diverge (rank 2's write failed, rank 0's succeeded). Rank 0's header then describes a slot that rank 2 never wrote (or wrote a different key into later after slot reuse). Worse: after a failed store, the slot goes back to `free` and is reassigned to a *different* key; rank 0's stale header for the old key sits in the same slot. On restart, rank 0's header for that slot reflects whichever write happened last *on rank 0*, which may not match what the scheduler's rebuilt index thinks, and may not match other ranks at all. The seq-equality check at load time catches some of this (job fails, blocks recomputed) — but that silently converts your 3–5 s warm reload into a 335 s cold prefill with no warning, and the index will claim a hit for keys that are garbage on some rank. This violates the connector contract: `lookup() == True` must mean the data is on *every* rank.
**Fix:** rebuild the index as the *intersection* across ranks: each rank reads its own headers, scheduler intersects by (slot, hash, seq). If that's too slow, at minimum write a scheduler-side journal (append-only, fsync'd with the same chunk) of successful `complete_store`/evictions and rebuild from that, using rank headers only as a consistency check. Also: the "a few seconds" rebuild estimate is optimistic — 160k × 4 KB random reads on cold page cache is more like tens of seconds; do it in the background and serve `False` until done, or mmap + sequential scan.

**B2. `reset_cache` is broken as specified (B/C).**
"Rebuild `free` = all slots, leave data" while the index is cleared means every slot is free *and* still holds a valid-looking header. The first new store into slot k overwrites it — fine — but until then, nothing prevents... actually the worse direction: `reset_cache` is called on config changes; if any worker still has in-flight jobs against old slots, the slot can be reassigned mid-read. The seq check catches it, but only after the read. And the proposal's "?" on truncate suggests this wasn't thought through.
**Fix:** `reset_cache` must drain in-flight jobs first (or fence them), then either truncate the slabs or bump a per-slab epoch stored in a small superblock that headers must match. Specify it; don't leave a "?" in a design doc.

**B3. Eviction-skipping can deadlock/starve stores, and the fallback is silent capacity loss (B).**
Rule 2: if all LRU candidates are in `inflight_load`, keys are dropped from the store. `inflight_load` is populated by `prepare_load` and cleared at `complete_load` — which for a 100k-token reload takes ~3 s of worker I/O. Under a workload with continuous reloads (the exact workload this tier exists for), the LRU tail can be permanently in-flight, and *all* new stores get dropped. The cache then stops accepting new data with no signal, and the LRU order is never advanced because nothing can be evicted. There's no deadlock, but there is livelock-ish starvation of stores and unbounded `inflight_store` growth is prevented only by dropping.
**Fix:** (1) bound `inflight_load` lifetime — if a load job exceeds a deadline, evictable anyway (the seq check already makes this safe, per rule 3). (2) If stores are dropped N steps in a row, log/metric it. (3) Consider evicting in-flight-load slots by *canceling* the load's claim on those keys (they'll fail the seq check and be recomputed) rather than dropping the store — stores are the scarce, valuable operation.

## Should-fix

**S1. Sizing arithmetic double-counts nothing but mis-allocates (E).**
`N_drafter = R * N_target` with R=4 assumes drafter blocks are stored at the same density as target blocks. But the drafter is a 2048-token sliding window: only the last 32 blocks per request are ever live, and the connector requires the drafter's last 33 blocks for a hit. Old drafter blocks beyond the window are dead weight — do they even get stored? If yes, you're spending 4/5 of your disk on data that can never produce a hit by itself and ages out instantly. If the intent is that only the trailing window is stored, the 4:1 ratio is wrong in the other direction (you need ≥33 drafter slots per *request*, not per target block, and the working set is request-count-bound, not token-bound). Also: a hit needs target prefix AND drafter tail; if the drafter slab fills and recycles its LRU (which it will, fast, since every request's tail is hot), hit rate collapses even with a fat target index.
**Fix:** size the drafter slab from concurrent-request count × 33 blocks + margin, not from R×N_target; state explicitly which drafter blocks are stored and add a metric for "target hit, drafter miss" — that's the failure mode to watch.

**S2. CRC on header only; payload corruption is undetected (D).**
A torn or bit-flipped payload passes all checks (magic, hash-of-key, seq, length are all header fields). You'll serve corrupt KV into the model. Today's per-file format presumably had the same hole, but a slab that's overwritten in-place makes silent cross-key contamination more likely (partial overwrite of a reused slot where the header write landed but the payload didn't).
**Fix:** CRC the payload too (cost is negligible vs 4 MB/block, ~1 GB/s+ for CRC32C with hardware support); or at least fdatasync ordering: write payload, fdatasync, then header, fdatasync — so a valid header implies a durable payload. The current "one fdatasync per chunk" doesn't give you that ordering guarantee.

**S3. `pump()` on the main thread + one fdatasync per chunk (F).**
fdatasync of a 48-block chunk (~190 MB target) can take hundreds of ms on consumer NVMe; if the sync is issued from the main thread's pump, you stall the engine step. The design says disk I/O is "thread pools" but the sync point is ambiguous. Also `pwritev` of header+payload as one iovec is fine, but 48 slots × 4 MB = 190 MB against 48 pinned bounce slots — is the bounce buffer the source? Then the sync must complete before those bounce slots are recycled, coupling GPU-side progress to disk latency.
**Fix:** issue fdatasync from the I/O thread pool; recycle bounce slots only after sync completion; state the expected worst-case sync latency and confirm it's off the engine-critical path.

**S4. Sequence numbers are per-scheduler-lifetime but slots are shared across restarts (C).**
"Restart continues from the max seen" — max seen *on rank 0*. If ranks diverged (B1), rank 2's max seq differs, and post-restart stores get seqs that may collide with or be lower than stale headers on some rank. Combined with B1 this weakens the seq check.
**Fix:** persist the seq counter in the superblock per slab, bumped on every store, and take the global max across ranks at rebuild.

## Nits

- Sparse growth to 150 GB on a shared NVMe with 4 ranks/node: check filesystem free-space handling; a rank that can't grow fails at store time with no fallback specified.
- Two slabs per group per rank = 8 files/node: fine, but `posix_fadvise(DONTNEED)` after *each chunk* also drops the just-written header you may re-read soon; consider WILLNEED on the load path.
- `lookup` touches LRU on every poll — the connector "re-queries on None," so hot in-flight keys get artificially promoted. Minor distortion of LRU order.
- No mention of metrics, alerting, or admin tooling at all (G): slot occupancy, eviction rate, store-drop count (B3), hit rate split target/drafter, rebuild time. A production tier needs all of these plus a documented drain/resize path (you can't change `disk_bytes_per_rank` without invalidating everything — say so).

## Verdict

**Build with changes** — the slab approach is the right call versus 8,500 files/rank and unbounded growth, and the alternative (per-block files + scheduler LRU shipping deletes) still needs the same scheduler-side index and has worse crash-consistency and directory-scan costs. But B1 (intersection or journal-based rebuild) and B3 (bounded in-flight-load lifetime) must be fixed before this is correct under failure; S1's drafter sizing needs to be re-derived, not carried over from the 4:1 ratio. The core slot/seq/eviction machinery is sound; the restart and starvation paths are where it currently breaks the connector's "True means every rank" contract.