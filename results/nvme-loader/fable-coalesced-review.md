# Fable coalesced reader reviews

I'll read the new coalesced module, the streaming changes, the loader opt-in, and the tests.

**Verdict:** no hard correctness blocker. Ownership, fencing, the mutable reader state, retained-alias accounting and cancellation are all sound. Two items should be fixed before the Spark run because either one turns a benign condition into a mid-stream worker abort with no fallback. Details follow, confirmed items first.

**Confirmed correct**

- **Stream ownership.** The pump thread sets its current stream to the upload stream before the first `next()`, so `RangeReader.__init__` captures the upload stream, the batch `torch.empty` is allocated on it, every tile copy is enqueued on it, and `fill` host-synchronizes it once per batch before the batch is yielded. `prefetch_weights` then records the consumer stream on the base allocation; typed views share that storage block, so the record covers them and the allocator will not recycle the block until the consumer stream's work at free time completes.
- **Tile reuse.** A tile's reads wait on the event recorded after that tile's previous upload, and the event is captured at submit time, so the later `slot[2]` reassignment cannot race it.
- **Mutable `fd` and `source_offset`.** Both change only in `fill`, and every tile awaits all of its futures, including on error, before the next tile or batch mutates them. `close()` shuts the pool down with `wait=True` before closing the descriptor. No race.
- **Retained aliases and budget.** Views keep the whole batch alive and `track` charges the batch's storage; `test_coalescing_and_retained_aliases` exercises exactly that. With the draft retaining 4.92 GB and three 64 MiB batches in flight, the 6 GiB limit holds; the two adjacent 1.9 GB tensors in shard 1 become their own batches and peak at about 3.8 GB.
- **Cancellation.** `GeneratorExit` can only arrive at `yield batch, output`, never inside `fill`; the producer's `finally` closes the reader, pending futures are cancelled, running ones complete within one tile, and the 30 s join covers it.
- **Layout math.** `batch.start` is aligned down, tile positions and job quanta are 4 KiB multiples, the typed view's byte offset is a multiple of the element size by the `plan_batches` check, and safetensors headers are 8-byte padded so absolute offsets satisfy it for every supported dtype. Cross-batch re-reads of the aligned-down prefix are at most 4095 bytes and harmless.

**Fix before Spark**

1. **`read_aligned` misclassifies legal short reads as EOF.** At `runtime/nvme_loader/spark_nvme/coalesced.py:77` any partial read whose running total is not a 4 KiB multiple raises `EOFError` unless all required bytes are already in. That rule is only right for direct I/O on a 4 KiB logical-block device. In buffered mode there is no alignment constraint, and on a 512-byte-sector NVMe a direct read may legally continue from a 512-aligned position. A false `EOFError` here surfaces on the pump, propagates through the consumer, and aborts the worker after weights have been consumed, so there is no native fallback. Pass `direct` into `read_aligned`, skip the check when buffered, and when direct compare against the device's logical block size rather than `ALIGN`.

2. **No eager probe of direct I/O support.** `RangeReader.fill` opens with `O_DIRECT` on the first batch, which runs inside the first `next()` on the pump, so an `EINVAL` from an unsupported filesystem reaches the consumer as a hard failure after the fallback window has closed. The loader's eager block at `runtime/nvme_loader/spark_nvme/loader.py:178` should open and close the first file with `O_DIRECT` and, on `OSError`, select buffered mode rather than abort. The Sparks are ext4, so this is insurance, but the same code will meet other filesystems.

**Non-blocking notes**

- `plan_batches` can raise plain `ValueError` for overlapping or out-of-range tensors inside the eager block, which is not caught by the `UnsupportedStream`/`ImportError` handler. Since `safe_open` already validated the header, those can only mean a bug in the offset math, so failing loudly is acceptable; just be aware it is not a fallback path.
- The only file-tail case that matters is the last job of the last batch reading `align(length)` past EOF; Linux returns the short tail for both buffered and direct reads, `plan_batches` already guarantees the required range lies inside the file, and the direct-mode variant of `test_real_coalesced_retention_and_oversize_tensor` exercises it on a real filesystem. Nothing further needed there.
- The reader drains completely at every tile boundary before the next tile's jobs are submitted, so read overlap exists only within a tile. That is a throughput shape, not a correctness issue, and is outside this review's scope.

I'll take a quick look at the two changed pieces to make sure nothing new slipped in, then confirm.

Both fixes check out; no new correctness issue is apparent.

- **Short-read continuation.** Reopening `/proc/self/fd/N` without `O_DIRECT` gives a new open file description on the same inode, so a replaced or unlinked path cannot redirect the remainder, and a buffered read has no alignment constraint for the unaligned continuation. Reading into `view[done:]` may run past `required` into the alignment tail, which is harmless because `fill` copies only `length` bytes from the tile. Direct and buffered reads on an immutable file are coherent. Each job opens and closes its own descriptor, so the pool threads do not share it.
- **Filesystem probe.** One aligned 4 KiB direct read per `st_dev` before any weight is consumed, mapping `EINVAL`, `EOPNOTSUPP` and `ENOSYS` to `UnsupportedStream`, keeps the failure inside the fallback window. A file shorter than 4 KiB returns a short read rather than an error, so tiny fixtures do not trip it. Falling back to native rather than to buffered coalesced reads is a conservative policy choice, not a defect.

The harness memory-sample race is on the test side and does not affect the loader path. The coalesced backend is clear for the Spark run as far as this review goes.

I'll read the changed producer, `fill`, and the consumer's wait path, then check the prefetch pump's handling of the new `(batch, event, output)` shape.

The change is correct. I traced the three ordering edges that matter and the cancellation path; none regress.

**Upload to consumer.** `fill` enqueues every tile copy on the upload stream and returns the event recorded after the last one, so stream order makes that event cover the whole batch. The consumer calls `wait_event` on its current stream before it creates the typed views, and every loader kernel, and any `.item()` or `.cpu()` a loader might do, is issued on that same stream after the wait. The `prefetch_weights` pump still sees `value[1]` as the CUDA `output` tensor, so `record_stream` on the consumer stream is unchanged.

**Consumer free to block reuse.** The batch allocation belongs to the upload stream's pool. When the last view dies on the consumer thread, the allocator records an event on the consumer stream at free time, which is after the consumer's `wait_event` and its kernels, so a later `torch.empty` on the pump cannot obtain that block until the consumer's reads are done. If the consumer unwinds before calling `wait_event`, the block can only be reused by the upload stream itself, where the pending copy precedes any new copy in stream order, and `reader.close()` synchronizes the upload stream before the pinned tiles are released.

**Pinned tile reuse.** Each slot's fence is the event of the most recent copy out of that slot, and the producer host-synchronizes it before submitting the reads that overwrite the tile. The same event object serving as both slot fence and batch completion event is fine; events may be waited on any number of times, and re-recording never happens because every tile creates a new event. Persistent `next_slot` alternation across batches keeps the fence pairing correct across batch boundaries.

**Cancellation and bounds.** `GeneratorExit` still only lands at the producer's `yield`, `close()` waits for the pool and synchronizes the upload stream before the descriptor and tiles go away, and the one-item prefetch queue bounds pending uploads to at most the queued batch plus the one being filled. CPU and pinned modes return `None` for the event and take the synchronous copy path unchanged.

Two non-correctness notes. `torch.cuda.Event()` defaults to a spinning host wait, so the producer burns a core while blocked on a fence; `torch.cuda.Event(blocking=True)` yields the CPU instead, which matters only if the consumer thread is CPU-bound. And `upload_wait_seconds` is now measured only at close, so treat it as tail latency rather than a per-batch figure when reading the new metrics.

Nothing here needs to change before the parity rerun.
