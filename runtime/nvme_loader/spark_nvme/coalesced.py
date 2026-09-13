"""Original safetensors ranges -> bounded pinned tiles -> owning tensor batches.

Only the producer touches the read pool and upload stream. The native consumer
receives owning views after a stream dependency on each batch's upload event. Retained aliases keep
their entire backing allocation alive and are charged to the owned byte budget.
No quantizer/architecture semantics or prepared weight files are required.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import errno
import json
import logging
import os
import time

import torch

from .streaming import OwnedStorageBudget, UnsupportedStream, prefetch_weights, stamp, storage_read_bytes
from .transport import ALIGN, align

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Batch:
    file: str
    start: int
    end: int
    tensors: tuple

    @property
    def bytes(self):
        return self.end - self.start


def plan_batches(metadata, batch_bytes=64 << 20):
    if batch_bytes < ALIGN or batch_bytes % ALIGN:
        raise ValueError("batch size must be a positive 4096 multiple")
    tensors, stamps, _ = metadata
    by_file = {path: [] for path in stamps}
    for name, info in tensors.items():
        if info.bytes:
            itemsize = torch.empty((), dtype=info.dtype).element_size()
            if info.offset % itemsize:
                raise UnsupportedStream("unaligned tensor encoding needs native reconstruction")
            by_file[info.file].append((name, info))
    batches = []
    for path, rows in by_file.items():
        rows.sort(key=lambda row: row[1].offset)
        current = []
        start = end = 0
        for name, info in rows:
            if not 0 <= info.offset < info.offset + info.bytes <= stamps[path][2]:
                raise ValueError("tensor range outside checkpoint")
            if current and info.offset < end:
                raise ValueError("overlapping checkpoint tensors")
            if current and info.offset + info.bytes - start > batch_bytes:
                batches.append(Batch(path, start, end, tuple(current)))
                current = []
            if not current:
                start = info.offset // ALIGN * ALIGN
            end = info.offset + info.bytes
            current.append((name, info))
        if current:
            batches.append(Batch(path, start, end, tuple(current)))
    return batches


def read_aligned(fd, view, offset, required, direct=True):
    """Allow a final EOF-short direct read only after all required bytes arrived."""
    done = 0
    calls = 0
    buffered_fd = None
    try:
        while done < required:
            n = os.preadv(buffered_fd if buffered_fd is not None else fd,
                         [view[done:]], offset + done)
            calls += 1
            if not n:
                raise EOFError(f"short checkpoint read at {offset + done}")
            done += n
            if done < required and direct and buffered_fd is None:
                # POSIX permits short reads. Reopen the same inode without
                # O_DIRECT for a rare remainder instead of assuming its new
                # offset/address satisfies this filesystem's alignment rules.
                buffered_fd = os.open(f"/proc/self/fd/{fd}", os.O_RDONLY)
        return done, calls
    finally:
        if buffered_fd is not None:
            os.close(buffered_fd)


def check_direct_support(metadata):
    """Probe one aligned read per source filesystem before native placement."""
    owner = torch.empty(ALIGN * 2, dtype=torch.uint8)
    offset = (-owner.data_ptr()) % ALIGN
    view = memoryview(owner[offset:offset + ALIGN].numpy())
    checked = set()
    for path, before in metadata[1].items():
        if before[0] in checked:
            continue
        try:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            try:
                os.preadv(fd, [view], 0)
            finally:
                os.close(fd)
        except OSError as exc:
            if exc.errno in (errno.EINVAL, errno.EOPNOTSUPP, errno.ENOSYS):
                raise UnsupportedStream("filesystem does not support aligned direct reads") from exc
            raise
        checked.add(before[0])


class RangeReader:
    """Two reusable host tiles, concurrent aligned reads, event-fenced reuse."""
    def __init__(self, *, device, memory_limit, concurrency, tile_bytes, direct, metrics):
        if not 1 <= concurrency <= 128:
            raise ValueError("read concurrency must be 1..128")
        # Charge the extra alignment allocation, not only usable tile bytes.
        self.tile_bytes = min(tile_bytes, (memory_limit // 2 - ALIGN) // ALIGN * ALIGN)
        if self.tile_bytes < ALIGN:
            raise ValueError("read budget must fit two aligned host tiles")
        self.cuda = device == "cuda"
        self.stream = torch.cuda.current_stream() if self.cuda else None
        self.direct, self.metrics = direct, metrics
        self.slots = []
        for _ in range(2):
            owner = torch.empty(self.tile_bytes + ALIGN, dtype=torch.uint8,
                                pin_memory=device in ("cuda", "pinned"))
            offset = (-owner.data_ptr()) % ALIGN
            tensor = owner[offset:offset + self.tile_bytes]
            self.slots.append([tensor, memoryview(tensor.numpy()), None])
        self.pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="nvme-range")
        self.concurrency = concurrency
        self.fd = None
        self.path = None
        self.next_slot = 0
        metrics["reader_buffer_bytes"] = 2 * (self.tile_bytes + ALIGN)

    def read(self, slot, offset, length):
        return read_aligned(self.fd, slot[1][offset:offset + align(length)],
                            self.source_offset + offset, length, direct=self.direct)

    def fill(self, batch, output):
        if self.path != batch.file:
            if self.fd is not None:
                os.close(self.fd)
            self.fd = None
            self.fd = os.open(batch.file, os.O_RDONLY | (os.O_DIRECT if self.direct else 0))
            self.path = batch.file
        # Tile reads finish before source_offset changes. Uploads can overlap
        # the next tile's reads, and slot fences protect their source memory.
        final_event = None
        for position in range(0, batch.bytes, self.tile_bytes):
            slot = self.slots[self.next_slot % 2]
            self.next_slot += 1
            if slot[2] is not None:
                started = time.monotonic()
                slot[2].synchronize()
                self.metrics["staging_wait_seconds"] += time.monotonic() - started
            length = min(self.tile_bytes, batch.bytes - position)
            self.source_offset = batch.start + position
            quantum = max(ALIGN, align((length + self.concurrency - 1) // self.concurrency))
            futures = [self.pool.submit(self.read, slot, offset, min(quantum, length - offset))
                       for offset in range(0, length, quantum)]
            started = time.monotonic()
            # Wait for every submitted job even on failure before reusing state.
            error = None
            for future in futures:
                try:
                    received, calls = future.result()
                    self.metrics["read_bytes"] += received
                    self.metrics["read_calls"] += calls
                except BaseException as exc:
                    error = error or exc
            self.metrics["reader_wait_seconds"] += time.monotonic() - started
            if error is not None:
                raise error
            started = time.monotonic()
            output[position:position + length].copy_(slot[0][:length], non_blocking=self.cuda)
            if self.cuda:
                event = torch.cuda.Event()
                event.record(self.stream)
                slot[2] = event
                final_event = event
            self.metrics["upload_enqueue_seconds"] += time.monotonic() - started
            self.metrics["uploads"] += 1
        return final_event

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        try:
            if self.cuda:
                started = time.monotonic()
                self.stream.synchronize()
                self.metrics["upload_wait_seconds"] += time.monotonic() - started
        finally:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None


def coalesced_weights(files, metadata, *, concurrency=32, memory_limit=2 << 30,
                      owned_limit=6 << 30, batch_bytes=64 << 20, device="cpu",
                      direct=True, skip=lambda name: False, metrics=None):
    if device not in ("cpu", "pinned", "cuda") or owned_limit <= 0:
        raise ValueError("invalid coalesced device or owned budget")
    tensors, stamps, source_id = metadata
    if set(os.path.abspath(f) for f in files) != set(stamps):
        raise ValueError("checkpoint file inventory mismatch")
    batches = plan_batches(metadata, batch_bytes)
    if any(b.bytes > owned_limit for b in batches):
        raise UnsupportedStream("largest batch exceeds owned byte budget")
    metrics = metrics if metrics is not None else {}
    metrics.update(backend="coalesced", source_metadata_id=source_id, device=device,
                   direct=direct, concurrency=concurrency, batch_bytes=batch_bytes,
                   bytes=0, tensors=0, batches=0, empty_tensors=0, skipped_tensors=0,
                   read_bytes=0, read_calls=0, uploads=0, reader_wait_seconds=0.,
                   upload_enqueue_seconds=0., upload_wait_seconds=0.,
                   staging_wait_seconds=0.,
                   producer_backpressure_seconds=0., owned_limit_bytes=owned_limit,
                   payload_sha256=False)
    budget = OwnedStorageBudget(owned_limit)
    started = time.monotonic()
    initial_reads = storage_read_bytes()

    def produce():
        reader = None
        try:
            reader = RangeReader(device=device, memory_limit=memory_limit, concurrency=concurrency,
                                 tile_bytes=batch_bytes, direct=direct, metrics=metrics)
            for batch in batches:
                if stamp(batch.file) != stamps[batch.file]:
                    raise ValueError("checkpoint changed before batch read")
                budget.reserve(batch.bytes)
                output = torch.empty(batch.bytes, dtype=torch.uint8,
                                     device="cuda" if device == "cuda" else "cpu",
                                     pin_memory=device == "pinned")
                budget.track(output)
                event = reader.fill(batch, output)
                metrics["batches"] += 1
                tick = time.monotonic()
                yield (batch, event), output
                metrics["producer_backpressure_seconds"] += time.monotonic() - tick
                del output
        finally:
            if reader is not None:
                reader.close()

    iterator = prefetch_weights(produce(), device=device)
    last_log = started
    try:
        for (batch, event), output in iterator:
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
            for name, info in batch.tensors:
                if skip(name):
                    metrics["skipped_tensors"] += 1
                    continue
                offset = info.offset - batch.start
                tensor = output[offset:offset + info.bytes].view(info.dtype).reshape(info.shape)
                metrics["bytes"] += info.bytes
                metrics["tensors"] += 1
                yield name, tensor
                del tensor
            del output
            if time.monotonic() - last_log > 15:
                log.info("NVME_COALESCED_PROGRESS batches=%d bytes=%d seconds=%.3f",
                         metrics["batches"], metrics["bytes"], time.monotonic() - started)
                last_log = time.monotonic()
        for name, info in tensors.items():
            if info.bytes == 0:
                if skip(name):
                    metrics["skipped_tensors"] += 1
                else:
                    yield name, torch.empty(info.shape, dtype=info.dtype,
                                            device="cuda" if device == "cuda" else "cpu")
                    metrics["empty_tensors"] += 1
        for path, before in stamps.items():
            if stamp(path) != before:
                raise ValueError("checkpoint changed during coalesced streaming")
        metrics["complete"] = True
    finally:
        try:
            iterator.close()
        finally:
            metrics["seconds"] = time.monotonic() - started
            metrics["process_storage_read_bytes"] = storage_read_bytes() - initial_reads
            metrics["peak_owned_bytes"] = budget.peak
            budget.close()
        log.info("NVME_COALESCED %s", json.dumps(metrics, sort_keys=True))
