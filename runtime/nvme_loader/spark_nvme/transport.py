"""Bounded concurrent preadv -> pinned memory -> destination, with slot fencing.

One thread per outstanding read. No Python byte-string copy on the read path.
O_DIRECT avoids duplicating weights in the page cache on unified-memory GPUs.
The caller provides immutable, aligned extents, including their SHA256 digest.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
import hashlib
import os
import time

import torch

ALIGN = 4096


def align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


@dataclass(frozen=True)
class Extent:
    offset: int
    length: int
    digest: str
    destination: torch.Tensor


def read_exact(fd, view, offset):
    """preadv can legally return short reads; EOF must never become zeros."""
    done = 0
    while done < len(view):
        n = os.preadv(fd, [view[done:]], offset + done)
        if not n:
            raise EOFError(f"short NVMe read at {offset + done}")
        done += n


def stream(path, extents, *, chunk_bytes=4 << 20, depth=32, direct=True):
    if chunk_bytes <= 0 or chunk_bytes % ALIGN or not 1 <= depth <= 128:
        raise ValueError("chunk_bytes must be a positive 4096 multiple; depth 1..128")
    extents = iter(extents)
    first = next(extents, None)
    if first is None:
        return {"bytes": 0, "seconds": 0, "read_MB_s": 0, "depth": depth}
    device = first.destination.device
    cuda = device.type == "cuda"
    copy_stream = torch.cuda.Stream(device=device) if cuda else None
    if cuda:
        # The destination allocations were made on the caller's stream.
        copy_stream.wait_stream(torch.cuda.current_stream(device))
    slots = []
    for _ in range(depth):
        owner = torch.empty(chunk_bytes + ALIGN, dtype=torch.uint8,
                            device="cpu", pin_memory=cuda)
        start = (-owner.data_ptr()) % ALIGN
        tensor = owner[start:start + chunk_bytes]
        slots.append((tensor, memoryview(tensor.numpy()), None))
    flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
    fd = os.open(path, flags)
    total = 0
    started = time.monotonic()
    pending = {}
    tail = iter([first])

    def next_extent():
        nonlocal tail
        e = next(tail, None)
        if e is None:
            tail = extents
            e = next(tail, None)
        return e

    def read(slot, extent):
        tensor, view, event = slots[slot]
        if event is not None:
            event.synchronize()  # NEVER overwrite an in-flight H2D source.
        if extent.offset % ALIGN or not 0 < extent.length <= chunk_bytes:
            raise ValueError("invalid aligned extent")
        read_exact(fd, view[:align(extent.length)], extent.offset)
        if hashlib.sha256(view[:extent.length]).hexdigest() != extent.digest:
            raise ValueError(f"weight checksum mismatch at {extent.offset}")
        return slot, extent

    try:
        with ThreadPoolExecutor(max_workers=depth, thread_name_prefix="nvme") as pool:
            for slot in range(depth):
                if (e := next_extent()) is not None:
                    pending[pool.submit(read, slot, e)] = slot
            while pending:
                ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in ready:
                    pending.pop(future)
                    slot, e = future.result()
                    tensor, view, _ = slots[slot]
                    if e.destination.numel() != e.length or e.destination.dtype != torch.uint8:
                        raise ValueError("destination must be an exact byte view")
                    if e.destination.device != device:
                        raise ValueError("mixed destination devices")
                    if cuda:
                        with torch.cuda.stream(copy_stream):
                            e.destination.copy_(tensor[:e.length], non_blocking=True)
                            event = torch.cuda.Event()
                            event.record(copy_stream)
                        slots[slot] = tensor, view, event
                    else:
                        e.destination.copy_(tensor[:e.length])
                    total += e.length
                    if (nxt := next_extent()) is not None:
                        pending[pool.submit(read, slot, nxt)] = slot
    finally:
        # Includes failed/cancelled reads: keep staging allocations alive until
        # every submitted copy completes before unwinding to native fallback.
        if cuda:
            copy_stream.synchronize()
        os.close(fd)
    elapsed = time.monotonic() - started
    return {"bytes": total, "seconds": elapsed, "read_MB_s": total / elapsed / 1e6,
            "depth": depth, "staging_bytes": depth * (chunk_bytes + ALIGN),
            "direct": direct, "verified_sha256": True}
