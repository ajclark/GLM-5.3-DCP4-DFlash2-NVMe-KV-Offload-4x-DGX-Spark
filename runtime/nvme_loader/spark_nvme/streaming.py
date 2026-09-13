"""Concurrent original safetensors ingestion; no prepared weights required.

The native model loader owns placement. Returned tensors own their storage;
Run:ai's reusable buffers never escape this iterator. Source metadata is an
activation consistency check, not a cryptographic digest of weight payloads.
"""
from dataclasses import dataclass
from collections import Counter
import hashlib
import json
import logging
import os
from pathlib import Path
import struct
import queue
import threading
import time

import torch

from .artifact import digest

log = logging.getLogger(__name__)
DTYPES = {"BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8,
          "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
          "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32,
          "F64": torch.float64}
for label, name in (("F8_E4M3", "float8_e4m3fn"), ("F8_E5M2", "float8_e5m2"),
                    ("U16", "uint16"), ("U32", "uint32"), ("U64", "uint64")):
    if hasattr(torch, name):
        DTYPES[label] = getattr(torch, name)


class UnsupportedStream(ValueError):
    """Can fall back only before the first weight is consumed."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple
    dtype: torch.dtype
    bytes: int
    file: str = ""
    offset: int = 0


def stamp(path):
    st = os.stat(path)
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def storage_read_bytes():
    return int(dict(line.split(":", 1) for line in
                    Path("/proc/self/io").read_text().splitlines())["read_bytes"])


def checkpoint_metadata(files):
    from safetensors import safe_open
    tensors, stamps, identities = {}, {}, []
    for filename in files:
        # Retain symlink paths in the final stamp check: redirecting a source
        # link after inspection must not silently select different weights.
        path = str(Path(filename).absolute())
        if path in stamps:
            raise ValueError("duplicate checkpoint file")
        stamps[path] = stamp(path)
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError("truncated safetensors header")
            size = struct.unpack("<Q", raw)[0]
            if size > min(100_000_000, stamps[path][2] - 8):
                raise ValueError("invalid safetensors header length")
            header = f.read(size)
            if len(header) != size:
                raise ValueError("truncated safetensors header")
        # Let the official parser validate dtype/shape/offset coverage too.
        with safe_open(path, framework="pt", device="cpu") as source:
            offsets = json.loads(header)
            metadata = source.metadata() or {}
            if any("torchao" in str(x).lower() for x in metadata.items()):
                raise UnsupportedStream("tensor subclass reconstruction requires native loading")
            for name in source.keys():
                if name in tensors:
                    raise ValueError("duplicate checkpoint tensor: " + name)
                view = source.get_slice(name)
                label, shape = view.get_dtype(), tuple(view.get_shape())
                if label not in DTYPES:
                    raise UnsupportedStream("unsupported streaming dtype: " + label)
                n = torch.empty((), dtype=DTYPES[label]).element_size()
                for dim in shape:
                    n *= dim
                start, end = offsets[name]["data_offsets"]
                if end - start != n:
                    raise ValueError("checkpoint tensor byte count mismatch")
                tensors[name] = TensorInfo(name, shape, DTYPES[label], n, path, 8 + size + start)
        identities.append({"name": Path(path).name, "size": stamps[path][2],
                           "header_sha256": hashlib.sha256(header).hexdigest()})
        if stamp(path) != stamps[path]:
            raise ValueError("checkpoint changed during header inspection")
    if not tensors:
        raise ValueError("empty checkpoint file list")
    return tensors, stamps, digest(identities)


def prefetch_weights(iterator, device="cpu"):
    """One owning tensor of read/copy lookahead, with explicit cancellation.

    The producer alone advances the iterator. Model mutation stays on the
    caller thread. Run:ai tensors complete their upload before publication.
    Async batch callers must carry a completion event and wait on the consumer
    stream before accessing views. Both paths record the consumer stream here.
    """
    ready = queue.Queue(maxsize=1)
    slot = threading.Semaphore(1)
    stop = threading.Event()
    gpu = device == "cuda"
    cuda_device = torch.cuda.current_device() if gpu else None
    upload_stream = torch.cuda.Stream(device=cuda_device) if gpu else None
    if gpu:
        upload_stream.wait_stream(torch.cuda.current_stream(cuda_device))

    def put(item):
        while not stop.is_set():
            try:
                ready.put(item, timeout=.1)
                return
            except queue.Full:
                pass

    def pump():
        terminal = ("done", None)
        try:
            if gpu:
                torch.cuda.set_device(cuda_device)
                torch.cuda.set_stream(upload_stream)
            while not stop.is_set():
                if not slot.acquire(timeout=.1):
                    continue
                if stop.is_set():
                    break
                try:
                    value = next(iterator)
                except StopIteration:
                    break
                put(("tensor", value))
                del value
        except BaseException as exc:
            terminal = ("error", exc)
        finally:
            try:
                iterator.close()
            except BaseException as exc:
                terminal = ("error", exc)
        # Completion is visible only after producer cleanup succeeds.
        put(terminal)

    thread = threading.Thread(target=pump, name="nvme-stream-pump", daemon=True)
    thread.start()
    try:
        while True:
            kind, value = ready.get()
            slot.release()
            if kind == "error":
                raise value
            if kind == "done":
                break
            if value[1].is_cuda:
                value[1].record_stream(torch.cuda.current_stream(cuda_device))
            yield value
            del value
    finally:
        stop.set()
        thread.join(timeout=30)
        if thread.is_alive():
            raise RuntimeError("stream producer failed to stop; abort this worker")


class OwnedStorageBudget:
    """Track C++ storage lifetime, including aliases surviving a Python tensor.

    Never block waiting for a model to release tensors: it may need later
    weights to finish a fusion. Fail the fresh worker instead of deadlocking.
    """
    def __init__(self, limit):
        self.limit, self.live, self.peak = limit, 0, 0
        self.refs = []

    def collect(self):
        remaining = []
        for ref, size in self.refs:
            if torch.UntypedStorage._expired(ref):
                self.live -= size
                torch.UntypedStorage._free_weak_ref(ref)
            else:
                remaining.append((ref, size))
        self.refs = remaining

    def reserve(self, size):
        self.collect()
        if self.live + size > self.limit:
            raise MemoryError("native loader retained streaming tensors beyond owned byte budget")

    def track(self, tensor):
        storage = tensor.untyped_storage()
        size = storage.nbytes()
        self.refs.append((storage._weak_ref(), size))
        self.live += size
        self.peak = max(self.peak, self.live)

    def close(self):
        for ref, _ in self.refs:
            torch.UntypedStorage._free_weak_ref(ref)
        self.refs.clear()


def stream_weights(files, metadata, *, concurrency=32, memory_limit=2 << 30,
                   owned_limit=6 << 30, device="cpu",
                   skip=lambda name: False, metrics=None, streamer_factory=None):
    """Yield owning tensors as reads finish, like vLLM's Run:ai iterator.

    Source order is preserved by the surrounding DefaultModelLoader. Within
    one source, completion order is allowed by the native streaming interface.
    The CPU buffer budget is separate from model-retained/output allocations.
    """
    tensors, stamps, source_id = metadata
    if not 1 <= concurrency <= 128 or memory_limit <= 0 or owned_limit <= 0:
        raise ValueError("invalid streaming concurrency or memory budget")
    if device not in ("cpu", "pinned", "cuda"):
        raise ValueError("invalid streaming device")
    largest = max(t.bytes for t in tensors.values())
    if largest > memory_limit or largest > owned_limit:
        raise ValueError("streaming budgets must accommodate largest checkpoint tensor")
    if streamer_factory is None:
        from runai_model_streamer import SafetensorsStreamer
        streamer_factory = SafetensorsStreamer
    # Native initialization loads model sources sequentially in each worker.
    keys = {"RUNAI_STREAMER_CONCURRENCY": str(concurrency),
            "RUNAI_STREAMER_MEMORY_LIMIT": str(memory_limit),
            "RUNAI_STREAMER_DIST": "0"}
    previous = {k: os.environ.get(k) for k in keys}
    budget = OwnedStorageBudget(owned_limit)
    metrics = metrics if metrics is not None else {}
    metrics.update(source_metadata_id=source_id, tensors=0, bytes=0,
                   empty_tensors=0, skipped_tensors=0, reader_wait_seconds=0.,
                   placement_stage_seconds=0., producer_backpressure_seconds=0.,
                   reader_buffer_bytes=memory_limit, owned_limit_bytes=owned_limit,
                   device=device, concurrency=concurrency, payload_sha256=False)
    seen = set()
    remaining = Counter(t.file for t in tensors.values() if t.bytes)
    started = last_log = time.monotonic()
    initial_reads = storage_read_bytes()

    def own(tensor, info):
        budget.reserve(info.bytes)
        if device == "cuda":
            # Blocking copy finishes consumption of Run:ai's reusable host
            # buffer before get_tensors() can overwrite it. CUDA allocator
            # stream tracking handles the owning output's later lifetime.
            output = tensor.to(device=torch.device("cuda", torch.cuda.current_device()), copy=True)
        elif device == "pinned" and info.bytes:
            output = torch.empty(info.shape, dtype=info.dtype, pin_memory=True)
            output.copy_(tensor)
        else:
            output = tensor.clone()
        budget.track(output)
        return output

    try:
        os.environ.update(keys)
        with streamer_factory() as streamer:
            streamer.stream_files(list(files), device="cpu", is_distributed=False)
            iterator = iter(streamer.get_tensors())
            while True:
                tick = time.monotonic()
                try:
                    name, tensor = next(iterator)
                except StopIteration:
                    break
                metrics["reader_wait_seconds"] += time.monotonic() - tick
                if name not in tensors or name in seen:
                    raise ValueError("unexpected/duplicate streamed tensor: " + name)
                info = tensors[name]
                if tuple(tensor.shape) != info.shape or tensor.dtype != info.dtype:
                    raise ValueError("streamed tensor schema mismatch: " + name)
                seen.add(name)
                if info.bytes:
                    remaining[info.file] -= 1
                    if remaining[info.file] == 0:
                        # Finished source payloads are in owning buffers now;
                        # stop the page cache displacing Spark's unified RAM.
                        fd = os.open(info.file, os.O_RDONLY)
                        try:
                            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        finally:
                            os.close(fd)
                if skip(name):
                    metrics["skipped_tensors"] += 1
                    del tensor
                    continue
                tick = time.monotonic()
                output = own(tensor, info)
                del tensor
                metrics["placement_stage_seconds"] += time.monotonic() - tick
                metrics["bytes"] += info.bytes
                metrics["tensors"] += 1
                tick = time.monotonic()
                yield name, output
                metrics["producer_backpressure_seconds"] += time.monotonic() - tick
                del output
                if time.monotonic() - last_log >= 15:
                    metrics["process_storage_read_bytes"] = storage_read_bytes() - initial_reads
                    metrics["rss_bytes"] = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
                    if device == "cuda":
                        metrics["cuda_free_bytes"] = torch.cuda.mem_get_info()[0]
                    log.warning("NVME_STREAM_PROGRESS %s", json.dumps(metrics, sort_keys=True))
                    last_log = time.monotonic()
        # Run:ai 0.16.1 omits zero-element tensors. Recreate every missing empty
        # entry; a missing nonempty weight must fail rather than serve silently.
        for name, info in tensors.items():
            if name in seen:
                continue
            if info.bytes:
                raise ValueError("missing streamed tensor: " + name)
            if not skip(name):
                yield name, torch.empty(info.shape, dtype=info.dtype,
                                        device="cuda" if device == "cuda" else "cpu")
                metrics["empty_tensors"] += 1
        for path, before in stamps.items():
            if stamp(path) != before:
                raise ValueError("checkpoint changed during streaming: " + path)
        metrics["complete"] = True
    finally:
        metrics["seconds"] = time.monotonic() - started
        metrics["process_storage_read_bytes"] = storage_read_bytes() - initial_reads
        metrics["peak_owned_bytes"] = budget.peak
        budget.close()
        for k, value in previous.items():
            if value is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = value
        log.warning("NVME_STREAM %s", json.dumps(metrics, sort_keys=True))
