"""Atomic rank-local, pre-kernel weight artifacts. No pickle or GPU pointers.

The native model constructor owns tensors and their metadata; restore copies
bytes into those allocations only after the complete schema matches. Native
quantization/kernel postprocessing then runs exactly once, as on a cold load.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import logging
import time
import uuid

import torch

from .transport import Extent, align, stream

VERSION = 1
log = logging.getLogger(__name__)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def inventory(model):
    """Include aliases and nonpersistent buffers; describe physical storage."""
    tensors = dict(model.named_parameters(remove_duplicate=False))
    tensors.update(model.named_buffers(remove_duplicate=False))
    storages, schema, groups = [], {}, {}
    for name, t in sorted(tensors.items()):
        if t.device.type == "meta" or t.layout != torch.strided or t.is_quantized:
            raise ValueError(f"unsupported storage for {name}")
        storage = t.untyped_storage()
        key = (str(t.device), storage._cdata)
        if key not in groups:
            groups[key] = len(storages)
            raw = torch.empty(0, dtype=torch.uint8, device=t.device)
            raw.set_(storage, 0, (storage.nbytes(),), (1,))
            storages.append(raw)
        schema[name] = {"storage": groups[key], "dtype": str(t.dtype),
                        "shape": list(t.shape), "stride": list(t.stride()),
                        "offset": t.storage_offset(), "bytes": storage.nbytes()}
    return schema, storages


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(model, path, contract, *, chunk_bytes=4 << 20, expected_schema=None, loader_state=None):
    """Export BEFORE native postprocessing, with <= one chunk of host memory."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema, storages = inventory(model)
    if expected_schema is not None and schema != expected_schema:
        raise ValueError("native load changed tensor schema; requires a model adapter")
    if chunk_bytes <= 0 or chunk_bytes % 4096:
        raise ValueError("invalid chunk size")
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    tmp.mkdir()
    manifest = {"version": VERSION, "phase": "before-native-postprocess",
                "contract": contract, "schema": schema, "chunk_bytes": chunk_bytes,
                "chunks": [], "storage_bytes": [s.numel() for s in storages],
                "loader_state": loader_state or {}}
    try:
        with (tmp / "weights.bin").open("wb", buffering=0) as f:
            flushed = 0
            last_log = time.monotonic()
            for i, raw in enumerate(storages):
                for start in range(0, raw.numel(), chunk_bytes):
                    cpu = raw[start:start + chunk_bytes].to("cpu")
                    view = memoryview(cpu.numpy())
                    n = len(view)
                    manifest["chunks"].append({"storage": i, "start": start,
                        "length": n, "offset": f.tell(),
                        "sha256": hashlib.sha256(view).hexdigest()})
                    # FileIO.write may legally return a short write.
                    while view:
                        written = f.write(view)
                        if not written:
                            raise OSError("zero-length artifact write")
                        view = view[written:]
                    f.write(bytes(align(n) - n))
                    if f.tell() - flushed >= 256 << 20:
                        os.fdatasync(f.fileno())
                        os.posix_fadvise(f.fileno(), flushed, f.tell() - flushed,
                                         os.POSIX_FADV_DONTNEED)
                        flushed = f.tell()
                    if time.monotonic() - last_log > 15:
                        log.warning("NVME_PREPARE wrote %.2f GiB", f.tell() / 2**30)
                        last_log = time.monotonic()
            f.flush()
            os.fsync(f.fileno())
            manifest["file_bytes"] = f.tell()
        manifest["artifact_id"] = digest(manifest)
        with (tmp / "manifest.json").open("wb") as f:
            f.write(canonical(manifest)); f.flush(); os.fsync(f.fileno())
        fsync_directory(tmp)
        os.rename(tmp, path)
        fsync_directory(path.parent)
        return manifest
    except BaseException:
        shutil.rmtree(tmp)
        raise


def inspect(path, contract):
    path = Path(path)
    m = json.loads((path / "manifest.json").read_bytes())
    artifact_id = m.pop("artifact_id")
    if digest(m) != artifact_id:
        raise ValueError("manifest checksum mismatch")
    m["artifact_id"] = artifact_id
    if m["version"] != VERSION or m["phase"] != "before-native-postprocess":
        raise ValueError("unsupported artifact format")
    if m["contract"] != contract:
        raise ValueError("artifact runtime/model/topology contract mismatch")
    if (path / "weights.bin").stat().st_size != m["file_bytes"]:
        raise ValueError("artifact file size mismatch")
    # Validate complete coverage and bounds BEFORE any destination is written.
    cursors = [0] * len(m["storage_bytes"])
    file_cursor = 0
    for e in m["chunks"]:
        i, n = e["storage"], e["length"]
        if not 0 <= i < len(cursors) or e["start"] != cursors[i]:
            raise ValueError("invalid/overlapping artifact storage extent")
        if not 0 < n <= m["chunk_bytes"] or e["offset"] != file_cursor:
            raise ValueError("invalid artifact file extent")
        cursors[i] += n
        file_cursor += align(n)
    if cursors != m["storage_bytes"] or file_cursor != m["file_bytes"]:
        raise ValueError("incomplete artifact coverage")
    return m


def restore(model, path, contract, *, depth=32, direct=True):
    m = inspect(path, contract)
    schema, storages = inventory(model)
    if schema != m["schema"] or [s.numel() for s in storages] != m["storage_bytes"]:
        raise ValueError("constructed tensor/alias schema differs from artifact")
    extents = (Extent(e["offset"], e["length"], e["sha256"],
                      storages[e["storage"]][e["start"]:e["start"] + e["length"]])
               for e in m["chunks"])
    metrics = stream(Path(path) / "weights.bin", extents,
                     chunk_bytes=m["chunk_bytes"], depth=depth, direct=direct)
    metrics["artifact_id"] = m["artifact_id"]
    return metrics
