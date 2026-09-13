"""Real-file direct-I/O, typed-view ownership, and coalesced failure boundaries."""
import gc
import os
from pathlib import Path
import sys
import threading

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime/nvme_loader"))
from spark_nvme.streaming import checkpoint_metadata
from spark_nvme.coalesced import coalesced_weights, plan_batches, read_aligned, check_direct_support


@pytest.fixture
def checkpoint(tmp_path):
    expected = {}
    paths = []
    for shard in range(4):
        rows = {f"{shard}.packed": torch.arange(3073 * 65, dtype=torch.int32).reshape(3073, 65),
                f"{shard}.scale": torch.arange(513).to(torch.bfloat16),
                f"{shard}.scalar": torch.tensor(1.25),
                f"{shard}.shape": torch.tensor([3073, 65]),
                f"{shard}.empty": torch.empty(0, 17)}
        path = tmp_path / f"part-{shard}.safetensors"
        save_file(rows, path)
        paths.append(str(path))
        expected.update(rows)
    return paths, expected


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("depth", [1, 32])
def test_real_coalesced_retention_and_oversize_tensor(checkpoint, direct, depth):
    files, expected = checkpoint
    meta = checkpoint_metadata(files)
    metrics = {}
    actual = dict(coalesced_weights(files, meta, concurrency=depth, direct=direct,
                  batch_bytes=64 << 10, memory_limit=48 << 10, owned_limit=8 << 20, metrics=metrics))
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name
    assert metrics["complete"]
    assert metrics["reader_buffer_bytes"] <= 48 << 10
    assert metrics["read_bytes"] >= sum(t.numel() * t.element_size() for t in expected.values())
    assert metrics["peak_owned_bytes"] <= 8 << 20
    assert metrics["uploads"] > metrics["batches"]  # giant tensors use bounded tiles


def test_coalescing_and_retained_aliases(tmp_path):
    data = {f"weight.{i}": torch.arange(128, dtype=torch.int32) + i for i in range(80)}
    path = str(tmp_path / "weights.safetensors")
    save_file(data, path)
    metrics = {}
    iterator = coalesced_weights([path], checkpoint_metadata([path]), batch_bytes=16 << 10,
                                 memory_limit=64 << 10, owned_limit=128 << 10, metrics=metrics)
    aliases = {}
    for name, tensor in iterator:
        aliases[name] = tensor[1:]
    gc.collect()
    for name, alias in aliases.items():
        assert torch.equal(alias, data[name][1:])
    assert metrics["uploads"] < len(data) // 4
    assert metrics["peak_owned_bytes"] > sum(t.numel() * 4 for t in data.values())


def test_early_close_stops_pool_and_retained_view_survives(checkpoint):
    files, expected = checkpoint
    before = {t.ident for t in threading.enumerate()}
    metrics = {}
    iterator = coalesced_weights(files, checkpoint_metadata(files), batch_bytes=64 << 10, metrics=metrics)
    name, tensor = next(iterator)
    iterator.close()
    assert torch.equal(tensor, expected[name])
    assert not metrics.get("complete")
    assert not [t for t in threading.enumerate() if t.ident not in before and t.name.startswith("nvme")]


def test_budget_failure_propagates_and_does_not_deadlock(checkpoint):
    files, _ = checkpoint
    with pytest.raises(MemoryError, match="retained"):
        dict(coalesced_weights(files, checkpoint_metadata(files), batch_bytes=64 << 10,
                              owned_limit=1500 << 10))


def test_changed_source_rejected(checkpoint):
    files, _ = checkpoint
    meta = checkpoint_metadata(files)
    iterator = coalesced_weights(files, meta, batch_bytes=64 << 10)
    next(iterator)
    stat = os.stat(files[-1])
    os.utime(files[-1], ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000))
    with pytest.raises(ValueError, match="changed"):
        dict(iterator)


def test_truncated_read_and_valid_eof_tail(tmp_path):
    path = tmp_path / "raw"
    path.write_bytes(b"x" * 4199)
    fd = os.open(path, os.O_RDONLY)
    try:
        data = bytearray(8192)
        assert read_aligned(fd, memoryview(data), 0, 4199)[0] == 4199
        with pytest.raises(EOFError):
            read_aligned(fd, memoryview(data), 0, 4200)
    finally:
        os.close(fd)


def test_skip_empty_and_small_memory_budget(checkpoint):
    files, expected = checkpoint
    meta = checkpoint_metadata(files)
    actual = dict(coalesced_weights(files, meta, skip=lambda n: n.endswith(("packed", "empty"))))
    assert actual.keys() == {n for n in expected if not n.endswith(("packed", "empty"))}
    with pytest.raises(ValueError, match="two aligned"):
        list(coalesced_weights(files, meta, memory_limit=8192))
    assert all(b.bytes > 0 for b in plan_batches(meta))


@pytest.mark.parametrize("direct", [True, False])
def test_legal_unaligned_short_reads_are_completed(tmp_path, monkeypatch, direct):
    path = tmp_path / "short-read"
    data = bytes(range(256)) * 32
    path.write_bytes(data)
    native = os.preadv
    calls = 0
    def short(fd, buffers, offset):
        nonlocal calls
        calls += 1
        return native(fd, [buffers[0][:513] if calls == 1 else buffers[0]], offset)
    monkeypatch.setattr(os, "preadv", short)
    fd = os.open(path, os.O_RDONLY)
    try:
        output = bytearray(len(data))
        assert read_aligned(fd, memoryview(output), 0, len(data), direct=direct)[0] == len(data)
        assert output == data
        assert calls == 2
    finally:
        os.close(fd)


def test_direct_support_failure_is_an_early_fallback(checkpoint, monkeypatch):
    import errno
    from spark_nvme.streaming import UnsupportedStream
    files, _ = checkpoint
    meta = checkpoint_metadata(files)
    def unsupported(*args):
        raise OSError(errno.EINVAL, "unsupported direct read")
    monkeypatch.setattr(os, "preadv", unsupported)
    with pytest.raises(UnsupportedStream, match="filesystem"):
        check_direct_support(meta)
