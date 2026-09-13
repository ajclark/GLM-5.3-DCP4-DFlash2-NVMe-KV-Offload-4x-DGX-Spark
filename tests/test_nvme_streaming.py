"""Original-shard streaming correctness against official safetensors tensors."""
import gc
import os
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime/nvme_loader"))
from spark_nvme.streaming import checkpoint_metadata, stream_weights, OwnedStorageBudget
from spark_nvme.streaming import prefetch_weights


@pytest.fixture
def checkpoint(tmp_path):
    expected = {"packed": torch.arange(1024 * 65, dtype=torch.int32).reshape(1024, 65),
                "scale": torch.arange(31, dtype=torch.float32).to(torch.bfloat16),
                "shape": torch.tensor([1024, 520], dtype=torch.int64),
                "scalar": torch.tensor(2.25), "empty": torch.empty(0, 17)}
    files = []
    for i, (name, tensor) in enumerate(expected.items()):
        file = str(tmp_path / f"part-{i}.safetensors")
        save_file({name: tensor}, file)
        files.append(file)
    return files, expected


@pytest.mark.parametrize("concurrency", [1, 32])
def test_real_streamer_retention_empty_scalar_and_dtype(checkpoint, concurrency):
    pytest.importorskip("runai_model_streamer")
    files, expected = checkpoint
    metrics = {}
    actual = dict(stream_weights(files, checkpoint_metadata(files), concurrency=concurrency,
                  memory_limit=300000, owned_limit=600000, metrics=metrics))
    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name
    assert metrics["complete"] and metrics["empty_tensors"] == 1
    assert metrics["bytes"] == sum(t.numel() * t.element_size() for t in expected.values())


def test_storage_budget_follows_alias_not_python_tensor_lifetime():
    budget = OwnedStorageBudget(16)
    tensor = torch.ones(4)
    alias = tensor[1:]
    budget.track(tensor)
    del tensor
    gc.collect()
    with pytest.raises(MemoryError, match="retained"):
        budget.reserve(4)
    del alias
    gc.collect()
    budget.reserve(16)
    assert budget.live == 0
    budget.close()


def test_metadata_rejects_duplicate_and_truncated_files(checkpoint):
    files, _ = checkpoint
    with pytest.raises(ValueError, match="duplicate checkpoint file"):
        checkpoint_metadata([files[0], files[0]])
    Path(files[0]).write_bytes(b"invalid")
    with pytest.raises(ValueError, match="truncated"):
        checkpoint_metadata(files)


class FakeStreamer:
    rows = []
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def stream_files(self, *args, **kwargs): pass
    def get_tensors(self): yield from self.rows


def test_missing_nonempty_and_mutating_source_fail(checkpoint):
    files, expected = checkpoint
    metadata = checkpoint_metadata(files)
    FakeStreamer.rows = []
    with pytest.raises(ValueError, match="missing streamed tensor"):
        list(stream_weights(files, metadata, streamer_factory=FakeStreamer))
    FakeStreamer.rows = list(expected.items())
    iterator = stream_weights(files, metadata, streamer_factory=FakeStreamer)
    next(iterator)
    stat = os.stat(files[0])
    os.utime(files[0], ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="changed during streaming"):
        list(iterator)


def test_schema_failure_and_early_close_restore_environment(checkpoint, monkeypatch):
    files, expected = checkpoint
    metadata = checkpoint_metadata(files)
    monkeypatch.setenv("RUNAI_STREAMER_CONCURRENCY", "9")
    FakeStreamer.rows = [("packed", torch.ones(7))]
    with pytest.raises(ValueError, match="schema mismatch"):
        list(stream_weights(files, metadata, streamer_factory=FakeStreamer))
    assert os.environ["RUNAI_STREAMER_CONCURRENCY"] == "9"
    FakeStreamer.rows = list(expected.items())
    metrics = {}
    iterator = stream_weights(files, metadata, metrics=metrics, streamer_factory=FakeStreamer)
    first = next(iterator)
    iterator.close()
    assert not metrics.get("complete")
    assert torch.equal(first[1], expected[first[0]])
    assert os.environ["RUNAI_STREAMER_CONCURRENCY"] == "9"


def test_skip_and_insufficient_buffer(checkpoint):
    files, expected = checkpoint
    FakeStreamer.rows = list(expected.items())
    meta = checkpoint_metadata(files)
    actual = dict(stream_weights(files, meta, streamer_factory=FakeStreamer,
                                skip=lambda name: name in ("packed", "empty")))
    assert actual.keys() == expected.keys() - {"packed", "empty"}
    with pytest.raises(ValueError, match="largest checkpoint tensor"):
        list(stream_weights(files, meta, memory_limit=16, streamer_factory=FakeStreamer))


def test_prefetch_complete_failure_and_cancel(checkpoint, monkeypatch):
    files, expected = checkpoint
    meta = checkpoint_metadata(files)
    monkeypatch.setenv("RUNAI_STREAMER_CONCURRENCY", "11")
    FakeStreamer.rows = list(expected.items())
    source = stream_weights(files, meta, streamer_factory=FakeStreamer)
    assert dict(prefetch_weights(source)).keys() == expected.keys()
    FakeStreamer.rows = [("packed", torch.zeros(3))]
    with pytest.raises(ValueError, match="schema mismatch"):
        list(prefetch_weights(stream_weights(files, meta, streamer_factory=FakeStreamer)))
    FakeStreamer.rows = list(expected.items())
    iterator = prefetch_weights(stream_weights(files, meta, streamer_factory=FakeStreamer))
    first = next(iterator)
    iterator.close()
    assert torch.equal(first[1], expected[first[0]])
    assert os.environ["RUNAI_STREAMER_CONCURRENCY"] == "11"


def test_source_symlink_replacement_is_detected(checkpoint, tmp_path):
    files, expected = checkpoint
    link = tmp_path / "selected.safetensors"
    link.symlink_to(files[0])
    metadata = checkpoint_metadata([str(link)])
    FakeStreamer.rows = [("packed", expected["packed"])]
    iterator = stream_weights([str(link)], metadata, streamer_factory=FakeStreamer)
    next(iterator)
    link.unlink()
    link.symlink_to(files[1])
    with pytest.raises(ValueError, match="changed during streaming"):
        list(iterator)


def test_prefetch_cleanup_failure_is_not_reported_as_success():
    class BadCleanup:
        def __next__(self): raise StopIteration
        def close(self): raise RuntimeError("teardown failed")
    with pytest.raises(RuntimeError, match="teardown failed"):
        list(prefetch_weights(BadCleanup()))
