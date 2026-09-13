"""Byte correctness and failure boundaries, runnable without vLLM/CUDA."""
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime/nvme_loader"))
from spark_nvme.artifact import digest, inspect, publish, restore
from spark_nvme.identity import prepare_source, verify_source
from spark_nvme.transport import read_exact


class Aliased(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = torch.arange(10500, dtype=torch.float32).reshape(100, 105)
        self.weight = torch.nn.Parameter(base)
        self.tied = self.weight
        self.register_buffer("transpose", base.T, persistent=False)
        self.register_buffer("slice", base[2:8, 3:19])
        self.register_buffer("empty", torch.empty(0))
        self.register_buffer("scalar", torch.tensor(2.5))


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("depth", [1, 7])
def test_round_trip_aliases_strides_and_chunk_tail(tmp_path, direct, depth):
    m = Aliased()
    expected = m.weight.detach().clone()
    publish(m, tmp_path / "weights", {"model": "a"}, chunk_bytes=8192)
    with torch.no_grad():
        m.weight.fill_(-100)
        m.scalar.zero_()
    metrics = restore(m, tmp_path / "weights", {"model": "a"}, depth=depth, direct=direct)
    assert metrics["bytes"] == 10500 * 4 + 4
    assert torch.equal(m.weight, expected)
    assert torch.equal(m.transpose, expected.T)
    assert torch.equal(m.slice, expected[2:8, 3:19])
    assert m.scalar.item() == 2.5
    assert m.tied is m.weight


def test_compatibility_and_truncation_rejected_before_writes(tmp_path):
    m = Aliased()
    path = tmp_path / "weights"
    publish(m, path, {"model": "a"})
    before = m.weight.detach().clone()
    with pytest.raises(ValueError, match="contract"):
        restore(m, path, {"model": "b"})
    with (path / "weights.bin").open("r+b") as f:
        f.truncate(5)
    with pytest.raises(ValueError, match="file size"):
        restore(m, path, {"model": "a"})
    assert torch.equal(m.weight, before)


def test_payload_and_manifest_corruption(tmp_path):
    m = Aliased()
    path = tmp_path / "weights"
    manifest = publish(m, path, {})
    with (path / "weights.bin").open("r+b") as f:
        f.seek(manifest["chunks"][-1]["offset"]); f.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        restore(m, path, {}, depth=3)
    manifest = path / "manifest.json"
    data = json.loads(manifest.read_text())
    data["schema"].clear()
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        inspect(path, {})


def test_schema_rejection_and_atomic_publication(tmp_path):
    m = Aliased()
    path = tmp_path / "weights"
    with pytest.raises(ValueError, match="schema"):
        publish(m, path, {}, expected_schema={})
    assert not path.exists()
    publish(m, path, {})
    m.tied = torch.nn.Parameter(m.weight.detach().clone())
    with pytest.raises(ValueError, match="schema"):
        restore(m, path, {})
    with pytest.raises(FileExistsError):
        publish(m, path, {})


def test_short_read_retry_and_eof(monkeypatch):
    calls = []
    def read(fd, buffers, offset):
        calls.append(offset)
        n = min(len(buffers[0]), 2)
        buffers[0][:n] = b"x" * n
        return n
    monkeypatch.setattr("os.preadv", read)
    out = bytearray(5)
    read_exact(9, memoryview(out), 20)
    assert out == b"xxxxx" and calls == [20, 22, 24]
    monkeypatch.setattr("os.preadv", lambda *a: 0)
    with pytest.raises(EOFError):
        read_exact(9, memoryview(out), 0)


def test_source_identity_hashes_bytes_and_detects_replacement(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    weights = source / "weights.safetensors"
    weights.write_bytes(b"weights")
    identity = tmp_path / "identity.json"
    content = prepare_source(source, identity)
    assert verify_source(source, identity) == content["content_id"]
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_source(source, identity)


def test_loader_side_effect_guard_and_mutable_flags():
    from spark_nvme.loader import object_state, check_side_effects, scalar_state, scalar_delta
    m = torch.nn.Linear(3, 5)
    m.flags = [1]
    before, scalars = object_state(m), scalar_state(m)
    m.flags.append(2)
    check_side_effects(before, object_state(m), m)
    assert scalar_delta(scalars, scalar_state(m)) == {"": {"flags": [1, 2]}}
    m._derived_weight = torch.ones(3)
    with pytest.raises(ValueError, match="adapter required"):
        check_side_effects(before, object_state(m), m)


def _gate_worker(rank, root):
    from datetime import timedelta
    from spark_nvme.loader import activation_gate
    torch.distributed.init_process_group("gloo", init_method="file://" + root + "/rendezvous",
                                         rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        for kind in ("error", "identity"):
            receipt = {"rank": rank, "generation": "g", "error": None,
                       "content_id": "a", "compatibility_id": "compatible"}
            if rank == 1:
                if kind == "error":
                    receipt["error"] = "rank 1: checksum mismatch"
                else:
                    receipt["content_id"] = "b"
            try:
                activation_gate(receipt, torch.distributed.group.WORLD)
            except ValueError as e:
                Path(root, f"rank-{rank}-{kind}").write_text(str(e))
            else:
                raise AssertionError("activation admitted a failed/mismatched rank")
    finally:
        torch.distributed.destroy_process_group()


def test_distributed_error_and_identity_gate(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_gate_worker, args=(str(tmp_path),), nprocs=2, join=True)
    for rank in (0, 1):
        assert "rank 1: checksum mismatch" in (tmp_path / f"rank-{rank}-error").read_text()
        assert "identity mismatch" in (tmp_path / f"rank-{rank}-identity").read_text()
