"""Real CUDA fences, retained batch aliases, and oversized-tensor direct reads."""
import gc
import json
from pathlib import Path
import tempfile

import torch
from safetensors.torch import save_file
from spark_nvme.streaming import checkpoint_metadata
from spark_nvme.coalesced import coalesced_weights, RangeReader


with tempfile.TemporaryDirectory(dir="/nvme-artifacts") as root:
    files = []
    expected = {}
    for shard in range(3):
        rows = {f"{shard}.packed": torch.arange(2_000_003, dtype=torch.int32),
                f"{shard}.scale": torch.arange(513).to(torch.bfloat16),
                f"{shard}.scalar": torch.tensor(1.25),
                f"{shard}.empty": torch.empty(0, 3)}
        rows.update({f"{shard}.small.{i}": torch.arange(64, dtype=torch.int64) + i for i in range(40)})
        path = str(Path(root) / f"part-{shard}.safetensors")
        save_file(rows, path)
        files.append(path)
        expected.update(rows)
    meta = checkpoint_metadata(files)
    original_fill = RangeReader.fill
    pending_events = []
    def delayed_fill(reader, batch, output):
        # Force copies to remain pending when published, so a missing consumer
        # dependency cannot be hidden by a fast copy completing before dequeue.
        torch.cuda._sleep(20_000_000)
        event = original_fill(reader, batch, output)
        if event is not None:
            pending_events.append(not event.query())
        return event
    RangeReader.fill = delayed_fill
    consumer = torch.cuda.Stream()
    metrics = {}
    with torch.cuda.stream(consumer):
        actual = dict(coalesced_weights(files, meta, device="cuda", concurrency=32,
                      batch_bytes=1 << 20, memory_limit=512 << 10, owned_limit=64 << 20,
                      metrics=metrics))
        copies = {name: tensor.clone() for name, tensor in actual.items()}
        del actual
        gc.collect()
        # Put allocator pressure behind the recorded consumer work.
        temporary = torch.empty(32 << 20, device="cuda", dtype=torch.uint8).fill_(219)
        for name, tensor in copies.items():
            assert torch.equal(tensor.cpu(), expected[name]), name
        del temporary, copies
    consumer.synchronize()
    with torch.cuda.stream(consumer):
        iterator = coalesced_weights(files, meta, device="cuda", batch_bytes=1 << 20,
                                     memory_limit=512 << 10, owned_limit=64 << 20)
        name, tensor = next(iterator)
        alias = tensor.reshape(-1)
        del tensor
        iterator.close()
        assert torch.equal(alias.cpu(), expected[name].reshape(-1))
    consumer.synchronize()
    assert metrics["complete"]
    assert metrics["reader_buffer_bytes"] <= 512 << 10
    assert metrics["peak_owned_bytes"] <= 64 << 20
    assert any(pending_events), "test did not exercise an asynchronous event handoff"
    print(json.dumps({"passed": True, "cases": len(expected), "pending_event_handoffs": sum(pending_events),
                      "metrics": metrics}), flush=True)
