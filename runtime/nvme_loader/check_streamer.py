#!/usr/bin/env python3
"""CPU-only, small-fixture check of the installed Run:ai tensor stream.

Re-shards fixture tensors into temporary files, exercises bounded streaming,
and checks every tensor after buffer reuse and streamer shutdown. This does
not benchmark storage or establish GPU model-loader parity.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile


def check(directory):
    import torch
    from safetensors.torch import load_file, save_file
    from runai_model_streamer import SafetensorsStreamer
    torch.set_num_threads(1)
    expected = {}
    for path in sorted(Path(directory).glob("*.safetensors")):
        for name, tensor in load_file(str(path)).items():
            if name in expected:
                raise ValueError("duplicate tensor")
            expected[name] = tensor
    if not expected:
        raise ValueError("no fixture tensors")
    largest = max(t.numel() * t.element_size() for t in expected.values())
    os.environ["RUNAI_STREAMER_MEMORY_LIMIT"] = str(2 * largest)
    os.environ["RUNAI_STREAMER_CONCURRENCY"] = "32"
    actual = {}
    with tempfile.TemporaryDirectory(prefix="nvme-stream-fixture-") as temporary:
        partitions = [{} for _ in range(5)]
        for i, (name, tensor) in enumerate(expected.items()):
            partitions[i % len(partitions)][name] = tensor.contiguous()
        files = []
        for i, partition in enumerate(partitions):
            if partition:
                path = str(Path(temporary) / f"part-{i}.safetensors")
                save_file(partition, path)
                files.append(path)
        with SafetensorsStreamer() as streamer:
            streamer.stream_files(files, device="cpu", is_distributed=False)
            for name, tensor in streamer.get_tensors():
                if name in actual:
                    raise AssertionError("duplicate yield")
                # Match installed vLLM's ownership boundary. A naked view can
                # be overwritten on the next iteration of a bounded streamer.
                actual[name] = tensor.clone()
        missing = sorted(expected.keys() - actual.keys())
        unexpected = sorted(actual.keys() - expected.keys())
        assert not unexpected, unexpected
        for name, tensor in actual.items():
            assert torch.equal(tensor, expected[name]), name
    return {"model": Path(directory).name, "tensors": len(actual),
            "shards": len(files), "configured_stream_buffer_bytes": 2 * largest,
            "payload_bytes": sum(t.numel() * t.element_size() for t in expected.values()),
            "concurrency": 32, "retained_clone_equality": True,
            "complete": not missing, "missing_tensors": missing, "device": "cpu"}


def check_mixed_dtypes():
    import torch
    from safetensors.torch import save_file
    with tempfile.TemporaryDirectory(prefix="mixed-checkpoint-") as root:
        path = Path(root) / "mixed-dtypes"
        path.mkdir()
        save_file({
            "weight_packed": torch.arange(1024 * 256, dtype=torch.int32).reshape(1024, 256),
            "weight_scale": torch.arange(1024 * 16, dtype=torch.float32).to(torch.bfloat16).reshape(1024, 16),
            "weight_shape": torch.tensor([1024, 2048], dtype=torch.int64),
            "scalar": torch.tensor(1.25, dtype=torch.float32),
            "empty": torch.empty(0, dtype=torch.float32),
        }, str(path / "model.safetensors"))
        return check(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directories", nargs="+")
    args = p.parse_args()
    print(json.dumps([check(d) for d in args.directories] + [check_mixed_dtypes()], indent=2))
