#!/usr/bin/env python3
"""Read safetensors headers only; report source layout without reading weights.

This is an investigation utility, not a production format validator or TP planner.
Uses only the standard library so it can run on a serving host without CUDA.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct


def inspect_checkpoint(directory):
    root = Path(directory).resolve()
    index_path = root / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"] if index_path.exists() else None
    files = sorted(set(weight_map.values())) if weight_map is not None else sorted(
        p.name for p in root.glob("*.safetensors"))
    if not files:
        raise ValueError("no safetensors files")
    sizes, tensors, dtype_bytes = [], [], Counter()
    metadata_bytes = 0
    seen = set()
    expert_bytes = expert_count = 0
    examples = {}
    for name in files:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("shard outside model directory")
        size = path.stat().st_size
        sizes.append(size)
        with path.open("rb") as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise ValueError("truncated header length")
            length = struct.unpack("<Q", prefix)[0]
            if length > min(100_000_000, size - 8):
                raise ValueError("invalid header length")
            raw = f.read(length)
            if len(raw) != length:
                raise ValueError("truncated header")
            header = json.loads(raw)
        metadata_bytes += 8 + length
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if key in seen or (weight_map is not None and weight_map.get(key) != name):
                raise ValueError("duplicate or index/header mismatch")
            seen.add(key)
            start, end = value["data_offsets"]
            if not 0 <= start <= end <= size - 8 - length:
                raise ValueError("invalid tensor offsets")
            n = end - start
            tensors.append((n, key, name, value["shape"], value["dtype"]))
            dtype_bytes[value["dtype"]] += n
            if ".experts." in key:
                expert_count += 1
                expert_bytes += n
                # Names only label examples; no model semantics are inferred.
                suffix = key.split(".experts.", 1)[1].split(".", 1)[-1]
                examples.setdefault(suffix, {"name": key, "shape": value["shape"],
                                            "dtype": value["dtype"], "bytes": n})
    if weight_map is not None and seen != set(weight_map):
        raise ValueError("index tensors missing from shards")
    return {"directory": str(root), "shards": len(files), "file_bytes": sum(sizes),
            "header_bytes_read": metadata_bytes, "tensors": len(tensors),
            "tensor_bytes": sum(t[0] for t in tensors),
            "shard_bytes_min": min(sizes), "shard_bytes_max": max(sizes),
            "dtype_bytes": dict(dtype_bytes), "expert_named_tensors": expert_count,
            "expert_named_bytes": expert_bytes, "expert_examples": examples,
            "largest_tensors": [{"bytes": n, "name": key, "file": file,
                                 "shape": shape, "dtype": dtype}
                                for n, key, file, shape, dtype in sorted(tensors, reverse=True)[:8]],
            "payload_read": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directories", nargs="+")
    args = p.parse_args()
    print(json.dumps([inspect_checkpoint(d) for d in args.directories], indent=2))
