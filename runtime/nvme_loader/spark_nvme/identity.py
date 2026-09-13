"""Preparation-time content identity; startup verifies immutable source inventory."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

from .artifact import canonical, digest, fsync_directory


def source_inventory(path):
    root = Path(path)
    files = sorted(p for p in root.rglob("*") if p.is_file() and
                   p.suffix in (".safetensors", ".json", ".bin", ".pt", ".pth", ".gguf"))
    if not files:
        raise ValueError(f"no local checkpoint files at {root}")
    return [{"name": str(p.relative_to(root)), "size": p.stat().st_size,
             "mtime_ns": p.stat().st_mtime_ns} for p in files]


def prepare_source(path, output, workers=8):
    root = Path(path)
    before = source_inventory(root)

    def hash_file(entry):
        h = hashlib.sha256()
        with (root / entry["name"]).open("rb", buffering=0) as f:
            buffer = bytearray(4 << 20)
            view = memoryview(buffer)
            while (n := f.readinto(buffer)):
                h.update(view[:n])
                # Keep hashing from displacing the live model's small OS margin.
                os.posix_fadvise(f.fileno(), max(0, f.tell() - n), n, os.POSIX_FADV_DONTNEED)
        return {"name": entry["name"], "size": entry["size"], "sha256": h.hexdigest()}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashes = list(pool.map(hash_file, before))
    if source_inventory(root) != before:
        raise ValueError("checkpoint changed while content identity was prepared")
    identity = {"content_id": digest(hashes), "files": hashes, "inventory": before}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as f:
        f.write(canonical(identity)); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, output)
    fsync_directory(output.parent)
    return identity


def verify_source(path, identity_path):
    identity = json.loads(Path(identity_path).read_bytes())
    if digest(identity["files"]) != identity["content_id"]:
        raise ValueError("source identity manifest checksum mismatch")
    if source_inventory(path) != identity["inventory"]:
        raise ValueError("source checkpoint changed; prepare its content identity again")
    return identity["content_id"]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("source"); p.add_argument("output")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    print(prepare_source(a.source, a.output, a.workers)["content_id"], flush=True)
