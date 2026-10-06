#!/usr/bin/env python3
"""Vendored b12x must match PROVENANCE.json: every file byte-identical to upstream
b58f34ea except the reviewed local patches, whose current hashes are pinned too.

    python3 tests/test_vendored_b12x.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "b12x"


def main():
    prov = json.loads((ROOT / "PROVENANCE.json").read_text())
    on_disk = {str(p.relative_to(ROOT)) for p in (ROOT / "b12x").rglob("*") if p.is_file()
               and "__pycache__" not in p.parts}
    assert on_disk == set(prov["files"]), sorted(on_disk ^ set(prov["files"]))
    for rel, h in prov["files"].items():
        cur = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert cur == h["sha256"], f"{rel} changed: {cur}"
        if rel not in prov["local_patches"]:
            assert h["sha256"] == h["upstream_sha256"], f"{rel} differs from upstream without a recorded patch"
    print(f"vendored b12x ok: {len(on_disk)} files, local patches {sorted(prov['local_patches'])}")


if __name__ == "__main__":
    main()
