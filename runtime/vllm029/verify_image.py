#!/usr/bin/env python3
"""Verify every installed overlay before a cluster rollout (no CUDA imports)."""
import hashlib
from importlib.metadata import version
import json
import sys
from pathlib import Path

manifest=json.loads(Path('/opt/spark-vllm-manifest.json').read_text())
if version('vllm') != manifest['version']:
    raise SystemExit('Installed vLLM version differs from manifest')
root=Path('/usr/local/lib/python3.12/dist-packages/vllm')
for name,row in manifest['files'].items():
    expected=row['upstream_sha256'] if '--base' in sys.argv else row['sha256']
    if expected is None:
        continue
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
        raise SystemExit('Installed overlay mismatch: '+name)
for package,expected in manifest.get('dependency_versions',{}).items():
    if version(package)!=expected:
        raise SystemExit('Installed dependency version mismatch: '+package)
for name,row in manifest.get('dependency_files',{}).items():
    expected=row['upstream_sha256'] if '--base' in sys.argv else row['sha256']
    if hashlib.sha256((root.parent/name).read_bytes()).hexdigest()!=expected:
        raise SystemExit('Installed dependency overlay mismatch: '+name)
print(json.dumps(manifest,sort_keys=True))
