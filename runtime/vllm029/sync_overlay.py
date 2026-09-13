#!/usr/bin/env python3
"""Refresh the manifest after editing overlays; verify upstream bases on request."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--upstream', type=Path)
    args = parser.parse_args()
    path = ROOT / 'manifest.json'
    manifest = json.loads(path.read_text())
    shipped = {str(p.relative_to(ROOT / 'overlay/vllm'))
               for p in (ROOT / 'overlay/vllm').rglob('*.py')}
    if shipped != set(manifest['files']):
        raise SystemExit('overlay inventory differs from manifest')
    for rel, expected in manifest['files'].items():
        p = ROOT / 'overlay/vllm' / rel
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if args.check and digest != expected['sha256']:
            raise SystemExit(f'overlay hash mismatch: {rel}')
        expected['sha256'] = digest
        if args.upstream and expected['upstream_sha256']:
            base = args.upstream / 'vllm' / rel
            if hashlib.sha256(base.read_bytes()).hexdigest() != expected['upstream_sha256']:
                raise SystemExit(f'upstream base mismatch: {rel}')
    dependencies={str(p.relative_to(ROOT/'overlay')) for p in (ROOT/'overlay/flashinfer').rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts}
    if dependencies != set(manifest.get('dependency_files',{})):
        raise SystemExit('dependency overlay inventory differs from manifest')
    for rel,expected in manifest.get('dependency_files',{}).items():
        digest=hashlib.sha256((ROOT/'overlay'/rel).read_bytes()).hexdigest()
        if args.check and digest != expected['sha256']:
            raise SystemExit('dependency overlay hash mismatch: '+rel)
        expected['sha256']=digest
    if not args.check:
        path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Verified {len(manifest['files'])} vLLM {manifest['version']} and {len(dependencies)} dependency overlays")

if __name__ == '__main__':
    main()
