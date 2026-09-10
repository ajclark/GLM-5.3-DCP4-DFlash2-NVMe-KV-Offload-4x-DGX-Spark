#!/usr/bin/env python3
"""CPU-only RTN W8A16 overlays; never rewrites or copies original shards.

The pinned loader needs dense_indexed_loader.register() for mixed old/new
shards: its default safetensors iterator filters files, not tensor names.
See docs/research/DENSE-BYTES-PLAN.md for layout sources and admission gates.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import struct

INDEX = 'model.safetensors.index.json'
MANIFEST = 'dense-repack-manifest.json'
# Limit each full-width scratch array to <=1 MiB at float32/int32.
ROW_VALUES = 256 * 1024
DTYPE_BYTES = {'BF16': 2, 'F16': 2, 'F32': 4, 'F64': 8, 'I8': 1, 'U8': 1,
               'I16': 2, 'I32': 4, 'I64': 8, 'BOOL': 1, 'F8_E4M3': 1, 'F8_E5M2': 1}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def header(path):
    """Read only safetensors metadata; validate offsets before any payload read."""
    path = Path(path)
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated safetensors length')
        size = struct.unpack('<Q', prefix)[0]
        if not 2 <= size <= 64 * 1024**2:
            raise ValueError('safetensors header exceeds bound')
        raw = stream.read(size)
    if len(raw) != size:
        raise ValueError('truncated safetensors header')
    data = json.loads(raw, object_pairs_hook=unique_object)
    entries = {key: value for key, value in data.items() if key != '__metadata__'}
    end = 0
    for name, info in sorted(entries.items(), key=lambda item: item[1]['data_offsets']):
        shape, offsets = info['shape'], info['data_offsets']
        if (info['dtype'] not in DTYPE_BYTES or any(type(n) is not int or n < 0 for n in shape)
                or len(offsets) != 2 or any(type(n) is not int for n in offsets)
                or offsets[0] != end or offsets[1] - offsets[0] != math.prod(shape) * DTYPE_BYTES[info['dtype']]):
            raise ValueError('invalid safetensors tensor: ' + name)
        end = offsets[1]
    if path.stat().st_size != 8 + size + end:
        raise ValueError('safetensors payload length mismatch')
    return entries, 8 + size, hashlib.sha256(prefix + raw).hexdigest()


def shard_path(model, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.safetensors':
        raise ValueError('unsafe shard path: ' + name)
    return Path(model) / relative


def tensor_bytes(info):
    return info['data_offsets'][1] - info['data_offsets'][0]


def block_rows(n, k, requested=32):
    if k > ROW_VALUES:
        raise ValueError('matrix row exceeds bounded RTN scratch width')
    return min(n, requested, ROW_VALUES // k)


def load_tensor(path, name):
    """One tensor, one bounded read; no mmap pages retained from whole shards."""
    import torch
    entries, start, _ = header(path)
    info = entries[name]
    dtypes = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32,
              'I32': torch.int32, 'I64': torch.int64}
    if info['dtype'] not in dtypes:
        raise ValueError('unsupported conversion dtype: ' + info['dtype'])
    with Path(path).open('rb') as stream:
        stream.seek(start + info['data_offsets'][0])
        raw = bytearray(tensor_bytes(info))
        if stream.readinto(raw) != len(raw):
            raise ValueError('truncated tensor payload')
    return torch.frombuffer(raw, dtype=dtypes[info['dtype']]).reshape(info['shape'])


def write_tensors(path, tensors):
    """Serialize standard safetensors without a second copy of tensor payloads."""
    dtype_names = {'torch.bfloat16': 'BF16', 'torch.float16': 'F16', 'torch.float32': 'F32',
                   'torch.int32': 'I32', 'torch.int64': 'I64'}
    metadata, offset = {'__metadata__': {'format': 'pt'}}, 0
    for name, tensor in tensors.items():
        if tensor.device.type != 'cpu' or not tensor.is_contiguous():
            raise ValueError('writer requires contiguous CPU tensors')
        size = tensor.numel() * tensor.element_size()
        metadata[name] = {'dtype': dtype_names[str(tensor.dtype)], 'shape': list(tensor.shape),
                          'data_offsets': [offset, offset + size]}
        offset += size
    raw = json.dumps(metadata, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    with Path(path).open('xb') as stream:
        stream.write(struct.pack('<Q', len(raw)))
        stream.write(raw)
        for tensor in tensors.values():
            import torch
            stream.write(memoryview(tensor.view(torch.uint8).numpy()).cast('B'))


def quantize(weight, scheme, scale_dtype=None, row_block=32):
    """Round-to-nearest-even, symmetric [-127,127], stored as q+128 in LE lanes.

    Quantize against the rounded, serialized scale, so RTN and the loader's
    dequantization use exactly the same scale. Zero groups use scale one.
    Scratch is row-block bounded; only input and packed result are full-sized.
    """
    import torch
    if weight.device.type != 'cpu' or weight.ndim != 2 or row_block < 1:
        raise ValueError('expected a CPU matrix and a positive row block')
    n, k = weight.shape
    group = k if scheme == 'per-channel' else 128 if scheme == 'group-128' else 0
    if not group or not n or not k or k % 4 or k % group:
        raise ValueError('K must divide packing factor four and the requested group size')
    row_block = block_rows(n, k, row_block)
    scale_dtype = scale_dtype or weight.dtype
    if scale_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError('scales require a floating-point dtype')
    packed = torch.empty((n, k // 4), dtype=torch.int32)
    scales = torch.empty((n, k // group), dtype=scale_dtype)
    for begin in range(0, n, row_block):
        end = min(n, begin + row_block)
        values = weight[begin:end].float().reshape(end - begin, k // group, group)
        if not torch.isfinite(values).all():
            raise ValueError('cannot quantize nonfinite weights')
        maximum = values.abs().amax(dim=-1)
        scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127)
        scale = scale.to(scale_dtype)
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError('scale overflow/underflow; use a wider checkpoint dtype')
        scales[begin:end] = scale
        quantized = (values / scale.float().unsqueeze(-1)).round().clamp(-127, 127).to(torch.int32)
        lanes = (quantized.reshape(end - begin, k) + 128).reshape(end - begin, k // 4, 4)
        packed[begin:end] = lanes[..., 0] | (lanes[..., 1] << 8) | (lanes[..., 2] << 16) | (lanes[..., 3] << 24)
    return {'weight_packed': packed, 'weight_scale': scales,
            'weight_shape': torch.tensor([n, k], dtype=torch.int64)}


def matches(name, pattern):
    return bool(re.match(pattern[3:], name)) if pattern.startswith('re:') else name == pattern


def update_config(config, modules, scheme):
    result = copy.deepcopy(config)
    quant = result.setdefault('quantization_config', {})
    if quant.get('quant_method', 'compressed-tensors') != 'compressed-tensors' or quant.get('format', 'pack-quantized') != 'pack-quantized':
        raise ValueError('requires compressed-tensors pack-quantized config')
    # Subtract only selected modules from each broad ignore regex. This keeps
    # layer zero's OTHER projections, indexer norms/WK, routers, and MTP intact.
    exception = '(?!(?:' + '|'.join(re.escape(name) for name in sorted(modules)) + r')(?:$|\.))'
    ignored = []
    for pattern in quant.get('ignore', []):
        if not any(matches(name, pattern) for name in modules):
            ignored.append(pattern)
        elif pattern.startswith('re:'):
            ignored.append('re:' + exception + '(?:' + pattern[3:] + ')')
        elif pattern not in modules:
            ignored.append(pattern)
    groups = quant.get('config_groups', {})
    if 'dense_repack_w8a16' in groups:
        raise ValueError('output already contains a dense repack scheme')
    quant.update(quant_method='compressed-tensors', format='pack-quantized',
                 quantization_status='compressed', ignore=ignored,
                 config_groups={'dense_repack_w8a16': {
                     'targets': sorted(modules),
                     'weights': {'num_bits': 8, 'type': 'int', 'symmetric': True,
                                 'strategy': 'channel' if scheme == 'per-channel' else 'group',
                                 'group_size': -1 if scheme == 'per-channel' else 128,
                                 'dynamic': False}}, **groups})
    return result


def plan(model, patterns, scheme, max_tensor_mib=256):
    model = Path(model).resolve()
    config, index = read_json(model / 'config.json'), read_json(model / INDEX)
    regexes = [re.compile(pattern) for pattern in patterns]
    if not regexes or scheme not in ('per-channel', 'group-128') or max_tensor_mib <= 0:
        raise ValueError('nonempty regex list, valid scheme and positive memory bound required')
    headers, selected, pattern_hits = {}, [], [0] * len(regexes)
    for shard in sorted(set(index['weight_map'].values())):
        headers[shard] = header(shard_path(model, shard))
    original_bytes = 0
    for name, shard in index['weight_map'].items():
        if name not in headers[shard][0]:
            raise ValueError('index points to missing tensor: ' + name)
        info = headers[shard][0][name]
        original_bytes += tensor_bytes(info)
        hits = [bool(regex.fullmatch(name)) for regex in regexes]
        pattern_hits = [count + hit for count, hit in zip(pattern_hits, hits)]
        if not any(hits):
            continue
        if not name.endswith('.weight') or info['dtype'] not in ('BF16', 'F16', 'F32') or len(info['shape']) != 2:
            raise ValueError('selection must contain only unpacked floating matrices: ' + name)
        n, k = info['shape']
        group = k if scheme == 'per-channel' else 128
        if not n or not k or k % 4 or k % group:
            raise ValueError('incompatible packing/group shape: ' + name)
        scratch_bytes = block_rows(n, k) * k * 64
        if tensor_bytes(info) > max_tensor_mib * 1024**2:
            raise ValueError('tensor exceeds --max-tensor-mib: ' + name)
        # Exact on-disk scale dtype follows checkpoint execution dtype.
        dtype = {'bfloat16': 'BF16', 'float16': 'F16', 'float32': 'F32'}.get(
            config.get('dtype', config.get('torch_dtype')), info['dtype'])
        new_bytes = n * k + n * (k // group) * DTYPE_BYTES[dtype] + 16
        selected.append({'name': name, 'source_shard': shard, 'shape': [n, k],
                         'source_dtype': info['dtype'], 'scale_dtype': dtype,
                         'old_bytes': tensor_bytes(info), 'new_bytes': new_bytes,
                         'numeric_buffer_budget_bytes': tensor_bytes(info) + new_bytes + scratch_bytes,
                         'saved_bytes': tensor_bytes(info) - new_bytes})
        for suffix in ('weight_packed', 'weight_scale', 'weight_shape'):
            if name[:-6] + suffix in index['weight_map']:
                raise ValueError('destination tensor already exists: ' + name)
    if not all(pattern_hits):
        raise ValueError('every regex must match at least one tensor')
    selected.sort(key=lambda row: row['name'])
    modules = {row['name'][:-7] for row in selected}
    updated = update_config(config, modules, scheme)
    # Preserve the ignore decision for every unselected checkpoint module.
    for name in index['weight_map']:
        module = name.rsplit('.', 1)[0]
        before = any(matches(module, p) for p in config.get('quantization_config', {}).get('ignore', []))
        after = any(matches(module, p) for p in updated['quantization_config']['ignore'])
        if (module in modules and after) or (module not in modules and before != after):
            raise ValueError('ignore edit changed an unselected module: ' + module)
    old_bytes, new_bytes = sum(r['old_bytes'] for r in selected), sum(r['new_bytes'] for r in selected)
    report = {'format_version': 1, 'scheme': scheme, 'patterns': patterns, 'model': str(model),
              'selected': selected, 'selected_tensors': len(selected),
              'old_bytes': old_bytes, 'new_bytes': new_bytes, 'saved_bytes': old_bytes - new_bytes,
              'saved_ms_at_240GBs_if_replicated_and_used_once': (old_bytes - new_bytes) / 240e6,
              'original_logical_bytes': original_bytes, 'new_logical_bytes': original_bytes - old_bytes + new_bytes,
              'additional_tensor_disk_bytes': new_bytes, 'original_shards_copied': 0,
              'max_input_tensor_bytes': max(r['old_bytes'] for r in selected),
              'max_packed_tensor_bytes': max(r['new_bytes'] for r in selected),
              'max_numeric_buffer_budget_bytes': max(r['numeric_buffer_budget_bytes'] for r in selected),
              'requires_index_filtered_loader': True,
              'note': 'Source shards retain replaced names physically. Use dense_indexed_loader; the pinned default loader ignores per-tensor index filtering. Bytes exclude JSON headers; bandwidth figures are accounting, not measured savings.',
              'config_sha256': hashlib.sha256((model / 'config.json').read_bytes()).hexdigest(),
              'index_sha256': hashlib.sha256((model / INDEX).read_bytes()).hexdigest(),
              'source_header_sha256': {shard: data[2] for shard, data in headers.items()}}
    return report, updated, index


def sandbox_only():
    if re.search(r'(^|[.-])spark', socket.gethostname(), re.I) or platform.machine().lower() in ('aarch64', 'arm64'):
        raise RuntimeError('repacking is sandbox-only; refusing Spark/ARM hosts')


def repack(model, patterns, scheme, out, *, dry_run=False, max_tensor_mib=256):
    sandbox_only()
    report, config, index = plan(model, patterns, scheme, max_tensor_mib)
    if dry_run:
        return report
    import torch
    torch.set_num_threads(1)
    model, requested_out = Path(model).resolve(), Path(out).absolute()
    out = requested_out.resolve()
    if requested_out.exists() or requested_out.is_symlink() or model == out or model in out.parents:
        raise ValueError('output must be a new directory outside the source model')
    weight_map = dict(index['weight_map'])
    # No checkpoint or source metadata is ever opened for writing.
    out.mkdir(parents=True, exist_ok=False)
    try:
        for i, row in enumerate(report['selected']):
            path = shard_path(model, row['source_shard'])
            if header(path)[2] != report['source_header_sha256'][row['source_shard']]:
                raise ValueError('source header changed during repack')
            weight = load_tensor(path, row['name'])
            result = quantize(weight, scheme, {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32}[row['scale_dtype']])
            stem = row['name'][:-6]
            packed = {stem + suffix: tensor for suffix, tensor in result.items()}
            shard = f'dense-int8-{i:05d}.safetensors'
            if shard in index['weight_map'].values():
                raise ValueError('new shard name collides with source')
            write_tensors(out / shard, packed)
            row['output_shard'] = shard
            row['output_sha256'] = file_digest(out / shard)
            row['source_tensor_sha256'] = hashlib.sha256(memoryview(weight.contiguous().view(torch.uint8).numpy()).cast('B')).hexdigest()
            del weight_map[row['name']]
            weight_map.update({name: shard for name in packed})
            del weight, result, packed
        original_shards = set(index['weight_map'].values()) & set(weight_map.values())
        for shard in sorted(original_shards):
            destination = out / shard
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(shard_path(model, shard))
        # Keep tokenizer/generation metadata available without copying shards.
        for path in model.iterdir():
            if path.is_file() and path.name not in ('config.json', INDEX, MANIFEST) and path.suffix in ('.json', '.model', '.txt', '.jinja'):
                (out / path.name).symlink_to(path)
        (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        new_index = {**index, 'metadata': {**index.get('metadata', {}), 'total_size': report['new_logical_bytes']},
                     'weight_map': dict(sorted(weight_map.items()))}
        (out / INDEX).write_text(json.dumps(new_index, indent=2) + '\n')
        report['symlinked_shards'] = sorted(original_shards)
        (out / MANIFEST).write_text(json.dumps(report, indent=2) + '\n')
    except BaseException:
        shutil.rmtree(out)  # Only the new directory created by this invocation.
        raise
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model', type=Path, help='Local sandbox model directory')
    ap.add_argument('--tensor-regex', action='append', required=True, help='Full-match tensor name regex; repeat for multiple patterns')
    ap.add_argument('--scheme', choices=('per-channel', 'group-128'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dry-run', action='store_true', help='Headers/config only; write nothing and import no torch')
    ap.add_argument('--max-tensor-mib', type=int, default=256, help='Reject larger source tensors before allocating')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        print(json.dumps(repack(args.model, args.tensor_regex, args.scheme, args.out,
                                dry_run=args.dry_run, max_tensor_mib=args.max_tensor_mib), indent=2))
    except (ValueError, RuntimeError) as exc:
        ap.error(str(exc))


if __name__ == '__main__':
    main()
