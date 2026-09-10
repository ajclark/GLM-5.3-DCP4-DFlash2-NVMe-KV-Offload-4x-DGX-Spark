#!/usr/bin/env python3
"""Reproduce dense-byte accounting from saved headers; no checkpoint/GPU access."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

METADATA = Path(__file__).resolve().parents[1] / 'docs/research/dense-bytes-checkpoint-metadata.json'


def target_inventory(metadata, tp=4):
    rows = []
    for name, info in sorted(metadata['target']['tensors'].items()):
        layer_match = re.search(r'model\.layers\.(\d+)\.', name)
        layer = int(layer_match[1]) if layer_match else None
        mtp = layer is not None and layer >= metadata['target']['config']['num_hidden_layers']
        sharded = (name.startswith(('lm_head.', 'model.embed_tokens.')) or
                   bool(re.search(r'self_attn\.(?:q_b_proj|kv_b_proj|o_proj)\.|mlp\.(?:shared_experts\.)?(?:gate_proj|up_proj|down_proj)\.', name)))
        divisor = tp if sharded else 1
        if name.endswith('weight_shape') or (name.endswith('weight_scale') and info['shape'][-1] == 1 and
                                             ('.down_proj.' in name or '.o_proj.' in name)):
            divisor = 1
        resident = info['bytes'] // divisor
        stream, note = resident, 'one read of participating tensor'
        if mtp:
            family, stream, note = 'mtp_not_loaded', 0, 'main model skips layer 78; DFlash uses a separate checkpoint'
            resident = 0
        elif name.startswith('model.embed_tokens.'):
            family, stream, note = 'embedding_lookup', 0, 'lookup only; not a full matrix stream'
        elif '.kv_b_proj.' in name:
            family, stream, note = 'kv_b_checkpoint', 0, 'decode uses derived BF16 W_UK_T/W_UV instead'
        elif '.indexer.' in name:
            family = 'indexer_bf16'
        elif '.mlp.gate.' in name:
            family = 'router_bf16' if info['dtype'] == 'BF16' else 'router_bias_f32'
        elif name.startswith('lm_head.'):
            family = 'lm_head_bf16'
        elif layer == 0 and info['shape'] and len(info['shape']) == 2:
            family = 'layer0_bf16'
        elif name.endswith(('weight_packed', 'weight_scale', 'weight_shape')):
            family = 'dense_int8'
        else:
            family = 'norms_bf16'
        if name.endswith('weight_shape'):
            stream = 0
            note = 'loader shape metadata; no GEMM read'
        rows.append({'name': name, 'shape': info['shape'], 'dtype': info['dtype'],
                     'checkpoint_bytes': info['bytes'], 'tp_divisor': divisor, 'resident_bytes_per_rank': resident,
                     'stream_bytes_per_rank_cycle': stream, 'family': family, 'note': note})
    config = metadata['target']['config']
    h, latent = config['num_attention_heads'] // tp, config['kv_lora_rank']
    for layer in range(config['num_hidden_layers']):
        for suffix, shape in [('W_UK_T', [h, config['qk_nope_head_dim'], latent]),
                              ('W_UV', [h, latent, config['v_head_dim']])]:
            size = 2
            for n in shape:
                size *= n
            rows.append({'name': f'model.layers.{layer}.self_attn.mla_attn.{suffix}', 'shape': shape, 'dtype': 'BF16',
                         'resident_bytes_per_rank': size, 'stream_bytes_per_rank_cycle': size,
                         'family': 'kv_b_derived_bf16', 'note': 'dequantized/reshaped at load; not a second packed GEMM'})
    return rows


def draft_inventory(metadata, tp):
    rows = []
    for name, info in sorted(metadata['draft']['tensors'].items()):
        sharded = bool(re.search(r'(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$', name))
        divisor = tp if sharded else 1
        resident = info['bytes'] // divisor
        stream = 0 if name.endswith('_codebook') else resident
        rows.append({'name': name, 'shape': info['shape'], 'dtype': info['dtype'],
                     'tp_divisor': divisor, 'resident_bytes_per_rank': resident,
                     'stream_bytes_per_rank_cycle': stream,
                     'fp8_narrow_candidate': name == 'fc.weight' or bool(re.search(r'(?:self_attn\.o_proj|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$', name)),
                     'family': 'draft_checkpoint', 'note': 'codebooks are gathered, not streamed in full' if name.endswith('_codebook') else ''})
    cfg = metadata['draft']['config']
    fused = cfg['num_hidden_layers'] * 2 * (cfg['num_key_value_heads'] // tp) * cfg['head_dim'] * cfg['hidden_size'] * 2
    head = metadata['target']['tensors']['lm_head.weight']['bytes'] // tp
    rows += [{'name': '_fused_kv_weight', 'family': 'context_kv_copy', 'resident_bytes_per_rank': fused,
              'stream_bytes_per_rank_cycle': fused, 'note': 'context precompute reads an extra K/V copy each cycle'},
             {'name': 'lm_head.weight (shared target head)', 'family': 'shared_head', 'resident_bytes_per_rank': 0,
              'stream_bytes_per_rank_cycle': head, 'note': 'shared parameter, streamed again for draft candidate logits'}]
    return rows


def summarize(metadata):
    target = target_inventory(metadata)
    families = defaultdict(lambda: {'resident_bytes_per_rank': 0, 'stream_bytes_per_rank_cycle': 0})
    for row in target:
        for key in ('resident_bytes_per_rank', 'stream_bytes_per_rank_cycle'):
            families[row['family']][key] += row[key]
    for values in families.values():
        values['ms_at_240GBs'] = values['stream_bytes_per_rank_cycle'] / 240e6
    drafts = {}
    for tp in (4, 1):
        rows = draft_inventory(metadata, tp)
        stream = sum(r['stream_bytes_per_rank_cycle'] for r in rows)
        eligible = sum(r['stream_bytes_per_rank_cycle'] for r in rows if r.get('fp8_narrow_candidate'))
        drafts[str(tp)] = {'rows': rows, 'stream_bytes_per_rank_cycle': stream, 'ms_at_240GBs': stream / 240e6,
                           'narrow_fp8_saved_bytes_before_scales': eligible // 2,
                           'narrow_fp8_saved_ms_at_240GBs': eligible / 2 / 240e6}
    selected = [row for row in target if row['name'].endswith('.indexer.wq_b.weight') and row['stream_bytes_per_rank_cycle']]
    n, k = selected[0]['shape']
    return {'target_rows': target, 'target_families': dict(families), 'draft_tp': drafts,
            'indexer_wq_b': {'active_layers': len(selected), 'old_bytes': len(selected) * n * k * 2,
                             'per_channel_new_bytes': len(selected) * (n * k + n * 2),
                             'group128_new_bytes': len(selected) * (n * k + n * (k // 128) * 2)},
            'method': 'TP4 target; pinned V2 DFlash loader retains TP4, despite requested draft TP1. TP1 column is counterfactual. Matrix bytes count one full read per operation; excludes activation/KV/cache traffic, repeated tiles and cache reuse. Draft lookup tables/embeddings are not full-matrix streams. weight_shape is load metadata.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metadata', type=Path, default=METADATA)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    report = summarize(json.loads(args.metadata.read_text()))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'target_families': report['target_families'], 'indexer_wq_b': report['indexer_wq_b'],
                      'draft_tp': {tp: {k: v for k, v in data.items() if k != 'rows'} for tp, data in report['draft_tp'].items()}}, indent=2))


if __name__ == '__main__':
    main()
