"""Inspect first output divergences using recorded target token probabilities.

This describes observations at a shared token prefix. It cannot establish the
underlying source of numerical variation or repair a failed functional check.
"""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path


def record(row):
    probabilities, prompts = [], []
    for chunk in row['chunks']:
        data = chunk['data']
        if data.get('prompt_token_ids'):
            prompts.append(data['prompt_token_ids'])
        for choice in data.get('choices', []):
            probabilities.extend((choice.get('logprobs') or {}).get('content') or [])
    if not probabilities:
        return None
    if len(probabilities) != len(row['token_ids']):
        raise ValueError('logprob/token ID accounting differs for ' + row['label'])
    if not prompts or any(p != prompts[0] for p in prompts):
        raise ValueError('missing or inconsistent exact prompt token IDs')
    return {'label': row['label'], 'token_ids': row['token_ids'], 'probabilities': probabilities,
            'prompt_sha256': hashlib.sha256(json.dumps(prompts[0]).encode()).hexdigest()}


def first_divergence(a, b):
    if a['prompt_sha256'] != b['prompt_sha256']:
        raise ValueError('different prompt token sequences')
    pair = {'labels': [a['label'], b['label']]}
    index = next((i for i, (x, y) in enumerate(zip(a['token_ids'], b['token_ids'])) if x != y), None)
    if index is None:
        return {**pair, 'first_divergence': None, 'identical_output_ids': a['token_ids'] == b['token_ids']}
    observations = []
    for row in (a, b):
        probability = row['probabilities'][index]
        top = sorted(probability.get('top_logprobs') or [], key=lambda x: x['logprob'], reverse=True)
        observations.append({'label': row['label'], 'selected_token_id': row['token_ids'][index],
                             'selected_logprob': probability['logprob'], 'top_logprobs': top,
                             'top_two_logprob_gap': top[0]['logprob'] - top[1]['logprob'] if len(top) >= 2 else None})
    return {**pair, 'first_divergence': index, 'shared_output_prefix_tokens': index,
            'observations': observations}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    args = ap.parse_args()
    rows, sources = [], {}
    for path in sorted(args.directory.glob('*.json')):
        row = json.loads(path.read_text())
        if not isinstance(row, dict) or not {'chunks', 'label', 'token_ids'} <= row.keys():
            continue
        converted = record(row)
        if converted:
            rows.append(converted)
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(rows) < 2:
        raise ValueError('need at least two complete probability-bearing requests')
    result = {'source_sha256': sources, 'pairs': [first_divergence(a, b) for a, b in combinations(rows, 2)],
              'note': 'Reported probabilities are observations at identical prompt/output prefixes, not proof of a numerical root cause. Logprob capture changes diagnostic overhead; these rates are excluded from performance comparisons.'}
    (args.directory / 'logprob-divergence-report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'requests': len(rows), 'divergent_pairs': sum(p['first_divergence'] is not None for p in result['pairs'])}))


if __name__ == '__main__':
    main()
