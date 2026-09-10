"""Local result loading shared by deterministic and blind prose evaluations."""
import hashlib
import json
from pathlib import Path


def variant_name(row):
    if row.get('variant'):
        return row['variant']
    policy = row.get('policy', 'fixed')
    return f"fixed{row['cap']}" if policy == 'fixed' else policy


def read_results(directory, variant=None, repeats=None, prose_only=True):
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text())
    prompts = config['prompts']
    rows = {}
    for path in sorted(directory.glob('*.json')):
        row = json.loads(path.read_text())
        if not isinstance(row, dict) or not {'case', 'repeat', 'text', 'cap'} <= row.keys():
            continue
        name = variant_name(row)
        if variant is not None and name != variant:
            continue
        if repeats is not None and row['repeat'] not in repeats:
            continue
        if prose_only and not row['case'].startswith('prose_'):
            continue
        key = row['case'], row['repeat'], name
        if key in rows:
            raise ValueError('duplicate completion: ' + str(key))
        if row['case'] not in prompts or not isinstance(row['text'], str):
            raise ValueError('missing prompt or invalid completion')
        prompt = prompts[row['case']]
        if isinstance(prompt, dict):
            prompt = prompt['prompt']
        token_prompts = [chunk['data']['prompt_token_ids'] for chunk in row.get('chunks', [])
                         if chunk['data'].get('prompt_token_ids')]
        if token_prompts:
            if any(ids != token_prompts[0] for ids in token_prompts):
                raise ValueError('inconsistent prompt token IDs')
            digest = hashlib.sha256(json.dumps(token_prompts[0]).encode()).hexdigest()
            if row.get('prompt_token_sha256', digest) != digest:
                raise ValueError('saved prompt digest differs from prompt token IDs')
            row['prompt_token_sha256'] = digest
        rows[key] = {**row, 'variant': name, 'prompt': prompt, 'source': str(path),
                     'comparison_settings': {key: value for key, value in {
                         'tokens': config.get('tokens'),
                         'thinking': (bool(config.get('thinking')) or row['repeat'] in config.get('thinking_repeat', []))
                                     if 'thinking' in config or 'thinking_repeat' in config else None,
                     }.items() if value is not None},
                     'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if not rows:
        raise ValueError('no matching completions in ' + str(directory))
    expected_cases = config.get('cases')
    expected_cases = expected_cases.split(',') if isinstance(expected_cases, str) else expected_cases
    expected_cases = expected_cases or list(prompts)
    if prose_only:
        expected_cases = [case for case in expected_cases if case.startswith('prose_')]
    expected_repeats = repeats
    if expected_repeats is None:
        expected_repeats = range(config['repeats']) if type(config.get('repeats')) is int else sorted({key[1] for key in rows})
    names = {key[2] for key in rows}
    if any((case, repeat, name) not in rows for case in expected_cases
           for repeat in expected_repeats for name in names):
        raise ValueError('incomplete configured corpus/variant/repeat set')
    return rows


def paired_results(lossless, lossy, control='fixed7', treatment=None, repeats=None):
    left = read_results(lossless, control, repeats)
    right = read_results(lossy, treatment, repeats)
    if treatment is None:
        names = {key[2] for key in right} - {control}
        if len(names) != 1:
            raise ValueError('select one treatment variant explicitly')
        treatment = next(iter(names))
        right = {key: row for key, row in right.items() if key[2] == treatment}
    if treatment == control:
        raise ValueError('comparison requires two distinct variant names')
    left = {key[:2]: row for key, row in left.items()}
    right = {key[:2]: row for key, row in right.items()}
    if left.keys() != right.keys():
        raise ValueError('incomplete paired case/repeat set')
    if repeats is not None and any((case, repeat) not in left
                                  for case, _ in left for repeat in repeats):
        raise ValueError('incomplete requested repeats')
    pairs = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if a['prompt'] != b['prompt']:
            raise ValueError('paired completions have different prompts')
        for field in ('message_sha256', 'prompt_token_sha256'):
            if field in a and field in b and a[field] != b[field]:
                raise ValueError('paired completions have different prompts')
        if a['comparison_settings'] != b['comparison_settings']:
            raise ValueError('paired completions have different token budgets or thinking settings')
        pairs.append((a, b))
    return pairs
