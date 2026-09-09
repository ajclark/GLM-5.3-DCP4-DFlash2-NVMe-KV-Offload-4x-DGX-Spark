"""Paired reports for exact Pi payload controls; private prompts stay private."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from analyze_adaptive_spec import device_energy


def prompt_digest(row):
    prompts = [c['data']['prompt_token_ids'] for c in row['chunks']
               if c['data'].get('prompt_token_ids')]
    if not prompts or any(p != prompts[0] for p in prompts):
        raise ValueError('missing or inconsistent prompt token IDs')
    return hashlib.sha256(json.dumps(prompts[0]).encode()).hexdigest()


def summarize(runs, events=(), samples=None):
    studies = {r['study'] for r in runs}
    if len(studies) != 1 or not studies <= {'hints', 'confidence'}:
        raise ValueError('exactly one recognized study required')
    study = next(iter(studies))
    variants = ('off', 'on', 'wrong', 'fixed7') if study == 'hints' else ('off', 'on')
    indexed, details = {}, []
    traces = defaultdict(list)
    for event in events:
        if event.get('event') == 'verify':
            traces[event['label']].append(event)
    for row in runs:
        key = (row['case'], row['repeat'], row['variant'])
        if key in indexed or row['variant'] not in variants:
            raise ValueError('duplicate or unknown request variant')
        indexed[key] = row
        rows = traces[row['label']]
        caps = Counter(r['scheduled_k'] for r in rows)
        hints = [r for r in rows if r.get('hint_domain') is not None]
        packets = [r['confidence'] for r in rows if r.get('confidence')]
        details.append({k: row[k] for k in ('label', 'case', 'repeat', 'variant',
                                           'decode_tps', 'ttft', 'token_sha256')})
        details[-1].update(
            prompt_token_sha256=prompt_digest(row),
            output_tokens=len(row['token_ids']),
            finish_reasons=sorted({str(c.get('finish_reason')) for chunk in row['chunks']
                for c in chunk['data'].get('choices', []) if c.get('finish_reason')}),
            cap_counts=dict(caps), hint_domains=dict(Counter(r['hint_domain'] for r in hints)),
            hint_observations=[r['observations'] for r in hints],
            confidence_packets=len(packets),
            valid_confidence_packets=sum(bool(p.get('valid')) for p in packets),
            decode_device_j_per_token=device_energy(row, samples) if samples else None)
    detail_index = {(r['case'], r['repeat'], r['variant']): r for r in details}
    comparisons = []
    for case, repeat in sorted({key[:2] for key in indexed}):
        if any((case, repeat, variant) not in indexed for variant in variants):
            raise ValueError('incomplete paired variant set')
        group = [indexed[case, repeat, variant] for variant in variants]
        if len({prompt_digest(r) for r in group}) != 1 or len({r['message_sha256'] for r in group}) != 1:
            raise ValueError('paired requests have different prompts')
        baseline = detail_index[case, repeat, 'off']
        for variant in variants[1:]:
            treatment = detail_index[case, repeat, variant]
            pair = {'case': case, 'repeat': repeat, 'variant': variant,
                    'labels': [baseline['label'], treatment['label']],
                    'identical_output_ids': indexed[case, repeat, 'off']['token_ids'] == indexed[case, repeat, variant]['token_ids']}
            for metric in ('decode_tps', 'ttft', 'decode_device_j_per_token'):
                a, b = baseline[metric], treatment[metric]
                pair[metric + '_ratio'] = b / a if a is not None and b is not None and a > 0 and b > 0 else None
            comparisons.append(pair)
    summaries = {}
    for variant in variants[1:]:
        selected = [r for r in comparisons if r['variant'] == variant]
        summary = {'pairs': len(selected), 'identical_output_pairs': sum(r['identical_output_ids'] for r in selected)}
        for metric in ('decode_tps_ratio', 'ttft_ratio', 'decode_device_j_per_token_ratio'):
            groups = defaultdict(list)
            for pair in selected:
                if pair[metric] is not None:
                    groups[pair['case']].append(math.log(pair[metric]))
            means = {case: math.exp(statistics.mean(values)) for case, values in groups.items()}
            summary[metric] = math.exp(statistics.mean(math.log(v) for v in means.values())) if means else None
            summary[metric + '_by_prompt'] = means
        summaries[variant + '/off'] = summary
    return {'study': study, 'screening_only': True, 'summaries': summaries,
            'requests': details, 'comparisons': comparisons,
            'method': 'Within-prompt/repeat treatment over absent-hint or confidence-off control; average log ratios within each prompt, then equally across prompts.',
            'limitations': 'Small development screen without confidence intervals or promotion claim. Output identity is reported, not assumed. Device energy excludes TTFT and is not wall energy. No private prompt text or token IDs are included in this report.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    ap.add_argument('--trace', type=Path)
    ap.add_argument('--power', type=Path)
    args = ap.parse_args()
    runs, sources = [], {}
    for path in sorted(args.directory.glob('*.json')):
        row = json.loads(path.read_text())
        if isinstance(row, dict) and {'study', 'variant', 'chunks', 'message_sha256'} <= row.keys():
            runs.append(row)
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    events = [json.loads(s) for s in args.trace.read_text().splitlines()] if args.trace else []
    samples = defaultdict(list)
    if args.power:
        for line in args.power.read_text().splitlines():
            row = json.loads(line)
            samples[row['host']].append(row)
        for values in samples.values():
            values.sort(key=lambda r: r['received_at'])
    result = summarize(runs, events, samples)
    result['source_sha256'] = sources
    (args.directory / 'request-control-report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summaries'], indent=2))


if __name__ == '__main__':
    main()
