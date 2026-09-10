"""Trace-proven paired reports for Pi controls and bounded-lossy prose."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from analyze_adaptive_spec import device_energy, gaps, paired_groups
from adaptive_spec import lossy_xargs
from spec_prose_results import read_results


def prompt_digest(row):
    prompts = [c['data']['prompt_token_ids'] for c in row['chunks']
               if c['data'].get('prompt_token_ids')]
    if not prompts and row.get('private_prompt_ids_removed') is True:
        digest = row.get('prompt_token_sha256')
        if isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest):
            # Publication retains the hash verified against original private
            # tokens. Its original/published file hashes live in provenance.
            return digest
    if not prompts or any(p != prompts[0] for p in prompts):
        raise ValueError('missing or inconsistent prompt token IDs')
    digest = hashlib.sha256(json.dumps(prompts[0]).encode()).hexdigest()
    if row.get('prompt_token_sha256', digest) != digest:
        raise ValueError('saved prompt digest differs from actual prompt token IDs')
    return digest


def summarize(runs, events=(), samples=None):
    studies = {r['study'] for r in runs}
    if studies == {'lossy'}:
        return summarize_lossy(runs, events, samples)
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
        if not rows:
            raise ValueError('missing runtime control trace for ' + row['label'])
        if any(r.get('relaxed', 0) != 0 for r in rows):
            raise ValueError('lossless control unexpectedly relaxed')
        if any(r.get('eligible') is not True or r.get('dropped') != 0
               or r.get('writer_error') is not None for r in rows):
            raise ValueError('control lost C1 eligibility or complete telemetry')
        caps = Counter(r['scheduled_k'] for r in rows)
        hints = [r for r in rows if r.get('hint_domain') is not None]
        packets = [r['confidence'] for r in rows if r.get('confidence')]
        expected_mode = 'fixed' if study == 'confidence' or row['variant'] == 'fixed7' else 'adaptive'
        if any(r.get('mode') != expected_mode for r in rows):
            raise ValueError('runtime policy did not match the requested control')
        if expected_mode == 'fixed' and set(caps) != {7}:
            raise ValueError('fixed control did not actually use cap seven')
        if study == 'confidence':
            if hints or (row['variant'] == 'off' and packets):
                raise ValueError('disabled control unexpectedly activated')
            if row['variant'] == 'on' and (not packets or not all(p.get('valid') for p in packets)):
                raise ValueError('confidence-on control has no valid packet stream')
        else:
            if packets:
                raise ValueError('hint control unexpectedly collected confidence')
            if row['variant'] in ('off', 'fixed7') and hints:
                raise ValueError('disabled hint control unexpectedly activated')
            if row['variant'] in ('on', 'wrong'):
                workload = row['experiment_xargs']['spec_workload']
                domain = 'prose' if workload == 'prose' else 'code'
                if not hints or any(r['hint_domain'] != domain or r['observations'] >= 8 for r in hints):
                    raise ValueError('hint-on control did not apply the bounded intended prior')
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


def acceptance_curve(rows):
    """Prefix survival, with scheduled positions as denominators (not hazard)."""
    return [{'position': i + 1,
             'cycles': sum(r['scheduled_k'] > i for r in rows),
             'accepted': sum(r['accepted'] > i for r in rows),
             'acceptance': (sum(r['accepted'] > i for r in rows) /
                            sum(r['scheduled_k'] > i for r in rows))
                           if any(r['scheduled_k'] > i for r in rows) else None}
            for i in range(7)]


def cycle_statistics(sequences):
    # Keep request boundaries and censored cycles intact before joining the
    # following draft. Never bridge over a terminal/nonlearnable row.
    usable = lambda r: not r.get('terminal', False) and r.get('learned', True)
    rows = [r for seq in sequences for r in seq if usable(r)]
    following = {'relaxed': [], 'exact': []}
    for seq in sequences:
        for previous, current in zip(seq, seq[1:]):
            if usable(previous) and usable(current):
                following['relaxed' if previous.get('relaxed', 0) > 0 else 'exact'].append(current)
    relaxed = sum(r.get('relaxed', 0) for r in rows)
    positions = sum(r['scheduled_k'] for r in rows)
    by_position = []
    for i, point in enumerate(acceptance_curve(rows)):
        # A scalar count identifies positions only when zero or ALL accepted
        # tokens were relaxed. Otherwise report sharp bounds, never allocate
        # the relaxed count to the first n positions by assumption.
        lower = sum(r['accepted'] > i and r.get('relaxed', 0) == r['accepted'] for r in rows)
        upper = sum(r['accepted'] > i and r.get('relaxed', 0) > 0 for r in rows)
        by_position.append({**point, 'relaxed_accepts': lower if lower == upper else None,
                            'relaxed_accepts_bounds': [lower, upper]})
    return {'cycles': len(rows), 'excluded_cycles': sum(map(len, sequences)) - len(rows),
            'accepted_per_cycle': statistics.mean(r['accepted'] for r in rows) if rows else None,
            'emitted_per_cycle': statistics.mean(r['accepted'] + 1 for r in rows) if rows else None,
            'relaxed_accepts': relaxed,
            'relaxed_per_cycle': relaxed / len(rows) if rows else None,
            'scheduled_positions': positions,
            'relaxed_per_scheduled_position': relaxed / positions if positions else None,
            'per_position': by_position,
            'following_relaxed': acceptance_curve(following['relaxed']),
            'following_exact': acceptance_curve(following['exact'])}


def prove_lossy(row, rows):
    if not rows:
        raise ValueError('missing runtime control trace for ' + row['label'])
    expected = lossy_xargs(row['variant']) if row['variant'] != 'fixed7' else None
    for event in rows:
        if (event.get('eligible') is not True or event.get('dropped') != 0
                or event.get('writer_error') is not None):
            raise ValueError('control lost C1 eligibility or complete telemetry')
        if event.get('mode') != 'fixed' or event.get('scheduled_k') != 7:
            raise ValueError('lossy study requires actual fixed cap seven')
        accepted, relaxed = event.get('accepted'), event.get('relaxed', 0)
        if (type(accepted) is not int or not 0 <= accepted <= 7 or
                type(relaxed) is not int or not 0 <= relaxed <= accepted):
            raise ValueError('invalid accepted/relaxed cycle counts')
        if expected:
            for field in ('lossy_margin', 'lossy_min_p'):
                value = event.get(field)
                if (type(value) not in (int, float) or not math.isfinite(value) or
                        not math.isclose(value, expected['spec_' + field], rel_tol=1e-6, abs_tol=1e-8)):
                    raise ValueError('lossy trace does not echo requested ' + field)
            if event.get('lossy_enabled') != 1:
                raise ValueError('lossy trace is not enabled')
        elif relaxed != 0:
            raise ValueError('lossless control unexpectedly relaxed')
    if expected and not any(event.get('relaxed', 0) > 0 for event in rows):
        raise ValueError('lossy request has no relaxed accepts')


def performance(rows, samples):
    intervals = sorted(g for row in rows for g in gaps(row))
    energy = [v for row in rows if samples and (v := device_energy(row, samples)) is not None]
    return {'requests': len(rows),
            'decode_tps': statistics.mean(row['decode_tps'] for row in rows),
            'ttft': statistics.mean(row['ttft'] for row in rows),
            'p95_emission_gap_s': intervals[math.ceil(.95 * len(intervals)) - 1] if intervals else None,
            'decode_device_j_per_token': statistics.mean(energy) if energy else None}


def ending_statistics(row):
    ids = row['token_ids']
    grams = [tuple(ids[i:i + 4]) for i in range(max(0, len(ids) - 3))]
    return {'output_tokens': len(ids),
            'output_4gram_repetition': (len(grams) - len(set(grams))) / len(grams) if grams else None,
            'finish_reasons': sorted({c['finish_reason'] for chunk in row['chunks']
                                      for c in chunk['data'].get('choices', []) if c.get('finish_reason')})}


def summarize_lossy(runs, events, samples=None):
    variants = sorted({row['variant'] for row in runs} - {'fixed7'})
    if not variants or 'fixed7' not in {row['variant'] for row in runs}:
        raise ValueError('lossy study requires fixed7 and at least one lossy variant')
    for name in variants:
        lossy_xargs(name)
    traces = defaultdict(list)
    for event in events:
        if event.get('event') == 'verify':
            traces[event['label']].append(event)
    indexed, details, labels = {}, [], set()
    for row in runs:
        key = row['case'], row['repeat'], row['variant']
        if key in indexed or row['label'] in labels:
            raise ValueError('duplicate request variant or label')
        indexed[key] = row
        labels.add(row['label'])
        seq = traces[row['label']]
        # decision_ns identifies scheduling order, even with async feedback.
        if seq and all('decision_ns' in event for event in seq):
            seq.sort(key=lambda event: event['decision_ns'])
            if len({event['decision_ns'] for event in seq}) != len(seq):
                raise ValueError('duplicate verify cycle')
        prove_lossy(row, seq)
        details.append({**{k: row[k] for k in ('label', 'case', 'repeat', 'variant', 'token_sha256')},
                        'prompt_token_sha256': prompt_digest(row),
                        **ending_statistics(row),
                        **performance([row], samples), **cycle_statistics([seq])})
    comparisons = []
    for case, repeat in sorted({key[:2] for key in indexed}):
        if any((case, repeat, v) not in indexed for v in ['fixed7', *variants]):
            raise ValueError('incomplete paired variant set')
        group = [indexed[case, repeat, v] for v in ['fixed7', *variants]]
        if len({prompt_digest(r) for r in group}) != 1 or len({r['message_sha256'] for r in group}) != 1:
            raise ValueError('paired requests have different prompts')
        base = group[0]
        for treatment in group[1:]:
            a, b = base['token_ids'], treatment['token_ids']
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                         min(len(a), len(b)) if len(a) != len(b) else None)
            n = min(64, len(a), len(b))
            comparisons.append({'case': case, 'repeat': repeat, 'variant': treatment['variant'],
                                'first_divergence_position': first,
                                'agreement_first64': sum(x == y for x, y in zip(a[:n], b[:n])) / n if n else None,
                                'agreement_positions': n})
    def pooled(selected):
        return {**performance(selected, samples),
                **cycle_statistics([traces[row['label']] for row in selected])}
    summaries = {v: pooled([row for row in runs if row['variant'] == v]) for v in ['fixed7', *variants]}
    cases = {case: {v: pooled([r for r in runs if r['case'] == case and r['variant'] == v])
                    for v in ['fixed7', *variants]} for case in sorted({r['case'] for r in runs})}
    return {'study': 'lossy', 'screening_only': True, 'summaries': summaries, 'cases': cases,
            'requests': details, 'comparisons': comparisons,
            'paired': paired_groups([{**r, 'policy': 'fixed' if r['variant'] == 'fixed7' else r['variant'], 'cap': 7} for r in runs]),
            'method': 'Activation uses every verify row; cycle statistics exclude terminal/nonlearnable rows. Following-cycle curves join adjacent rows within each request in scheduling order, without bridging excluded rows. Curves are accepted-prefix survival over scheduled positions; pooled rates use sums of counts, not averages of request rates.',
            'limitations': 'Scalar relaxed counts cannot generally identify individual relaxed positions; per-position exact counts are null when ambiguous, with sharp count bounds. Relaxed per scheduled position is total relaxed / total scheduled positions. SSE gaps measure emitted blocks, not individual GPU tokens. Token agreement is with the paired greedy completion, not target-logit top-1 agreement on the divergent prefix. First divergence is zero-based. Four-gram repetition is (all token 4-grams minus distinct token 4-grams) / all token 4-grams, including reasoning when present. Finish reasons distinguish server stop/length, not whether prose is semantically finished. Prompt-clustered speed intervals do not alone establish quality or promotion; device energy is not wall energy.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    ap.add_argument('--trace', type=Path, required=True, help='Required proof that each requested treatment actually activated')
    ap.add_argument('--power', type=Path)
    args = ap.parse_args()
    runs, sources = [], {}
    for path in sorted(args.directory.glob('*.json')):
        row = json.loads(path.read_text())
        if isinstance(row, dict) and {'study', 'variant', 'chunks', 'message_sha256'} <= row.keys():
            runs.append(row)
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if runs and {row['study'] for row in runs} == {'lossy'}:
        # Do not silently shrink a screen when both arms of a configured prompt
        # or repeat are absent. Legacy Pi studies have a different declaration.
        config = json.loads((args.directory / 'config.json').read_text())
        read_results(args.directory, prose_only=config.get('prose_only', False))
        configured = config.get('variants')
        if configured and set(configured.split(',')) != {row['variant'] for row in runs}:
            raise ValueError('incomplete configured variant set')
    events = [json.loads(s) for s in args.trace.read_text().splitlines()] if args.trace else []
    sources[str(args.trace)] = hashlib.sha256(args.trace.read_bytes()).hexdigest()
    samples = defaultdict(list)
    if args.power:
        sources[str(args.power)] = hashlib.sha256(args.power.read_bytes()).hexdigest()
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
