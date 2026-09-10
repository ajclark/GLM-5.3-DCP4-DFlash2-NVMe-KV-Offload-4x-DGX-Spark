#!/usr/bin/env python3
"""Aggregate private per-call pi captures; never export request/response texts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from harness_capture import (CATEGORIES, TokenizationUnavailable, call_type, canonical,
                             category_counts, identify_calls, load_capture, make_tokenizer,
                             ratio, tokenizer_options, write_report)


def pooled(rows):
    keys = ('generation_tokens', 'spec_drafts', 'spec_accepted_tokens',
            'server_decode_s', 'server_prefill_s', 'wall_s', 'prompt_tokens')
    out = {key: sum(r.get(key) or 0 for r in rows) for key in keys}
    out.update(calls=len(rows), accepted_per_cycle=(1 + out['spec_accepted_tokens'] / out['spec_drafts']
               if out['spec_drafts'] else None),
               decode_tok_s=ratio(out['generation_tokens'], out['server_decode_s']),
               prefill_wall_share=ratio(out['server_prefill_s'], out['wall_s']),
               decode_wall_share=ratio(out['server_decode_s'], out['wall_s']))
    out['other_wall_s'] = out['wall_s'] - out['server_decode_s'] - out['server_prefill_s']
    return out


def prefix_stability(prior, current):
    """Compare literal message prefixes, including IDs and the tools schema."""
    old, new = prior.get('messages', []), current.get('messages', [])
    common = 0
    for a, b in zip(old, new):
        if a != b:
            break
        common += 1
    a = canonical(prior.get('tools', [])) + canonical(old)
    b = canonical(current.get('tools', [])) + canonical(new)
    leading = 0
    for x, y in zip(a, b):
        if x != y:
            break
        leading += 1
    return {'prior_messages': len(old), 'unchanged_leading_messages': common,
            'append_only_messages': common == len(old) and len(new) >= len(old),
            'tools_unchanged': prior.get('tools') == current.get('tools'),
            'serialized_character_prefix_fraction': ratio(leading, len(a))}


def category_fit(rows, shares):
    """Weighted drafts/token regression, not a within-call trace measurement.

    Adapted from call_attrib.py's normal equations. For token fractions f and
    emitted/cycle e, additive cycle cost is 1/e = sum(f_c/e_c), not sum(f_c*e_c).
    Singular designs and physically impossible estimates return no estimate.
    """
    n = len(CATEGORIES)
    A, b = [[0.] * n for _ in range(n)], [0.] * n
    used = 0
    for row, x in zip(rows, shares):
        d, a = row.get('spec_drafts', 0), row.get('spec_accepted_tokens', 0)
        if d < 5 or not sum(x):
            continue
        weight = d + a
        y = d / weight
        used += 1
        for i in range(n):
            b[i] += weight * x[i] * y
            for j in range(n):
                A[i][j] += weight * x[i] * x[j]
    scale = max((abs(x) for r in A for x in r), default=0.)
    pivots = []
    for i in range(n):
        p = max(range(i, n), key=lambda k: abs(A[k][i]))
        A[i], A[p], b[i], b[p] = A[p], A[i], b[p], b[i]
        pivots.append(abs(A[i][i]))
        if abs(A[i][i]) <= max(1e-12, scale * 1e-10):
            return {'status': 'unidentified: insufficient independent category mixtures', 'calls': used}
        for k in range(i + 1, n):
            f = A[k][i] / A[i][i]
            for j in range(i, n):
                A[k][j] -= f * A[i][j]
            b[k] -= f * b[i]
    cost = [0.] * n
    for i in reversed(range(n)):
        cost[i] = (b[i] - sum(A[i][j] * cost[j] for j in range(i + 1, n))) / A[i][i]
    estimates = {c: 1 / v if 1 / 8 <= v <= 1 else None for c, v in zip(CATEGORIES, cost)}
    return {'status': 'indicative mixture regression; null means outside physical K7 range',
            'calls': used, 'accepted_per_cycle': estimates,
            'pivot_ratio': min(pivots) / max(pivots),
            'method': 'weighted least squares of drafts/(drafts+accepted) on category token fractions'}


def analyze(rows, provenance=None, tokenizer=None, source_dir=None):
    identities = identify_calls(rows, source_dir)
    counts = [category_counts(r, tokenizer) for r in rows]
    shares = [[x / sum(c) if sum(c) else 0. for x in c] for c in counts]
    allocated = [[f * (r.get('generation_tokens') or 0) for f in s] for r, s in zip(rows, shares)]
    total = pooled(rows)
    calls, prior, sessions, types = [], {}, {}, defaultdict(list)
    for i, (row, identity, allocation) in enumerate(zip(rows, identities, allocated)):
        cache = row.get('prefix_cache_hit_rate')
        if cache is not None and (not isinstance(cache, (int, float)) or not 0 <= cache <= 1):
            raise ValueError(f'invalid prefix cache ratio at call {i + 1}')
        prompt = row.get('prompt_tokens') or 0
        entry = {**identity, 'call': i + 1, 'category_tokens_allocated': dict(zip(CATEGORIES, allocation)),
                 'call_type': call_type(row), **pooled([row]), 'ttft_s': row.get('mean_ttft_s'),
                 'prefix_cache_hit_rate': cache,
                 'estimated_cached_prompt_tokens': prompt * cache if cache is not None else None,
                 'estimated_uncached_prompt_tokens': prompt * (1 - cache) if cache is not None else None}
        session = identity['session']
        if session in prior:
            entry['prefix_stability'] = prefix_stability(prior[session], row)
        prior[session] = row
        calls.append(entry)
        sessions.setdefault(session, []).append(entry)
        types[call_type(row)].append(row)
    category = {}
    for k, name in enumerate(CATEGORIES):
        tokens = sum(a[k] for a in allocated)
        # Call-mixture proxy: allocate each call's cycle count by category shares.
        cycles = sum(s[k] * r.get('spec_drafts', 0) for s, r in zip(shares, rows))
        accepted = sum(s[k] * r.get('spec_accepted_tokens', 0) for s, r in zip(shares, rows))
        pure = [r for r, s in zip(rows, shares) if s[k] == 1]
        category[name] = {'allocated_tokens': tokens, 'token_share': ratio(tokens, total['generation_tokens']),
                          'call_mixture_accepted_per_cycle_proxy': 1 + accepted / cycles if cycles else None,
                          'pure_calls': pooled(pure)}
    session_rows = []
    for session, rs in sessions.items():
        known = [r for r in rs if r['estimated_uncached_prompt_tokens'] is not None]
        session_rows.append({'session': session, 'task': rs[0]['task'], 'run': rs[0].get('run'),
                             **pooled(rs), 'task_wall_s': rs[0].get('task_wall_s'),
                             'official_pass': rs[0].get('official_pass'),
                             'estimated_uncached_prompt_tokens': sum(r['estimated_uncached_prompt_tokens'] for r in known),
                             'cache_measurement_calls': len(known)})
    cache_rows = [r for r in calls if r['prefix_cache_hit_rate'] is not None]
    total.update(estimated_uncached_prompt_tokens=sum(r['estimated_uncached_prompt_tokens'] for r in cache_rows),
                 estimated_cached_prompt_tokens=sum(r['estimated_cached_prompt_tokens'] for r in cache_rows),
                 cache_measurement_calls=len(cache_rows),
                 prompt_weighted_cache_hit_rate=ratio(sum(r['estimated_cached_prompt_tokens'] for r in cache_rows),
                                                      sum(r['prompt_tokens'] for r in cache_rows)),
                 unique_tasks=len({r['task'] for r in calls}), task_sessions=len(sessions),
                 completed_task_wall_s=sum(r['task_wall_s'] or 0 for r in session_rows),
                 completed_task_sessions=sum(r['task_wall_s'] is not None for r in session_rows))
    transitions = [r['prefix_stability'] for r in calls if 'prefix_stability' in r]
    cohorts = {}
    groups = {'not_length_limited': [i for i, r in enumerate(rows) if r.get('finish_reason') != 'length'],
              'length_limited': [i for i, r in enumerate(rows) if r.get('finish_reason') == 'length']}
    for run in sorted({m['run'] for m in identities if m.get('run') is not None}):
        groups[f'run{run}'] = [i for i, m in enumerate(identities) if m.get('run') == run]
    for name, indices in groups.items():
        subset = pooled([rows[i] for i in indices])
        subset['category_token_shares'] = {
            c: ratio(sum(allocated[i][k] for i in indices), subset['generation_tokens'])
            for k, c in enumerate(CATEGORIES)}
        cohorts[name] = subset
    return {'source': provenance, 'cohorts': cohorts, 'token_basis': 'CPU /tokenize; independently encoded <=2048-character chunks'
            if tokenizer else 'character-share allocation to usage tokens; estimated, not tokenizer counts',
            'notes': [
                'accepted/cycle includes the bonus token: 1 + accepted drafts / draft cycles',
                'Category cycle proxies are call-composition allocations, not within-call measurements; pure calls are measured.',
                'Token category allocations normalize parsed response counts to generation_tokens; wrappers and delimiters are unassigned proportionally.',
                'Cache tokens are estimated as prompt_tokens * the captured GLOBAL counter hit ratio; exact per-call cached usage was not captured.',
                'Server prefill/decode are counter deltas; TTFT overlaps prefill and is never added to it.',
                'Per-call wall excludes proxy metrics scrape overhead; task wall includes client/tool/proxy overhead.',
                'Identity is pseudonymous; absent runner artifacts, sessions are inferred and cannot establish actual run membership.',
            ], 'total': total, 'categories': category, 'category_fit': category_fit(rows, shares),
            'call_types': {k: pooled(v) for k, v in types.items()}, 'tasks': session_rows, 'calls': calls,
            'prefix_stability': {'transitions': len(transitions),
                                 'append_only_transitions': sum(s['append_only_messages'] for s in transitions),
                                 'unchanged_tools_transitions': sum(s['tools_unchanged'] for s in transitions),
                                 'mean_character_prefix_fraction': ratio(sum(s['serialized_character_prefix_fraction'] or 0 for s in transitions), len(transitions))},
            'counter_contamination_suspect_calls': sum(r.get('api_calls') != 1 for r in rows),
            'timing_overlap_suspect_calls': sum(r['other_wall_s'] < -.05 for r in calls)}


def fmt(value, digits=3):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def markdown(report):
    t = report['total']
    lines = ['# Private pi capture: aggregate attribution', '',
             f"{t['calls']} calls; {t['unique_tasks']} task identities; {t['task_sessions']} task sessions. "
             f"Pooled decode {fmt(t['decode_tok_s'])} tok/s; emitted/cycle {fmt(t['accepted_per_cycle'])}.",
             '', 'Token basis: ' + report['token_basis'] + '.', '',
             '| Category | Allocated tokens | Token share | Call-mixture cycle proxy | Pure-call cycle |',
             '|---|---:|---:|---:|---:|']
    for name, v in report['categories'].items():
        lines.append(f"| {name} | {fmt(v['allocated_tokens'], 1)} | {fmt(v['token_share'])} | {fmt(v['call_mixture_accepted_per_cycle_proxy'])} | {fmt(v['pure_calls']['accepted_per_cycle'])} |")
    lines += ['', f"Prefill {fmt(t['server_prefill_s'])} s ({fmt(t['prefill_wall_share'] * 100 if t['prefill_wall_share'] is not None else None, 1)}% of call wall); "
              f"decode {fmt(t['server_decode_s'])} s ({fmt(t['decode_wall_share'] * 100 if t['decode_wall_share'] is not None else None, 1)}%).",
              '', '| Task | Session | Run | Calls | Call wall s | Task wall s | Uncached prompt estimate |', '|---|---|---:|---:|---:|---:|---:|']
    for r in report['tasks']:
        lines.append(f"| {r['task']} | {r['session']} | {r['run'] or 'n/a'} | {r['calls']} | {fmt(r['wall_s'])} | {fmt(r['task_wall_s'])} | {fmt(r['estimated_uncached_prompt_tokens'], 1)} |")
    lines += ['', '| Task | Session | Call within task | TTFT s | Cache hit ratio | Prompt tokens | Uncached estimate | Emitted/cycle |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in report['calls']:
        lines.append(f"| {r['task']} | {r['session']} | {r['call_in_task']} | {fmt(r['ttft_s'])} | {fmt(r['prefix_cache_hit_rate'])} | {r['prompt_tokens']} | {fmt(r['estimated_uncached_prompt_tokens'], 1)} | {fmt(r['accepted_per_cycle'])} |")
    lines += ['', 'Cautions:', ''] + ['- ' + s for s in report['notes']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--out', type=Path, required=True, help='Aggregate JSON; Markdown uses the same stem')
    tokenizer_options(parser)
    args = parser.parse_args()
    rows, provenance = load_capture(args.capture)
    unavailable = None
    try:
        tokenizer = make_tokenizer(args)
        report = analyze(rows, provenance, tokenizer, args.capture.parent)
    except TokenizationUnavailable as error:
        unavailable = str(error)
        report = analyze(rows, provenance, source_dir=args.capture.parent)
    if unavailable:
        report['notes'].append(unavailable)
    write_report(args.out, report, markdown(report))
    print(json.dumps(report['total'], indent=2))


if __name__ == '__main__':
    main()
