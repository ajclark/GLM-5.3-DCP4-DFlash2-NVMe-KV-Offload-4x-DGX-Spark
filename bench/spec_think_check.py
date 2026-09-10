#!/usr/bin/env python3
"""Compare thinking-on fixed7/think-lossy completions locally, including code checks."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

from adaptive_spec import lossy_xargs
import spec_code_check
from spec_prose_results import read_results
from spec_think_utils import TOKEN_HEURISTIC, completion_accounting, completion_parts, thinking_enabled


def ratio(candidate, control):
    return candidate / control if control else None


def evaluate_answer(row):
    if thinking_enabled(row) is not True:
        raise ValueError('think quality comparison requires thinking on for every request')
    options = {'thinking': True, 'reasoning_field_present': row.get('reasoning_field_present', bool(row.get('reasoning_text')))}
    _, visible = completion_parts(row['text'], row.get('reasoning_text', ''), **options)
    counts = completion_accounting(row['text'], row.get('reasoning_text', ''), row.get('usage'), **options)
    result = {**counts, 'source': row['source'], 'source_sha256': row['source_sha256'],
              'empty_visible_answer': not visible.strip(),
              'finish_reasons': sorted({c['finish_reason'] for chunk in row.get('chunks', [])
                                       for c in chunk['data'].get('choices', []) if c.get('finish_reason')})}
    if row['case'].startswith('code'):
        # Invoke the existing bounded checker on VISIBLE text, never reasoning.
        try:
            message = spec_code_check.check(visible)
            check = {'passed': True, 'detail': message}
        except (ValueError, SyntaxError, RuntimeError, subprocess.TimeoutExpired) as exc:
            check = {'passed': False, 'detail': str(exc)[-2000:]}
        result['code_check'] = {**check, 'prompt_matches_checker_contract': row['prompt'].strip() == spec_code_check.PROMPT.strip()}
    return result


def compare(control_dir, treatment_dir, treatment=None, repeats=None):
    control = read_results(control_dir, 'fixed7', repeats, prose_only=False)
    candidates = read_results(treatment_dir, treatment, repeats, prose_only=False)
    if treatment is None:
        names = {key[2] for key in candidates if key[2].startswith('lossy-think-')}
        if len(names) != 1:
            raise ValueError('select exactly one think treatment with --treatment')
        treatment = next(iter(names))
    if lossy_xargs(treatment).get('spec_lossy_scope') != 'think':
        raise ValueError('treatment must be a lossy-think variant')
    control = {key[:2]: row for key, row in control.items()}
    candidates = {key[:2]: row for key, row in candidates.items() if key[2] == treatment}
    if control.keys() != candidates.keys():
        raise ValueError('incomplete paired case/repeat set')
    pairs = []
    for case, repeat in sorted(control):
        a, b = control[case, repeat], candidates[case, repeat]
        if a['prompt'] != b['prompt']:
            raise ValueError('paired completions have different prompts')
        for field in ('message_sha256', 'prompt_token_sha256'):
            if field in a and field in b and a[field] != b[field]:
                raise ValueError('paired completions have different prompts')
        if a['comparison_settings'] != b['comparison_settings'] or a.get('max_tokens') != b.get('max_tokens'):
            raise ValueError('paired completions have different thinking settings or completion budgets')
        left, right = evaluate_answer(a), evaluate_answer(b)
        pairs.append({'case': case, 'repeat': repeat, 'control': left, 'treatment': right,
                      'reasoning_token_count_ratio': ratio(right['reasoning_tokens'], left['reasoning_tokens']),
                      'final_answer_length_ratio': ratio(right['visible_answer_words'], left['visible_answer_words'])})
    totals = {}
    for side in ('control', 'treatment'):
        rows = [pair[side] for pair in pairs]
        checks = [row['code_check'] for row in rows if 'code_check' in row]
        applicable = [check for check in checks if check['prompt_matches_checker_contract']]
        totals[side] = {'variant': 'fixed7' if side == 'control' else treatment, 'requests': len(rows),
                        'reasoning_tokens': sum(r['reasoning_tokens'] for r in rows),
                        'reasoning_count_sources': dict(Counter(r['reasoning_tokens_source'] for r in rows)),
                        'estimated_reasoning_requests': sum(r['reasoning_tokens_estimated'] for r in rows),
                        'final_answer_words': sum(r['visible_answer_words'] for r in rows),
                        'final_answer_tokens_estimate': sum(r['visible_answer_tokens_estimate'] for r in rows),
                        'empty_visible_answers': sum(r['empty_visible_answer'] for r in rows),
                        'length_stopped_requests': sum('length' in r['finish_reasons'] for r in rows),
                        'code_checks': {'passed': sum(c['passed'] for c in checks), 'total': len(checks),
                                        'applicable_passed': sum(c['passed'] for c in applicable),
                                        'applicable_total': len(applicable)}}
    return {'control': totals['control'], 'treatment': totals['treatment'], 'pairs': pairs,
            'reasoning_token_count_ratio': ratio(totals['treatment']['reasoning_tokens'], totals['control']['reasoning_tokens']),
            'final_answer_length_ratio': ratio(totals['treatment']['final_answer_words'], totals['control']['final_answer_words']),
            'method': 'Ratios are treatment/control of pooled counts, not means of per-request ratios. Final-answer length is Unicode words. Zero control denominator yields null. ' + TOKEN_HEURISTIC,
            'limitations': 'Provider reasoning-token usage is preferred; otherwise reasoning field or raw think spans use the word heuristic. Estimates are unsuitable for exact acceptance attribution, especially code/non-English text. Visible answers follow the final </think>, or use parsed content when reasoning is separate; an unclosed raw think span yields no answer. The existing code checker tests only lower_bound, merge_intervals, run_length and stable_unique as pure Python functions; other code prompts are still checked but their failures do not establish task incorrectness. This quality report does not replace trace activation proof.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('control', type=Path)
    ap.add_argument('treatment_dir', type=Path)
    ap.add_argument('--treatment', help='Required when the directory contains multiple lossy-think variants')
    ap.add_argument('--repeat', type=int, action='append', help='Select zero-based thinking-on repeats')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    result = compare(args.control, args.treatment_dir, args.treatment, args.repeat)
    text = json.dumps(result, indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
