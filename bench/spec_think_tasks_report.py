#!/usr/bin/env python3
"""Local-only, private-text-free task quality/speed report for think-lossy holds."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
import statistics as stats

ARMS = {'control': None, 'm2.5': 2.5, 'm5.0': 5.0}


class InputError(ValueError):
    """Messages are fixed diagnostics, never captured input or exception text."""


def number(value, *, optional=False):
    if value is None and optional:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise InputError('invalid or missing nonnegative numeric measurement')
    return value


def read_json(path, partial, pending):
    try:
        before = path.stat()
        value = json.loads(path.read_bytes())
        after = path.stat()
    except (OSError, ValueError):
        if partial:
            pending.append('missing_or_unfinished_json')
            return None
        raise InputError('missing or unfinished JSON; use --allow-partial for an active experiment') from None
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        if not partial:
            raise InputError('input changed during reading; use --allow-partial')
        pending.append('json_changed_during_read')
    return value


def read_jsonl(path, partial, pending):
    try:
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
    except OSError:
        if partial:
            pending.append('missing_jsonl')
            return []
        raise InputError('missing JSONL input') from None
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        if not partial:
            raise InputError('input changed during reading; use --allow-partial')
        pending.append('jsonl_changed_during_read')
    lines, result = data.splitlines(), []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError()
        except ValueError:
            if partial and i == len(lines) - 1 and not data.endswith(b'\n'):
                pending.append('unfinished_jsonl_tail')
                break
            raise InputError('invalid JSONL record') from None
        result.append(row)
    return result


def safe_task(row):
    if not isinstance(row, dict) or not re.fullmatch(r'HumanEval/\d+', str(row.get('task_id', ''))):
        raise InputError('invalid public HumanEval task identifier')
    if type(row.get('check', {}).get('passed')) is not bool:
        raise InputError('missing official check.passed outcome')
    out = {k: number(row.get(k)) for k in ('start', 'end', 'seconds', 'output_tokens', 'reasoning_tokens')}
    if out['end'] < out['start']:
        raise InputError('task end precedes start')
    out.update(task_id=row['task_id'], passed=row['check']['passed'],
               errors=bool(row.get('errors')), api_calls=len(row.get('requests', [])),
               budgets=set(), length_calls=0)
    return out


def safe_call(row, arm):
    out = {k: number(row.get(k)) for k in ('t', 'wall_s', 'generation_tokens', 'server_decode_s',
                                         'spec_drafts', 'spec_accepted_tokens', 'api_calls')}
    # Only stored counts are exported; PRIVATE strings are not even interpolated.
    for name in ('reasoning_chars', 'content_chars', 'toolcall_chars'):
        out[name] = number(row.get(name, 0))
    params = row.get('request_params') or {}
    budget = number(params.get('max_completion_tokens', params.get('max_tokens')), optional=True)
    out.update(arm=arm, budget=budget, length=row.get('finish_reason') == 'length',
               thinking_on=(params.get('chat_template_kwargs') or {}).get('enable_thinking') is True)
    return out


def load_tree(root, arms, passes, expected_tasks, partial=False):
    """Check completion markers before opening private task/call files by default."""
    root = Path(root)
    pending, issues, cells, tasks, calls = [], [], {}, {}, []
    paths = {(arm, p): root / f'{arm}-pass{p}' / f'{arm}-summary.json' for arm in arms for p in passes}
    if not partial and any(not path.is_file() for path in paths.values()):
        raise InputError('experiment completion summaries are missing; use --allow-partial while it runs')
    for (arm, p), path in paths.items():
        summary = read_json(path, partial, pending)
        if summary is not None:
            if not isinstance(summary, list) or len(summary) != 1:
                raise InputError('expected one lane summary per arm/pass')
            summary = summary[0]
            if summary.get('lane') != arm or summary.get('concurrency') != 1:
                raise InputError('lane or concurrency differs from the requested C1 arm')
        cell_dir = path.parent / arm / 'c01'
        cell_tasks = []
        for task_path in sorted(cell_dir.glob('HumanEval_*/result.json')):
            row = read_json(task_path, partial, pending)
            if row is None:
                continue
            task = safe_task(row)
            key = (arm, p, task['task_id'])
            if key in tasks:
                raise InputError('duplicate task in one arm/pass')
            task.update(arm=arm, pass_number=p)
            tasks[key] = task
            cell_tasks.append(task)
        if len(cell_tasks) != expected_tasks or summary is None:
            if not partial:
                raise InputError('incomplete task set; use --allow-partial for an active experiment')
            pending.append('incomplete_arm_pass')
        if summary is not None:
            duplicate = read_json(cell_dir / 'summary.json', partial, pending)
            if duplicate is not None and duplicate != summary:
                issues.append(f'{arm}/pass{p}:summary_copies_disagree')
            for key, actual in [('tasks', len(cell_tasks)), ('passed', sum(t['passed'] for t in cell_tasks)),
                                ('errors', sum(t['errors'] for t in cell_tasks)),
                                *[(k, sum(t[k] for t in cell_tasks)) for k in ('output_tokens', 'reasoning_tokens', 'api_calls')]]:
                if summary.get(key) != actual:
                    issues.append(f'{arm}/pass{p}:summary_{key}_mismatch')
            if summary.get('validity_flags'):
                issues.append(f'{arm}/pass{p}:summary_validity_flags_present')
        cells[arm, p] = {'tasks': cell_tasks, 'summary_wall_s': number(summary.get('wall_s')) if summary else None}
    for arm in arms:
        raw_calls = read_jsonl(root / f'proxy-{arm}-calls.jsonl', partial, pending)
        for raw in raw_calls:
            call = safe_call(raw, arm)
            matches = [t for (a, _, _), t in tasks.items() if a == arm and t['start'] <= call['t'] <= t['end']]
            if len(matches) > 1:
                issues.append(f'{arm}:ambiguous_call_task_window')
            if len(matches) == 1:
                task = matches[0]
                call.update(pass_number=task['pass_number'], task_id=task['task_id'])
                task['budgets'].add(call['budget'])
                task['length_calls'] += call['length']
            elif partial:
                pending.append('unassigned_proxy_call')
            else:
                issues.append(f'{arm}:unassigned_proxy_call')
            calls.append(call)
    for (arm, p), cell in cells.items():
        selected = [c for c in calls if c['arm'] == arm and c.get('pass_number') == p]
        if sum(c['api_calls'] for c in selected) != sum(t['api_calls'] for t in cell['tasks']):
            issues.append(f'{arm}/pass{p}:proxy_client_call_count_mismatch')
    task_sets = [{t['task_id'] for t in cell['tasks']} for cell in cells.values()]
    if task_sets and any(s != task_sets[0] for s in task_sets):
        (pending if partial else issues).append('unmatched_task_sets')
    return cells, tasks, calls, pending, issues


def divide(a, b):
    return a / b if b else None


def aggregate(tasks, calls, walls):
    total = {k: sum(t[k] for t in tasks) for k in ('output_tokens', 'reasoning_tokens', 'api_calls')}
    drafts = sum(c['spec_drafts'] for c in calls)
    accepted = sum(c['spec_accepted_tokens'] for c in calls)
    generation = sum(c['generation_tokens'] for c in calls)
    decode = sum(c['server_decode_s'] for c in calls)
    chars = sum(c[k] for c in calls for k in ('reasoning_chars', 'content_chars', 'toolcall_chars'))
    budgets = {c['budget'] for c in calls if c['budget'] is not None}
    return {**total, 'tasks': len(tasks), 'passed': sum(t['passed'] for t in tasks),
            'errors': sum(t['errors'] for t in tasks), 'wall_s': sum(w for w in walls if w is not None),
            'wall_complete': all(w is not None for w in walls),
            'median_task_s': stats.median(t['seconds'] for t in tasks) if tasks else None,
            'proxy_calls': len(calls), 'proxy_api_calls': sum(c['api_calls'] for c in calls),
            'length_calls': sum(c['length'] for c in calls),
            'length_bound_tasks': sum(t['length_calls'] > 0 for t in tasks),
            'pooled_decode_tok_s': divide(generation, decode), 'proxy_generation_tokens': generation,
            'server_decode_s': decode, 'spec_drafts': drafts, 'spec_accepted_tokens': accepted,
            'accepted_per_cycle': 1 + accepted / drafts if drafts else None,
            'reasoning_chars_share': divide(sum(c['reasoning_chars'] for c in calls), chars),
            'completion_budgets': sorted(budgets), 'unknown_budget_calls': sum(c['budget'] is None for c in calls),
            'thinking_not_confirmed_calls': sum(not c['thinking_on'] for c in calls)}


def geomean(values):
    return 0. if 0 in values else math.exp(stats.mean(math.log(v) for v in values))


def ratio_statistics(pairs, field, seed, samples):
    by_task = defaultdict(list)
    for pair in pairs:
        value = pair[field]
        if value is not None:
            by_task[pair['task_id']].append(value)
    # Equal task weight; both passes stay together in every bootstrap draw.
    cluster_means = [geomean(v) for v in by_task.values()]
    raw = [v for vs in by_task.values() for v in vs]
    rng = random.Random(seed)
    boot = sorted(geomean(rng.choices(cluster_means, k=len(cluster_means)))
                  for _ in range(samples)) if len(cluster_means) > 1 else []
    return {'geomean': geomean(cluster_means) if cluster_means else None,
            'median': stats.median(raw) if raw else None, 'task_clusters': len(cluster_means),
            'pairs': len(raw), 'undefined_denominator_pairs': len(pairs) - len(raw),
            'bootstrap_95': [boot[int(.025 * samples)], boot[min(samples - 1, int(.975 * samples))]] if boot else None}


def compare(tasks, treatment, passes, seed, samples):
    pairs, matrix, missing = [], Counter(), 0
    for p in passes:
        ids = {task for arm, pass_number, task in tasks if arm in ('control', treatment) and pass_number == p}
        for task_id in sorted(ids):
            control, treated = tasks.get(('control', p, task_id)), tasks.get((treatment, p, task_id))
            if control is None or treated is None:
                missing += 1
                continue
            key = ('both_pass' if treated['passed'] else 'control_only') if control['passed'] else (
                'treatment_only' if treated['passed'] else 'both_fail')
            matrix[key] += 1
            cb, tb = control['budgets'], treated['budgets']
            budget = 'unknown' if not cb or not tb or None in cb or None in tb else (
                'matched' if cb == tb and len(cb) == 1 else 'mismatch')
            pairs.append({'task_id': task_id, 'pass': p,
                          'control_passed': control['passed'], 'treatment_passed': treated['passed'],
                          'wall_ratio': divide(treated['seconds'], control['seconds']),
                          'reasoning_token_ratio': divide(treated['reasoning_tokens'], control['reasoning_tokens']),
                          'output_token_ratio': divide(treated['output_tokens'], control['output_tokens']),
                          'budget_guard': budget,
                          'control_length_bound': control['length_calls'] > 0,
                          'treatment_length_bound': treated['length_calls'] > 0})
    return {'treatment': treatment, 'control': 'control', 'missing_pairs': missing, 'pairs': pairs,
            'pass_agreement': {k: matrix[k] for k in ('both_pass', 'control_only', 'treatment_only', 'both_fail')},
            'budget_mismatch_pairs': sum(p['budget_guard'] == 'mismatch' for p in pairs),
            'budget_unknown_pairs': sum(p['budget_guard'] == 'unknown' for p in pairs),
            'ratios': {field: ratio_statistics(pairs, field, seed, samples)
                       for field in ('wall_ratio', 'reasoning_token_ratio', 'output_token_ratio')}}


def activation(events, calls, arms):
    proofs = {arm: {'verify_rows': 0, 'relaxed': 0, 'failures': Counter()} for arm in arms}
    other = ignored = invalid_time = 0
    for row in events:
        if row.get('event') != 'verify':
            ignored += 1
            continue
        try:
            time = number(row.get('time'))
        except InputError:
            invalid_time += 1
            continue
        owners = {c['arm'] for c in calls if c['t'] <= time <= c['t'] + c['wall_s']}
        if not owners:
            other += 1
            continue
        label = row.get('label')
        for arm in owners:
            proof = proofs[arm]
            expected = '' if arm == 'control' else 'think-tasks-' + arm
            if len(owners) > 1:
                proof['failures']['overlapping_arm_windows'] += 1
            if label != expected:
                proof['failures']['label_or_arm_window_mismatch'] += 1
                continue
            proof['verify_rows'] += 1
            try:
                relaxed = number(row.get('relaxed'))
            except InputError:
                proof['failures']['missing_or_invalid_relaxed'] += 1
                continue
            proof['relaxed'] += relaxed
            if arm == 'control':
                if relaxed != 0:
                    proof['failures']['control_relaxed'] += 1
                if row.get('lossy_margin') is not None or 'lossy_margin' not in row:
                    proof['failures']['control_margin_not_null'] += 1
            else:
                if row.get('lossy_margin') != ARMS[arm]:
                    proof['failures']['margin_mismatch'] += 1
                if row.get('lossy_scope') != 'think':
                    proof['failures']['scope_mismatch'] += 1
                if row.get('lossy_enabled') is not True:
                    proof['failures']['lossy_not_enabled'] += 1
            if row.get('eligible') is not True or row.get('dropped') != 0 or row.get('writer_error') is not None:
                proof['failures']['incomplete_or_ineligible_telemetry'] += 1
    for arm, proof in proofs.items():
        if not proof['verify_rows']:
            proof['failures']['missing_label_rows'] += 1
        if arm != 'control' and proof['relaxed'] <= 0:
            proof['failures']['no_relaxation'] += 1
        proof['failures'] = dict(proof['failures'])
        proof['proven'] = not proof['failures'] and invalid_time == 0
    return {'status': 'proven' if all(p['proven'] for p in proofs.values()) else 'failed',
            'arms': proofs, 'other_traffic_verify_rows': other, 'non_verify_rows': ignored,
            'invalid_time_rows': invalid_time}


def build_report(root, *, arms=('control', 'm2.5', 'm5.0'), passes=(1, 2), expected_tasks=12,
                 trace=None, no_trace=False, allow_partial=False, seed=20260910, bootstrap_samples=10000):
    if bool(trace) == bool(no_trace):
        raise InputError('provide --trace or explicitly --no-trace')
    if not arms or len(set(arms)) != len(arms) or 'control' not in arms or any(a not in ARMS for a in arms):
        raise InputError('select control and recognized treatment arms')
    if len(arms) < 2 or expected_tasks < 1 or bootstrap_samples < 100 or not passes or len(set(passes)) != len(passes) or any(p < 1 for p in passes):
        raise InputError('invalid task/pass/bootstrap configuration')
    if trace and not Path(trace).is_file():
        raise InputError('trace is missing; explicitly use --no-trace to mark activation unproven')
    cells, tasks, calls, pending, issues = load_tree(root, arms, passes, expected_tasks, allow_partial)
    per_pass, pooled = {}, {}
    for arm in arms:
        per_pass[arm] = {}
        for p in passes:
            cell = cells[arm, p]
            per_pass[arm][str(p)] = aggregate(cell['tasks'], [c for c in calls if c['arm'] == arm and c.get('pass_number') == p], [cell['summary_wall_s']])
        selected_tasks = [t for (a, _, _), t in tasks.items() if a == arm]
        selected_calls = [c for c in calls if c['arm'] == arm and 'pass_number' in c]
        pooled[arm] = aggregate(selected_tasks, selected_calls, [cells[arm, p]['summary_wall_s'] for p in passes])
    comparisons = {arm: compare(tasks, arm, passes, seed, bootstrap_samples) for arm in arms if arm != 'control'}
    for arm, comparison in comparisons.items():
        if comparison['budget_mismatch_pairs']:
            issues.append(f'{arm}:completion_budget_mismatch')
        if comparison['budget_unknown_pairs']:
            issues.append(f'{arm}:completion_budget_unknown')
    proof = {'status': 'unproven', 'reason': 'explicit --no-trace'} if no_trace else activation(read_jsonl(Path(trace), allow_partial, pending), calls, arms)
    status = 'invalid' if issues or proof['status'] == 'failed' else (
        'partial' if pending else 'activation_unproven' if no_trace else 'complete')
    return {'status': status, 'promotion_ready': status == 'complete',
            'header': 'Hold 9 protocol: pi thinking high; PI_HE_MAX_TOKENS=8192, reduced from production 32768. Observed budget deviations are reported below.',
            'expected_tasks_per_arm_pass': expected_tasks, 'passes': list(passes),
            'allow_partial': allow_partial, 'pending_counts': dict(Counter(pending)), 'issues': dict(Counter(issues)),
            'per_arm_pass': per_pass, 'pooled_per_arm': pooled, 'comparisons': comparisons, 'activation': proof,
            'budget_deviation_arms': [a for a, r in pooled.items() if r['completion_budgets'] != [8192] or r['unknown_budget_calls']],
            'unassigned_proxy_calls': sum('pass_number' not in c for c in calls),
            'bootstrap': {'seed': seed, 'samples': bootstrap_samples, 'unit': 'task_id; paired passes remain clustered'},
            'notes': ['Official check.passed only; check output, errors, request payloads and private texts are never copied.',
                      'Wall is summed lane-summary elapsed time; pooled median is over individual task times.',
                      'Pooled decode = sum proxy generation_tokens / sum proxy server_decode_s.',
                      'Accepted/cycle = 1 + sum spec_accepted_tokens / sum spec_drafts; includes the bonus token.',
                      'Reasoning character share uses saved counts; it is not a tokenizer-derived token share.',
                      'Ratios are treatment/control; below one means less time or fewer tokens, not necessarily better quality.',
                      'Geomean gives each task equal weight; 95% percentile bootstrap resamples task clusters with all available paired passes.',
                      'Zero denominators are undefined and counted; zero numerators produce zero ratios/geomeans.',
                      'Trace time uses proxy t as call START; windows are inclusive [t,t+wall_s].',
                      'Verify rows outside every proxy window are other traffic, including known labels; labels inside the wrong arm window fail.',
                      'Length-bound tasks have at least one proxy finish_reason=length; budgets must match for each paired task.',
                      'Partial input may be an inconsistent live snapshot and is never promotion-ready.']}


def fmt(value):
    return 'n/a' if value is None else f'{value:.4g}'


def markdown(report):
    lines = ['# Task-level think-lossy report', '', report['header'], '',
             f"Status: **{report['status']}**. Activation: **{report['activation']['status']}**.", '',
             '| Arm | Pass | Tasks | Passed | Errors | Wall s | Median task s | Output | Reasoning | Calls | Length calls/tasks | Decode tok/s | Accepted/cycle | Reasoning char share |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, passes in report['per_arm_pass'].items():
        for p, r in [*passes.items(), ('pooled', report['pooled_per_arm'][arm])]:
            values = [arm, p, *[fmt(r[k]) for k in ('tasks', 'passed', 'errors', 'wall_s', 'median_task_s', 'output_tokens', 'reasoning_tokens', 'api_calls')],
                      f"{r['length_calls']}/{r['length_bound_tasks']}", *[fmt(r[k]) for k in ('pooled_decode_tok_s', 'accepted_per_cycle', 'reasoning_chars_share')]]
            lines.append('| ' + ' | '.join(values) + ' |')
    for arm, comp in report['comparisons'].items():
        lines += ['', f'## {arm} / control', '', 'Pass agreement: ' + json.dumps(comp['pass_agreement']) + '.', '',
                  '| Ratio | Task geomean | Pair median | Bootstrap 95% | Undefined pairs |', '|---|---:|---:|---|---:|']
        for field, r in comp['ratios'].items():
            interval = 'n/a' if r['bootstrap_95'] is None else ', '.join(fmt(x) for x in r['bootstrap_95'])
            lines.append(f"| {field} | {fmt(r['geomean'])} | {fmt(r['median'])} | {interval} | {r['undefined_denominator_pairs']} |")
        lines += ['', '| Task | Pass | Control pass | Treatment pass | Wall ratio | Reasoning ratio | Output ratio | Budget | Length-bound C/T |', '|---|---:|---|---|---:|---:|---:|---|---|']
        for p in comp['pairs']:
            lines.append('| ' + ' | '.join([p['task_id'], str(p['pass']), str(p['control_passed']), str(p['treatment_passed']),
                         *[fmt(p[k]) for k in ('wall_ratio', 'reasoning_token_ratio', 'output_token_ratio')],
                         p['budget_guard'], f"{p['control_length_bound']}/{p['treatment_length_bound']}"]) + ' |')
    lines += ['', '## Activation and guards', '', '```json', json.dumps({k: report[k] for k in ('activation', 'issues', 'pending_counts', 'budget_deviation_arms', 'unassigned_proxy_calls')}, indent=2), '```', '', 'Conventions:', '']
    lines += ['- ' + n for n in report['notes']]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path, required=True, help='Output prefix: writes PREFIX.json and PREFIX.md')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--trace', type=Path)
    group.add_argument('--no-trace', action='store_true')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--arms', default='control,m2.5,m5.0')
    parser.add_argument('--passes', default='1,2')
    parser.add_argument('--expected-tasks', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20260910)
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    args = parser.parse_args(argv)
    try:
        try:
            passes = tuple(int(p) for p in args.passes.split(','))
        except ValueError:
            raise InputError('passes must be comma-separated positive integers') from None
        result = build_report(args.directory, arms=tuple(args.arms.split(',')), passes=passes,
                              expected_tasks=args.expected_tasks, trace=args.trace, no_trace=args.no_trace,
                              allow_partial=args.allow_partial, seed=args.seed, bootstrap_samples=args.bootstrap_samples)
        outputs = [Path(str(args.out) + ext) for ext in ('.json', '.md')]
        # A prefix must never overwrite an input, even when the user makes a typo.
        if any(p.name in ('summary.json', 'result.json') or p.name.endswith('-summary.json')
               or p.resolve() == (args.trace.resolve() if args.trace else None) for p in outputs):
            raise InputError('output prefix would overwrite input')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        outputs[0].write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        outputs[1].write_text(markdown(result))
    except InputError as error:
        parser.exit(2, f'Cannot report: {error}\n')
    print(f"Task report: {result['status']}; activation {result['activation']['status']}.")
    return 1 if result['status'] == 'invalid' else 0


if __name__ == '__main__':
    raise SystemExit(main())
