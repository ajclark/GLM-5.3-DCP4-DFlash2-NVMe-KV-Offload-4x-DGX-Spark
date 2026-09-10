"""Synthetic hold-9 fixtures only: no live result tree or endpoint reads."""
import copy
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import spec_think_tasks_report as report

SECRET = 'PRIVATE_SYNTHETIC_MARKER_DO_NOT_EXPORT'
ARMS = ('control', 'm2.5')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / 'hold'
    root.mkdir()
    traces, calls = [], {a: [] for a in ARMS}
    for p in (1, 2):
        for arm_i, arm in enumerate(ARMS):
            folder = root / f'{arm}-pass{p}'
            cell = folder / arm / 'c01'
            tasks = []
            for n in range(3):
                start = p * 1000 + arm_i * 100 + n * 20
                seconds = 10 if arm == 'control' else 5
                passed = n != (2 if arm == 'control' else 1)
                task = {'task_id': f'HumanEval/{n}', 'start': start, 'end': start + seconds,
                        'seconds': seconds, 'request_seconds': 4., 'post_first_seconds': 3.,
                        'output_tokens': 100 if arm == 'control' else 50,
                        'reasoning_tokens': 40 if arm == 'control' else 10,
                        'input_tokens': 1000, 'requests': [{'payload': SECRET}], 'errors': [],
                        'check': {'passed': passed, 'result': 'passed' if passed else 'failed',
                                  'returncode': 0 if passed else 1, 'output': SECRET}}
                dump(cell / f'HumanEval_{n}' / 'result.json', task)
                tasks.append(task)
                call = {'t': start + 1, 'wall_s': 3., 'task': SECRET,
                        'generation_tokens': task['output_tokens'], 'server_decode_s': 2. if n == 0 else 4.,
                        'api_calls': 1, 'spec_drafts': 10, 'spec_accepted_tokens': 50,
                        'reasoning_chars': 20, 'content_chars': 10, 'toolcall_chars': 70,
                        'finish_reason': 'length' if n == 2 else 'tool_calls',
                        'request_params': {'max_completion_tokens': 8192, 'temperature': 0, 'top_p': 1,
                                           'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': 'high'}},
                        'reasoning_text': SECRET, 'content_text': SECRET, 'tool_args_text': SECRET,
                        'messages': [{'role': 'user', 'content': SECRET}], 'tools': [{'private': SECRET}],
                        'content_tail': SECRET, 'reasoning_tail': SECRET, 'tool_arg_sample': SECRET}
                calls[arm].append(call)
                traces.append({'event': 'verify', 'time': start + 2, 'label': '' if arm == 'control' else 'think-tasks-m2.5',
                               'request': SECRET, 'relaxed': 0 if arm == 'control' else 2,
                               'lossy_margin': None if arm == 'control' else 2.5,
                               'lossy_scope': 'all' if arm == 'control' else 'think',
                               'lossy_enabled': arm != 'control', 'eligible': True, 'dropped': 0,
                               'writer_error': None, 'accepted': 5, 'sampled': 6, 'scheduled_k': 7,
                               'cycle_ms': 150, 'terminal': False})
            summary = {'lane': arm, 'concurrency': 1, 'tasks': 3,
                       'passed': sum(t['check']['passed'] for t in tasks), 'errors': 0,
                       'output_tokens': sum(t['output_tokens'] for t in tasks),
                       'reasoning_tokens': sum(t['reasoning_tokens'] for t in tasks),
                       'wall_s': tasks[-1]['end'] - tasks[0]['start'], 'api_calls': 3,
                       'validity_flags': [], 'metric_delta': {'private': SECRET}}
            dump(folder / f'{arm}-summary.json', [summary])
            dump(cell / 'summary.json', summary)
    for arm, rows in calls.items():
        jsonl(root / f'proxy-{arm}-calls.jsonl', rows)
    trace = tmp_path / 'trace.jsonl'
    jsonl(trace, traces)
    return root, trace


def build(experiment, **kwargs):
    root, trace = experiment
    options = dict(arms=ARMS, expected_tasks=3, trace=trace, bootstrap_samples=300)
    options.update(kwargs)
    return report.build_report(root, **options)


def mutate_trace(experiment, change):
    path = experiment[1]
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    change(rows)
    jsonl(path, rows)


def test_complete_pooling_pairs_and_clustered_bootstrap(experiment):
    r = build(experiment)
    assert r['status'] == 'complete'
    assert r['promotion_ready'] is True
    assert r['activation']['status'] == 'proven'
    assert r['per_arm_pass']['control']['1']['tasks'] == 3
    c = r['pooled_per_arm']['control']
    assert (c['tasks'], c['passed'], c['api_calls'], c['wall_s']) == (6, 4, 6, 100)
    assert c['pooled_decode_tok_s'] == 600 / 20
    assert c['accepted_per_cycle'] == 6
    assert c['reasoning_chars_share'] == .2
    assert c['length_calls'] == c['length_bound_tasks'] == 2
    comp = r['comparisons']['m2.5']
    assert comp['pass_agreement'] == {'both_pass': 2, 'control_only': 2, 'treatment_only': 2, 'both_fail': 0}
    assert comp['ratios']['wall_ratio']['geomean'] == .5
    assert comp['ratios']['wall_ratio']['bootstrap_95'] == [.5, .5]
    assert comp['ratios']['reasoning_token_ratio']['geomean'] == .25
    assert comp['ratios']['wall_ratio']['task_clusters'] == 3
    assert comp['ratios']['wall_ratio']['pairs'] == 6
    assert r['bootstrap']['unit'].startswith('task_id')


def test_control_relaxed_failure(experiment):
    mutate_trace(experiment, lambda rows: rows[0].update(relaxed=1))
    r = build(experiment)
    assert r['status'] == 'invalid'
    assert r['activation']['arms']['control']['failures']['control_relaxed'] == 1


def test_missing_label_failure_even_with_other_correct_rows(experiment):
    mutate_trace(experiment, lambda rows: rows[3].update(label=''))
    r = build(experiment)
    assert r['activation']['status'] == 'failed'
    assert r['activation']['arms']['m2.5']['failures']['label_or_arm_window_mismatch'] == 1


def test_no_treatment_labels_fails(experiment):
    def change(rows):
        for row in rows:
            if row['label']:
                row['label'] = ''
    mutate_trace(experiment, change)
    r = build(experiment)
    assert r['activation']['arms']['m2.5']['failures']['missing_label_rows'] == 1


def test_time_window_violation_wrong_arm(experiment):
    # A treatment-labeled row during a real control call cannot prove treatment.
    mutate_trace(experiment, lambda rows: rows[3].update(time=rows[0]['time']))
    r = build(experiment)
    assert r['status'] == 'invalid'
    assert r['activation']['arms']['control']['failures']['label_or_arm_window_mismatch'] == 1


def test_outside_every_window_is_other_traffic_even_known_label(experiment):
    mutate_trace(experiment, lambda rows: rows.extend([{**rows[3], 'time': 99999, 'relaxed': 999},
                                                       {'event': 'unrelated', 'private': SECRET}]))
    r = build(experiment)
    assert r['status'] == 'complete'
    assert r['activation']['other_traffic_verify_rows'] == 1
    assert r['activation']['non_verify_rows'] == 1
    assert r['activation']['arms']['m2.5']['relaxed'] == 12


@pytest.mark.parametrize('fields,key', [({'lossy_margin': 5.}, 'margin_mismatch'),
    ({'lossy_scope': 'all'}, 'scope_mismatch'), ({'lossy_enabled': False}, 'lossy_not_enabled'),
    ({'dropped': 1}, 'incomplete_or_ineligible_telemetry'),
    ({'writer_error': SECRET}, 'incomplete_or_ineligible_telemetry')])
def test_treatment_contract_failures(experiment, fields, key):
    mutate_trace(experiment, lambda rows: rows[3].update(fields))
    r = build(experiment)
    assert r['activation']['arms']['m2.5']['failures'][key] == 1
    assert SECRET not in json.dumps(r)


def test_zero_relaxation_is_unproven(experiment):
    def change(rows):
        for row in rows:
            row['relaxed'] = 0
    mutate_trace(experiment, change)
    r = build(experiment)
    assert r['activation']['arms']['m2.5']['failures']['no_relaxation'] == 1


def test_partial_missing_pass_and_preflight_before_private_reads(experiment, monkeypatch):
    root, trace = experiment
    shutil.rmtree(root / 'm2.5-pass2')
    with pytest.raises(report.InputError, match='completion summaries'):
        build(experiment)
    r = build(experiment, allow_partial=True)
    assert r['status'] == 'partial'
    assert r['promotion_ready'] is False
    assert r['comparisons']['m2.5']['missing_pairs'] == 3
    assert r['pooled_per_arm']['m2.5']['tasks'] == 3
    assert r['unassigned_proxy_calls'] == 3
    assert r['pooled_per_arm']['m2.5']['wall_complete'] is False


def test_partial_truncated_jsonl_tail(experiment):
    path = experiment[0] / 'proxy-control-calls.jsonl'
    with path.open('a') as stream:
        stream.write('{"private":"' + SECRET)
    r = build(experiment, allow_partial=True)
    assert r['pending_counts']['unfinished_jsonl_tail'] == 1
    assert r['status'] == 'partial'
    with pytest.raises(report.InputError, match='JSONL'):
        build(experiment)


def test_missing_trace_requires_explicit_no_trace(experiment):
    with pytest.raises(report.InputError, match='--trace'):
        build(experiment, trace=None)
    with pytest.raises(report.InputError, match='trace is missing'):
        build(experiment, trace=experiment[0] / 'absent.jsonl')
    r = build(experiment, trace=None, no_trace=True)
    assert r['status'] == 'activation_unproven'
    assert r['activation']['status'] == 'unproven'
    assert r['promotion_ready'] is False


def test_budget_mismatch_guard(experiment):
    path = experiment[0] / 'proxy-m2.5-calls.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    rows[0]['request_params']['max_completion_tokens'] = 32768
    jsonl(path, rows)
    r = build(experiment)
    assert r['status'] == 'invalid'
    assert r['comparisons']['m2.5']['budget_mismatch_pairs'] == 1
    assert r['budget_deviation_arms'] == ['m2.5']


def test_zero_denominator_and_zero_numerator_are_explicit():
    pairs = [{'task_id': 'HumanEval/0', 'x': None}, {'task_id': 'HumanEval/1', 'x': 0},
             {'task_id': 'HumanEval/2', 'x': 2}]
    r = report.ratio_statistics(pairs, 'x', 1, 100)
    assert r['undefined_denominator_pairs'] == 1
    assert r['geomean'] == 0
    assert r['median'] == 1


def test_bootstrap_clusters_repeats_and_is_seeded():
    pairs = [{'task_id': 'HumanEval/0', 'x': 1}, {'task_id': 'HumanEval/0', 'x': 4},
             {'task_id': 'HumanEval/1', 'x': 8}]
    r = report.ratio_statistics(pairs, 'x', 9, 1000)
    assert r == report.ratio_statistics(pairs, 'x', 9, 1000)
    assert r['geomean'] == pytest.approx(4)  # equal task weights, not pair weights
    assert r['bootstrap_95'] == pytest.approx([2, 8])


def test_cli_private_text_guard_json_markdown_and_stdout(experiment, tmp_path, capsys):
    root, trace = experiment
    prefix = tmp_path / 'aggregate-report'
    rc = report.main([str(root), '--out', str(prefix), '--trace', str(trace),
                      '--arms', 'control,m2.5', '--expected-tasks', '3', '--bootstrap-samples', '100'])
    assert rc == 0
    rendered = Path(str(prefix) + '.json').read_text() + Path(str(prefix) + '.md').read_text() + capsys.readouterr().out
    assert SECRET not in rendered
    assert '8192' in rendered and '32768' in rendered
    assert 'bonus token' in rendered
    assert 'HumanEval/0' in rendered


def test_cli_validation_failure_still_writes_reviewable_report(experiment, tmp_path):
    mutate_trace(experiment, lambda rows: rows[0].update(relaxed=1))
    prefix = tmp_path / 'invalid-report'
    rc = report.main([str(experiment[0]), '--out', str(prefix), '--trace', str(experiment[1]),
                      '--arms', 'control,m2.5', '--expected-tasks', '3', '--bootstrap-samples', '100'])
    assert rc == 1
    assert json.loads(Path(str(prefix) + '.json').read_text())['status'] == 'invalid'


def test_summary_validity_flag_text_not_exported(experiment):
    root, _ = experiment
    folder = root / 'control-pass1'
    summary = json.loads((folder / 'control-summary.json').read_text())[0]
    summary['validity_flags'] = [SECRET]
    dump(folder / 'control-summary.json', [summary])
    dump(folder / 'control' / 'c01' / 'summary.json', summary)
    r = build(experiment)
    assert r['status'] == 'invalid'
    assert SECRET not in json.dumps(r) + report.markdown(r)
