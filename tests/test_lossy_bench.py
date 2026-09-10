"""Offline wire, ordering and trace-proof regression tests for the lossy bench."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import adaptive_spec as bench
from spec_request_report import cycle_statistics, ending_statistics, summarize

GRID = [f'lossy-m{m}{p}' for m in ('0.5', '1.0', '1.5', '2.5') for p in ('', '-p0.1')]


@pytest.mark.parametrize('name', ['fixed1', 'fixed3', 'fixed5', 'fixed7', 'adaptive', 'shadow'])
def test_existing_variant_requests_are_byte_identical(name):
    mode, cap = bench.parse_variants(name)[0]
    # Literal original request schema and insertion order, including absence of
    # lossy fields. Comparing JSON bytes catches numeric and ordering changes.
    original = {'model': 'glm-5.3', 'messages': [{'role': 'user', 'content': 'a prompt'}],
                'temperature': 0, 'seed': 42, 'max_tokens': 256,
                'stream': True, 'stream_options': {'include_usage': True},
                'return_token_ids': True, 'chat_template_kwargs': {'enable_thinking': False},
                'vllm_xargs': {'spec_policy': mode, 'spec_verify_cap': cap, 'spec_label': 'label'}}
    assert json.dumps(bench.variant_body('a prompt', mode, cap, 'label', 256)).encode() == json.dumps(original).encode()


@pytest.mark.parametrize('name,margin,min_p', [(v, float(v.split('m')[1].split('-')[0]), .1 if '-p' in v else 0.) for v in GRID])
def test_lossy_variants_add_only_three_numeric_fields_to_fixed7(name, margin, min_p):
    mode, cap = bench.parse_variants(name)[0]
    body = bench.variant_body('a prompt', mode, cap, 'label', 256)
    for key, expected in [('spec_lossy_margin', margin), ('spec_lossy_min_p', min_p)]:
        assert type(body['vllm_xargs'][key]) is float
        assert body['vllm_xargs'].pop(key) == expected
    assert body['vllm_xargs'].pop('spec_lossy_rank') == 2
    assert body == bench.request_body('a prompt', 7, 'label', 256)


@pytest.mark.parametrize('name', ['lossy-m0', 'lossy-m5.1', 'lossy-m1-p0.5', 'lossy-m1-p-1',
                                 'lossy-mnan', 'lossy-minf', 'lossy-m-1', 'lossy-m1-r3',
                                 'lossy-m1-pnan', 'lossy-m1/../../x', 'fixed2'])
def test_bad_variants_fail_before_requests(name):
    with pytest.raises(ValueError):
        bench.parse_variants(name)


@pytest.mark.parametrize('corpus,count', [('spec-development.json', 6), ('spec-heldout.json', 15)])
def test_prose_selector_uses_exact_corpus_subset(corpus, count):
    prompts = json.loads((ROOT / 'bench' / corpus).read_text())
    selected = bench.select_cases(prompts, prose_only=True)
    assert len(selected) == count and all(name.startswith('prose_') for name in selected)
    assert bench.select_cases(prompts, ','.join(list(prompts)[:1] + selected[:1]), True) == selected[:1]
    with pytest.raises(ValueError):
        bench.select_cases(prompts, 'absent', True)


def run_offline(monkeypatch, tmp_path, variants, corpus='spec-development.json', repeats=2, extra=()):
    bodies = []
    class Guard:
        def __init__(self, path): pass
        def start(self): return self
        def preflight(self, **kwargs): return {}
        def close(self): pass
    def generate(base, body, guard, deadline):
        bodies.append(copy.deepcopy(body))
        return {'text': 'prose', 'token_ids': [1, 2], 'decode_tps': 20, 'ttft': .5,
                'token_sha256': 'hash', 'usage': {'completion_tokens': 2},
                'chunks': [{'data': {'prompt_token_ids': [10, 11]}}]}
    monkeypatch.setattr(bench, 'MemoryGuard', Guard)
    monkeypatch.setattr(bench, 'generate', generate)
    monkeypatch.setattr(bench, 'idle_check', lambda base: None)
    monkeypatch.setattr(bench, 'spec_metrics', lambda base: {})
    out = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['adaptive_spec', '--out', str(out), '--corpus', str(ROOT / 'bench' / corpus),
                                    '--prose-only', '--variants', variants, '--repeats', str(repeats), *extra])
    bench.main()
    return bodies, out


def test_development_grid_has_96_lossy_requests_and_12_interleaved_anchors(monkeypatch, tmp_path):
    bodies, out = run_offline(monkeypatch, tmp_path, ','.join(['fixed7', *GRID]))
    assert len(bodies) == 108
    assert sum('spec_lossy_margin' not in body['vllm_xargs'] for body in bodies) == 12
    for i in range(0, 108, 9):
        block = bodies[i:i + 9]
        assert len({body['messages'][0]['content'] for body in block}) == 1
        assert sum('spec_lossy_margin' not in body['vllm_xargs'] for body in block) == 1
    results = [json.loads(path.read_text()) for path in out.glob('*.json') if path.name != 'config.json']
    assert len({(r['case'], r['repeat'], r['variant']) for r in results}) == 108
    assert all(row['study'] == 'lossy' and row['message_sha256'] for row in results)
    assert {row['policy'] for row in results} == {'fixed', *GRID}


def test_heldout_is_15_by_3_paired_ab_ba_with_one_thinking_repeat(monkeypatch, tmp_path):
    bodies, _ = run_offline(monkeypatch, tmp_path, 'fixed7,lossy-m1.0', 'spec-heldout.json', 3,
                            ['--thinking-repeat', '2'])
    assert len(bodies) == 90
    for repeat in range(3):
        for case in range(15):
            pair = bodies[(repeat * 15 + case) * 2:(repeat * 15 + case + 1) * 2]
            assert ('spec_lossy_margin' in pair[0]['vllm_xargs']) == bool((case + repeat) % 2)
            assert pair[0]['messages'] == pair[1]['messages']
            assert all(body['chat_template_kwargs']['enable_thinking'] == (repeat == 2) for body in pair)


def lossy_pair(case='prose_a', repeat=0):
    rows = []
    for variant in ('fixed7', 'lossy-m1.0-p0.1'):
        rows.append({'study': 'lossy', 'variant': variant, 'policy': 'fixed' if variant == 'fixed7' else variant,
                     'cap': 7, 'case': case, 'repeat': repeat, 'label': f'{case}-{repeat}-{variant}',
                     'decode_tps': 10 if variant == 'fixed7' else 12, 'ttft': .5,
                     'message_sha256': 'messages', 'token_sha256': 'ids', 'token_ids': [1, 2, 3],
                     'chunks': [{'seconds': .5, 'data': {'prompt_token_ids': [5, 6], 'choices': [{'token_ids': [1]}]}},
                                {'seconds': .6, 'data': {'choices': [{'token_ids': [2, 3]}]}}]})
    return rows


def verify(row, accepted=2, relaxed=None, **extra):
    lossy = row['variant'] != 'fixed7'
    return {'event': 'verify', 'label': row['label'], 'mode': 'fixed', 'eligible': True,
            'scheduled_k': 7, 'dropped': 0, 'writer_error': None, 'accepted': accepted,
            **({'lossy_margin': 1., 'lossy_min_p': .1, 'lossy_enabled': True,
                'relaxed': 1 if relaxed is None else relaxed} if lossy else {}), **extra}


def test_lossy_proof_pools_counts_and_reports_paired_divergence():
    runs = lossy_pair()
    runs[1]['token_ids'] = [1, 9, 3]
    events = [verify(runs[0]), verify(runs[1]), verify(runs[1], accepted=4, relaxed=0)]
    result = summarize(runs, events)
    pooled = result['summaries']['lossy-m1.0-p0.1']
    assert pooled['relaxed_per_cycle'] == .5
    assert pooled['relaxed_per_scheduled_position'] == pytest.approx(1 / 14)
    assert pooled['accepted_per_cycle'] == 3
    assert pooled['p95_emission_gap_s'] == pytest.approx(.1)
    assert result['comparisons'][0]['first_divergence_position'] == 1
    assert result['comparisons'][0]['agreement_first64'] == pytest.approx(2 / 3)
    assert result['paired']['prose/lossy-m1.0-p0.1']['paired_geomean_ratio'] == pytest.approx(1.2)
    assert pooled['per_position'][0]['relaxed_accepts'] is None
    assert pooled['per_position'][0]['relaxed_accepts_bounds'] == [0, 1]


@pytest.mark.parametrize('field,value', [('lossy_margin', None), ('lossy_margin', 2.), ('lossy_margin', float('nan')),
                                        ('lossy_min_p', 0.), ('lossy_enabled', False), ('relaxed', 0),
                                        ('relaxed', -1), ('relaxed', 3), ('relaxed', True),
                                        ('eligible', False), ('dropped', 1), ('writer_error', 'full'),
                                        ('scheduled_k', 3), ('mode', 'adaptive')])
def test_inactive_wrong_or_incomplete_lossy_trace_fails(field, value):
    runs = lossy_pair()
    events = [verify(row) for row in runs]
    events[1][field] = value
    with pytest.raises(ValueError):
        summarize(runs, events)


def test_every_control_row_must_be_exact_and_every_lossy_request_must_fire():
    runs = lossy_pair() + lossy_pair(repeat=1)
    events = [verify(row) for row in runs]
    summarize(runs, events)  # Older control traces omit all lossy fields.
    events[2]['relaxed'] = 1
    with pytest.raises(ValueError, match='control unexpectedly relaxed'):
        summarize(runs, events)
    events[2].pop('relaxed')
    events[3]['relaxed'] = 0
    with pytest.raises(ValueError, match='no relaxed accepts'):
        summarize(runs, events)
    events.pop()
    with pytest.raises(ValueError, match='missing runtime'):
        summarize(runs, events)


def test_following_curve_is_next_cycle_and_does_not_bridge_requests_or_censoring():
    seq = [{'scheduled_k': 7, 'accepted': a, 'relaxed': r} for a, r in [(2, 1), (4, 0), (1, 0), (7, 7)]]
    seq[-1]['terminal'] = True
    result = cycle_statistics([seq, [{'scheduled_k': 7, 'accepted': 0}]])
    assert result['cycles'] == 4 and result['excluded_cycles'] == 1
    assert [r['accepted'] for r in result['following_relaxed']] == [1, 1, 1, 1, 0, 0, 0]
    assert [r['accepted'] for r in result['following_exact']] == [1, 0, 0, 0, 0, 0, 0]
    assert all(r['cycles'] == 1 for r in result['following_exact'])
    assert result['relaxed_per_cycle'] == .25
    seq[1]['learned'] = False
    result = cycle_statistics([seq])
    assert all(r['cycles'] == 0 for r in result['following_relaxed'] + result['following_exact'])


def test_scheduling_order_wins_over_async_trace_order_and_duplicate_cycles_fail():
    runs = lossy_pair()
    first = verify(runs[1], accepted=2, decision_ns=10)
    second = verify(runs[1], accepted=5, relaxed=0, decision_ns=20)
    events = [verify(runs[0]), second, first]
    result = summarize(runs, events)['summaries'][runs[1]['variant']]
    assert result['following_relaxed'][4]['acceptance'] == 1
    assert result['following_relaxed'][5]['acceptance'] == 0
    with pytest.raises(ValueError, match='duplicate verify'):
        summarize(runs, events + [first])


def test_missing_mismatched_and_duplicate_lossy_pairs_fail():
    runs = lossy_pair()
    events = [verify(row) for row in runs]
    with pytest.raises(ValueError, match='duplicate'):
        summarize(runs + [runs[0]], events)
    runs += lossy_pair(repeat=1)[:1]
    with pytest.raises(ValueError, match='incomplete'):
        summarize(runs, events + [verify(runs[-1])])
    runs = lossy_pair()
    runs[1]['message_sha256'] = 'different'
    with pytest.raises(ValueError, match='different prompts'):
        summarize(runs, events)


def test_pooled_relaxation_uses_cycles_not_equal_request_weights():
    runs = lossy_pair() + lossy_pair(repeat=1)
    events = [verify(row) for row in runs] + [verify(runs[-1], relaxed=0) for _ in range(8)]
    result = summarize(runs, events)['summaries'][runs[1]['variant']]
    assert result['relaxed_per_cycle'] == .2  # (1+1)/(1+9), not (1+1/9)/2.
    exact = cycle_statistics([[{'scheduled_k': 7, 'accepted': 2, 'relaxed': 2}]])
    assert [r['relaxed_accepts'] for r in exact['per_position']] == [1, 1, 0, 0, 0, 0, 0]


def test_long_generation_reports_token_repetition_and_server_end_reason():
    row = lossy_pair()[0]
    row['token_ids'] = [1, 2, 3, 4] * 3
    row['chunks'][-1]['data']['choices'][0]['finish_reason'] = 'length'
    ending = ending_statistics(row)
    assert ending['output_4gram_repetition'] == pytest.approx(5 / 9)
    assert ending['finish_reasons'] == ['length']
    assert ending['output_tokens'] == 12


def test_repo_context_prose_selector_is_applied_after_build(monkeypatch, tmp_path):
    import spec_repo_context
    monkeypatch.setattr(spec_repo_context, 'build', lambda *args: {'code_repo_context': 'code', 'prose_repo_context': 'prose'})
    bodies, _ = run_offline(monkeypatch, tmp_path, 'fixed7,lossy-m1.0',
                            extra=['--repo-context', '32000'])
    assert len(bodies) == 4
    assert all(body['messages'][0]['content'] == 'prose' for body in bodies)


def test_constrained_corpus_is_unwrapped_before_generation(monkeypatch, tmp_path):
    bodies, _ = run_offline(monkeypatch, tmp_path, 'fixed7,lossy-m1.0',
                            'spec-prose-checks.json', 1, ['--tokens', '2048'])
    assert len(bodies) == 40
    assert all(isinstance(body['messages'][0]['content'], str) and body['max_tokens'] == 2048 for body in bodies)


def test_request_report_cli_consumes_runner_artifacts_and_refuses_missing_whole_repeat(monkeypatch, tmp_path):
    import spec_request_report
    _, out = run_offline(monkeypatch, tmp_path, 'fixed7,lossy-m1.0', extra=['--cases', 'prose_heat'])
    files = [path for path in out.glob('*.json') if path.name != 'config.json']
    rows = [json.loads(path.read_text()) for path in files]
    events = [verify(row, **({'lossy_min_p': 0.} if row['variant'] != 'fixed7' else {})) for row in rows]
    trace = tmp_path / 'trace.jsonl'
    trace.write_text(''.join(json.dumps(event) + '\n' for event in events))
    monkeypatch.setattr(sys, 'argv', ['spec_request_report', str(out), '--trace', str(trace)])
    spec_request_report.main()
    report = json.loads((out / 'request-control-report.json').read_text())
    assert report['study'] == 'lossy' and len(report['requests']) == 4
    assert str(trace) in report['source_sha256']
    for path, row in zip(files, rows):
        if row['repeat'] == 1:
            path.unlink()
    with pytest.raises(ValueError, match='incomplete'):
        spec_request_report.main()
