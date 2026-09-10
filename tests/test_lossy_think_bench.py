"""Think-scoped bench contracts, entirely offline; no model/tokenize calls."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import adaptive_spec as bench
import spec_code_check
import spec_request_report as report
import spec_think_check as quality
from spec_think_utils import completion_accounting, completion_parts

ARMS = 'fixed7,lossy-m2.5,lossy-think-m2.5,lossy-think-m5.0'


@pytest.mark.parametrize('name,margin,min_p', [('lossy-think-m2.5', 2.5, 0.), ('lossy-think-m5.0-p0.1', 5., .1)])
def test_think_variants_add_only_scope_to_existing_lossy_bytes(name, margin, min_p):
    body = bench.variant_body('p', name, 7, 'label', 1024)
    assert body['vllm_xargs'].pop('spec_lossy_scope') == 'think'
    original_name = name.replace('lossy-think-', 'lossy-')
    assert json.dumps(body).encode() == json.dumps(bench.variant_body('p', original_name, 7, 'label', 1024)).encode()
    assert body['vllm_xargs']['spec_lossy_margin'] == margin
    assert body['vllm_xargs']['spec_lossy_min_p'] == min_p
    assert bench.parse_variants(name) == [(name, 7)]


@pytest.mark.parametrize('name', ['lossy-m2.5', 'lossy-m5.0-p0.1'])
def test_existing_lossy_request_bytes_and_implicit_all_scope_remain_identical(name):
    expected = bench.request_body('p', 7, 'label', 1024)
    expected['vllm_xargs'].update(spec_lossy_margin=2.5 if name == 'lossy-m2.5' else 5.,
                                  spec_lossy_rank=2, spec_lossy_min_p=0. if name == 'lossy-m2.5' else .1)
    assert json.dumps(bench.variant_body('p', name, 7, 'label', 1024)).encode() == json.dumps(expected).encode()


@pytest.mark.parametrize('name', ['lossy-think-m0', 'lossy-think-m6', 'lossy-think-mnan',
                                 'lossy-think-m1-p0.5', 'lossy-Think-m1', 'lossy-think-m1-scope-all'])
def test_invalid_think_variants_fail(name):
    with pytest.raises(ValueError):
        bench.parse_variants(name)


class Guard:
    def __init__(self, *args): pass
    def start(self): return self
    def check(self): pass
    def preflight(self, **kwargs): return {}
    def close(self): pass


def result(variant='fixed7', thinking=True, case='prose_heat', repeat=0, duration=1.):
    return {'study': 'lossy', 'variant': variant, 'policy': 'fixed' if variant == 'fixed7' else variant,
            'thinking': thinking, 'max_tokens': 1024, 'case': case, 'repeat': repeat, 'cap': 7,
            'label': f'{case}-{repeat}-{variant}', 'text': 'Visible answer.', 'reasoning_text': 'Think carefully.',
            'token_ids': [1, 2, 3, 4], 'token_sha256': 'hash', 'message_sha256': 'prompt',
            'decode_tps': 2 / duration, 'ttft': .5, 'usage': {'completion_tokens': 4},
            'chunks': [{'seconds': .5, 'data': {'prompt_token_ids': [10, 11], 'choices': [{'token_ids': [1, 2]}]}},
                       {'seconds': .5 + duration, 'data': {'choices': [{'token_ids': [3, 4], 'finish_reason': 'stop'}]}}]}


def trace(row, relaxed=None, **extra):
    xargs = bench.lossy_xargs(row['variant']) if row['variant'] != 'fixed7' else {}
    event = {'event': 'verify', 'label': row['label'], 'mode': 'fixed', 'scheduled_k': 7,
             'eligible': True, 'dropped': 0, 'writer_error': None, 'accepted': 2,
             'relaxed': (int(bool(xargs) and (row['thinking'] or xargs.get('spec_lossy_scope') != 'think'))
                         if relaxed is None else relaxed)}
    if xargs:
        event.update(lossy_margin=xargs['spec_lossy_margin'], lossy_min_p=xargs['spec_lossy_min_p'],
                     lossy_scope=xargs.get('spec_lossy_scope', 'all'), lossy_enabled=1)
    return {**event, **extra}


@pytest.mark.parametrize('flags,arms,repeats,count', [(['--thinking'], ARMS, 2, 96),
    (['--thinking-repeat', '1'], ARMS, 2, 96), ([], 'fixed7,lossy-think-m5.0', 1, 24)])
def test_full_cli_schedule_same_thinking_and_budget_every_arm(monkeypatch, tmp_path, flags, arms, repeats, count):
    bodies = []
    def generate(base, body, guard, deadline):
        bodies.append(copy.deepcopy(body))
        return result()
    monkeypatch.setattr(bench, 'MemoryGuard', Guard)
    monkeypatch.setattr(bench, 'generate', generate)
    monkeypatch.setattr(bench, 'idle_check', lambda *a: None)
    monkeypatch.setattr(bench, 'spec_metrics', lambda *a: {})
    monkeypatch.setattr(bench.urllib.request, 'urlopen', lambda *a, **kw: pytest.fail('network forbidden'))
    out = tmp_path / 'think-dev'
    monkeypatch.setattr(sys, 'argv', ['adaptive_spec', '--out', str(out), '--corpus', str(ROOT/'bench/spec-development.json'),
        '--variants', arms, '--tokens', '1024', '--repeats', str(repeats), *flags])
    bench.main()
    assert len(bodies) == count
    rows = [json.loads(p.read_text()) for p in out.glob('*.json') if p.name != 'config.json']
    for row in rows:
        expected = '--thinking' in flags or ('--thinking-repeat' in flags and row['repeat'] == 1)
        assert row['thinking'] is expected and row['max_tokens'] == 1024
    width = len(arms.split(','))
    for start in range(0, count, width):
        block = bodies[start:start + width]
        assert len({b['max_tokens'] for b in block}) == 1
        assert len({b['chat_template_kwargs']['enable_thinking'] for b in block}) == 1
        assert len({b['messages'][0]['content'] for b in block}) == 1
        if width == 2:
            assert ('spec_lossy_scope' in block[0]['vllm_xargs']) == bool((start // width) % 2)


@pytest.mark.parametrize('separate,usage,expected,estimated', [
    ('reasoning', {'completion_tokens_details': {'reasoning_tokens': 7}}, 7, False),
    ('reasoning_content', {}, 10, True), (None, {}, 10, True)])
def test_streamed_reasoning_usage_field_or_raw_span_without_tokenize(monkeypatch, separate, usage, expected, estimated):
    words = ' '.join(['think'] * 10)
    delta = {separate: words, 'content': 'visible answer'} if separate else {'content': '<think>' + words + '</think>visible answer'}
    chunks = [{'choices': [{'delta': delta, 'token_ids': list(range(10))} ]},
              {'choices': [{'delta': {}, 'token_ids': [10, 11]}]},
              {'usage': {'completion_tokens': 12, **usage}, 'choices': []}]
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __iter__(self):
            return iter([('data: ' + json.dumps(c) + '\n').encode() for c in chunks])
    calls = []
    def open_mock(request, **kwargs):
        calls.append(request.full_url)
        assert request.full_url == 'http://offline.invalid/v1/chat/completions'
        return Response()
    monkeypatch.setattr(bench.urllib.request, 'urlopen', open_mock)
    body = bench.variant_body('p', 'lossy-think-m2.5', 7, 'l', 1024)
    body['chat_template_kwargs']['enable_thinking'] = True
    row = bench.generate('http://offline.invalid', body, Guard())
    assert calls == ['http://offline.invalid/v1/chat/completions']
    assert row['thinking'] is True and row['max_tokens'] == 1024
    assert row['reasoning_tokens'] == expected and row['reasoning_tokens_estimated'] is estimated
    assert '1.04' in row['reasoning_token_heuristic'] and row['visible_answer_words'] == 2


@pytest.mark.parametrize('text,reasoning,expected,visible', [
    ('a b</think>answer', '', 'a b', 'answer'),
    ('<think>a b', '', 'a b', ''),
    ('<think>a</think><think>b', '', 'a\nb', ''),
    ('answer', 'separate', 'separate', 'answer'),
    ('answer', '', '', 'answer'),
    ('<think>a</think>answer', 'a', 'a', 'answer')])
def test_visible_answer_and_prefilled_or_unclosed_spans(text, reasoning, expected, visible):
    assert completion_parts(text, reasoning) == (expected, visible)


def test_prefilled_unclosed_raw_span_is_not_a_visible_answer():
    assert completion_parts('still thinking', thinking=True) == ('still thinking', '')
    assert completion_parts('answer', thinking=True, reasoning_field_present=True) == ('', 'answer')
    assert completion_accounting('still thinking', thinking=True)['reasoning_tokens'] == 2


@pytest.mark.parametrize('bad', [-1, True, 100, '3'])
def test_invalid_provider_reasoning_counts_fail(bad):
    with pytest.raises(ValueError):
        completion_accounting('answer', 'thought', {'completion_tokens': 10, 'reasoning_tokens': bad})


@pytest.mark.parametrize('thinking', [True, False])
def test_think_proof_requires_firing_only_when_thinking_on(thinking):
    runs = [result(thinking=thinking), result('lossy-think-m2.5', thinking)]
    events = [trace(r) for r in runs]
    summary = report.summarize(runs, events)
    key = 'on' if thinking else 'off'
    assert summary['thinking_splits']['lossy-think-m2.5'][key]['pooled_decode_tps'] == 2
    events[1]['relaxed'] = 0 if thinking else 1
    with pytest.raises(ValueError, match='no relaxed accepts' if thinking else 'thinking-off.*relaxed'):
        report.summarize(runs, events)


@pytest.mark.parametrize('scope', [None, 'all', 'invalid'])
def test_think_scope_must_be_echoed_even_for_zero_relaxation(scope):
    runs = [result(thinking=False), result('lossy-think-m5.0', False)]
    events = [trace(r) for r in runs]
    events[1]['lossy_scope'] = scope
    with pytest.raises(ValueError, match='lossy_scope'):
        report.summarize(runs, events)
    events[1].pop('lossy_scope')
    with pytest.raises(ValueError, match='lossy_scope'):
        report.summarize(runs, events)


def test_think_off_must_still_be_enabled_and_controls_exact():
    runs = [result(thinking=False), result('lossy-think-m5.0', False)]
    events = [trace(r) for r in runs]
    events[1]['lossy_enabled'] = 0
    with pytest.raises(ValueError, match='not enabled'):
        report.summarize(runs, events)
    events[1]['lossy_enabled'] = 1
    events[0]['relaxed'] = 1
    with pytest.raises(ValueError, match='control unexpectedly relaxed'):
        report.summarize(runs, events)


def test_thinking_split_pools_tokens_and_seconds_not_request_rates():
    runs = [result(v, on, repeat=i, duration=d) for i, (on, d) in enumerate([(True, .25), (True, 1.), (False, 2.)])
            for v in ('fixed7', 'lossy-think-m2.5')]
    summary = report.summarize(runs, [trace(r) for r in runs])
    split = summary['thinking_splits']['lossy-think-m2.5']
    assert split['on']['pooled_decode_tps'] == 4 / 1.25
    assert split['off']['pooled_decode_tps'] == 1
    assert split['unknown'] is None


@pytest.mark.parametrize('field,value', [('thinking', False), ('max_tokens', 512)])
def test_pairs_reject_thinking_or_budget_mismatch(field, value):
    runs = [result(), result('lossy-think-m2.5')]
    runs[0][field] = value
    with pytest.raises(ValueError, match='different thinking'):
        report.summarize(runs, [trace(r) for r in runs])


def test_missing_thinking_cannot_masquerade_as_off():
    row = result('lossy-think-m5.0', False)
    event = trace(row)
    row.pop('thinking')
    with pytest.raises(ValueError, match='explicit thinking'):
        report.prove_lossy(row, [event])


GOOD_CODE = '''def lower_bound(values, target):
    return next_index(values, target)
def next_index(values, target):
    for i, value in enumerate(values):
        if value >= target: return i
    return len(values)
def merge_intervals(intervals):
    result = []
    for a, b in sorted(intervals):
        if result and a <= result[-1][1]: result[-1] = (result[-1][0], max(b, result[-1][1]))
        else: result.append((a, b))
    return result
def run_length(values):
    result = []
    for v in values:
        if result and result[-1][0] == v: result[-1] = (v, result[-1][1] + 1)
        else: result.append((v, 1))
    return result
def stable_unique(values):
    return list(dict.fromkeys(values))
'''


def write_directory(path, variant, text, reasoning='', thinking=True, prompt=spec_code_check.PROMPT):
    path.mkdir()
    (path/'config.json').write_text(json.dumps({'prompts': {'code_functions': prompt}, 'tokens': 1024, 'thinking': thinking, 'repeats': 1}))
    row = result(variant, thinking, 'code_functions')
    row.update(text=text, reasoning_text=reasoning)
    row['reasoning_field_present'] = '<think>' not in text and '</think>' not in text
    (path/'result.json').write_text(json.dumps(row))
    return path


def test_checker_runs_both_visible_answers_and_pools_length_ratios(tmp_path, monkeypatch):
    left = write_directory(tmp_path/'control', 'fixed7', '<think>two words</think>' + GOOD_CODE)
    right = write_directory(tmp_path/'treatment', 'lossy-think-m2.5', GOOD_CODE, 'four separate reasoning words')
    monkeypatch.setattr(bench.urllib.request, 'urlopen', lambda *a, **kw: pytest.fail('network forbidden'))
    output = quality.compare(left, right)
    assert output['reasoning_token_count_ratio'] == 2
    assert output['final_answer_length_ratio'] == 1
    assert output['control']['code_checks']['passed'] == output['treatment']['code_checks']['passed'] == 1
    assert output['control']['code_checks']['applicable_total'] == 1
    out = tmp_path/'quality.json'
    monkeypatch.setattr(sys, 'argv', ['spec_think_check', str(left), str(right), '--out', str(out)])
    quality.main()
    assert json.loads(out.read_text())['control']['requests'] == 1


def test_checker_retains_failure_empty_answer_and_inapplicable_contract(tmp_path):
    left = write_directory(tmp_path/'control', 'fixed7', GOOD_CODE, prompt='Return SQL')
    right = write_directory(tmp_path/'treatment', 'lossy-think-m5.0', '<think>still thinking', prompt='Return SQL')
    output = quality.compare(left, right)
    assert output['reasoning_token_count_ratio'] is None
    assert output['final_answer_length_ratio'] == 0
    assert output['treatment']['empty_visible_answers'] == 1
    assert output['treatment']['code_checks'] == {'passed': 0, 'total': 1, 'applicable_passed': 0, 'applicable_total': 0}


def test_checker_rejects_thinking_off_and_non_think_variant(tmp_path):
    left = write_directory(tmp_path/'control', 'fixed7', GOOD_CODE, thinking=False)
    right = write_directory(tmp_path/'treatment', 'lossy-think-m5.0', GOOD_CODE, thinking=False)
    with pytest.raises(ValueError, match='thinking on'):
        quality.compare(left, right)
    with pytest.raises(ValueError):
        quality.compare(left, right, treatment='lossy-m2.5')
