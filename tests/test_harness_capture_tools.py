"""Private capture tooling: synthetic data only; network paths use a fake opener."""
import copy
import io
import json
from pathlib import Path
import sys
from urllib.error import URLError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import harness_capture as capture
import harness_call_attrib as attrib
import toolarg_copy_screen as screen


class Words:
    def __init__(self):
        self.ids = {}

    def encode(self, value):
        return [self.ids.setdefault(word, len(self.ids)) for word in value.split()]


def row(**changes):
    value = dict(t=1., api_calls=1, messages=[{'role': 'user', 'content': 'PRIVATE TASK'}],
                 tools=[], tool_names=['tool'], tool_args_text='a b c d e f g h i j k l m n o p q r s t',
                 reasoning_text='', content_text='', generation_tokens=40,
                 prompt_tokens=100, prefix_cache_hit_rate=.75, spec_drafts=10,
                 spec_accepted_tokens=30, accept_rate_by_pos=[1., .8, .5, .3, .2, .1, .1],
                 server_decode_s=2., server_prefill_s=1., wall_s=3.5, mean_ttft_s=1.2)
    value.update(changes)
    return value


def test_attribution_pooling_cache_and_no_double_count_ttft():
    a = row()
    b = row(t=2., generation_tokens=80, spec_drafts=10, spec_accepted_tokens=70,
            server_decode_s=10., messages=a['messages'] + [{'role': 'tool', 'content': 'PRIVATE RESULT'}])
    result = attrib.analyze([a, b], tokenizer=Words())
    assert result['total']['decode_tok_s'] == 10
    assert result['total']['accepted_per_cycle'] == 6
    assert result['total']['estimated_uncached_prompt_tokens'] == 50
    assert result['total']['server_prefill_s'] == 2
    assert result['total']['prefill_wall_share'] == 2 / 7
    assert result['total']['other_wall_s'] == -7  # flag overlapping global counters; do not clamp
    assert result['calls'][1]['call_in_task'] == 2
    assert result['prefix_stability']['append_only_transitions'] == 1
    assert result['categories']['tool_args']['token_share'] == 1
    output = json.dumps(result) + attrib.markdown(result)
    assert 'PRIVATE' not in output
    assert a['tool_args_text'] not in output


def test_cached_missing_is_unknown_not_cold():
    result = attrib.analyze([row(prefix_cache_hit_rate=None)])
    assert result['calls'][0]['estimated_uncached_prompt_tokens'] is None
    assert result['total']['prompt_weighted_cache_hit_rate'] is None
    assert result['total']['cache_measurement_calls'] == 0


def test_composition_normalized_to_generation_and_pure_measurement():
    result = attrib.analyze([row(reasoning_text='r r r r', tool_args_text='a a', content_text='c')], tokenizer=Words())
    assert sum(r['allocated_tokens'] for r in result['categories'].values()) == pytest.approx(40)
    assert result['categories']['reasoning']['token_share'] == pytest.approx(4 / 7)
    assert result['categories']['reasoning']['pure_calls']['calls'] == 0
    assert 'unidentified' in result['category_fit']['status']


def test_harmonic_regression_recovers_known_category_cycle_cost():
    rows = [row(spec_drafts=10, spec_accepted_tokens=a) for a in (10, 30, 70)]
    fit = attrib.category_fit(rows, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert fit['accepted_per_cycle'] == pytest.approx(dict(reasoning=2, content=4, tool_args=8))


def test_private_artifacts_supply_run_and_actual_task_wall(tmp_path):
    path = tmp_path / 'run2' / 'private-name' / 'result.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'start': 0, 'end': 10, 'seconds': 10, 'check': {'passed': True}, 'secret': 'PRIVATE'}))
    result = attrib.analyze([row()], source_dir=tmp_path)
    assert result['tasks'][0]['run'] == 2
    assert result['tasks'][0]['official_pass'] is True
    assert result['total']['completed_task_wall_s'] == 10
    assert 'private-name' not in json.dumps(result)


def test_session_resets_and_task_identity():
    a = row()
    b = row(t=2., messages=a['messages'] + [{'role': 'tool', 'content': 'x'}])
    c = row(t=3.)
    ids = capture.identify_calls([a, b, c])
    assert len({x['task'] for x in ids}) == 1
    assert [x['call_in_task'] for x in ids] == [1, 2, 1]
    assert ids[0]['session'] != ids[2]['session']


def test_prefix_changes_and_tools_invalidation():
    a = row()
    b = row(messages=[{'role': 'user', 'content': 'changed'}], tools=[{'type': 'function'}])
    result = attrib.prefix_stability(a, b)
    assert result['unchanged_leading_messages'] == 0
    assert result['tools_unchanged'] is False
    assert result['append_only_messages'] is False


def test_flatten_order_and_literal_arguments():
    value = row(tools=[{'schema': 'FIRST'}], messages=[{'role': 'developer', 'content': 'SECOND'},
                {'role': 'user', 'content': [{'type': 'text', 'text': 'THIRD'}]},
                {'role': 'assistant', 'content': None, 'tool_calls': [{'function': {'name': 'FOURTH', 'arguments': '{"FIFTH":1}'}}]},
                {'role': 'tool', 'content': 'SIXTH'}])
    text = capture.flatten_context(value)
    assert [text.index(x) for x in ('FIRST', 'SECOND', 'THIRD', 'FOURTH', 'FIFTH', 'SIXTH')] == sorted(text.index(x) for x in ('FIRST', 'SECOND', 'THIRD', 'FOURTH', 'FIFTH', 'SIXTH'))
    assert '{"FIFTH":1}' in text


def test_copy_match_uses_most_recent_not_oracle_best():
    anchor = [1, 2, 3, 4]
    future = list(range(10, 30))
    context = anchor + future + [100] + anchor + [999] * 20
    result = screen.score_tokens(context, anchor + future, 4, (4,))
    # Earlier perfect match must not override the most recent bad continuation.
    assert result['agreement_histogram_capped_at_8']['0'] >= 1
    assert result['candidate_emitted_per_cycle'] < 8
    index = screen.context_index(context, (4,))
    assert context[index[4][tuple(anchor)]] == 999


def test_copy_scores_eight_and_charges_bad_proposals():
    output = list(range(40))
    good = screen.score_tokens(output, output, 5, (4,))
    bad = screen.score_tokens(list(range(4)) + [999] * 40, output, 5, (4,))
    assert good['correct_copy_ge8_positions'] == 29
    assert good['candidate_emitted_per_cycle'] > 5
    assert bad['candidate_emitted_per_cycle'] < 5
    assert good['tail_positions_using_dflash'] == 7
    assert good['candidate_emitted_per_cycle'] == pytest.approx((4 * 5 + 29 * 8 + 7 * 5) / 40)


def test_longest_anchor_precedes_recent_shorter_anchor():
    output = list(range(60))
    context = output[:16] + [999] * 20 + output[12:16] + output[16:]
    result = screen.score_tokens(context, output, 5)
    assert result['anchor_widths'].get('16', 0) > 0
    assert result['agreement_histogram_capped_at_8'].get('0', 0) > 0


def test_short_output_no_match_fallback():
    result = screen.score_tokens(list(range(30)), [1, 2], 4)
    assert result['positions'] == 0
    assert result['candidate_emitted_per_cycle'] == 4
    assert result['copy_ge8_share'] is None


def test_pooled_gain_uses_cycle_cost_not_arithmetic_mean():
    rows = [row(generation_tokens=100), row(generation_tokens=100)]
    calls = [{**screen.score_tokens([], [], 2), 'allocated_tool_tokens': 100,
              'candidate_emitted_per_cycle': 4},
             {**screen.score_tokens([], [], 8), 'allocated_tool_tokens': 0}]
    p = screen.pool(calls, rows)
    assert p['baseline_all_emitted_per_cycle'] == 200 / (100 / 2 + 100 / 8)
    assert p['candidate_all_emitted_per_cycle'] == 200 / (100 / 4 + 100 / 8)
    assert p['gate_above_3_percent'] is True


@pytest.mark.parametrize('curve', [[], [1] * 6, [1] * 8, [1, .8, .9, .3, .2, .1, .1], [float('nan')] * 7, [None] * 7])
def test_invalid_curves_fail_closed(curve):
    with pytest.raises(ValueError):
        screen.validate_curve(row(accept_rate_by_pos=curve))


def test_curve_counter_disagreement_fails_closed():
    with pytest.raises(ValueError, match='disagrees'):
        screen.validate_curve(row(spec_accepted_tokens=60))


def test_copy_output_is_aggregate_only():
    value = row()
    result = screen.screen([value], Words(), {'records': 1})
    serialized = json.dumps(result) + screen.markdown(result)
    assert value['tool_args_text'] not in serialized
    assert 'PRIVATE TASK' not in serialized
    assert set(result['policies']) == {'n4', 'n8', 'n16', 'longest'}


class Response(io.BytesIO):
    status = 200


def test_tokenizer_health_first_chunks_cache_and_no_generation():
    calls = []
    def opener(request, timeout):
        calls.append(request)
        if request.full_url.endswith('/health'):
            return Response(b'')
        payload = json.loads(request.data)
        assert payload['add_special_tokens'] is False
        assert len(payload['prompt']) <= 4
        return Response(json.dumps({'tokens': [ord(c) for c in payload['prompt']]}).encode())
    tokenizer = capture.CpuTokenizer('http://spark-06c4.local:8000', no_hold=True, opener=opener, chunk_chars=4)
    assert tokenizer.encode('abcdefgh') == list(map(ord, 'abcdefgh'))
    count = len(calls)
    assert tokenizer.encode('abcd') == list(map(ord, 'abcd'))
    assert len(calls) == count
    assert [r.full_url.rsplit('/', 1)[1] for r in calls] == ['health', 'health', 'tokenize', 'health', 'tokenize']


def test_hold_attestation_required_without_any_network():
    with pytest.raises(capture.TokenizationUnavailable, match='no-hold'):
        capture.CpuTokenizer('http://spark-06c4.local:8000', opener=lambda *a: pytest.fail('network'))


def test_failed_health_never_tokenizes_or_leaks_error():
    calls = []
    def opener(request, timeout):
        calls.append(request.full_url)
        raise URLError('PRIVATE ERROR')
    with pytest.raises(capture.TokenizationUnavailable, match='health') as error:
        capture.CpuTokenizer('http://spark-06c4.local:8000', no_hold=True, opener=opener)
    assert len(calls) == 1
    assert 'PRIVATE' not in str(error.value)


def test_health_failure_mid_run_discards_screen(tmp_path, monkeypatch):
    class Failing:
        def encode(self, text):
            raise capture.TokenizationUnavailable('tokenization skipped: health check failed')
    source, out = tmp_path / 'capture.jsonl', tmp_path / 'screen.json'
    source.write_text(json.dumps(row()) + '\n')
    monkeypatch.setattr(screen, 'make_tokenizer', lambda args: Failing())
    monkeypatch.setattr(sys, 'argv', ['screen', str(source), '--out', str(out), '--tokenize', '--no-hold'])
    screen.main()
    result = json.loads(out.read_text())
    assert result['status'] == 'skipped'
    assert 'policies' not in result


def test_no_tokenizer_no_fake_token_results():
    result = screen.screen([row()], None)
    assert result['status'] == 'skipped'
    assert 'policies' not in result


def test_invalid_capture_error_contains_only_line_number(tmp_path):
    path = tmp_path / 'input.jsonl'
    path.write_text('{"PRIVATE":')
    with pytest.raises(ValueError, match='record 1') as error:
        capture.load_capture(path)
    assert 'PRIVATE' not in str(error.value)


def test_gitignore_private_patterns_present():
    patterns = (Path(__file__).resolve().parents[1] / '.gitignore').read_text().splitlines()
    assert 'results/harness-capture-*/proxy_*.jsonl' in patterns
    assert 'results/harness-capture-*/run*/' in patterns
