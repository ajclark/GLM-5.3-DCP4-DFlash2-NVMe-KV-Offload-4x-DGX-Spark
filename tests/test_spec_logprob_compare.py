import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from spec_logprob_compare import first_divergence, record


def test_only_first_shared_prefix_divergence_is_compared():
    probs = [{'logprob': -.5, 'top_logprobs': [{'token': 'a', 'logprob': -.5}, {'token': 'b', 'logprob': -.5001}]}] * 3
    a = {'label': 'a', 'prompt_sha256': 'same', 'token_ids': [10, 20, 30], 'probabilities': probs}
    b = {**a, 'label': 'b', 'token_ids': [10, 40, 50]}
    row = first_divergence(a, b)
    assert row['first_divergence'] == 1
    assert [r['selected_token_id'] for r in row['observations']] == [20, 40]
    assert row['observations'][0]['top_two_logprob_gap'] == pytest.approx(.0001)
    assert first_divergence(a, a)['identical_output_ids'] is True
    with pytest.raises(ValueError, match='different prompt'):
        first_divergence(a, {**b, 'prompt_sha256': 'different'})


def test_misaligned_logprob_records_are_rejected():
    row = {'label': 'x', 'token_ids': [1, 2], 'chunks': [{'data': {'prompt_token_ids': [3],
        'choices': [{'logprobs': {'content': [{'logprob': -.1}]}}]}}]}
    with pytest.raises(ValueError, match='accounting differs'):
        record(row)
