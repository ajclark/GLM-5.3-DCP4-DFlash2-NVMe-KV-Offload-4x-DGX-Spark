import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from spec_request_report import summarize


def pair(case='code_a', repeat=0, ratio=2):
    base = {'study': 'confidence', 'variant': 'off', 'case': case, 'repeat': repeat,
            'label': f'{case}-{repeat}-off', 'decode_tps': 10, 'ttft': .5,
            'token_sha256': 'output-hash', 'token_ids': [8, 9],
            'message_sha256': 'private-prompt-hash',
            'chunks': [{'data': {'prompt_token_ids': [123, 456], 'choices': []}}]}
    on = copy.deepcopy(base)
    on.update(variant='on', label=f'{case}-{repeat}-on', decode_tps=10 * ratio)
    return [base, on]


def test_pairs_are_exact_and_prompt_weighting_does_not_count_repeats_as_prompts():
    rows = pair(ratio=4) + pair(repeat=1, ratio=1) + pair('prose_b', ratio=.5)
    result = summarize(rows)
    assert result['summaries']['on/off']['decode_tps_ratio'] == pytest.approx(1)
    assert result['summaries']['on/off']['identical_output_pairs'] == 3
    published = json.dumps(result)
    assert '123, 456' not in published and 'prompt_token_ids' not in published
    changed = copy.deepcopy(rows)
    changed[1]['chunks'][0]['data']['prompt_token_ids'][0] = 789
    with pytest.raises(ValueError, match='different prompts'):
        summarize(changed)


def test_missing_duplicate_and_unmatched_controls_are_not_silently_paired():
    rows = pair()
    with pytest.raises(ValueError, match='incomplete'):
        summarize(rows[:1])
    with pytest.raises(ValueError, match='duplicate'):
        summarize(rows + rows[:1])
    rows[1]['message_sha256'] = 'changed-system-context'
    with pytest.raises(ValueError, match='different prompts'):
        summarize(rows)
