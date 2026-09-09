import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from spec_hint_screen import conditional_prior, survival, screen


def test_censored_tail_does_not_invent_conditional_failures():
    assert conditional_prior([0] * 100)[1:] == [.5] * 6
    assert survival([.5] * 7) == [.5 ** n for n in range(1, 8)]


def test_leave_one_prompt_out_cannot_train_on_the_evaluated_case():
    costs = {'1': 1, '3': 1.4, '5': 1.8, '7': 2.2}
    cases = {'code_a': [7, 7], 'code_b': [5, 6], 'prose_a': [0, 1], 'prose_b': [1, 2]}
    before = next(row for row in screen(cases, costs) if row['case'] == 'code_a')
    cases['code_a'] = [0] * 100
    after = next(row for row in screen(cases, costs) if row['case'] == 'code_a')
    assert before['train_priors'] == after['train_priors']
    assert before['caps'] == after['caps']
    with pytest.raises(ValueError):
        screen({'code_a': [1], 'prose_a': [1]}, costs)
