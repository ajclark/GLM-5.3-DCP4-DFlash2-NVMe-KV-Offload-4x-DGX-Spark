"""Server workload priors preserve runtime eligibility and censored learning."""
import copy
import json

import pytest

from spec_harness import load_policy
from test_adaptive_spec import Request, config

P = load_policy()
SIGNATURE = {'tp': 4, 'dcp': 2, 'max_model_len': 180224, 'draft_capacity': 7}


def table():
    return {'config': SIGNATURE, 'context_range': [0, 4096], 'prior_strength': 2,
            'domains': {'code': [.9] * 7, 'prose': [.1] * 7}}


def hint(**changes):
    return dict(spec_policy='adaptive', spec_workload='prose', spec_phase='user_turn',
                spec_hint_strength='weak', **changes)


@pytest.fixture
def policy(monkeypatch, tmp_path):
    monkeypatch.setenv('GLM_SPEC_POLICY', 'shadow')
    for key in ('GLM_SPEC_COSTS', 'GLM_SPEC_TRACE', 'GLM_SPEC_VERIFY_CAP'):
        monkeypatch.delenv(key, raising=False)
    p = tmp_path / 'priors.json'
    p.write_text(json.dumps(table()))
    monkeypatch.setenv('GLM_SPEC_HINT_PRIORS', str(p))
    out = P.VerificationPolicy(config())
    out.costs = {1: 95, 3: 115, 5: 130, 7: 145}
    out.cost_context = (0, 4096)
    return out


def test_prior_only_cold_start_changes_choice_without_inventing_trials(policy):
    request = Request()
    request.sampling_params.extra_args = hint()
    assert policy.select(request, True, 1) == 1
    stats = policy.states[request]['stats']
    assert stats.observations == 0 and stats.trials == [0.] * 7
    assert policy.chosen[request.request_id]['hint_domain'] == 'prose'
    for step in range(2, 16):
        assert policy.select(request, True, step) == 1
    assert policy.select(request, True, 16) == 7  # Mandatory full-width probe.


@pytest.mark.parametrize('field,value', [
    ('spec_workload', 'mixed'), ('spec_workload', 'unknown'),
    ('spec_workload', 'prose;force_cap=1'), ('spec_hint_strength', 'strong'),
    ('spec_hint_strength', 'abstain'), ('spec_phase', 'tool_followup'),
    ('spec_use_hints', False), ('spec_use_hints', 1), ('spec_use_hints', 'true')])
def test_ambiguous_untrusted_or_disabled_hints_keep_original_warmup(policy, field, value):
    request = Request()
    request.sampling_params.extra_args = {**hint(), field: value}
    assert policy.select(request, True, 1) == 7
    assert policy.chosen[request.request_id]['hint_domain'] is None


@pytest.mark.parametrize('reason', ['c2', 'context', 'costs', 'sampling', 'prefill', 'fixed', 'no_server_table'])
def test_hint_does_not_bypass_lane_and_request_eligibility(policy, reason):
    request = Request()
    request.sampling_params.extra_args = hint()
    if reason == 'context': request.num_tokens = 5000
    if reason == 'costs': policy.costs = {}
    if reason == 'sampling': request.sampling_params.temperature = .7
    if reason == 'prefill': request.is_prefill_chunk = True
    if reason == 'fixed': request.sampling_params.extra_args['spec_policy'] = 'fixed'
    if reason == 'no_server_table': policy.hint_priors = None
    assert policy.select(request, reason != 'c2', 1) == 7
    assert policy.chosen[request.request_id]['hint_domain'] is None


def test_wrong_hint_yields_to_actual_acceptance_and_separate_requests(policy):
    request = Request()
    request.sampling_params.extra_args = hint()
    assert policy.select(request, True, 1) == 1
    stats = policy.states[request]['stats']
    for step in range(2, 150):
        cap = policy.select(request, True, step)
        stats.observe(cap, cap)
    assert policy.select(request, True, 150) == 7
    other = Request(request_id='other')
    other.sampling_params.extra_args = hint()
    assert policy.select(other, True, 151) == 1
    assert policy.states[other]['stats'].observations == 0


@pytest.mark.parametrize('change', ['lane', 'bounds', 'weight', 'length', 'nan', 'bool'])
def test_invalid_server_tables_rejected(change):
    data = copy.deepcopy(table())
    if change == 'lane': data['config']['dcp'] = 4
    if change == 'bounds': data['context_range'] = [0, 180225]
    if change == 'weight': data['prior_strength'] = 20
    if change == 'length': data['domains']['prose'] = [.5]
    if change == 'nan': data['domains']['prose'][0] = float('nan')
    if change == 'bool': data['domains']['prose'][0] = True
    with pytest.raises(ValueError, match='Invalid server workload prior table'):
        P.HintPriors.parse(data, SIGNATURE)


def test_default_choice_matches_legacy_warmup_even_with_regularizing_prior():
    stats = P.PrefixStats(prior=(.1,) * 7)
    costs = {1: 95, 3: 115, 5: 130, 7: 145}
    for step in range(1, 9):
        assert stats.choose(costs, step)[0] == 7
        stats.observe(7, 0)
    assert stats.choose(costs, 9)[0] == 1
