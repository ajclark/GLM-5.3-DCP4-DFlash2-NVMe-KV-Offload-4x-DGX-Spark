"""Exercise the actual live OpenAPI xargs shape before runtime switch parsing."""
import json
from pathlib import Path

from pydantic import TypeAdapter
import pytest

from spec_harness import load_policy
from test_spec_confidence_trace import C, new_request
from test_spec_hints import SIGNATURE, table


WIRE = TypeAdapter(dict[str, str | int | float | list[str | int | float]])
P = load_policy()


def test_adapter_matches_the_captured_live_vllm_xargs_schema():
    schema = json.loads((Path(__file__).parent / 'fixtures/spec_hints/vllm-xargs-schema.json').read_text())
    assert WIRE.json_schema() == schema['anyOf'][0]


@pytest.mark.parametrize('value,enabled', [(True, True), (False, False), (1, True),
    (0, False), (1.0, False), (2, False), ('true', False), ('1', False)])
def test_wire_normalization_reaches_both_runtime_controls(value, enabled):
    xargs = WIRE.validate_python({'spec_use_hints': value, 'spec_confidence_trace': value,
        'spec_workload': 'prose', 'spec_hint_strength': 'weak', 'spec_phase': 'user_turn'})
    if type(value) is bool:
        assert type(xargs['spec_confidence_trace']) is int
    data = new_request()
    data.sampling_params.extra_args = xargs
    assert C.request_eligible(data) is enabled
    priors = P.HintPriors.parse(table(), SIGNATURE)
    assert (priors.select(xargs, 312)[1] == 'prose') is enabled
