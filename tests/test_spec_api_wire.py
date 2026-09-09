"""Exercise the deployed API field type before admitting a benchmark request."""
import json
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'bench'))
from adaptive_spec import add_costs,request_body

pydantic=pytest.importorskip('pydantic')


def test_context_table_roundtrips_through_the_pinned_api_field():
    fixture=json.loads((ROOT/'tests/fixtures/spec_runtime/api-xargs.json').read_text())
    assert fixture['vllm_xargs']=='dict[str, str | int | float | list[str | int | float]] | None'
    adapter=pydantic.TypeAdapter(dict[str,str|int|float|list[str|int|float]]|None)
    costs={'config':{'tp':4,'dcp':2,'max_model_len':180224,'draft_capacity':7},
           'lane':'tp4-dcp2-l180224-k7','context_range':[0,180224],
           'calibration_points':[{'context':220,'cycle_ms':{'1':96,'3':116,'5':130,'7':145}}]}
    with pytest.raises(pydantic.ValidationError):
        adapter.validate_python({'spec_cost_table':costs})
    body=request_body('test',7,'wire',256)
    add_costs(body,costs)
    admitted=adapter.validate_json(json.dumps(body['vllm_xargs']))
    assert json.loads(admitted['spec_cost_table'])==costs
