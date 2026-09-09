import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))

from spec_harness import load_policy
from test_adaptive_spec import Request,config

P=load_policy()


def table():
    return {'context_range':[0,10000],'calibration_points':[
        {'context':100,'cycle_ms':{'1':100,'3':110,'5':120,'7':140},'conditional_acceptance_prior':[.5]*7},
        {'context':9900,'cycle_ms':{'1':120,'3':130,'5':140,'7':160},'conditional_acceptance_prior':[.75]*7}]}


def test_interpolation_bounds_and_immutable_step_costs():
    curve=P.CostCurve.parse(table(),10000)
    costs,prior=curve.at(5000)
    assert costs=={1:110,3:120,5:130,7:150}
    assert prior==(.625,)*7
    costs[1]=999
    assert curve.at(5000)[0][1]==110
    assert curve.at(0)[0][1]==100
    assert curve.at(9999)[0][1]==120
    assert curve.at(10000)==({},None)
    assert curve.at(-1)==({},None)


@pytest.mark.parametrize('change',[
    lambda d:d.update(context_range=[0,10001]),
    lambda d:d.update(calibration_points=[]),
    lambda d:d['calibration_points'][1].update(context=100),
    lambda d:d['calibration_points'][0]['cycle_ms'].update({'1':float('nan')}),
    lambda d:d['calibration_points'][0].update(conditional_acceptance_prior=[True]*7),
    lambda d:d['calibration_points'][0].update(prior_strength=5),
])
def test_invalid_curves_are_rejected(change):
    data=table();change(data)
    with pytest.raises(ValueError,match='cost curve'):
        P.CostCurve.parse(data,10000)


def test_curve_controls_are_bound_to_request_and_lane(monkeypatch):
    monkeypatch.setenv('GLM_SPEC_POLICY','shadow')
    monkeypatch.delenv('GLM_SPEC_COSTS',raising=False)
    p=P.VerificationPolicy(config());r=Request(num_tokens=5000)
    data={**table(),'config':p.signature,'lane':p.lane}
    r.sampling_params.extra_args={'spec_policy':'adaptive','spec_cost_table':data}
    p.select(r,True,1)
    assert p.chosen[r.request_id]['costs_ms'][1]==110
    compiled=p.states[r]['curve']
    p.select(r,True,2)
    assert p.states[r]['curve'] is compiled
    r.sampling_params.extra_args['spec_cost_table']='invalid'
    p.select(r,True,3)
    assert p.chosen[r.request_id]['costs_ms']=={}
    data=copy.deepcopy(data);data['lane']='wrong'
    r.sampling_params.extra_args['spec_cost_table']=data
    p.select(r,True,4)
    assert p.chosen[r.request_id]['costs_ms']=={}


def test_boot_curve_keeps_its_global_context_bounds(tmp_path,monkeypatch):
    data={**table(),'config':{'tp':4,'dcp':2,'max_model_len':180224,'draft_capacity':7}}
    path=tmp_path/'costs.json';path.write_text(json.dumps(data))
    monkeypatch.setenv('GLM_SPEC_POLICY','shadow')
    monkeypatch.setenv('GLM_SPEC_COSTS',str(path))
    p=P.VerificationPolicy(config());r=Request(num_tokens=5000)
    p.select(r,True,1)
    assert p.chosen[r.request_id]['costs_ms'][1]==110
    r.num_tokens=10000;p.select(r,True,2)
    assert p.chosen[r.request_id]['costs_ms']=={}


def test_encoded_api_table_is_decoded_once_per_request(monkeypatch):
    monkeypatch.setenv('GLM_SPEC_POLICY','shadow')
    monkeypatch.delenv('GLM_SPEC_COSTS',raising=False)
    p=P.VerificationPolicy(config());r=Request(num_tokens=5000)
    data={**table(),'config':p.signature,'lane':p.lane}
    from adaptive_spec import add_costs,request_body
    body=request_body('test',7,'wire-test',256)
    add_costs(body,data)
    assert isinstance(body['vllm_xargs']['spec_cost_table'],str)
    r.sampling_params.extra_args=body['vllm_xargs']
    p.select(r,True,1)
    assert p.chosen[r.request_id]['costs_ms'][1]==110
    compiled=p.states[r]['curve']
    monkeypatch.setattr(P.json,'loads',lambda _:(_ for _ in ()).throw(AssertionError('decoded twice')))
    p.select(r,True,2)
    assert p.states[r]['curve'] is compiled


@pytest.mark.parametrize('encoded',('not json','[]','null','"scalar"','['*2000,' '*17000))
def test_invalid_or_oversized_encoded_table_falls_back(monkeypatch,encoded):
    monkeypatch.setenv('GLM_SPEC_POLICY','shadow')
    monkeypatch.delenv('GLM_SPEC_COSTS',raising=False)
    p=P.VerificationPolicy(config());r=Request(num_tokens=5000)
    r.sampling_params.extra_args={'spec_policy':'adaptive','spec_cost_table':encoded}
    assert p.select(r,True,1)==7
    assert p.chosen[r.request_id]['costs_ms']=={}
