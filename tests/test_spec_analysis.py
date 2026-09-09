"""Ensure adaptive/fixed baselines stay separate and sampled energy is bounded."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from analyze_adaptive_spec import device_energy, paired_groups


def test_prompt_pairing_does_not_pool_adaptive_with_fixed_seven():
    rows = [dict(case=case,repeat=r,cap=7,policy=mode,decode_tps=rate)
            for case in ('prose_a','prose_b') for r in (0,1,2)
            for mode,rate in (('fixed',20),('adaptive',24))]
    result = paired_groups(rows)['prose/adaptive']
    assert result['prompts']==2 and result['pairs']==6
    assert result['paired_geomean_ratio']==pytest.approx(1.2)
    assert result['prompt_bootstrap_95']==pytest.approx([1.2,1.2])


def test_device_energy_interpolates_and_refuses_missing_node_or_time_gap():
    row = dict(started_at=100,token_ids=[1,2,3,4,5],chunks=[
        {'seconds':1,'data':{'choices':[{'token_ids':[1]}]}},
        {'seconds':3,'data':{'choices':[{'token_ids':[2,3,4,5]}]}}])
    samples = {h:[{'received_at':100,'power_w':10},{'received_at':104,'power_w':30}]
               for h in ('a','b','c','d')}
    assert device_energy(row,samples)==pytest.approx(40)
    assert device_energy(row,{k:v for k,v in samples.items() if k!='d'}) is None
    samples['a'][1]['received_at']=110
    assert device_energy(row,samples) is None


def test_single_prompt_has_no_meaningful_prompt_bootstrap_interval():
    rows = [dict(case='prose_a',repeat=0,cap=7,policy=mode,decode_tps=rate)
            for mode,rate in (('fixed',20),('adaptive',24))]
    result = paired_groups(rows)['prose/adaptive']
    assert result['paired_geomean_ratio']==pytest.approx(1.2)
    assert result['prompt_bootstrap_95'] is None
