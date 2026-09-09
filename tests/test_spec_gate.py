import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from evaluate_spec_gate import aggregate_decode_rates, audit_pairs, performance_gates


def recordings():
    return [dict(case=case,repeat=repeat,cap=7,policy=mode,ttft=1,
                 decode_tps=20 if mode=='fixed' else rate,
                 chunks=[{'seconds':i*.15,'data':{'choices':[{'token_ids':[i]}]}}
                         for i in range(12)])
            for case,rate in (('code_a',20),('code_b',20),('prose_a',24),('prose_b',22))
            for repeat in range(3) for mode in ('fixed','adaptive')]


def test_missing_and_duplicate_pairs_cannot_pass_a_complete_evaluation():
    rows=recordings();cases=['code_a','code_b','prose_a','prose_b']
    audit_pairs(rows,cases,3)
    for incomplete in (rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]]):
        with pytest.raises(ValueError,match='pairs'):
            audit_pairs(incomplete,cases,3)


def test_throughput_gain_does_not_hide_matched_case_latency_regression():
    rows=recordings()
    gates,_,_=performance_gates(rows)
    assert all(gates.values())
    rows=copy.deepcopy(rows)
    for row in rows:
        if row['case']=='prose_a' and row['policy']=='adaptive':
            row['ttft']=1.1
    gates,_,latency=performance_gates(rows)
    assert gates['prose_throughput'] and gates['coding_noninferiority']
    assert not gates['ttft'] and gates['p95_emission_gap']
    assert latency['prose_a']['ttft']==1.1


def test_aggregate_rate_excludes_first_speculative_burst_and_prefill():
    row={'case':'code_a','cap':7,'policy':'fixed','token_ids':[1,2,3,4,5],
         'chunks':[{'seconds':20,'data':{'choices':[{'token_ids':[1,2,3]}]}},
                   {'seconds':20.5,'data':{'choices':[{'token_ids':[4,5]}]}}]}
    result=aggregate_decode_rates([row,row])['code/fixed7']
    assert result=={'decoded_tokens':4,'decode_seconds':1,'tokens_per_second':4}
