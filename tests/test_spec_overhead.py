import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from spec_overhead_report import compare,validate_count,clock_summary


def row(seconds):
    return {'case':'count_numbers','repeat':0,'cap':7,'policy':'fixed','text':'\n'.join(map(str,range(1,181))),
            'token_ids':[1,2],'started_at':100,'chunks':[{'seconds':t,'data':{'choices':[{'token_ids':[1]}]}} for t in (1,1+seconds)]}


def test_overhead_uses_decode_time_and_requires_identical_output():
    result=compare([row(10.2)],[row(10)])
    assert result['paired_geomean_time_ratio']==pytest.approx(1.02)
    assert result['prompt_bootstrap_95'] is None
    other=row(10);other['token_ids']=[1,3]
    with pytest.raises(ValueError,match='tokens differ'):compare([row(10)], [other])


def test_overhead_rejects_missing_pairs_and_bad_counting():
    with pytest.raises(ValueError,match='Missing'):compare([row(10)],[])
    with pytest.raises(ValueError,match='complete'):validate_count('count_evens','2\n6')
    validate_count('count_triples','\n'.join(' '.join([str(i)]*3) for i in range(1,101)))


def test_sparse_clock_samples_cannot_establish_matched_clock_conditions():
    samples={'node':[{'received_at':t,'graphics_mhz':1995} for t in range(100,113,2)]}
    assert clock_summary([row(10)],samples)['node']['complete_coverage']
    samples['node']=samples['node'][:2]+samples['node'][-1:]
    assert not clock_summary([row(10)],samples)['node']['complete_coverage']
