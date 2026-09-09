import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from spec_concurrency import overlap_rate


def row(chunks):
    return {'started_at':100,'chunks':[{'seconds':time,'data':{'choices':[{'token_ids':[1]*n}]}} for time,n in chunks]}


def test_c2_measurement_excludes_prefill_and_single_stream_tails():
    result=overlap_rate([row([(1,2),(2,3),(3,1)]),row([(1.5,1),(2.5,2),(4,1)])])
    assert result['tokens']==6 and result['seconds']==1.5 and result['tokens_per_second']==4
    with pytest.raises(ValueError,match='shared'):
        overlap_rate([row([(1,2),(2,3)]),row([(3,1),(4,2)])])
