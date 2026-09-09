import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from spec_energy_report import integrate


def test_whole_request_energy_includes_prefill_and_requires_complete_coverage():
    samples={host:[{'received_at':t,'power_w':10} for t in (0,4,8)] for host in ('a','b','c','d')}
    assert integrate(samples,1,7)==pytest.approx(240)
    assert integrate(samples,4,7)==pytest.approx(120)
    assert integrate(samples,1,9) is None
    assert integrate({k:v for k,v in samples.items() if k!='d'},1,7) is None
