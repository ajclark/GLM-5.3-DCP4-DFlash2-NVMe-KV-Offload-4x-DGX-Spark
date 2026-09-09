import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("spec_memory", Path(__file__).resolve().parents[1]/"bench/spec_memory.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def sample(**kw):
    row = dict(time=100, boot="same", available_mib=2700, swap_used_mib=2800,
               swap_in_mib=100, swap_out_mib=100, oom_kill=0,
               full={"avg10":0, "total":100}, some={"avg10":0, "total":200})
    row.update(kw)
    return row


def test_existing_swap_is_not_new_pressure():
    assert m.pressure_reason(sample(time=102), sample()) is None


def test_pressure_is_rejected_before_experiment():
    for row in (sample(available_mib=511), sample(time=102, available_mib=900, swap_out_mib=700),
                sample(boot="changed"), sample(oom_kill=1),
                sample(available_mib=1500, full={"avg10":11,"total":100}),
                sample(time=102, available_mib=900, full={"avg10":0,"total":1100000})):
        assert m.pressure_reason(row, sample())


def test_normal_checkpoint_reclaim_is_allowed():
    before = sample(available_mib=15272, time=100)
    after = sample(available_mib=15390,time=102,swap_out_mib=133,
                   full={"avg10":1,"total":38000})
    assert m.pressure_reason(after,before,loading=True) is None
    assert m.pressure_reason(sample(available_mib=900)) is None


def test_sustained_pressure_requires_low_headroom():
    for available, expected in ((15000,False),(1500,True)):
        tracker = m.PressureTracker()
        for i in range(6):
            reason = tracker.observe(sample(time=100+2*i, available_mib=available,
                                            swap_out_mib=100+120*i,
                                            full={"avg10":3,"total":100+1000*i}),loading=True)
        assert bool(reason) is expected
