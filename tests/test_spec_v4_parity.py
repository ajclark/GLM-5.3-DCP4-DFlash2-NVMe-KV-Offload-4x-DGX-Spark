"""The new hint-disabled controller must preserve the frozen V4 decisions."""
import importlib.util
from pathlib import Path
import random
import sys

from spec_harness import load_policy

P = load_policy()
path = Path(__file__).parent / 'fixtures/spec_hints/adaptive_v4.py'
spec = importlib.util.spec_from_file_location('frozen_adaptive_v4', path)
V4 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = V4
spec.loader.exec_module(V4)


def test_disabled_hints_match_frozen_v4_across_drifting_acceptance():
    rng = random.Random(20260909)
    for prior in (None, (.1,) * 7, (.8,) * 7):
        old, new = V4.PrefixStats(prior=prior), P.PrefixStats(prior=prior)
        costs = {1: 95.94, 3: 116.28, 5: 130.48, 7: 145.11}
        for step in range(1, 1500):
            before = old.choose(costs, step)
            after = new.choose(costs, step)
            assert before == after
            cap = before[0]
            probability = (.1, .8, .45)[(step // 100) % 3]
            accepted = 0
            while accepted < cap and rng.random() < probability:
                accepted += 1
            old.observe(cap, accepted)
            new.observe(cap, accepted)
            assert new.trials == old.trials and new.successes == old.successes
