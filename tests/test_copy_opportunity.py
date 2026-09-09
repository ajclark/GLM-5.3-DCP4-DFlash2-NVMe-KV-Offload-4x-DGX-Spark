import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from copy_opportunity import CopyIndex, propose


def test_copy_uses_only_existing_continuation():
    history = [1,2,3,4,10,11,12,13,14,15,16,90,1,2,3,4]
    assert propose(history) == ([10,11,12,13,14,15,16],4)
    assert propose([1,2,3,4,5,6,7,8]) == ([],0)


def test_copy_prefers_longest_then_most_recent_match():
    suffix = list(range(20,36))
    history = suffix+[1]*7+[90]+suffix[-4:]+[2]*7+[90]+suffix
    assert propose(history) == ([1]*7,16)
    assert propose(history,widths=(4,)) == ([2]*7,4)


def test_lookback_bound_and_incomplete_continuation():
    history = [1,2,3,4]+[5]*7+[90]*30+[1,2,3,4]
    assert propose(history,lookback=20) == ([],0)
    assert propose(history) == ([5]*7,4)
    assert propose([1,2,3,4]+[5]*2+[1,2,3,4]) == ([],0)


def test_incremental_index_matches_exhaustive_causal_search():
    import random
    rng = random.Random(42)
    for lookback in (10,32,128,180224):
        index = CopyIndex(lookback=lookback)
        history = []
        for _ in range(120):
            tokens = [rng.randrange(3) for _ in range(rng.randrange(1,25))]
            if len(history)>40 and rng.random()<.5:
                start = rng.randrange(len(history)-32)
                tokens = history[start:start+32]
            history.extend(tokens)
            index.extend(tokens)
            assert index.propose() == propose(history,lookback=lookback)
            assert all(len(q)<=lookback for q in index.queues.values())


def test_full_window_index_finds_early_repository_continuation():
    prefix = [1,2,3,4]+list(range(10,17))+[99]*40000
    tail = [1,2,3,4]
    short = CopyIndex()
    full = CopyIndex(lookback=180224)
    for index in (short,full):
        index.extend(prefix)
        index.extend(tail)
    assert short.propose() == ([],0)
    assert full.propose() == (list(range(10,17)),4)
