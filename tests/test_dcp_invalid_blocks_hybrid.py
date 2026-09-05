"""The engine scheduler's invalid-block recovery (used when an NVMe tier load
fails on a rank) unpacked a single KV-cache group and crashed with two. The
overlay walks every group with its own tokens-per-block and truncates at the
earliest invalid position."""
import pathlib
from types import SimpleNamespace

from harness import ROOT, extract_methods

SCHED = ROOT / "overlay/vllm/v1/core/sched/scheduler.py"
FORK = pathlib.Path.home() / "lmcache-mg/spark-src/vllm/v1/core/sched/scheduler.py"


def make_self(block_ids_by_req, sizes=(64, 64), dcp=(4, 1)):
    managers = [SimpleNamespace(block_size=s, dcp_world_size=d) for s, d in zip(sizes, dcp)]
    km = SimpleNamespace(get_block_ids=lambda rid: block_ids_by_req[rid],
                         coordinator=SimpleNamespace(single_type_managers=managers))
    return SimpleNamespace(block_size=256, kv_cache_manager=km)


def run(fn, self_, requests, invalid, evict=True):
    return fn(self_, requests, invalid, num_scheduled_tokens={}, evict_blocks=evict)


def test_two_groups_truncate_at_earliest_invalid_token():
    fn = extract_methods(SCHED, "Scheduler", ["_update_requests_with_invalid_blocks"])["_update_requests_with_invalid_blocks"]
    # request r: 1024 computed tokens = 4 target blocks (256) + 16 drafter blocks (64)
    ids = {"r": ([10, 11, 12, 13], list(range(100, 116)))}
    req = SimpleNamespace(request_id="r", num_computed_tokens=1024)
    affected, n_tokens, evict = run(fn, make_self(ids), [req], invalid={112})     # drafter block 12 -> token 768
    assert affected == {"r"} and req.num_computed_tokens == 768 and n_tokens == 256
    assert evict == {112, 113, 114, 115}
    req = SimpleNamespace(request_id="r", num_computed_tokens=1024)
    affected, n_tokens, evict = run(fn, make_self(ids), [req], invalid={11, 115})  # target block 1 -> token 256 wins
    assert req.num_computed_tokens == 256 and n_tokens == 768 and {11, 12, 13, 115} <= evict
    # a drafter-only failure at an interior 64-token position aligns down to the 256 scheduler block
    req = SimpleNamespace(request_id="r", num_computed_tokens=1024)
    run(fn, make_self(ids), [req], invalid={101})                                  # token 64 -> 0
    assert req.num_computed_tokens == 0
    # no invalid block: untouched
    req = SimpleNamespace(request_id="r", num_computed_tokens=1024)
    assert run(fn, make_self(ids), [req], invalid={999}) == (set(), 0, set()) and req.num_computed_tokens == 1024


def test_shared_invalid_block_recomputed_once():
    fn = extract_methods(SCHED, "Scheduler", ["_update_requests_with_invalid_blocks"])["_update_requests_with_invalid_blocks"]
    ids = {"a": ([10, 11], [100, 101, 102, 103, 104, 105, 106, 107]), "b": ([10, 11], [100, 101, 102, 103, 104, 105, 106, 107])}
    ra = SimpleNamespace(request_id="a", num_computed_tokens=512); rb = SimpleNamespace(request_id="b", num_computed_tokens=512)
    affected, n_tokens, _ = run(fn, make_self(ids), [ra, rb], invalid={11}, evict=False)
    assert affected == {"a", "b"} and ra.num_computed_tokens == 256 and rb.num_computed_tokens == 512 and n_tokens == 256


def test_fork_still_has_the_single_group_assumption():
    src = FORK.read_text()
    assert "(req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)" in src   # the overlay's reason to exist
    assert "DCP overlay: hybrid-aware" in SCHED.read_text()
