"""The fork's connector scheduler must eventually store every block of a
chunked prefill except the volatile trailing block of an eagle group.

The baseline held the trailing block back at each step but advanced the
store progress index past it, so one block per step per group was never
stored. A later request that presented the whole prompt in one step (a GPU
prefix hit) refilled the holes, which is why in-process reloads looked
fine; after a restart every prefix lookup broke at the first hole. This
test drives the real `_build_store_jobs` over a 100k-token prefill in
1,700-token steps with both groups flagged eagle, as DFlash does.
"""
import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import nvme_harness  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
REL = "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"


class _AnyModule(types.ModuleType):
    """A module that hands out an empty class for any name (import-time only)."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        cls = type(name, (), {})
        setattr(self, name, cls)
        return cls


def _load(variant):
    nvme_harness.build_vllm()
    for name in ("vllm.distributed.kv_events", "vllm.distributed.kv_transfer.kv_connector.utils",
                 "vllm.v1.core.kv_cache_manager", "vllm.v1.request"):
        sys.modules.setdefault(name, _AnyModule(name))
    import vllm.utils.math_utils as mu
    import vllm.v1.kv_cache_interface as kci
    import vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics as metrics
    mu.round_down = lambda x, m: (x // m) * m
    if not hasattr(kci, "MambaSpec"):
        kci.MambaSpec = type("MambaSpec", (), {})
    if not hasattr(metrics, "_TransferMetricName"):
        metrics._TransferMetricName = str
    path = ROOT / variant / REL
    spec = importlib.util.spec_from_file_location(f"offloading_scheduler_{variant}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeManager:
    def __init__(self, mod):
        self.mod = mod
        self.calls = []

    def prepare_store(self, keys, req_context):
        keys = list(keys)
        self.calls.append(keys)
        from vllm.v1.kv_offload.base import PrepareStoreOutput
        return PrepareStoreOutput(keys_to_store=keys, store_spec=SimpleNamespace(keys=keys), evicted_keys=[])

    def touch(self, keys, req_context):
        pass

    def stored(self, group):
        from vllm.v1.kv_offload.base import get_offload_group_idx
        return {k for call in self.calls for k in call if get_offload_group_idx(k) == group}


N_TOKENS = 99_975          # the probe's 100k prompt
STEP = 1_700               # tokens per prefill step (what the cluster showed: ~6.6 target rows/step)
HASH_BLOCK = 16


def _drive(mod, eagle, prompt_only=True):
    """Build a scheduler around the real config/state classes and run a chunked prefill."""
    from vllm.v1.kv_offload.base import ReqContext, RequestOffloadingContext
    target = mod.GroupOffloadConfig(group_idx=0, gpu_block_size=64, offloaded_block_size=256, hash_block_size_factor=16,
                                    sliding_window_size_in_blocks=None, alignment_block_count=None, is_eagle_group=eagle)
    drafter = mod.GroupOffloadConfig(group_idx=1, gpu_block_size=16, offloaded_block_size=64, hash_block_size_factor=4,
                                     sliding_window_size_in_blocks=32, alignment_block_count=None, is_eagle_group=eagle)
    config = mod.SchedulerOffloadConfig(kv_group_configs=(target, drafter), block_size_factor=4, num_workers=4,
                                        offload_prompt_only=prompt_only)
    sched = object.__new__(mod.OffloadingConnectorScheduler)
    sched.config = config
    sched.manager = _FakeManager(mod)
    sched._jobs = {}
    sched._block_id_to_pending_jobs = {}
    sched._job_counter = 0
    sched._req_status = {}
    n_hash = -(-N_TOKENS // HASH_BLOCK)
    req = SimpleNamespace(request_id="r1", block_hashes=[i.to_bytes(32, "big") for i in range(1, n_hash + 1)],
                          num_tokens=N_TOKENS, num_prompt_tokens=N_TOKENS, num_computed_tokens=0, kv_transfer_params=None)
    st = mod.RequestOffloadState(config=config, req=req, req_context=ReqContext("r1", None),
                                 offloading_context=RequestOffloadingContext())
    st.update_offload_keys()
    for gs, gpu_block in zip(st.group_states, (64, 16)):
        gs.block_ids.extend(range(1, -(-N_TOKENS // gpu_block) + 1))   # non-zero GPU ids for the whole prompt
    sched._req_status["r1"] = st
    steps = 0
    t = 0
    while t < N_TOKENS:
        chunk = min(STEP, N_TOKENS - t)
        req.num_computed_tokens = t
        jobs = sched._build_store_jobs(SimpleNamespace(num_scheduled_tokens={"r1": chunk}))
        for job in jobs.values():   # every key the job carries must be a key of this request
            assert set(job.transfer_spec[1].keys) <= set(st.group_states[0].offload_keys) | set(st.group_states[1].offload_keys)
        t += chunk
        steps += 1
    return sched, st, steps


@pytest.mark.parametrize("eagle", [True, False])
def test_overlay_stores_every_block_but_the_volatile_tail(eagle):
    mod = _load("overlay")
    sched, st, steps = _drive(mod, eagle)
    for g, gs in enumerate(st.group_states):
        full = N_TOKENS // sched.config.kv_group_configs[g].offloaded_block_size
        expected = set(gs.offload_keys[: full - 1 if eagle else full])
        missing = expected - sched.manager.stored(g)
        assert not missing, f"group {g}: {len(missing)} of {len(expected)} blocks never stored over {steps} steps"
        # and never the trailing block of an eagle group
        if eagle:
            assert gs.offload_keys[full - 1] not in sched.manager.stored(g)
        assert gs.next_stored_block_idx == (full - 1 if eagle else full)


def test_overlay_keeps_non_eagle_semantics_identical_to_baseline():
    """Without eagle groups the fix must be a no-op: same keys in the same order."""
    over, base = _load("overlay"), _load("baseline")
    so, _, _ = _drive(over, eagle=False)
    sb, _, _ = _drive(base, eagle=False)
    assert so.manager.calls == sb.manager.calls


def test_baseline_witness_loses_one_block_per_step():
    """Pins the bug the overlay fixes; if the fork changes, this tells us."""
    mod = _load("baseline")
    sched, st, steps = _drive(mod, eagle=True)
    for g, gs in enumerate(st.group_states):
        full = N_TOKENS // sched.config.kv_group_configs[g].offloaded_block_size
        missing = set(gs.offload_keys[: full - 1]) - sched.manager.stored(g)
        assert len(missing) == steps - 1, (g, len(missing), steps)


def test_overlay_marker_and_stage():
    text = (ROOT / "overlay" / REL).read_text()
    assert "DCP overlay: eagle trailing block is revisited" in text
    assert (ROOT / "stage/glm-dcp/offloading_scheduler.py").read_bytes() == (ROOT / "overlay" / REL).read_bytes()
