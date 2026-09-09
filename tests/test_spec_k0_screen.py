"""K0 sampler/input correctness and an explicit asynchronous parking model."""
import hashlib
import importlib.util
import json
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from harness import OVERLAY, ROOT, extract, extract_methods
from spec_harness import FIXTURES as RUNTIME, source_class
from test_spec_runtime import INPUT, Mode, graph_classes, sched_classes, BookkeepingRequest

spec = importlib.util.spec_from_file_location("spec_k0_screen", ROOT / "bench/spec_k0_screen.py")
k0 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = k0
spec.loader.exec_module(k0)
FIXTURES = ROOT / "tests/fixtures/spec_k0"


def ints(values):
    return torch.tensor(values, dtype=torch.int32)


def test_pinned_k0_source_hashes_and_actual_combined_feature_dimensions():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for name, row in manifest["files"].items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == row["sha256"]
    config = NS(**json.loads((FIXTURES / "draft-config.json").read_text()))
    def linear(**kwargs):
        layer = torch.nn.Identity()
        layer.input_size, layer.output_size = kwargs["input_size"], kwargs["output_size"]
        return layer
    namespace = {
        "nn": torch.nn, "get_draft_quant_config": lambda cfg: None,
        "get_current_vllm_config": lambda: NS(cache_config=None),
        "VocabParallelEmbedding": lambda *a, **kw: torch.nn.Identity(),
        "RMSNorm": lambda *a, **kw: torch.nn.Identity(), "ReplicatedLinear": linear,
        "support_torch_compile": lambda cls: cls,
        "maybe_prefix": lambda prefix, name: f"{prefix}.{name}" if prefix else name,
    }
    cls = source_class(FIXTURES / "qwen3_dflash.py", "DFlashQwen3Model", ["__init__"], namespace)
    cls.decoder_layer_cls = staticmethod(lambda *a, **kw: torch.nn.Identity())
    model = cls(vllm_config=NS(speculative_config=NS(draft_model_config=NS(hf_config=config)),
                             model_config=NS(dtype=torch.bfloat16)))
    assert len(model.layers) == 6
    assert model.fc.input_size == 36864
    assert model.fc.output_size == 6144
    assert config.sliding_window == 2048


@pytest.mark.parametrize("position", [61, 2047, 100055, 170125])
def test_k0_inputs_ignore_stale_draft_buffer_and_commit_one_target_token(position):
    mapping, cu = ints([0]), ints([0, 1])
    drafts = ints([[-777] * 7])
    ids = ints([-999] * 16)
    positions = torch.full((16,), -999, dtype=torch.int64)
    seq_lens = ints([-999] * 12)
    computed, total, last = ints([position]), ints([position + 1]), ints([9])
    INPUT["prepare_pos_seq_lens"](mapping, cu, computed, positions, seq_lens)
    indices = INPUT["combine_sampled_and_draft_tokens"](
        ids, mapping, last, cu, seq_lens, ints([32]), drafts, cu, 1)
    assert indices.tolist() == [0]
    assert ids.tolist() == [9] + [-999] * 15
    assert positions[0].item() == position
    sampled, rejected = INPUT["get_num_sampled_and_rejected"](
        ints([1]), seq_lens[:1], cu, mapping, ints([32]))
    assert sampled.item() == 1 and rejected.item() == 0
    all_ids = torch.full((1, position + 16), -1, dtype=torch.int32)
    INPUT["post_update"](mapping, computed, last, None, ints([[25]]), sampled,
                         rejected, cu, all_ids, total)
    assert computed.item() == position + 1
    assert total.item() == position + 2 and last.item() == 25
    assert all_ids[0, position + 1].item() == 25


@pytest.mark.parametrize("num_drafts,expected", [(0, "ordinary"), (1, "rejection")])
def test_actual_worker_chooses_ordinary_sampler_for_all_zero_draft_batch(num_drafts, expected):
    fn = extract_methods(OVERLAY / "v1/worker/gpu/model_runner.py", "GPUModelRunner", ["sample"],
                         {"SamplerOutput": object, "InputBatch": object, "GrammarOutput": object})["sample"]
    calls = []
    output = NS(num_sampled=ints([1]), num_rejected=ints([0]))
    def ordinary(*args):
        calls.append("ordinary")
        return output
    def rejection(*args):
        calls.append("rejection")
        return output
    worker = NS(model=NS(compute_logits=lambda hidden: hidden), sampler=ordinary,
                rejection_sampler=rejection, speculator=NS(draft_logits=None))
    fn(worker, torch.ones(1, 4), NS(logits_indices=ints([0]), num_draft_tokens=num_drafts), None)
    assert calls == [expected]


@pytest.mark.parametrize("caps,acceptance", [((0,), (0,)), ((0, 7), (0, 0)),
    ((7, 0), (7, 0)), ((0, 3, 0), (0, 1, 0))])
def test_full_rejection_pipeline_handles_zero_rows_in_mixed_batches(caps, acceptance):
    gumbel = extract(FIXTURES / "gumbel.py", ["gumbel_block_argmax"])
    names = ["_compute_block_max_and_sumexp", "_compute_global_lse",
             "_compute_block_stats_kernel", "_rejection_kernel", "_resample_kernel",
             "_insert_resampled_kernel", "rejection_sample"]
    reject = extract(RUNTIME / "rejection_sampler_utils.py", names, gumbel)["rejection_sample"]
    sizes = [k + 1 for k in caps]
    cu = ints([0] + torch.cumsum(ints(sizes), 0).tolist())
    n = sum(sizes)
    logits = torch.full((n, 32), -10.)
    draft_input, mappings, local = [], [], []
    for req, (cap, accepted) in enumerate(zip(caps, acceptance)):
        start = int(cu[req])
        draft_input += [2] + list(range(10, 10 + cap))
        mappings += [req] * (cap + 1)
        local += list(range(cap + 1))
        for j in range(cap + 1):
            winner = 26 if j == cap else (25 if j == accepted else 10 + j)
            logits[start + j, winner] = 10.
    sampled, counts = reject(logits, None, ints(draft_input), cu,
        torch.arange(100055, 100055 + n), ints(list(range(len(caps)))), ints(mappings),
        ints(local), torch.zeros(len(caps)), ints([13] * len(caps)), 7)
    assert counts.tolist() == [a + 1 for a in acceptance]
    for req, (cap, accepted) in enumerate(zip(caps, acceptance)):
        expected = list(range(10, 10 + accepted)) + [26 if accepted == cap else 25]
        assert sampled[req, :accepted + 1].tolist() == expected


def test_current_graph_dispatch_is_not_true_m1_but_candidate_extension_is_feasible(monkeypatch):
    monkeypatch.setenv("GLM_SPEC_POLICY", "shadow")
    _, target = graph_classes()
    def manager():
        mgr = target()
        mgr.compilation_config = NS(cudagraph_capture_sizes=[8, 16, 24, 32])
        mgr.cudagraph_mode = Mode.FULL_AND_PIECEWISE
        mgr.max_num_reqs, mgr.decode_query_len, mgr.lora_capture_cases = 12, 8, [0]
        mgr._candidates, mgr._capture_descs = {}, {}
        mgr._lora_dispatch_map, mgr._max_lora_case, mgr._graphs_captured = {}, 0, True
        mgr._init_candidates()
        return mgr
    original = manager().dispatch(1, 1, 1, 0)
    assert original.cg_mode == Mode.PIECEWISE and original.num_tokens == 2
    # Change only the CPU-extracted test namespace, never a runtime file.
    monkeypatch.setitem(target._init_candidates.__globals__, "extra_capture_sizes",
                        lambda is_target: (1, 2, 4, 6) if is_target else ())
    proposed = manager().dispatch(1, 1, 1, 0)
    assert proposed.cg_mode == Mode.FULL and proposed.num_tokens == 1
    assert proposed.uniform_token_count == 1 and proposed.num_reqs == 1


def test_zero_cap_bookkeeping_can_follow_an_inflight_rejection_but_placeholders_return():
    cls, _, rollback = sched_classes()
    request = BookkeepingRequest()
    scheduler = cls()
    scheduler.requests = {request.request_id: request}
    scheduler.defer_block_free = scheduler.enable_return_routed_experts = False
    scheduler._inflight_prefills = set()
    scheduler.current_step, scheduler.pp_size, scheduler.num_sampled_tokens_per_step = 1, 1, 1
    scheduler.use_v2_model_runner = True
    scheduler.kv_cache_manager = NS(cache_blocks=lambda *args: None)
    scheduler.make_spec_decoding_stats = lambda *args, **kwargs: None
    outputs = []
    for cap in (7, 0):
        output = NS(num_scheduled_tokens={request.request_id: cap + 1},
            scheduled_spec_decode_tokens={request.request_id: [-1] * cap} if cap else {},
            num_spec_tokens_to_schedule=7, has_structured_output_requests=False,
            pending_structured_output_tokens=False, num_invalid_spec_tokens=0)
        outputs.append(output)
        scheduler._update_after_schedule(output)
    assert request.num_output_placeholders == 9
    # A single empty entry does not park future steps: unmodified AsyncScheduler
    # restores the global seven-proposal placeholder for the next schedule.
    assert request.spec_token_ids == [-1] * 7
    for cap, output in zip((7, 0), outputs):
        exec(rollback, {"self": scheduler, "request": request,
            "scheduled_spec_token_ids": [-1] * cap, "generated_token_ids": [25],
            "spec_decoding_stats": None, "scheduler_output": output,
            "req_id": request.request_id})
        scheduler._update_request_with_output(request, [25])
    assert request.num_output_placeholders == 0
    assert request.num_computed_tokens == request.num_tokens - 1


@pytest.mark.parametrize("strategy", ["eager", "deferred"])
def test_park_arm_resume_survives_two_queued_steps_and_refreshes_anchor(strategy):
    model = k0.ParkingModel(strategy=strategy)
    old = [model.schedule(), model.schedule()]
    model.park()
    parked = [model.schedule(), model.schedule()]
    assert [s.cap for s in old + parked] == [7, 7, 0, 0]
    for step in old:
        model.execute(step, accepted=2)
    passes = model.query_passes
    for step in parked:
        model.execute(step)
    assert model.query_passes == passes and model.proposal_anchor is None
    model.arm()
    arming = [model.schedule(), model.schedule()]
    ack = model.execute(arming[0])
    old_anchor = ack.anchor
    assert model.receive(ack)
    resumed = model.schedule(3)
    assert resumed.cap == 3
    # Already queued K0 steps MUST still run drafting while ARMED.
    model.execute(arming[1])
    assert model.proposal_anchor > old_anchor
    model.execute(resumed, accepted=3)
    assert model.proposal_anchor == model.computed


def test_consuming_one_shot_arm_proposal_after_queued_k0_is_rejected():
    model = k0.ParkingModel()
    model.park()
    model.execute(model.schedule())
    model.arm()
    first, queued = model.schedule(), model.schedule()
    ack = model.execute(first)
    model.receive(ack)
    resumed = model.schedule(3)
    model.execute(queued)
    model.proposal_anchor = ack.anchor  # emulate the unsafe one-shot design
    with pytest.raises(ValueError, match="stale/missing proposal"):
        model.execute(resumed, 1)


def test_old_arm_ack_cannot_override_a_later_park_or_reused_request_epoch():
    model = k0.ParkingModel()
    model.park()
    model.execute(model.schedule())
    model.arm()
    ack = model.execute(model.schedule())
    model.park()
    assert not model.receive(ack)
    replacement = k0.ParkingModel(epoch=1)
    replacement.arm()
    assert not replacement.receive(ack)


def test_cancel_keeps_retention_lease_until_pending_worker_steps_drain():
    model = k0.ParkingModel(strategy="deferred")
    model.park()
    queued = [model.schedule(), model.schedule()]
    model.cancel()
    assert not model.retention_lease_releasable
    for step in queued:
        ack = model.execute(step)
        assert ack.reason == "cancelled" and not model.receive(ack)
    assert model.retention_lease_releasable and model.computed == 4096


def test_deferred_retention_is_bounded_and_replay_uses_new_physical_slots():
    model = k0.ParkingModel(strategy="deferred", window=16, slack=8, chunk=5)
    model.park()
    for _ in range(80):
        model.execute(model.schedule())
    assert len(model.rows) <= 24
    model.arm()
    model.execute(model.schedule())
    assert model.rebuilt_positions == list(range(model.computed - 16, model.computed))
    assert max(map(len, model.rebuild_chunks)) <= 5
    old = k0.slots_for_positions(model.rebuilt_positions, list(range(100)), 64)
    new = k0.slots_for_positions(model.rebuilt_positions, list(range(200, 300)), 64)
    assert all(b - a == 200 * 64 for a, b in zip(old, new))


def test_deferred_rows_are_immutable_and_cannot_publish_stale_draft_kv():
    model = k0.ParkingModel(strategy="deferred")
    model.park()
    step = model.schedule()
    values = [[1, 2, 3]]
    model.execute(step, combined_rows=values)
    values[0][0] = 999  # retained storage must not alias a reused output buffer
    assert model.rows[4096] == (1, 2, 3)
    with pytest.raises(ValueError, match="poison cache reuse"):
        model.publish_prefix(model.computed)
    model.arm()
    assert model.receive(model.execute(model.schedule()))
    model.publish_prefix(model.computed)


def test_missing_base_cache_requires_full_live_window_before_rearming():
    model = k0.ParkingModel(strategy="deferred", window=16)
    model.park()
    model.execute(model.schedule())
    model.materialized = None  # simulate invalidated base ownership
    model.arm()
    ack = model.execute(model.schedule())
    assert not ack.drafting_ready and not model.receive(ack)
    assert model.schedule().cap == 0


def test_combined_feature_ring_fits_but_raw_aux_and_full_replay_peak_do_not():
    config = json.loads((FIXTURES / "draft-config.json").read_text())
    budget = k0.memory_budget(config)
    assert budget["combined_bytes_per_token"] == 12288
    assert budget["raw_aux_bytes_per_token"] == 73728
    assert budget["combined_ring_bytes"] == 25264128
    assert budget["combined_chunk_fits_budget"]
    assert budget["raw_aux_ring_bytes"] > budget["budget_bytes"]
    assert budget["estimated_ring_plus_full_window_scratch_bytes"] > budget["budget_bytes"]
    # Conservative unsharded draft-head case still fits when replay is chunked.
    assert k0.memory_budget(config, kv_heads_per_rank=8)["combined_chunk_fits_budget"]


def test_k0_timing_is_unknown_and_scenarios_include_strict_reentry_break_even():
    costs = {"1": 100., "3": 120., "5": 140., "7": 160.}
    zero_acceptance = [0.] * 7
    unknown = k0.break_even(costs, zero_acceptance)
    assert unknown["strict_k0_ms_per_token_ceiling_before_resume"] == 100
    assert unknown["true_k0_timing"] == "unmeasured"
    assert "scenario_assumptions_not_measurements" not in unknown
    scenario = k0.break_even(costs, zero_acceptance, 90, 50, 5)["scenario_assumptions_not_measurements"]
    assert scenario["minimum_tokens_for_strict_time_saving"] == 6
    assert scenario["time_saved_ms"] == 0
    impossible = k0.break_even(costs, [1.] * 7, 90, 50)["scenario_assumptions_not_measurements"]
    assert impossible["minimum_tokens_for_strict_time_saving"] is None
