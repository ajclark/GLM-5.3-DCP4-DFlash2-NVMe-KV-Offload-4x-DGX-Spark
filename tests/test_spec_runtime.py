"""Exercise deployed kernels and scheduler bookkeeping at variable verify caps.

Triton runs in CPU interpretation mode. These tests do not establish numerical
parity of the full target model or validate CUDA graph replay on the Sparks.
"""
import ast
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from itertools import product
from types import SimpleNamespace as NS

import pytest
import torch

from harness import OVERLAY, extract
from spec_harness import FIXTURES, load_policy, source_class
from test_adaptive_spec import Request

P = load_policy()
INPUT = extract(FIXTURES / "input_batch.py", [
    "_combine_sampled_and_draft_tokens_kernel", "combine_sampled_and_draft_tokens",
    "_get_num_sampled_and_rejected_kernel", "get_num_sampled_and_rejected",
    "_post_update_kernel", "post_update",
    "_prepare_pos_seq_lens_kernel", "prepare_pos_seq_lens",
])
REJECT = extract(FIXTURES / "rejection_sampler_utils.py", ["_rejection_kernel"])["_rejection_kernel"]
DRAFT = extract(FIXTURES / "dflash_speculator.py", ["_prepare_dflash_inputs_kernel"])["_prepare_dflash_inputs_kernel"]


def ints(values):
    return torch.tensor(values, dtype=torch.int32)


@pytest.mark.parametrize("caps", [(1,), (3,), (5,), (7,), (1, 7), (3, 5, 1, 7)])
def test_target_inputs_and_positions_use_actual_lengths(caps):
    # Deliberately permute the persistent request slots to catch stride errors.
    mapping = ints(list(reversed(range(len(caps)))))
    sizes = [k+1 for k in caps]
    cu = ints([0] + list(torch.cumsum(ints(sizes), 0).tolist()))
    drafts = torch.arange(len(caps)*7, dtype=torch.int32).reshape(-1, 7) + 100
    original_drafts = drafts.clone()
    sampled = ints(list(range(1000, 1000+len(caps))))
    ids = torch.full((sum(sizes)+16,), -999, dtype=torch.int32)
    positions = torch.full_like(ids, -999)
    seq_lens = torch.full((12,), -999, dtype=torch.int32)
    computed = ints([61+i for i in range(len(caps))])
    INPUT["prepare_pos_seq_lens"](mapping, cu, computed, positions, seq_lens)
    indices = INPUT["combine_sampled_and_draft_tokens"](
        ids, mapping, sampled, cu, seq_lens, ints([32]*len(caps)), drafts, cu, sum(sizes))
    assert indices.tolist() == list(range(sum(sizes)))
    for i, (slot, k) in enumerate(zip(mapping.tolist(), caps)):
        start, end = cu[i:i+2].tolist()
        assert ids[start:end].tolist() == [sampled[slot].item()] + drafts[slot, :k].tolist()
        assert positions[start:end].tolist() == list(range(computed[slot], computed[slot]+k+1))
    assert torch.equal(drafts, original_drafts)
    assert ids[sum(sizes):].eq(-999).all()
    assert seq_lens[len(caps):].eq(0).all()


@pytest.mark.parametrize("k,a", [(k,a) for k in P.CAPS for a in range(k+1)])
def test_greedy_acceptance_rollback_and_next_draft(k, a):
    # Rejection stage consumes the actual cu_num_logits; storage stays width 8.
    draft_input = ints([99] + list(range(10, 10+k)))
    target_argmax = ints(list(range(10, 10+k)) + [50]).reshape(-1, 1)
    if a < k:
        target_argmax[a] = 51
    sampled = torch.full((1, 8), -999, dtype=torch.int32)
    rejected = ints([-1])
    lse = torch.zeros(1)
    maxima = torch.ones((k+1, 1))
    cu, mapping = ints([0, k+1]), ints([0])
    REJECT[(1,)](
        sampled, 8, rejected, lse, lse.clone(), None, 0,
        target_argmax, 1, maxima, 1, None, 0, draft_input,
        None, 0, 0, None, 0, None, 0, cu, mapping,
        torch.zeros(1), ints([1]), ints(list(range(k+1))), None, 1,
        PADDED_VOCAB_NUM_BLOCKS=1, HAS_DRAFT_LOGITS=False, SYNTHETIC_MODE=False)
    assert rejected.item() == a
    assert sampled[0, :a].tolist() == list(range(10, 10+a))
    if a < k:
        assert sampled[0, a].item() == 51
    else:
        # The separate resampling stage supplies the full-acceptance bonus.
        sampled[0, a] = 50
    assert sampled[0, a+1:].eq(-999).all()
    num_sampled, num_rejected = INPUT["get_num_sampled_and_rejected"](
        ints([a+1]), ints([61+k+1]), cu, mapping, ints([32]))
    assert num_rejected.item() == k-a
    computed, total, last = ints([61]), ints([62]), ints([99])
    all_ids = torch.full((1, 128), -1, dtype=torch.int32)
    INPUT["post_update"](mapping, computed, last, None, sampled,
                          num_sampled, num_rejected, cu, all_ids, total)
    assert computed.item() == 62+a
    assert total.item() == 63+a
    assert last.item() == (51 if a < k else 50)
    assert all_ids[0, 62:63+a].tolist() == sampled[0, :a+1].tolist()
    # Rejection must move the draft anchor back, while still proposing 7 tokens.
    out_ids = torch.full((16,), -999, dtype=torch.int32)
    out_pos, out_slots = out_ids.clone(), out_ids.clone()
    out_cu, out_seq = ints([-1]*3), ints([-1]*2)
    ctx_pos, ctx_slots = ints([-1]*(k+1)), ints([-1]*(k+1))
    sample_idx, sample_pos, sample_map = ints([-1]*14), ints([-1]*14), ints([-1]*14)
    block_table = ints([[10, 11, 12, 13]])
    DRAFT[(1, 1)](
        out_ids, out_pos, out_cu, out_seq, out_slots, ctx_pos, ctx_slots,
        sample_idx, sample_pos, sample_map, ints(list(range(61, 62+k))),
        cu, mapping, last, ints([-1]), num_sampled, num_rejected,
        block_table, 4, 777, 64, 8, 7, 2, 16, PAD_SLOT_ID=-1, BLOCK_SIZE=32)
    assert out_ids[:8].tolist() == [last.item()] + [777]*7
    assert out_pos[:8].tolist() == list(range(62+a, 70+a))
    assert out_seq.tolist() == [70+a, 0]
    assert out_cu.tolist() == [0, 8, 8]
    assert sample_idx[:7].tolist() == list(range(1, 8))
    assert out_slots[8:].eq(-1).all()


class Mode(IntEnum):
    NONE = 0
    FULL = 1
    PIECEWISE = 2
    FULL_AND_PIECEWISE = 3

    def decode_mode(self):
        return Mode.FULL if self == Mode.FULL_AND_PIECEWISE else self

    def mixed_mode(self):
        return Mode.PIECEWISE if self == Mode.FULL_AND_PIECEWISE else self

    def separate_routine(self):
        return self == Mode.FULL_AND_PIECEWISE


def graph_classes():
    path = OVERLAY / "v1/worker/gpu/cudagraph_utils.py"
    ns = extract(path, ["BatchExecutionDescriptor", "_is_compatible"],
                 {"dataclass": dataclass, "CUDAGraphMode": Mode})
    ns.update(defaultdict=defaultdict, product=product, extra_capture_sizes=P.extra_capture_sizes)
    base = source_class(path, "CudaGraphManager", ["_init_candidates", "dispatch", "_resolve_effective_loras"], ns)
    target = type("ModelCudaGraphManager", (base,), {})
    ns["ModelCudaGraphManager"] = target
    return base, target


@pytest.mark.parametrize("mode", ["off", "shadow"])
@pytest.mark.parametrize("cg_mode", [Mode.FULL,Mode.FULL_AND_PIECEWISE])
def test_only_target_gets_small_graphs(monkeypatch, mode, cg_mode):
    monkeypatch.setenv("GLM_SPEC_POLICY", mode)
    base, target = graph_classes()
    for cls in (base, target):
        mgr = cls()
        mgr.compilation_config = NS(cudagraph_capture_sizes=[8, 16, 24, 32])
        mgr.cudagraph_mode = cg_mode
        mgr.max_num_reqs = 12
        mgr.decode_query_len = 8
        mgr.lora_capture_cases = [0]
        mgr._candidates, mgr._capture_descs = {}, {}
        mgr._lora_dispatch_map, mgr._max_lora_case = {}, 0
        mgr._graphs_captured = True
        mgr._init_candidates()
        for k in P.CAPS:
            desc = mgr.dispatch(1, k+1, k+1, 0)
            expected_mode = (Mode.PIECEWISE if cg_mode == Mode.FULL_AND_PIECEWISE and k!=7
                             and (mode=='off' or cls is base) else Mode.FULL)
            assert desc.cg_mode == expected_mode
            assert desc.num_tokens == (k+1 if mode != "off" and cls is target else 8)
            if mode != 'off' and cls is target and cg_mode == Mode.FULL_AND_PIECEWISE:
                assert desc.num_reqs == 1 and desc.uniform_token_count == k+1
        assert mgr.dispatch(2, 16, 8, 0).num_tokens == 16
        # A mixed two-request batch must never use one of the uniform C1 graphs.
        if cg_mode == Mode.FULL_AND_PIECEWISE:
            assert mgr.dispatch(2,6,None,0).cg_mode == Mode.PIECEWISE


class BookkeepingRequest(Request):
    def __init__(self):
        super().__init__()
        self.num_computed_tokens = 99
        self.num_output_placeholders = 0
        self.spec_token_ids = [-1]*7
        self.status = "running"

    @property
    def num_tokens_with_spec(self):
        return self.num_tokens + len(self.spec_token_ids)


def sched_classes():
    path = OVERLAY / "v1/core/sched/scheduler.py"
    ns = {"RequestStatus": NS(RUNNING="running")}
    base = source_class(path, "Scheduler", ["_update_after_schedule"], ns, bases=[])
    def append(self, request, tokens):
        request.num_tokens += len(tokens)
        return tokens, False
    base._update_request_with_output = append
    async_cls = source_class(FIXTURES / "async_scheduler.py", "AsyncScheduler",
                             ["_update_after_schedule", "_update_request_with_output"], ns)
    # Execute the exact scheduling cap/budget statements and rejection branch.
    tree = ast.parse(path.read_text())
    schedule = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "schedule")
    loop = next(n for n in ast.walk(schedule) if isinstance(n, ast.While))
    start = next(i for i,n in enumerate(loop.body) if isinstance(n, ast.If) and "_adaptive_spec.select" in ast.unparse(n))
    fragment = ast.Module(body=loop.body[start:start+2], type_ignores=[])
    budget_code = compile(ast.fix_missing_locations(fragment), str(path), "exec")
    update = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "update_from_output")
    reject = next(n for n in ast.walk(update) if isinstance(n, ast.If) and ast.unparse(n.test).startswith("scheduled_spec_token_ids and"))
    rollback_code = compile(ast.Module(body=[reject], type_ignores=[]), str(path), "exec")
    return async_cls, budget_code, rollback_code


@pytest.mark.parametrize("k,a,next_k", [(k,a,next_k) for k in P.CAPS for a in range(k+1) for next_k in P.CAPS])
def test_two_inflight_steps_keep_accounting_exact(k, a, next_k):
    cls, budget, rollback = sched_classes()
    r = BookkeepingRequest()
    shared = r.spec_token_ids
    s = cls()
    s.requests, s.running, s.waiting, s.skipped_waiting = {r.request_id:r}, [r], [], []
    s.defer_block_free = s.enable_return_routed_experts = False
    s._inflight_prefills = set()
    s.current_step, s.pp_size, s.num_sampled_tokens_per_step = 1, 1, 1
    s.use_v2_model_runner = True
    s.kv_cache_manager = NS(cache_blocks=lambda *args: None)
    s.make_spec_decoding_stats = lambda *args, **kwargs: None
    outputs = []
    for cap in (k, next_k):
        s._adaptive_spec = NS(select=lambda *args, **kwargs: cap)
        env = {"self":s, "request":r}
        exec(budget, env)
        assert shared == [-1]*7
        assert env["num_new_tokens"] == cap+1
        out = NS(num_scheduled_tokens={r.request_id:cap+1},
                 scheduled_spec_decode_tokens={r.request_id:r.spec_token_ids},
                 num_spec_tokens_to_schedule=7, has_structured_output_requests=False,
                 pending_structured_output_tokens=False, num_invalid_spec_tokens=0)
        outputs.append(out)
        s._update_after_schedule(out)
        assert len(r.spec_token_ids) == 7
    for cap, accepted, out in ((k, a, outputs[0]), (next_k, next_k, outputs[1])):
        tokens = list(range(1000, 1001+accepted))
        exec(rollback, {"self":s, "request":r, "scheduled_spec_token_ids":[-1]*cap,
                        "generated_token_ids":tokens, "spec_decoding_stats":None,
                        "scheduler_output":out, "req_id":r.request_id})
        s._update_request_with_output(r, tokens)
        assert r.num_computed_tokens == r.num_tokens - 1 + r.num_output_placeholders
    assert r.num_output_placeholders == 0
