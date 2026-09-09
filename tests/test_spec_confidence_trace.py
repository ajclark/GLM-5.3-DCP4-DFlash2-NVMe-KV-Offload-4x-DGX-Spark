"""Actual async-copy methods, proposal ownership and causal trace joins."""
import ast
import contextlib
import copy
import importlib.util
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from harness import BASELINE, OVERLAY, ROOT, extract_methods
from spec_harness import source_class, load_policy
from test_adaptive_spec import Request, config, schedule
from test_spec_runtime import DRAFT, INPUT, ints


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = load("confidence_trace_test", OVERLAY / "v1/spec_decode/confidence_trace.py")
CONVERT = load("confidence_convert_test", ROOT / "bench/spec_confidence_convert.py")
SCREEN = load("confidence_screen_trace_test", ROOT / "bench/spec_confidence_screen.py")
P = load_policy()


def new_request(rid="r", **sp_changes):
    return NS(req_id=rid, sampling_params=NS(temperature=0, extra_args={}, **sp_changes))


def batch(k=7, anchor=100055, rid="r", padded=16):
    return NS(num_reqs=1, req_ids=[rid], has_structured_output_reqs=False,
        is_prefilling_np=np.array([False]), num_draft_tokens=k, num_tokens=k + 1,
        num_tokens_after_padding=padded, num_reqs_after_padding=12,
        positions=torch.tensor(list(range(anchor, anchor + k + 1)) + [-999] * (padded-k-1)),
        input_ids=torch.tensor([99] + list(range(10, 10+k)) + [-999] * (padded-k-1), dtype=torch.int32),
        idx_mapping=torch.tensor([4]), query_start_loc=torch.tensor([0,k+1]))


def speculator(anchor=100055):
    return NS(_selector_scores=torch.arange(12*7*16,dtype=torch.float32).reshape(12,7,16),
        sample_pos=torch.tensor(list(range(anchor+1,anchor+8)) + [-111] * 77),
        draft_tokens=torch.tensor([list(range(10,17))] + [[-222]*7]*11, dtype=torch.int64))


def prepared(k=7):
    collector = C.ConfidenceCollector()
    collector.add(new_request())
    b, draft = batch(k), speculator()
    assert collector.capture(b, draft) is None  # no owned previous proposal
    collector.proposed(b)
    return collector, b, draft


def packet(k=7):
    collector, b, draft = prepared(k)
    return collector.capture(b, draft)


@pytest.mark.parametrize("rank,flag,want", [(0,"1",True),(1,"1",False),(0,"0",False),(0,"true",False)])
def test_opt_in_and_target_output_tp_rank_are_required(monkeypatch, rank, flag, want):
    monkeypatch.setenv("GLM_SPEC_CONFIDENCE_TRACE", flag)
    assert C.enabled(config(), rank) == want


@pytest.mark.parametrize("key,value", [("temperature",.8),("structured_outputs",object()),
    ("presence_penalty",1),("frequency_penalty",1),("repetition_penalty",1.1),
    ("allowed_token_ids",[3]),("bad_words",["bad"]),("logit_bias",{1:2})])
def test_request_filters_match_unsupported_sampling_contract(key,value):
    data = new_request()
    setattr(data.sampling_params,key,value)
    assert not C.request_eligible(data)


@pytest.mark.parametrize("value", [False,0,1,"false","true"])
def test_only_boolean_true_or_omission_allows_request_capture(value):
    data = new_request()
    data.sampling_params.extra_args = {"spec_confidence_trace":value}
    assert not C.request_eligible(data)


@pytest.mark.parametrize("k", [1,3,5,7])
def test_compact_clones_survive_source_reuse_and_ignore_padding(k):
    collector,b,draft = prepared(k)
    captured = collector.capture(b,draft)
    assert captured["meta"]["proposal_age_steps"] == 1
    assert captured["tensors"]["realized_scores"].shape == (7,16)
    assert sum(v.numel()*v.element_size() for v in captured["tensors"].values()) <= 656
    draft._selector_scores.fill_(123456)
    draft.sample_pos.fill_(-1)
    draft.draft_tokens.fill_(-1)
    b.positions.fill_(-1)
    b.input_ids.fill_(-1)
    row = C.finish_packet(captured)["r"]
    assert row["valid"] and row["anchor"] == 100055
    assert row["target_tokens"] == [99] + list(range(10,10+k))
    assert row["realized_scores"][0][0] == 0


@pytest.mark.parametrize("cause", ["c2","prefill","structured","zero","odd","owner"])
def test_ineligible_batch_invalidates_previous_ownership(cause):
    collector,b,draft = prepared()
    if cause == "c2":
        b.num_reqs=2; b.req_ids=["r","second"]
    if cause == "prefill": b.is_prefilling_np[0]=True
    if cause == "structured": b.has_structured_output_reqs=True
    if cause == "zero": b.num_draft_tokens=0
    if cause == "odd": b.num_tokens=1024
    if cause == "owner": b.req_ids=["other"]
    assert collector.capture(b,draft) is None and collector.previous is None


def test_c2_admission_even_with_one_scheduled_row_requires_new_proposal():
    collector,b,draft = prepared()
    collector.add(new_request("other"))
    assert collector.capture(b,draft) is None
    collector.remove("other")
    assert collector.capture(b,draft) is None
    collector.proposed(b)
    assert collector.capture(b,draft) is not None


def test_removal_preemption_and_request_reuse_change_epoch():
    collector,b,draft = prepared()
    previous = copy.deepcopy(collector.previous)
    collector.remove("r")
    assert collector.previous is None and not collector.requests
    collector.add(new_request())
    collector.previous = previous  # emulates stale queued metadata
    assert collector.capture(b,draft) is None
    collector.proposed(b)
    row = collector.capture(b,draft)
    assert row["meta"]["epoch"] > previous["epoch"]


def test_per_request_capture_budget_does_not_allocate_past_limit():
    collector,b,draft = prepared()
    for i in range(C.PACKET_LIMIT):
        assert collector.capture(b,draft) is not None
        collector.proposed(b)
    # Removing the sources makes any accidental clone raise immediately.
    assert collector.capture(b,None) is None
    assert collector.requests["r"]["captured"] == 128
    assert collector.previous is None


@pytest.mark.parametrize("k,accepted", [(1,0),(3,1),(5,3),(7,7)])
def test_actual_preparation_sample_positions_match_next_verified_suffix(k,accepted):
    start=100055
    anchor=start+accepted+1
    output_ids,output_pos,output_slots=[torch.full((16,),-999,dtype=torch.int32) for _ in range(3)]
    output_cu,output_seq=ints([-1]*3),ints([-1]*2)
    context_pos,context_slots=ints([-1]*(k+1)),ints([-1]*(k+1))
    sample_idx,sample_pos,sample_map=ints([-1]*14),ints([-1]*14),ints([-1]*14)
    cu,mapping=ints([0,k+1]),ints([0])
    DRAFT[(1,1)](output_ids,output_pos,output_cu,output_seq,output_slots,
        context_pos,context_slots,sample_idx,sample_pos,sample_map,
        ints(list(range(start,start+k+1))),cu,mapping,ints([99]),ints([-1]),
        ints([accepted+1]),ints([k-accepted]),torch.arange(2048,dtype=torch.int32).reshape(1,-1),
        2048,777,64,8,7,2,16,PAD_SLOT_ID=-1,BLOCK_SIZE=32)
    assert sample_pos[:7].tolist()==list(range(anchor+1,anchor+8))
    b=batch(k,anchor)
    seq_lens=ints([-1]*12)
    INPUT["prepare_pos_seq_lens"](mapping,cu,ints([anchor]),b.positions,seq_lens)
    draft=speculator(anchor)
    draft.sample_pos[:7]=sample_pos[:7]
    INPUT["combine_sampled_and_draft_tokens"](b.input_ids,mapping,ints([99]),cu,
        seq_lens,ints([32]),draft.draft_tokens,cu,k+1)
    collector=C.ConfidenceCollector(); collector.add(new_request())
    collector.capture(b,draft); collector.proposed(b)
    row=C.finish_packet(collector.capture(b,draft))["r"]
    assert row["valid"] and row["anchor"]==anchor


def test_actual_runner_add_remove_and_preempt_maintain_incarnation():
    cls=source_class(OVERLAY/"v1/worker/gpu/model_runner.py","GPUModelRunner",
        ["_remove_request","add_requests","finish_requests"],{},bases=["object"])
    runner=cls(); runner._spec_confidence=C.ConfidenceCollector()
    indices={}
    runner.req_states=NS(req_id_to_index=indices,
        remove_request=lambda rid:indices.pop(rid,None),
        add_request=lambda **kwargs:indices.update({kwargs["req_id"]:4}),
        apply_staged_writes=lambda:None)
    runner.model_state=NS(remove_request=lambda rid:None,add_request=lambda *args:None,
                         apply_staged_writes=lambda:None)
    runner.pp_handler=runner.encoder_cache=runner.prompt_logprobs_worker=None
    runner.lora_state=NS(remove_request=lambda rid:None,add_request=lambda *args:None)
    runner.block_tables=NS(append_block_ids=lambda *args,**kwargs:None)
    runner.is_last_pp_rank=False; runner.sampler=None
    data=new_request(); data.sampling_params.max_tokens=256
    data.prompt_token_ids=data.prefill_token_ids=[1,2,3]
    data.num_computed_tokens=0; data.block_ids=[]; data.lora_request=None
    output=NS(scheduled_new_reqs=[data])
    runner.add_requests(output)
    first=runner._spec_confidence.requests["r"]["epoch"]
    runner.add_requests(output)  # streaming replacement uses remove then add
    assert runner._spec_confidence.requests["r"]["epoch"]>first
    runner.finish_requests(NS(finished_req_ids=set(),preempted_req_ids={"r"}))
    assert not indices and not runner._spec_confidence.requests


@pytest.mark.parametrize("kind,reason", [("anchor","anchor_or_position_mismatch"),
    ("position","anchor_or_position_mismatch"),("draft","verified_draft_token_mismatch"),
    ("nan","nonfinite_score")])
def test_bad_gpu_diagnostic_data_abstains_without_changing_target(kind,reason):
    value = packet(3)
    t = value["tensors"]
    original = t["target_tokens"].clone()
    if kind=="anchor": t["sample_positions"][0]+=1
    if kind=="position": t["sample_positions"][6]+=2
    if kind=="draft": t["draft_tokens"][0]+=1
    if kind=="nan": t["realized_scores"][0,0]=float("nan")
    row = C.finish_packet(value)["r"]
    assert not row["valid"] and row["invalid_reason"]==reason
    assert "realized_scores" not in row
    assert torch.equal(original,t["target_tokens"])


def fake_async(monkeypatch, events):
    pending=[]
    class Host:
        def __init__(self): self.value=None
        def copy_(self, value, non_blocking):
            assert non_blocking
            events.append("confidence_copy")
            pending.append(lambda: setattr(self,"value",value.clone()))
            return self
        def tolist(self):
            assert self.value is not None, "read before existing event completion"
            return self.value.tolist()
    def empty_like(tensor,device,pin_memory):
        assert device=="cpu" and pin_memory
        return Host()
    monkeypatch.setattr(C,"torch",NS(empty_like=empty_like,float32=torch.float32,
                                    int64=torch.int64,int32=torch.int32))
    class Event:
        def __init__(self): events.append("new_event")
        def record(self,stream): events.append("record")
        def synchronize(self):
            events.append("synchronize")
            for fn in pending: fn()
            pending.clear()
    class Stream:
        def wait_stream(self,other): events.append("wait_existing_main")
    @contextlib.contextmanager
    def stream(copy_stream,main_stream):
        yield
    def finish(value):
        events.append("finish_packet")
        return C.finish_packet(value)
    cls=source_class(OVERLAY/"v1/worker/gpu/async_utils.py","AsyncOutput",
        ["__init__","get_output"], {
            "torch":NS(cuda=NS(Event=Event)),"stream":stream,
            "async_copy_to_np":lambda value:value.numpy(),
            "packet_to_host":C.packet_to_host,"finish_packet":finish,
        },bases=["object"])
    return cls,Stream()


@pytest.mark.parametrize("capture", [False,True])
def test_actual_async_methods_use_one_existing_event_and_preserve_sampled_output(monkeypatch,capture):
    value=packet(3) if capture else None
    events=[]
    cls,copy_stream=fake_async(monkeypatch,events)
    runner=NS(prompt_logprobs_dict={},req_ids=["r"],spec_confidence=None)
    sampler=NS(sampled_token_ids=torch.tensor([[10,11,25,-1,-1,-1,-1,-1]]),
               logprobs_tensors=None,num_nans=None)
    result=cls(runner,sampler,torch.tensor([3]),object(),copy_stream,confidence_packet=value)
    assert events.count("new_event")==1 and events.count("record")==1
    assert events.count("wait_existing_main")==1 and "synchronize" not in events
    assert events.count("confidence_copy")== (5 if capture else 0)
    if capture:
        assert result.confidence_packet is value  # source clones retained
    out=result.get_output()
    assert out.sampled_token_ids==[[10,11,25]]
    assert events.count("synchronize")==1
    if capture:
        assert events.index("synchronize") < events.index("finish_packet")
        assert out.spec_confidence["r"]["valid"]
    else:
        assert out.spec_confidence is None


@pytest.mark.parametrize("capture",[False,True])
def test_actual_runner_clones_before_copy_record_and_next_proposal(monkeypatch,capture):
    collector,b,draft=prepared(3)
    events=[]
    cls,copy_stream=fake_async(monkeypatch,events)
    def propose(*args,**kwargs):
        events.append("propose")
        draft._selector_scores.fill_(-888)
        draft.sample_pos.fill_(-1)
        draft.draft_tokens.fill_(-1)
        return draft.draft_tokens[:1]
    draft.propose=propose; draft.supports_mm_inputs=False
    sampling=NS(sampled_token_ids=torch.tensor([[10,11,25,-1,-1,-1,-1,-1]]),
                logprobs_tensors=None,num_nans=None)
    func=source_class(OVERLAY/"v1/worker/gpu/model_runner.py","GPUModelRunner",["sample_tokens"],
        {"torch":torch,"step_eplb_after":lambda:lambda fn:fn,
         "AsyncOutput":cls,"ModelRunnerOutput":lambda **kw:NS(**kw)},bases=["object"])
    runner=func()
    runner.execute_model_state=NS(input_batch=b,attn_metadata=None,slot_mappings_by_layer=None,
        hidden_states=torch.zeros(4,2),aux_hidden_states=None,finished_req_ids=set())
    runner.is_last_pp_rank=True; runner.pp_handler=None
    runner.sample=lambda *args:(sampling,torch.tensor([3]),torch.tensor([1]))
    runner.prompt_logprobs_worker=NS(compute_prompt_logprobs=lambda *args:{})
    runner.model=NS(compute_logits=lambda x:x)
    runner.req_states=NS(all_token_ids=NS(gpu=None),num_computed_tokens=NS(gpu=None),
        prompt_len=NS(np=None),last_sampled_tokens=None,next_prefill_tokens=None,
        draft_tokens=torch.zeros(12,7,dtype=torch.int64))
    runner._spec_confidence=collector if capture else None; runner.speculator=draft
    runner.main_stream=object(); runner.output_copy_stream=copy_stream
    runner.postprocess_sampled=lambda *args:events.append("postprocess")
    runner.sampler=NS(sampling_states=NS(temperature=NS(gpu=None),seeds=NS(gpu=None)))
    runner.num_speculative_steps=7
    runner.draft_tokens_handler=NS(set_draft_tokens=lambda *args:None)
    runner.kv_connector=NS(post_forward=lambda *args:None)
    out=runner.sample_tokens(None)
    assert events.index("record") < events.index("postprocess") < events.index("propose")
    assert "synchronize" not in events
    result=out.get_output()
    assert result.sampled_token_ids==[[10,11,25]]
    if capture:
        assert result.spec_confidence["r"]["valid"]
        assert result.spec_confidence["r"]["realized_scores"][0][0]==0
        assert collector.previous["proposal_id"]==2
        assert out.confidence_packet is None and out.confidence_cpu is None
    else:
        assert getattr(result,"spec_confidence",None) is None
        assert "confidence_copy" not in events


def trace_row(monkeypatch, *, with_packet=True, terminal=False, sampled=None):
    monkeypatch.setenv("GLM_SPEC_POLICY","shadow")
    for key in ("GLM_SPEC_TRACE","GLM_SPEC_COSTS","GLM_SPEC_HINT_PRIORS","GLM_SPEC_VERIFY_CAP"):
        monkeypatch.delenv(key,raising=False)
    policy=P.VerificationPolicy(config())
    emitted=[]
    policy.sink=NS(emit=emitted.append,dropped=0,error=None)
    policy.costs={1:100,3:120,5:140,7:160}; policy.cost_context=(0,180224)
    request=Request(request_id="r",max_tokens=22 if terminal else 1024)
    sequence=iter([100,200])
    monkeypatch.setattr(P.time,"monotonic_ns",lambda:next(sequence))
    output=schedule(policy,request,k=3)
    runner=NS(req_id_to_index={"r":0},sampled_token_ids=[sampled if sampled is not None else [10,11,25]],
        spec_confidence=C.finish_packet(packet(3)) if with_packet else None)
    policy.complete(output,runner,{"r":request})
    return emitted[0],policy.states[request]["stats"]


def test_actual_policy_join_records_local_receipt_and_leaves_learning_unchanged(monkeypatch):
    row,stats=trace_row(monkeypatch)
    assert row["decision_ns"]==100 and row["receipt_ns"]==200
    assert row["confidence"]["learnable"]
    assert row["request"]!="r" and "request" not in row["confidence"]
    plain,plain_stats=trace_row(monkeypatch,with_packet=False)
    assert stats==plain_stats and row["cap"]==plain["cap"]
    assert "confidence" not in plain


def test_terminal_gate_remains_authoritative_for_confidence(monkeypatch):
    row,stats=trace_row(monkeypatch,terminal=True)
    assert not row["learned"] and not row["confidence"]["learnable"]
    assert stats.observations==0


def test_accepted_prefix_mismatch_invalidates_diagnostic_without_changing_learning(monkeypatch):
    row,stats=trace_row(monkeypatch,sampled=[999,11,25])
    assert row["learned"] and stats.observations==1
    assert not row["confidence"]["valid"] and not row["confidence"]["learnable"]
    assert row["confidence"]["invalid_reason"]=="acceptance_join_mismatch"
    assert "realized_scores" not in row["confidence"]


def test_pinned_base_speculator_allocates_int64_draft_tokens():
    path=ROOT/"tests/fixtures/spec_confidence/base_speculator.py"
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="DraftModelSpeculator")
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
    statement=next(n for n in init.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Attribute) and t.attr=="draft_tokens" for t in n.targets))
    state=NS(max_num_reqs=12,num_speculative_steps=7)
    exec(compile(ast.Module(body=[statement],type_ignores=[]),str(path),"exec"),
         {"self":state,"torch":torch,"device":"cpu"})
    assert state.draft_tokens.shape==(12,7) and state.draft_tokens.dtype==torch.int64


def test_converter_preserves_receipt_causality_and_prompt_repeat_group(monkeypatch):
    row,_=trace_row(monkeypatch)
    joined,summary=CONVERT.convert([row],runtime_id="repair-r1",run_id="boot1",case_map={"":"code_a"})
    assert summary["converted"]==1
    value=joined[0]
    SCREEN.validate_records(joined)
    assert value["feature_available_ns"]==value["feedback_available_ns"]==200
    assert SCREEN.select_feature(value,joined) is None
    assert SCREEN.select_feature(value,joined,causal=False)["age"]==0
    assert value["case"]=="code_a" and value["runtime_id"]=="repair-r1"


@pytest.mark.parametrize("fault",["decision","receipt","anchor","tokens","case","duplicate"])
def test_converter_rejects_fabricated_availability_or_wrong_join(monkeypatch,fault):
    row,_=trace_row(monkeypatch)
    cases={"":"code"}
    if fault=="decision": row.pop("decision_ns")
    if fault=="receipt": row["receipt_ns"]=99
    if fault=="anchor": row["confidence"]["anchor"]+=1
    if fault=="tokens": row["confidence"]["draft_tokens"][0]+=1
    if fault=="case": cases={}
    with pytest.raises(ValueError):
        CONVERT.convert([row,row] if fault=="duplicate" else [row],
                        runtime_id="repaired",run_id="boot",case_map=cases)


def test_converter_counts_dropped_invalid_terminal_and_uncalibrated_rows(monkeypatch):
    row,_=trace_row(monkeypatch)
    variants=[copy.deepcopy(row) for _ in range(4)]
    variants[0].pop("confidence")
    variants[1]["confidence"]["valid"]=False
    variants[2]["learned"]=False
    variants[3]["costs_ms"]={}
    values,summary=CONVERT.convert(variants,runtime_id="repaired",run_id="boot",case_map={"":"code"})
    assert not values and summary["excluded"]=={
        "no_packet":1,"invalid_packet":1,"not_learnable":1,"missing_measured_costs":1}


def test_formal_output_field_and_unmodified_baseline_are_portable():
    baseline=ast.parse((BASELINE/"v1/outputs.py").read_text())
    overlay=ast.parse((OVERLAY/"v1/outputs.py").read_text())
    def fields(tree):
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="ModelRunnerOutput")
        return {n.target.id for n in cls.body if isinstance(n,ast.AnnAssign)}
    assert fields(overlay)-fields(baseline)=={"spec_confidence"}
    assert (BASELINE/"v1/worker/gpu/async_utils.py").read_bytes()==(
        ROOT/"tests/fixtures/spec_confidence/async_utils.py").read_bytes()
