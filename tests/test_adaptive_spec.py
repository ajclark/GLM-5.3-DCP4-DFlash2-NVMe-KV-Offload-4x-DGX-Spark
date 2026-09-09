import gc
import json
import math
from types import SimpleNamespace as NS

import pytest

from spec_harness import load_policy

P = load_policy()


class Request:
    def __init__(self, **overrides):
        self.request_id = "test-rid"
        self.sampling_params = NS(temperature=0, extra_args={})
        self.is_prefill_chunk = False
        self.use_structured_output = False
        self.async_tokens_to_discard = 0
        self.num_output_tokens = 20
        self.num_tokens = 100
        self.max_tokens = 1024
        self.finished = False
        self.__dict__.update(overrides)

    def is_finished(self):
        return self.finished


def config():
    return NS(use_v2_model_runner=True, num_speculative_tokens=7,
              speculative_config=NS(method="dflash", rejection_sample_method="standard"),
              parallel_config=NS(pipeline_parallel_size=1, data_parallel_size=1,
                                 tensor_parallel_size=4, decode_context_parallel_size=2),
              scheduler_config=NS(async_scheduling=True),
              model_config=NS(max_model_len=180224))


@pytest.fixture
def policy(monkeypatch):
    for key in ("GLM_SPEC_TRACE", "GLM_SPEC_COSTS", "GLM_SPEC_VERIFY_CAP"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GLM_SPEC_POLICY", "shadow")
    return P.VerificationPolicy(config())


def schedule(policy, request, k=7, step=1, c1=True):
    request.sampling_params.extra_args = {"spec_policy": "fixed", "spec_verify_cap": k}
    policy.begin()
    cap = policy.select(request, c1, step)
    out = NS(scheduled_spec_decode_tokens={request.request_id: [-1]*cap},
             num_scheduled_tokens={request.request_id: cap+1})
    policy.scheduled(out)
    return out


def complete(policy, request, output, accepted, **overrides):
    runner = NS(req_id_to_index={request.request_id: 0},
                sampled_token_ids=[list(range(1000, 1001+accepted))])
    runner.__dict__.update(overrides)
    policy.complete(output, runner, {request.request_id: request})


def test_censored_tail_and_observed_rejection():
    stats = P.PrefixStats()
    stats.observe(3, 3, decay=1)
    assert stats.trials == [1, 1, 1, 0, 0, 0, 0]
    assert stats.successes == [1, 1, 1, 0, 0, 0, 0]
    stats.observe(3, 1, decay=1)
    assert stats.trials == [2, 2, 1, 0, 0, 0, 0]
    assert stats.successes == [2, 1, 1, 0, 0, 0, 0]
    assert stats.expected(3) == 3
    assert stats.expected(7) == 5  # optimistic bound on the unobserved tail


def test_censoring_does_not_bias_the_tail_toward_short_rejections():
    stats = P.PrefixStats()
    # A is 0 or 7 with equal probability. Verify 3 on 15/16 cycles.
    # Counting known tail failures from short rejections but omitting censored
    # successes would incorrectly estimate tail survival as 1/17, not 1/2.
    for _ in range(10):
        for k in [3]*15+[7]:
            stats.observe(k,0,decay=1)
            stats.observe(k,k,decay=1)
    assert stats.expected(7) == 4.5
    assert stats.expected(3) == 2.5


def test_prior_regularizes_unknown_tails_without_inventing_observations():
    stats=P.PrefixStats(prior=(.5,)*7)
    for _ in range(40):
        stats.observe(3,3)
    assert stats.trials[3:]==[0.]*4
    assert stats.successes[3:]==[0.]*4
    assert stats.expected(7)>stats.expected(3)
    assert stats.expected(7)<stats.expected(3)+1


def test_prior_limits_rechecking_but_recovers_for_predictable_output():
    stats=P.PrefixStats(prior=(.75,2/3,.5,.1,.5,.5,.5))
    costs={1:95,3:115,5:130,7:145}
    chosen=[]
    for step in range(1,321):
        cap,_=stats.choose(costs,step)
        chosen.append(cap)
        stats.observe(cap,min(cap,(step-1)%4))
    assert chosen[80:].count(7)/len(chosen[80:])<=.12
    for step in range(321,449):
        cap,_=stats.choose(costs,step)
        stats.observe(cap,cap)
    assert stats.choose(costs,449)[0]==7


def test_request_prior_requires_valid_cost_context(policy):
    request=Request()
    x=dict(spec_policy='adaptive',spec_cycle_ms=[95,115,130,145],
           spec_cost_lane=policy.lane,spec_cost_context=[0,512],
           spec_acceptance_prior=[.5]*7)
    request.sampling_params.extra_args=x
    policy.select(request,True,1)
    assert policy.states[request]['stats'].prior==(.5,)*7
    x['spec_acceptance_prior']=[True]*7
    policy.select(request,True,2)
    assert policy.states[request]['stats'].prior is None
    x['spec_acceptance_prior']=[.5]*7
    request.num_tokens=513
    assert policy.select(request,True,3)==7
    assert policy.states[request]['stats'].prior is None


def test_trace_writer_blocks_after_flushing_idle_data(tmp_path,monkeypatch):
    import threading
    original=P.queue.Queue
    class ObservedQueue(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.initial_idle=threading.Event()
            self.flushed_idle=threading.Event()
            self.idle_calls=0
        def get(self,block=True,timeout=None):
            if timeout is None:
                self.idle_calls+=1
                (self.initial_idle if self.idle_calls==1 else self.flushed_idle).set()
            return super().get(block,timeout)
    monkeypatch.setattr(P.queue,'Queue',ObservedQueue)
    path=tmp_path/'idle.jsonl'
    sink=P.JsonlSink(str(path))
    try:
        assert sink.queue.initial_idle.wait(1)
        sink.emit({'sample':1})
        assert sink.queue.flushed_idle.wait(2)
        assert json.loads(path.read_text())=={'sample':1}
        assert sink.queue.idle_calls==2
    finally:
        sink.close()


def test_warmup_probe_costs_and_adaptation():
    s = P.PrefixStats()
    costs = {1: 100, 3: 110, 5: 120, 7: 140}
    for _ in range(7):
        s.observe(7, 0)
    assert s.choose(costs, 7) == (7, False)
    s.observe(7, 0)
    assert s.choose(costs, 8) == (1, False)
    assert s.choose(costs, 16) == (7, True)
    assert s.cap == 1  # a probe doesn't erase the chosen cap
    for _ in range(300):
        s.observe(7, 7)
    assert s.choose(costs, 317) == (7, False)
    assert s.choose({}, 318) == (7, False)


def test_hysteresis():
    s = P.PrefixStats()
    for _ in range(10):
        s.observe(7, 0)
    assert s.choose({1: 99, 3: 100, 5: 100, 7: 100}, 11)[0] == 7
    assert s.choose({1: 96, 3: 100, 5: 100, 7: 100}, 12)[0] == 1


@pytest.mark.parametrize("field,value", [
    ("is_prefill_chunk", True), ("use_structured_output", True),
    ("resumable", True), ("lora_request", object()), ("mm_features", [1]),
    ("async_tokens_to_discard", 1),
])
def test_ineligible_requests_fall_back(policy, field, value):
    request = Request(**{field: value})
    assert len(schedule(policy, request, 1).scheduled_spec_decode_tokens[request.request_id]) == 7


@pytest.mark.parametrize("field,value", [
    ("temperature", 0.7), ("presence_penalty", 1), ("frequency_penalty", 1),
    ("repetition_penalty", 1.1), ("allowed_token_ids", [1]),
    ("bad_words", ["a"]), ("logit_bias", {1: 2}),
])
def test_sampling_fallback(policy, field, value):
    request = Request()
    setattr(request.sampling_params, field, value)
    assert len(schedule(policy, request, 1).scheduled_spec_decode_tokens[request.request_id]) == 7


@pytest.mark.parametrize("k", P.CAPS)
def test_every_acceptance_boundary_and_delayed_feedback(policy, k):
    request = Request()
    outputs = [schedule(policy, request, k, step=a+1) for a in range(k+1)]
    assert policy.states[request]["stats"].observations == 0
    for a, output in enumerate(outputs):
        complete(policy, request, output, a)
    assert policy.states[request]["stats"].observations == k+1
    assert not policy.pending


def test_concurrency_fallback_and_return(policy):
    request = Request()
    for c1, expected in ((True, 1), (False, 7), (True, 1)):
        out = schedule(policy, request, 1, c1=c1)
        assert len(out.scheduled_spec_decode_tokens[request.request_id]) == expected
        complete(policy, request, out, 0)
    assert policy.states[request]["stats"].observations == 2


@pytest.mark.parametrize("kind", ["abort", "discard", "terminal", "eos", "stop", "kv_failure", "id_reuse"])
def test_stale_or_terminal_feedback_never_trains(policy, kind):
    request = Request()
    out = schedule(policy, request, 3)
    runner_args = {}
    if kind == "abort":
        request.finished = True
    elif kind == "discard":
        request.async_tokens_to_discard = 1
    elif kind == "terminal":
        request.num_output_tokens = request.max_tokens - 1
    elif kind == "eos":
        request.sampling_params.all_stop_token_ids = {1000}
    elif kind == "stop":
        request.sampling_params.stop = ["END"]
    elif kind == "kv_failure":
        runner_args["kv_connector_output"] = NS(invalid_block_ids={1})
    elif kind == "id_reuse":
        complete(policy, Request(), out, 0)
    complete(policy, request, out, 0, **runner_args)
    assert policy.states[request]["stats"].observations == 0
    assert not policy.pending


def test_finished_requests_do_not_accumulate(policy):
    for i in range(100):
        request = Request(request_id=str(i))
        complete(policy, request, schedule(policy, request, 1), 0)
    del request
    gc.collect()
    assert not policy.states


def test_cost_table_validation(policy, monkeypatch, tmp_path):
    path = tmp_path / "costs.json"
    monkeypatch.setenv("GLM_SPEC_COSTS", str(path))
    table = {"config": {"tp": 4, "dcp": 2, "max_model_len": 180224, "draft_capacity": 7},
             "context_range": [0,4096],
             "cycle_ms": {str(k): 100+k for k in P.CAPS}}
    path.write_text(json.dumps(table))
    assert P.VerificationPolicy(config()).costs[7] == 107
    for mutate in (lambda t: t["config"].update(dcp=4),
                   lambda t: t["cycle_ms"].update({"1": 0}),
                   lambda t: t["cycle_ms"].update({"1": float("nan")})):
        bad = json.loads(json.dumps(table))
        mutate(bad)
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            P.VerificationPolicy(config())


def test_bounded_json_writer(tmp_path):
    path = tmp_path / "trace.jsonl"
    sink = P.JsonlSink(str(path))
    for i in range(200):
        sink.emit({"n": i})
    sink.close()
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    assert [r["n"] for r in rows[:-1]] == list(range(200))
    assert rows[-1] == {"event": "writer_close", "dropped": 0}
    sink.emit({})
    assert sink.dropped == 1


def test_request_scoped_measured_costs_preserve_boot_defaults(policy):
    request = Request()
    state = {"stats":P.PrefixStats(),"scheduled":10}
    policy.states[request] = state
    for _ in range(10):
        state['stats'].observe(7,0)
    request.sampling_params.extra_args = dict(spec_policy='adaptive',
        spec_cost_lane=policy.lane,spec_cycle_ms=[100,110,120,140],spec_cost_context=[0,4096])
    assert policy.select(request,True,1) == 1
    assert policy.costs == {}
    assert policy.chosen[request.request_id]['costs_ms'][1] == 100
    request.sampling_params.extra_args['spec_cost_lane'] = 'wrong-lane'
    assert policy.select(request,True,2) == 7
    request.sampling_params.extra_args.update(spec_cost_lane=policy.lane,spec_cycle_ms=[0,1,2,3])
    assert policy.select(request,True,3) == 7
    request.sampling_params.extra_args['spec_cycle_ms'] = [100,110,120,140]
    request.num_tokens = 4100
    assert policy.select(request,True,4) == 7
