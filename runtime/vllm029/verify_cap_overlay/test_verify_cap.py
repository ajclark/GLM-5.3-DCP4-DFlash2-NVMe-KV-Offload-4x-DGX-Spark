#!/usr/bin/env python3
"""CPU unit tests for the verify-cap controller (vLLM modules stubbed).

    python3 test_verify_cap.py      (needs torch)
"""
import collections
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


class FakeTP:
    rank_in_group, world_size, ranks, cpu_group = 0, 1, [0], None


def load_module(batch_max=4):
    os.environ["VLLM_VERIFY_CAP_BATCH_MAX"] = str(batch_max)
    vllm = types.ModuleType("vllm")
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: types.SimpleNamespace(info=lambda *a, **k: None,
                                                                warning=lambda *a, **k: None)
    ps = types.ModuleType("vllm.distributed.parallel_state")
    ps.get_tp_group = lambda: FakeTP()
    sys.modules.update({"vllm": vllm, "vllm.logger": logger_mod, "vllm.distributed": types.ModuleType("vllm.distributed"),
                        "vllm.distributed.parallel_state": ps})
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "verify_cap", HERE / "vllm/v1/worker/gpu/spec_decode/verify_cap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C1 = [{"context": 1000, "cycle_ms": {"1": 100, "3": 117, "5": 131, "7": 145}},
      {"context": 100000, "cycle_ms": {"1": 104, "3": 121, "5": 135, "7": 149}}]
C2 = [{"context": 1000, "cycle_ms": {"1": 115, "3": 150, "5": 180, "7": 205}}]


def make(mod, fixed=None, batch_costs=True, control=None, policy=None):
    d = tempfile.mkdtemp()
    cal = {"bins": 4, "binning": "neglog10_1mp_over4",
           "q": [[0.2, 0.5, 0.8, 0.97]] * 7}
    costs = {"points": C1}
    if batch_costs:
        costs["batch"] = {"2": {"points": C2}}
    Path(d, "cal.json").write_text(json.dumps(cal))
    Path(d, "costs.json").write_text(json.dumps(costs))
    os.environ.update(VLLM_VERIFY_CAP="1", VLLM_VERIFY_CAP_CAL=f"{d}/cal.json",
                      VLLM_VERIFY_CAP_COSTS=f"{d}/costs.json")
    for k in ("VLLM_VERIFY_CAP_FIXED", "VLLM_VERIFY_CAP_CONTROL", "VLLM_VERIFY_CAP_POLICY",
              "VLLM_VERIFY_CAP_LOCAL_DECIDE", "VLLM_VERIFY_CAP_EARLY", "VLLM_VERIFY_CAP_SYNC_EVERY",
              "VLLM_VERIFY_CAP_GAP_EVENTS"):
        os.environ.pop(k, None)
    if fixed:
        os.environ["VLLM_VERIFY_CAP_FIXED"] = str(fixed)
    if policy:
        os.environ["VLLM_VERIFY_CAP_POLICY"] = policy
    if control is not None:
        Path(d, "control.json").write_text(json.dumps(control))
        os.environ["VLLM_VERIFY_CAP_CONTROL"] = f"{d}/control.json"
    cfg = types.SimpleNamespace(
        speculative_config=types.SimpleNamespace(method="dflash"), num_speculative_tokens=7,
        scheduler_config=types.SimpleNamespace(max_num_seqs=6))
    vc = mod.VerifyCap(cfg, torch.device("cpu"))
    vc.tmpdir = d
    return vc


def sched_out(rids=("r",), ntok=8, nspec=7, structured=False):
    so = types.SimpleNamespace(num_scheduled_tokens={}, scheduled_spec_decode_tokens={},
                               total_num_scheduled_tokens=0, has_structured_output_requests=structured)
    for rid in rids:
        so.num_scheduled_tokens[rid] = ntok
        so.scheduled_spec_decode_tokens[rid] = [-1] * nspec
        so.total_num_scheduled_tokens += ntok
    return so


RIDS = ("r", "s", "t", "u", "v", "w")


def runner(ctx=5000):
    rs = types.SimpleNamespace(req_id_to_index={rid: i for i, rid in enumerate(RIDS)},
                               num_computed_tokens_np=np.array([ctx] * 6))
    return types.SimpleNamespace(req_states=rs)


def primed(vc, p, rids=("r",)):
    vc.step = 5
    for rid in rids:
        slot = RIDS.index(rid)
        vc.p_cpu[slot] = torch.tensor(p)
        vc.slot_step[slot] = 5
        vc.eligible[rid] = True
    vc.event = types.SimpleNamespace(synchronize=lambda: None)


def choose1(vc, p, ctx=5000, bias=0.0):
    vc.lam.clear()
    return vc._choose([vc._expected(p, bias)], vc.costs.at(ctx), 1)


def main():
    mod = load_module()
    assert mod.graph_shapes() == tuple((n, q) for n in range(1, 5) for q in (2, 4, 6))
    vc = make(mod)
    # cost interpolation, per batch size
    c = vc.costs.at(50500)
    assert abs(c[7] - 147) < 1e-6 and vc.costs.at(10)[1] == 100 and vc.costs.at(10**6)[7] == 149
    assert vc.costs.at(5000, 2)[7] == 205 and vc.costs.at(5000, 3) is None
    # confident draft -> verify all 7; unconfident -> 1 (both policies)
    for pol in ("lambda", "ratio"):
        vc.ctl.policy = pol
        assert choose1(vc, [0.99999] * 7) == 7, pol
        assert choose1(vc, [0.3] * 7) == 1, pol
        assert choose1(vc, [0.99999, 0.99999, 0.3, 0.3, 0.3, 0.3, 0.3]) in (1, 3), pol
    vc.ctl.policy = "lambda"
    # lambda: the first decision equals the ratio rule; lambda then tracks the chosen ratio
    vc.lam.clear()
    e = vc._expected([0.9] * 7, 0.0)
    k = vc._choose([e], vc.costs.at(5000), 1)
    ratio_k = max(mod.CAPS, key=lambda kk: (e[kk] / vc.costs.at(5000)[kk], kk))
    assert k == ratio_k and 1 in vc.lam
    # a low lambda (slow history) favours verifying more drafts than the ratio rule
    vc.lam[1] = 0.001
    assert vc._choose([e], vc.costs.at(5000), 1) >= ratio_k
    # trim applies the chosen cap to the scheduler output
    vc.lam.clear()
    primed(vc, [0.3] * 7)
    so = sched_out()
    vc.trim(so, runner())
    assert so.scheduled_spec_decode_tokens["r"] == [-1] and so.num_scheduled_tokens["r"] == 2
    assert so.total_num_scheduled_tokens == 2
    # batch of two: one cap for both, uniform shape
    vb = make(mod)
    primed(vb, [0.3] * 7, rids=("r", "s"))
    so = sched_out(rids=("r", "s"))
    vb.trim(so, runner())
    assert so.num_scheduled_tokens == {"r": 2, "s": 2} and so.total_num_scheduled_tokens == 4
    assert all(len(v) == 1 for v in so.scheduled_spec_decode_tokens.values())
    assert vb.counts[(2, 1)] == 1
    # a batch with no cost table stays at 7; so does a batch when control turns batching off
    primed(vb, [0.3] * 7, rids=("r", "s", "t"))
    so = sched_out(rids=("r", "s", "t"))
    vb.trim(so, runner())
    assert so.total_num_scheduled_tokens == 24
    primed(vb, [0.3] * 7, rids=("r", "s"))
    vb.ctl.batch = False
    so = sched_out(rids=("r", "s"))
    vb.trim(so, runner())
    assert so.total_num_scheduled_tokens == 16
    vb.ctl.batch = True
    # untouched: > batch max, structured output, stale draft, ineligible member, prefill chunk
    for kw, prep in (({"rids": RIDS[:5]}, None), ({"structured": True}, None), ({}, "stale"),
                     ({"rids": ("r", "s")}, "inelig"), ({"ntok": 5, "nspec": 0}, None)):
        rids = kw.get("rids", ("r",))
        primed(vb, [0.3] * 7, rids=rids)
        if prep == "stale":
            vb.slot_step[0] = 4
        if prep == "inelig":
            vb.eligible["s"] = False
        so = sched_out(**kw)
        before = (dict(so.num_scheduled_tokens), so.total_num_scheduled_tokens)
        vb.trim(so, runner())
        assert (dict(so.num_scheduled_tokens), so.total_num_scheduled_tokens) == before, (kw, prep)
    # batch max 1 = single requests only
    mod1 = load_module(batch_max=1)
    assert mod1.graph_shapes() == ((1, 2), (1, 4), (1, 6))
    v1 = make(mod1)
    primed(v1, [0.3] * 7, rids=("r", "s"))
    so = sched_out(rids=("r", "s"))
    v1.trim(so, runner())
    assert so.total_num_scheduled_tokens == 16
    mod = load_module()
    # fixed cap overrides the policy, for single requests and batches
    vf = make(mod, fixed=3)
    for rids in (("r",), ("r", "s")):
        primed(vf, [0.99999] * 7, rids=rids)
        so = sched_out(rids=rids)
        vf.trim(so, runner())
        assert all(v == 4 for v in so.num_scheduled_tokens.values()), rids
        assert all(len(v) == 3 for v in so.scheduled_spec_decode_tokens.values())
    # mode off -> 7 everywhere
    vo = make(mod, control={"mode": "off"})
    primed(vo, [0.3] * 7)
    so = sched_out()
    vo.trim(so, runner())
    assert so.total_num_scheduled_tokens == 8
    # runtime control: fixed mode, then back to auto with a new cost table
    vr = make(mod, batch_costs=False, control={"mode": "fixed", "fixed_k": 5})
    assert vr.ctl.mode == "fixed" and vr.ctl.fixed_k == 5
    primed(vr, [0.3] * 7, rids=("r", "s"))
    so = sched_out(rids=("r", "s"))
    vr.trim(so, runner())
    assert so.total_num_scheduled_tokens == 12        # fixed applies to batches without a table
    Path(vr.tmpdir, "new-costs.json").write_text(json.dumps({"points": C1, "batch": {"2": {"points": C2}}}))
    time.sleep(0.02)
    Path(vr.tmpdir, "control.json").write_text(json.dumps({"mode": "auto", "costs": "new-costs.json"}))
    vr.ctl._next_check = 0.0
    primed(vr, [0.3] * 7, rids=("r", "s"))
    so = sched_out(rids=("r", "s"))
    vr.trim(so, runner())
    assert vr.ctl.mode == "auto" and vr.costs.at(5000, 2) is not None
    assert so.total_num_scheduled_tokens == 4
    # a bad control edit is ignored
    Path(vr.tmpdir, "control.json").write_text(json.dumps({"mode": "bogus"}))
    vr.ctl._next_check = 0.0
    vr._poll_control()
    assert vr.ctl.mode == "auto"
    # live periods: only between consecutive steps, written to periods.json
    vr.prev_decision = None
    vr.step = 10
    vr._note_period(100.0, 2, 1, 5000)
    vr.step = 11
    vr._note_period(100.120, 2, 1, 5000)
    vr.step = 13
    vr._note_period(100.240, 2, 1, 5000)                 # a step ran in between: skipped
    assert list(vr.periods[(2, 1, 0)]) == [120.00000000000455] or abs(vr.periods[(2, 1, 0)][0] - 120) < 1e-6
    assert len(vr.periods[(2, 1, 0)]) == 1
    vr._dump_periods()
    per = json.loads(Path(vr.tmpdir, "periods.json").read_text())
    assert per["periods"]["n2_k1_ctx0"]["count"] == 1
    # eligibility
    sp = types.SimpleNamespace(temperature=0, structured_outputs=None, logit_bias=None, allowed_token_ids=None,
                               bad_words=None, presence_penalty=0, frequency_penalty=0, repetition_penalty=1)
    vc.on_new_request("a", sp)
    vc.on_new_request("b", types.SimpleNamespace(**{**sp.__dict__, "temperature": 0.7}))
    assert vc.eligible["a"] and not vc.eligible["b"]
    vc.on_finished({"a", "b"})
    assert "a" not in vc.eligible
    # record(): scatter p_sel by slot and mark the step
    scores = torch.full((4, 7, 16), -10.0)
    scores[:, :, 0] = 10.0
    spec = types.SimpleNamespace(_selector_scores=scores)
    ib = types.SimpleNamespace(num_reqs=1, idx_mapping=torch.tensor([3]), idx_mapping_np=np.array([3]))
    vc.event = None
    torch.cuda.Event = lambda: types.SimpleNamespace(record=lambda: None, synchronize=lambda: None)
    vc.p_gpu = vc.p_gpu.cpu()
    vc.p_cpu = torch.zeros_like(vc.p_gpu)
    s0 = vc.step
    vc.record(spec, ib)
    assert vc.step == s0 + 1 and vc.slot_step[3] == vc.step and float(vc.p_cpu[3, 0]) > 0.999
    # online correction: always-accepted drafts raise the request bias and lengthen K;
    # early rejections lower it
    vl = make(mod)
    st = {"bias": 0.0, "p": [0.9] * 7, "k": 3}
    for _ in range(30):
        vl._learn(st, 4)                    # all 3 verified drafts accepted
    assert st["bias"] > 0.5, st
    p_mid = [0.99] * 7
    k_before = choose1(vl, p_mid, 5000, 0.0)
    k_after = choose1(vl, p_mid, 5000, vl.bias_global + st["bias"])
    assert k_after >= k_before, (k_before, k_after)
    st2 = {"bias": 0.0, "p": [0.99] * 7, "k": 7}
    for _ in range(30):
        vl._learn(st2, 1)                   # first draft rejected every time
    assert st2["bias"] < -0.5, st2
    # learning happens across consecutive trims of the same request, also inside batches
    for rids in (("r",), ("r", "s")):
        vt = make(mod)
        primed(vt, [0.99] * 7, rids=rids)
        vt.trim(sched_out(rids=rids), runner())
        vt.acc_cpu[0] = 1                   # sampled 1 => the first verified draft was rejected
        vt.step += 1
        for rid in rids:
            vt.slot_step[RIDS.index(rid)] = vt.step
        b0 = vt.req["r"]["bias"]
        vt.trim(sched_out(rids=rids), runner())
        assert vt.req["r"]["bias"] < b0, rids
    # speculative preparation: pre_trim applies the predicted cap, resolve re-applies
    # the decided cap only when it differs
    vs = make(mod, policy="ratio")
    primed(vs, [0.3] * 7)
    vs.hist["r"] = collections.deque([3, 3, 1], maxlen=8)
    so = sched_out()
    vs.pre_trim(so, runner())
    assert so.num_scheduled_tokens["r"] == 4 and so.total_num_scheduled_tokens == 4     # predicted K=3
    redo = vs.resolve(so, runner())                     # low confidence -> K=1
    assert redo and so.num_scheduled_tokens["r"] == 2 and so.scheduled_spec_decode_tokens["r"] == [-1]
    assert vs.mispredicts == 1
    primed(vs, [0.3] * 7)
    vs.hist["r"] = collections.deque([1, 1], maxlen=8)
    so = sched_out()
    vs.pre_trim(so, runner())
    assert not vs.resolve(so, runner()) and so.num_scheduled_tokens["r"] == 2   # matched, no redo
    # K=7 decided after a short prediction restores the full draft list
    primed(vs, [0.99999] * 7)
    vs.hist["r"] = collections.deque([3], maxlen=8)
    so = sched_out()
    vs.pre_trim(so, runner())
    assert vs.resolve(so, runner()) and so.num_scheduled_tokens["r"] == 8 and len(so.scheduled_spec_decode_tokens["r"]) == 7
    # batch prediction comes from the per-batch-size history
    primed(vs, [0.3] * 7, rids=("r", "s"))
    vs.hist_n[2] = collections.deque([1, 1, 3], maxlen=8)
    so = sched_out(rids=("r", "s"))
    vs.pre_trim(so, runner())
    assert so.total_num_scheduled_tokens == 4
    assert not vs.resolve(so, runner())
    print("verify_cap tests passed")


def test_verify_cap():
    """pytest entry point (python3 -m pytest test_verify_cap.py -q)."""
    main()


if __name__ == "__main__":
    main()
