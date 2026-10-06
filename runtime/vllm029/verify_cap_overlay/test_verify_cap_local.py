#!/usr/bin/env python3
"""CPU tests for the step-4 verify-cap changes: rank-local decision (with rank 0's p_sel
shared on the device), early decision, control sync points, the rank check and its
fallback, the GPU gap timer and the extended log line.

    python3 -m pytest test_verify_cap_local.py -q      (needs torch with gloo)

The multi-rank test runs 4 real processes over gloo; the "device" broadcast is a gloo
broadcast of the same CPU tensor, standing in for the TP PyNccl communicator.
"""
import importlib.util
import json
import os
import socket
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parent
RIDS = ("r", "s", "t", "u", "v", "w")
STEP_ENV = ("VLLM_VERIFY_CAP_LOCAL_DECIDE", "VLLM_VERIFY_CAP_EARLY", "VLLM_VERIFY_CAP_SYNC_EVERY",
            "VLLM_VERIFY_CAP_GAP_EVENTS", "VLLM_VERIFY_CAP_FIXED", "VLLM_VERIFY_CAP_CONTROL",
            "VLLM_VERIFY_CAP_POLICY")
C1 = [{"context": 1000, "cycle_ms": {"1": 100, "3": 117, "5": 131, "7": 145}},
      {"context": 100000, "cycle_ms": {"1": 104, "3": 121, "5": 135, "7": 149}}]
C2 = [{"context": 1000, "cycle_ms": {"1": 115, "3": 150, "5": 180, "7": 205}}]
C3 = [{"context": 1000, "cycle_ms": {"1": 130, "3": 175, "5": 215, "7": 250}}]
LOGS: list = []


class FakeEvent:
    """torch.cuda.Event stand-in; `ready` is what query() reports."""
    ready = True

    def __init__(self, *a, **k):
        pass

    def record(self):
        pass

    def synchronize(self):
        pass

    def query(self):
        return FakeEvent.ready


class FakeTP:
    def __init__(self, rank=0, world=1, group=None, comm=None):
        self.rank_in_group, self.world_size, self.ranks = rank, world, list(range(world))
        self.cpu_group = group
        if comm is not None:
            self.device_communicator = types.SimpleNamespace(pynccl_comm=comm)


def load(tp):
    """verify_cap.py with vllm stubbed and get_tp_group() -> tp."""
    vllm = types.ModuleType("vllm")
    logger_mod = types.ModuleType("vllm.logger")

    def _log(level):
        return lambda msg, *a, **k: LOGS.append((level, msg % a if a else msg))
    logger_mod.init_logger = lambda name: types.SimpleNamespace(info=_log("info"), warning=_log("warning"),
                                                                error=_log("error"))
    ps = types.ModuleType("vllm.distributed.parallel_state")
    ps.get_tp_group = lambda: tp
    sys.modules.update({"vllm": vllm, "vllm.logger": logger_mod, "vllm.distributed": types.ModuleType("vllm.distributed"),
                        "vllm.distributed.parallel_state": ps})
    torch.cuda.Event = FakeEvent
    spec = importlib.util.spec_from_file_location("verify_cap", HERE / "vllm/v1/worker/gpu/spec_decode/verify_cap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_tables(d):
    cal = {"bins": 4, "binning": "neglog10_1mp_over4", "q": [[0.2, 0.5, 0.8, 0.97]] * 7}
    Path(d, "cal.json").write_text(json.dumps(cal))
    Path(d, "costs.json").write_text(json.dumps({"points": C1, "batch": {"2": {"points": C2}}}))
    Path(d, "costs3.json").write_text(json.dumps({"points": C1, "batch": {"2": {"points": C2}, "3": {"points": C3}}}))


def make(tp, d=None, local=False, early=False, sync_every=None, control=None, control_path=None, policy=None):
    """A VerifyCap for tp; control (a dict) is written to d/control.json."""
    d = d or tempfile.mkdtemp()
    if not Path(d, "cal.json").exists():
        write_tables(d)
    for k in STEP_ENV:
        os.environ.pop(k, None)
    os.environ.update(VLLM_VERIFY_CAP="1", VLLM_VERIFY_CAP_CAL=f"{d}/cal.json",
                      VLLM_VERIFY_CAP_COSTS=f"{d}/costs.json", VLLM_VERIFY_CAP_BATCH_MAX="4")
    if local:
        os.environ["VLLM_VERIFY_CAP_LOCAL_DECIDE"] = "1"
    if early:
        os.environ["VLLM_VERIFY_CAP_EARLY"] = "1"
    if sync_every:
        os.environ["VLLM_VERIFY_CAP_SYNC_EVERY"] = str(sync_every)
    if policy:
        os.environ["VLLM_VERIFY_CAP_POLICY"] = policy
    if control is not None:
        Path(d, "control.json").write_text(json.dumps(control))
        control_path = f"{d}/control.json"
    if control_path:
        os.environ["VLLM_VERIFY_CAP_CONTROL"] = control_path
    mod = load(tp)
    cfg = types.SimpleNamespace(speculative_config=types.SimpleNamespace(method="dflash"), num_speculative_tokens=7,
                                scheduler_config=types.SimpleNamespace(max_num_seqs=6))
    try:
        vc = mod.VerifyCap(cfg, torch.device("cpu"))
    finally:
        for k in STEP_ENV:
            os.environ.pop(k, None)
    vc.tmpdir = d
    greedy = types.SimpleNamespace(temperature=0, structured_outputs=None, logit_bias=None, allowed_token_ids=None,
                                   bad_words=None, presence_penalty=0, frequency_penalty=0, repetition_penalty=1)
    for rid in RIDS:
        vc.on_new_request(rid, greedy)
    return mod, vc


def sched_out(rids):
    so = types.SimpleNamespace(num_scheduled_tokens={}, scheduled_spec_decode_tokens={},
                               total_num_scheduled_tokens=0, has_structured_output_requests=False)
    for rid in rids:
        so.num_scheduled_tokens[rid] = 8
        so.scheduled_spec_decode_tokens[rid] = [-1] * 7
        so.total_num_scheduled_tokens += 8
    return so


def runner(ctx=5000):
    rs = types.SimpleNamespace(req_id_to_index={rid: i for i, rid in enumerate(RIDS)},
                               num_computed_tokens_np=np.array([ctx] * len(RIDS)))
    return types.SimpleNamespace(req_states=rs)


def schedule(t):
    """The batch at step t: mostly 1-3 requests, sometimes 4, and every 17th step 5
    (above the batch max: drafted but not capped)."""
    if t % 17 == 16:
        return RIDS[:5]
    return RIDS[:(1, 1, 2, 3, 1, 4, 2, 1)[(t // 6) % 8]]


def inputs(t, rank_noise=0.0, rank=0):
    """Selector scores and sampled counts for step t, identical on every rank except for
    `rank_noise` (the per-rank last-bits drift of the drafter's hidden state)."""
    rng = np.random.default_rng(1000 + t)
    rids = schedule(t)
    n = len(rids)
    s0 = rng.choice([4.905, 4.905, 9.0, 1.0, 6.0], size=(n, 7)) + rng.uniform(-0.03, 0.03, size=(n, 7))
    if rank_noise:
        s0 = s0 + np.random.default_rng(77 + rank * 1000 + t).normal(0.0, rank_noise, size=(n, 7))
    scores = torch.zeros(6, 7, 16)
    scores[:n, :, 0] = torch.tensor(s0, dtype=torch.float32)
    num_sampled = torch.tensor(rng.integers(1, 9, size=n), dtype=torch.int32)
    return rids, scores, num_sampled


def cycle(vc, rids, scores, num_sampled, rn, ready=True):
    """One model_runner cycle as the verify cap sees it: record() after drafting, then the
    next step's pre_trim -> (prepare) -> resolve -> before_launch -> launch -> after_launch.
    Returns the verified cap (7 when the step is not capped)."""
    slots = [RIDS.index(r) for r in rids]
    ib = types.SimpleNamespace(num_reqs=len(rids), idx_mapping=torch.tensor(slots), idx_mapping_np=np.array(slots))
    vc.record(types.SimpleNamespace(_selector_scores=scores), ib, num_sampled)
    FakeEvent.ready = ready
    so = sched_out(rids)
    vc.pre_trim(so, rn)
    redo = vc.resolve(so, rn)
    vc.before_launch()
    vc.after_launch()
    return so.num_scheduled_tokens[rids[0]] - 1, redo


def state(vc):
    return (vc.bias_global, sorted(vc.lam.items()), sorted((r, s["bias"], s["k"]) for r, s in vc.req.items()))


# -- single process ----------------------------------------------------------------------
def test_local_decide_matches_rank0_decide():
    """World 1: local decide (late, early, early when ready) makes the same decisions with
    the same learned state as today's rank-0 path, for both policies."""
    for policy in ("ratio", "lambda"):
        runs = {}
        for name, kw, ready in (("rank0", {}, True), ("local", {"local": True}, True),
                                ("early", {"local": True, "early": True}, True),
                                ("early-mixed", {"local": True, "early": True}, None)):
            d = tempfile.mkdtemp()
            _, vc = make(FakeTP(), d, policy=policy, **kw)
            rn, ks, redos = runner(), [], 0
            for t in range(400):
                rids, scores, ns = inputs(t)
                k, redo = cycle(vc, rids, scores, ns, rn, ready=(t % 3 != 0) if ready is None else ready)
                ks.append(k)
                redos += redo
            runs[name] = (ks, state(vc), redos, vc)
        ref = runs["rank0"]
        assert len(set(ref[0])) >= 3, ref[0][:40]            # the inputs exercise several caps
        for name in ("local", "early", "early-mixed"):
            assert runs[name][0] == ref[0], (policy, name)
            assert runs[name][1] == ref[1], (policy, name)
        assert ref[2] > 0 and runs["local"][2] == ref[2]
        assert runs["early"][2] == 0                          # decided in pre_trim: never prepared twice
        assert runs["early"][3].n_early == runs["early"][3].n_dec
        assert 0 < runs["early-mixed"][2] < ref[2]
        assert runs["local"][3].checks == runs["local"][3].n_dec // 32 and runs["local"][3].check_fail is None


def test_local_decide_needs_pynccl():
    """World 2 without a PyNccl communicator: local decide turns itself off before any
    collective (consistently on every rank) and the rank-0 path runs."""
    _, vc = make(FakeTP(rank=1, world=2), local=True)
    assert not vc.local and any("needs the TP group's PyNccl" in m for _, m in LOGS)


def test_rank0_path_unchanged_world2():
    """LOCAL_DECIDE=0: one gloo broadcast of K per decision, no device broadcast, rank 0
    decides alone (others take its K)."""
    calls = {"gloo": 0, "nccl": 0}

    class Comm:
        disabled = False

        def broadcast(self, t, src):
            calls["nccl"] += 1

    orig = torch.distributed.broadcast

    def fake_bcast(t, src, group):
        calls["gloo"] += 1
        t[0] = 5                                             # "rank 0 said K=5"
    torch.distributed.broadcast = fake_bcast
    try:
        _, vc = make(FakeTP(rank=1, world=2, comm=Comm()))
        rn = runner()
        for t in range(12):
            rids, scores, ns = inputs(t)
            if len(rids) <= 4:
                assert cycle(vc, rids, scores, ns, rn)[0] == 5
    finally:
        torch.distributed.broadcast = orig
    assert calls["nccl"] == 0 and calls["gloo"] == sum(len(schedule(t)) <= 4 for t in range(12))
    assert vc.req == {} and vc.n_dec == 0 and not vc.local


CONFIDENT = torch.zeros(6, 7, 16)
CONFIDENT[:, :, 0] = 12.0                                    # p_sel ~ 0.9999: auto verifies all 7


def test_control_applies_at_sync_points():
    """Local decide: a control change waits for the next sync point and applies from the
    following decision; cost tables travel with it and reset lambda; a bad edit is ignored."""
    d = tempfile.mkdtemp()
    _, vc = make(FakeTP(), d, local=True, sync_every=4, control={"mode": "auto", "policy": "ratio"})
    assert vc.ctl.policy == "ratio" and vc.ctl_seq == 1
    rn = runner()
    for t in range(2):                                        # decisions 1, 2
        cycle(vc, ("r",), CONFIDENT, torch.tensor([3], dtype=torch.int32), rn)
    Path(d, "control.json").write_text(json.dumps({"mode": "fixed", "fixed_k": 3}))
    os.utime(Path(d, "control.json"), (1e9, 1e9))
    vc.ctl_src._next_check = 0.0
    ks = []
    for t in range(4):                                        # decisions 3..6
        ks.append(cycle(vc, ("r",), CONFIDENT, torch.tensor([3], dtype=torch.int32), rn)[0])
    assert ks[0] == 7 and ks[1] == 7                          # 3, 4: before the sync after decision 4
    assert ks[2] == 3 and ks[3] == 3                          # 5, 6: fixed K=3
    assert vc.ctl.mode == "fixed" and vc.ctl_seq == 2
    # costs + lambda reset
    vc.lam[2] = 1.0
    Path(d, "control.json").write_text(json.dumps({"mode": "auto", "costs": "costs3.json"}))
    os.utime(Path(d, "control.json"), (2e9, 2e9))
    vc.ctl_src._next_check = 0.0
    for t in range(2):                                        # decisions 7, 8 -> sync
        cycle(vc, ("r",), CONFIDENT, torch.tensor([3], dtype=torch.int32), rn)
    assert vc.ctl.mode == "auto" and vc.costs.at(5000, 3) is not None and 2 not in vc.lam
    # a bad edit (and a missing cost file) changes nothing
    Path(d, "control.json").write_text(json.dumps({"mode": "bogus"}))
    os.utime(Path(d, "control.json"), (3e9, 3e9))
    vc.ctl_src._next_check = 0.0
    for t in range(4):
        cycle(vc, ("r",), CONFIDENT, torch.tensor([3], dtype=torch.int32), rn)
    assert vc.ctl.mode == "auto" and vc.ctl_seq == 3
    Path(d, "control.json").write_text(json.dumps({"mode": "auto", "costs": "missing.json"}))
    os.utime(Path(d, "control.json"), (4e9, 4e9))
    vc.ctl_src._next_check = 0.0
    for t in range(4):
        cycle(vc, ("r",), CONFIDENT, torch.tensor([3], dtype=torch.int32), rn)
    assert vc.ctl_seq == 4 and vc.costs.at(5000, 3) is not None


def test_one_ulp_flips_a_bin():
    """Why the p_sel broadcast is needed: values a hair apart across a calibration bin edge
    give different q, so per-rank p_sel drift could give per-rank caps."""
    _, vc = make(FakeTP())
    assert vc._q(0, 0.9 - 1e-7, 0.0) != vc._q(0, 0.9 + 1e-7, 0.0)


def test_gap_timer():
    clock = {"t": 0.0}

    class Ev:
        def __init__(self):
            self.t, self.done = None, False

        def record(self):
            self.t, self.done = clock["t"], False

        def query(self):
            return self.done

        def elapsed_time(self, other):
            return other.t - self.t

    mod = load(FakeTP())
    g = mod.GapTimer(make_event=Ev)
    for step in range(6):
        g.after_launch()
        clock["t"] += 90.0                                    # verify + drafter
        g.record()
        clock["t"] += 2.5                                     # host still preparing
        if step == 3:
            g.record()                                        # an uncapped step in between: dropped
        g.before_launch()
        for evs in g.ev:
            for e in evs:
                e.done = True
    g.after_launch()
    assert g.n == 5 and abs(g.gap_ms / g.n - 2.5) < 1e-9 and abs(g.draft_ms / g.n - 90.0) < 1e-9
    assert "draft end->verify start 2.50 ms" in g.summary()


def test_log_line():
    for local in (False, True):
        LOGS.clear()
        _, vc = make(FakeTP(), local=local, sync_every=2)
        rn = runner()
        for t in range(5):
            rids, scores, ns = inputs(t)
            cycle(vc, rids, scores, ns, rn)
        vc._log_window(1e12)
        line = [m for lvl, m in LOGS if m.startswith("Verify cap: ")][-1]
        assert "per decision: wait" in line and "re-prepared" in line and "pre_trim->launch" in line
        if local:
            assert "decide local" in line and "rank check ok 2/2" in line and "gpu off" in line
        else:
            assert "decide rank0" in line and "rank check off" in line
    vc.check_fail = (64, [1, 2])
    vc._log_window(2e12)
    assert "MISMATCH at decision 64" in LOGS[-1][1]


# -- four ranks over gloo ----------------------------------------------------------------
STEPS, CHANGE_AT, PERTURB_AT, BACK_AT, SYNC = 120, 30, 70, 90, 4


class GlooComm:
    """The TP PyNccl communicator's broadcast, over gloo on CPU tensors."""
    disabled = False

    def __init__(self, group):
        self.group, self.calls = group, 0

    def broadcast(self, t, src):
        self.calls += 1
        dist.broadcast(t, src=src, group=self.group)


def _rank_main(rank, world, port, d):
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    group = dist.new_group(list(range(world)), backend="gloo")
    comm = GlooComm(group)
    tp = FakeTP(rank, world, group, comm)
    ctl = {"control": {"mode": "auto", "policy": "ratio"}} if rank == 0 else {"control_path": f"{d}/r{rank}/none.json"}
    # odd ranks also decide early, and their drafts are "ready" at pre_trim on a schedule
    # of their own: where a rank decides must not change what it decides
    _, vc = make(tp, f"{d}/r{rank}", local=True, early=rank % 2 == 1, sync_every=SYNC, **ctl)
    rn = runner()
    out = {"k": [], "policy0": vc.ctl.policy, "mode_change": None, "fallback": None, "calls_at_fallback": None}
    for t in range(STEPS):
        if rank == 0 and t in (CHANGE_AT, BACK_AT):
            ctl_file = Path(vc.tmpdir, "control.json")
            ctl_file.write_text(json.dumps({"mode": "fixed", "fixed_k": 3} if t == CHANGE_AT else {"mode": "auto"}))
            os.utime(ctl_file, (1e9 + t, 1e9 + t))
            (vc.ctl_src or vc.ctl)._next_check = 0.0
        if rank == 2 and t == PERTURB_AT:
            vc.bias_global += 1e-12                       # last-bit drift in one rank's state
        rids, scores, ns = inputs(t, rank_noise=2e-4, rank=rank)
        local_before = vc.local
        k, _ = cycle(vc, rids, scores, ns, rn, ready=bool((t + rank) % 2))
        out["k"].append(k)
        if out["mode_change"] is None and vc.ctl.mode == "fixed":
            out["mode_change"] = vc.n_dec
        if local_before and not vc.local:
            out["fallback"] = vc.n_dec
            out["calls_at_fallback"] = comm.calls
    out.update(calls=comm.calls, check_fail=vc.check_fail, n_dec=vc.n_dec, checks=vc.checks, drift=vc.drift)
    Path(d, f"rank{rank}.json").write_text(json.dumps(out))
    dist.barrier()
    dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_four_ranks_over_gloo():
    """Ranks whose local p_sel drift apart still decide alike (rank 0's p_sel is shared),
    adopt rank 0's control at the same decision, detect a state mismatch at the next sync
    point, and all fall back to rank-0 decide + broadcast together."""
    world, d = 4, tempfile.mkdtemp()
    for r in range(world):
        Path(d, f"r{r}").mkdir()
        write_tables(f"{d}/r{r}")
    mp.spawn(_rank_main, args=(world, _free_port(), d), nprocs=world, join=True)
    res = [json.loads(Path(d, f"rank{r}.json").read_text()) for r in range(world)]
    ks = [r["k"] for r in res]
    assert all(k == ks[0] for k in ks), [k[:40] for k in ks]
    assert all(r["policy0"] == "ratio" for r in res)          # rank 0's control reached every rank at start
    # control change: adopted at the same sync point (a multiple of SYNC) on every rank,
    # used from the next decision
    mc = {r["mode_change"] for r in res}
    assert len(mc) == 1 and None not in mc and mc.pop() % SYNC == 0
    # the perturbation is caught at the next sync point, everywhere at once
    fb = {r["fallback"] for r in res}
    assert len(fb) == 1 and None not in fb and fb.pop() % SYNC == 0
    assert all(r["check_fail"] is not None for r in res)
    # one device broadcast per drafted step while local (none after the fallback)
    assert all(r["calls"] == r["calls_at_fallback"] for r in res)
    assert len({r["calls"] for r in res}) == 1
    # the drift diagnostic: every rank sees every rank's count; rank 0 never differs from itself,
    # the others (noisy p_sel) do
    drift = res[0]["drift"]
    assert all(r["drift"] == drift for r in res) and drift[0][0] == 0 and drift[0][1] > 0
    assert all(d > 0 for d, _ in drift[1:])
    # decisions before the control change match a single-rank reference fed rank 0's inputs
    _, ref = make(FakeTP(), control={"mode": "auto", "policy": "ratio"})
    rn, ref_k = runner(), []
    for t in range(CHANGE_AT):
        rids, scores, ns = inputs(t, rank_noise=2e-4, rank=0)
        ref_k.append(cycle(ref, rids, scores, ns, rn)[0])
    assert ks[0][:CHANGE_AT] == ref_k
    # and some rank would have decided differently on its own p_sel (the hazard the broadcast removes)
    differ = 0
    for r in range(1, world):
        _, own = make(FakeTP(), control={"mode": "auto", "policy": "ratio"})
        own_k = [cycle(own, *inputs(t, rank_noise=2e-4, rank=r), rn)[0] for t in range(CHANGE_AT)]
        differ += own_k != ref_k
    assert differ >= 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
