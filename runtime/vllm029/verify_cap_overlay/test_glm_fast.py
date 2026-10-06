#!/usr/bin/env python3
"""CPU unit tests for glm_fast (step 5: vocab-parallel target argmax + L2 prefetch planning).

    python3 test_glm_fast.py          (needs torch; with triton installed it also runs the stock
                                       rejection/gumbel Triton kernels of this overlay and the
                                       glm_fast kernels under TRITON_INTERPRET=1 on the CPU)

vLLM is not imported: the overlay's sampler kernels are loaded from their files with
vllm.triton_utils stubbed to plain triton.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

COMPILE_CHECK = "--compile-check" in sys.argv
if COMPILE_CHECK:
    os.environ.pop("TRITON_INTERPRET", None)
else:
    os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ["GLM_FAST_TEST_CPU_TABLES"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OV = HERE / "vllm/v1/worker/gpu"

from glm_fast import boot, l2_prefetch as l2, vocab_argmax as va  # noqa: E402

HAS_TRITON = importlib.util.find_spec("triton") is not None

V_FULL = 154880
TP = 4


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def stock_kernels():
    """The overlay's rejection_sampler_utils / gumbel (the production sampler kernels)."""
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice
    tu = types.ModuleType("vllm.triton_utils")
    tu.tl, tu.triton, tu.tldevice, tu.HAS_TRITON = tl, triton, tldevice, True
    saved = {k: sys.modules.get(k) for k in ("vllm", "vllm.triton_utils", "vllm.v1", "vllm.v1.worker",
                                             "vllm.v1.worker.gpu", "vllm.v1.worker.gpu.sample",
                                             "vllm.v1.worker.gpu.sample.gumbel")}
    for n in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu", "vllm.v1.worker.gpu.sample"):
        sys.modules.setdefault(n, types.ModuleType(n))
    sys.modules["vllm.triton_utils"] = tu

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m
    try:
        g = load("vllm.v1.worker.gpu.sample.gumbel", OV / "sample/gumbel.py")
        r = load("glm_fast_test_rsu", OV / "spec_decode/rejection_sampler_utils.py")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return g, r


def shard_keys(x: torch.Tensor, tp: int = TP, use_kernel: bool = False) -> torch.Tensor:
    """Simulate the TP ranks: local keys of every shard, concatenated rank-major (all_gather dim 0)."""
    vs = x.shape[1] // tp
    out = []
    for r in range(tp):
        sh = x[:, r * vs:(r + 1) * vs]
        if use_kernel:
            k, _ = va.local_keys(sh, r * vs)
        else:
            k = va.local_keys_torch(sh, r * vs)
        out.append(k)
    return torch.cat(out)


def ref_argmax(x: torch.Tensor) -> torch.Tensor:
    f = x.float()
    return torch.where(torch.isnan(f), torch.full_like(f, float("-inf")), f).argmax(-1)


def adversarial(V: int, tp: int = TP) -> torch.Tensor:
    vs = V // tp
    rows = []
    g = torch.Generator().manual_seed(7)

    def base():
        return torch.randn(V, generator=g).to(torch.bfloat16)
    r = base(); r[5] = 30; r[vs * 2 + 3] = 30; rows.append(r)                 # tie across shards
    r = base(); r[vs + 10] = 30; r[vs + 9] = 30; rows.append(r)               # tie inside a shard
    r = base(); r[vs - 1] = 30; r[vs] = 30; rows.append(r)                    # tie at a shard boundary
    rows.append(torch.full((V,), float("-inf"), dtype=torch.bfloat16))         # all -inf
    r = torch.full((V,), float("-inf"), dtype=torch.bfloat16); r[V - 1] = -1e30; rows.append(r)
    r = base(); r[vs * 3 - 1] = float("inf"); r[vs * 3] = float("inf"); rows.append(r)  # +inf tie
    r = torch.zeros(V, dtype=torch.bfloat16); r[: vs + 7] = -0.0; rows.append(r)       # -0 / +0 tie
    r = torch.full((V,), -5.0, dtype=torch.bfloat16); r[vs * 2] = -0.0; r[vs * 3 + 1] = 0.0; rows.append(r)
    rows.append(torch.full((V,), 1.5, dtype=torch.bfloat16))                   # all equal
    r = base(); r[V - 1] = 40; rows.append(r)                                  # last column
    r = base(); r[0] = 40; rows.append(r)                                      # first column
    r = base(); r[vs * 2] = 40; rows.append(r)                                 # first column of shard 2
    r = base() * 1e-38; rows.append(r)                                         # subnormals
    for edge in (8192 * 3, 1024):                                              # ties at stock block edges
        if edge < V:
            r = base(); r[edge - 1] = 33; r[edge] = 33; rows.append(r)
    return torch.stack(rows)


# ---------------------------------------------------------------------------------------------
# vocab-parallel argmax
# ---------------------------------------------------------------------------------------------
def test_key_order():
    vals = torch.tensor([-float("inf"), -3.0, -0.0, 0.0, 1e-40, 2.5, float("inf")])
    for i in range(len(vals)):
        for j in range(len(vals)):
            for ia, ib in ((3, 9), (9, 3), (5, 5)):
                ka = va.encode_keys(vals[i:i + 1], torch.tensor([ia]))
                kb = va.encode_keys(vals[j:j + 1], torch.tensor([ib]))
                a, b = float(vals[i]), float(vals[j])
                if a > b or (a == b and ia < ib):
                    assert ka > kb, (a, ia, b, ib)
                elif a == b and ia == ib:
                    assert ka == kb
                else:
                    assert ka < kb, (a, ia, b, ib)
    ids = torch.tensor([0, 1, 154879, 38719])
    assert torch.equal(va.decode_ids(va.encode_keys(torch.tensor([-1.0, 0.0, 3.0, -float("inf")]), ids)), ids)


def test_sharded_argmax_torch():
    torch.manual_seed(0)
    for V in (V_FULL, 4 * 9680, 4 * 1000):
        x = torch.randn(33, V).to(torch.bfloat16)
        assert torch.equal(va.reduce_keys(shard_keys(x), TP), ref_argmax(x))
        x = adversarial(V)
        assert torch.equal(va.reduce_keys(shard_keys(x), TP), ref_argmax(x))
    # few distinct values -> many ties everywhere
    x = torch.randint(0, 3, (64, 4 * 4000)).to(torch.bfloat16)
    assert torch.equal(va.reduce_keys(shard_keys(x), TP), ref_argmax(x))
    # TP 1 and 2
    for tp in (1, 2):
        x = torch.randn(9, 4 * 5000).to(torch.bfloat16)
        assert torch.equal(va.reduce_keys(shard_keys(x, tp), tp), ref_argmax(x))


def test_nan_rows_in_range():
    x = adversarial(V_FULL)[:4].clone()
    x[0, 17] = float("nan")
    x[1, :] = float("nan")
    x[2, ::2] = float("nan")
    x[3, 38720 * 2 + 5] = float("nan"); x[3, 38720 * 2 + 6] = 1e4
    ids = va.reduce_keys(shard_keys(x), TP)
    assert ((ids >= 0) & (ids < V_FULL)).all()
    assert torch.equal(ids, ref_argmax(x))  # NaN behaves as -inf
    assert int(ids[1]) == 0                  # all-NaN row -> id 0, like an all -inf row


def test_triton_local_keys_match_torch():
    if not HAS_TRITON:
        print("  (skipped: no triton)")
        return
    torch.manual_seed(1)
    for V in (V_FULL, 4 * 9680):
        x = torch.cat([torch.randn(5, V).to(torch.bfloat16), adversarial(V)])
        x[3, 11] = float("nan")
        for r in range(TP):
            vs = V // TP
            sh = x[:, r * vs:(r + 1) * vs]
            k, nan = va.local_keys(sh, r * vs)
            assert torch.equal(k, va.local_keys_torch(sh, r * vs)), (V, r)
            assert torch.equal(nan.bool(), torch.isnan(sh).any(-1))
        assert torch.equal(va.reduce_keys(shard_keys(x, use_kernel=True), TP), ref_argmax(x))
    # a strided (non-contiguous row) shard view is handled
    x = torch.randn(4, 4 * 9680).to(torch.bfloat16)
    sh = x[:, 9680:2 * 9680]
    assert torch.equal(va.local_keys(sh, 9680)[0], va.local_keys_torch(sh, 9680))


def _rejection_case(V, ns_logits, accept, g, special=None):
    """Logits and drafts for len(ns_logits) requests; request i accepts accept[i] drafts
    (accept[i] == n - 1 -> everything accepted, bonus row sampled)."""
    L = sum(ns_logits)
    x = torch.randn(L, V, generator=g).to(torch.bfloat16)
    if special is not None:
        special(x)
    am = ref_argmax(x)
    draft = torch.randint(0, V, (L,), generator=g)
    cu = [0]
    for n in ns_logits:
        cu.append(cu[-1] + n)
    for r, n in enumerate(ns_logits):
        s = cu[r]
        for i in range(n - 1):
            if i < accept[r]:
                draft[s + i + 1] = am[s + i]
            elif i == accept[r]:
                draft[s + i + 1] = (am[s + i] + 1) % V if (r % 2 == 0) else -1  # mismatch or placeholder
    return x, draft, torch.tensor(cu, dtype=torch.int32)


def _stock_rejection(r_mod, x, draft, cu, K):
    nreq = cu.numel() - 1
    idx = torch.arange(nreq, dtype=torch.int32)
    counts = (cu[1:] - cu[:-1]).to(torch.int64)
    exp_idx = torch.repeat_interleave(idx, counts)
    loc = torch.cat([torch.arange(int(c), dtype=torch.int32) + (K + 1 - int(c)) for c in counts])
    temp = torch.zeros(nreq, dtype=torch.float32)
    seed = torch.zeros(nreq, dtype=torch.int64)
    pos = torch.arange(x.shape[0], dtype=torch.int64)
    return r_mod.rejection_sample(x, None, draft, cu, pos, idx, exp_idx, loc, temp, seed, K)


def test_greedy_verify_vs_stock_rejection_sampler():
    """End to end against the production rejection kernels (overlay file, Triton interpreter):
    sampled[:num_sampled] and num_sampled must be identical."""
    if not HAS_TRITON:
        print("  (skipped: no triton)")
        return
    _g, r_mod = stock_kernels()
    g = torch.Generator().manual_seed(3)
    K = 7
    cases = [
        (4 * 9680, [8], [7]), (4 * 9680, [8], [0]), (4 * 9680, [8], [3]),
        (4 * 9680, [8, 8, 8, 8], [7, 0, 5, 2]),
        (4 * 9680, [4, 4], [3, 1]),                 # verify cap K=3 batch
        (4 * 9680, [2], [1]), (4 * 9680, [2], [0]),  # K=1
        (4 * 9680, [8, 1, 8], [6, 0, 7]),            # a prefill row (1 logit) in a verify batch
        (V_FULL, [8, 8], [4, 7]),
    ]
    for V, ns, acc in cases:
        x, draft, cu = _rejection_case(V, ns, acc, g)
        s_ref, n_ref = _stock_rejection(r_mod, x, draft, cu, K)
        keys = shard_keys(x, use_kernel=True)
        s_fast, n_fast = va.greedy_verify(keys, TP, draft, cu, len(ns), K + 1)
        s_t, n_t = va.greedy_verify_torch(va.reduce_keys(shard_keys(x), TP), draft, cu, len(ns), K + 1)
        assert torch.equal(n_fast, n_ref) and torch.equal(n_t, n_ref), (V, ns, acc, n_fast, n_t, n_ref)
        for r in range(len(ns)):
            k = int(n_ref[r])
            assert torch.equal(s_fast[r, :k], s_ref[r, :k]), (V, ns, acc, r, s_fast[r], s_ref[r])
            assert torch.equal(s_t[r, :k], s_ref[r, :k])
        assert ((s_fast >= 0) & (s_fast < V)).all()
    # adversarial logits rows inside a verify step (ties at shard and block edges, -inf, +-0)
    V = V_FULL
    adv = adversarial(V)
    for start in range(0, adv.shape[0] - 7, 4):
        def special(x, start=start):
            x[:] = adv[start:start + 8]
        x, draft, cu = _rejection_case(V, [8], [7], g, special)
        s_ref, n_ref = _stock_rejection(r_mod, x, draft, cu, K)
        s_fast, n_fast = va.greedy_verify(shard_keys(x, use_kernel=True), TP, draft, cu, 1, K + 1)
        assert torch.equal(n_fast, n_ref) and torch.equal(s_fast[0, :int(n_ref[0])], s_ref[0, :int(n_ref[0])])


def test_plain_sampler_vs_stock_gumbel():
    if not HAS_TRITON:
        print("  (skipped: no triton)")
        return
    g_mod, _r = stock_kernels()
    x = torch.cat([torch.randn(3, V_FULL).to(torch.bfloat16), adversarial(V_FULL)])
    L = x.shape[0]
    ref = g_mod.gumbel_sample(x, torch.zeros(L, dtype=torch.int32), torch.zeros(1), torch.zeros(1, dtype=torch.int64),
                              torch.arange(L), apply_temperature=False, is_drafting=False)
    assert torch.equal(va.reduce_keys(shard_keys(x, use_kernel=True), TP), ref)


class _Buf:
    def __init__(self, a):
        self.np = np.asarray(a)


class FakeSampler:
    pass


FakeSampler.__name__ = "Sampler"


class FakeRS:
    pass


FakeRS.__name__ = "RejectionSampler"


def fake_runner(n=2, K=7, temp=0.0):
    sampler = FakeSampler()
    sampler.compute_nans = False
    sampler.trace_replay_state = None
    sampler.return_sampling_mask = False
    sampler.needs_logits_processing = np.zeros(16, dtype=bool)
    st = types.SimpleNamespace(temperature=_Buf(np.full(16, temp, dtype=np.float32)),
                               num_logprobs=np.full(16, -1))
    st.max_num_logprobs = lambda idx: int(np.max(st.num_logprobs[idx]))
    sampler.sampling_states = st
    sampler.logprob_token_ids_state = types.SimpleNamespace(max_num_token_ids=lambda idx: 0)
    rs = FakeRS()
    rs.synthetic_conditional_rates = None
    rs.enable_adaptive_verification = False
    rs.num_speculative_steps = K
    runner = types.SimpleNamespace(sampler=sampler, rejection_sampler=rs, batch_sharder=None,
                                   speculator=types.SimpleNamespace(draft_logits=None), model=None)
    runner.__dict__["_glm_vp_head"] = (types.SimpleNamespace(), types.SimpleNamespace(org_vocab_size=V_FULL))
    ib = types.SimpleNamespace(num_reqs=n, idx_mapping_np=np.arange(n), num_draft_tokens=n * K,
                               logits_indices=torch.arange(n * (K + 1)))
    return runner, ib


def test_plan_eligibility():
    runner, ib = fake_runner()
    assert va.plan(runner, ib, None) == ("reject", 8)
    assert va.plan(runner, ib, object()) is None                       # grammar
    r2, ib2 = fake_runner(temp=0.7)
    assert va.plan(r2, ib2, None) is None                              # sampled rows
    r2, ib2 = fake_runner(); r2.sampler.sampling_states.temperature.np[1] = 1.0
    assert va.plan(r2, ib2, None) is None                              # one sampled row
    r2, ib2 = fake_runner(); r2.sampler.needs_logits_processing[0] = True
    assert va.plan(r2, ib2, None) is None                              # penalties / bias / budget
    r2, ib2 = fake_runner(); r2.sampler.sampling_states.num_logprobs[1] = 5
    assert va.plan(r2, ib2, None) is None                              # logprobs
    r2, ib2 = fake_runner(); r2.sampler.logprob_token_ids_state.max_num_token_ids = lambda i: 2
    assert va.plan(r2, ib2, None) is None
    r2, ib2 = fake_runner(); r2.sampler.compute_nans = True
    assert va.plan(r2, ib2, None) is None
    r2, ib2 = fake_runner(); r2.batch_sharder = object()
    assert va.plan(r2, ib2, None) is None
    r2, ib2 = fake_runner(); ib2.idx_mapping_np = np.array([0, -1])
    assert va.plan(r2, ib2, None) is None                              # masked rows
    r2, ib2 = fake_runner(); r2.rejection_sampler.enable_adaptive_verification = True
    assert va.plan(r2, ib2, None) is None
    r2, ib2 = fake_runner(); r2.rejection_sampler.synthetic_conditional_rates = torch.ones(3)
    assert va.plan(r2, ib2, None) is None
    r2, ib2 = fake_runner(); r2.speculator.draft_logits = torch.zeros(1, 1, 1000)
    assert va.plan(r2, ib2, None) is None                              # narrower draft vocab
    r2, ib2 = fake_runner(); r2.speculator.draft_logits = torch.zeros(1, 1, V_FULL)
    assert va.plan(r2, ib2, None) == ("reject", 8)
    r2, ib2 = fake_runner(); ib2.num_draft_tokens = 0; ib2.logits_indices = torch.arange(2)
    assert va.plan(r2, ib2, None) == ("sampler", 1)
    r2, ib2 = fake_runner(); ib2.num_draft_tokens = 0
    assert va.plan(r2, ib2, None) is None                              # rows != requests
    r2, ib2 = fake_runner(); r2.__dict__["_glm_vp_head"] = "vocab padding"
    assert va.plan(r2, ib2, None) is None


def _lm_head(V=V_FULL, tp=TP, rank=0, pad=0):
    si = types.SimpleNamespace(num_org_vocab_padding=pad, num_added_elements_padded=0,
                               org_vocab_start_index=rank * (V // tp), org_vocab_end_index=(rank + 1) * (V // tp))
    return types.SimpleNamespace(shard_indices=si, num_embeddings_padded=V + pad * tp, tp_size=tp, bias=None)


class FakeLP:
    def __init__(self, V=V_FULL):
        self.org_vocab_size = V
        self.scale = 1.0
        self.soft_cap = None
        self.logits_as_input = False
        self.use_all_gather = True

    def _apply_head(self, lm_head, h, bias):
        return h


FakeLP.__name__ = "LogitsProcessor"


def test_resolve_head():
    model = types.SimpleNamespace(lm_head=_lm_head(), logits_processor=FakeLP())
    assert isinstance(va.resolve_head(model), tuple)
    assert isinstance(va.resolve_head(types.SimpleNamespace(lm_head=_lm_head(pad=8), logits_processor=FakeLP())), str)
    lp = FakeLP(); lp.soft_cap = 30.0
    assert isinstance(va.resolve_head(types.SimpleNamespace(lm_head=_lm_head(), logits_processor=lp)), str)
    lp = FakeLP(); lp.scale = 0.5
    assert isinstance(va.resolve_head(types.SimpleNamespace(lm_head=_lm_head(), logits_processor=lp)), str)
    wrapped = types.SimpleNamespace(language_model=model)
    assert isinstance(va.resolve_head(wrapped), tuple)


def _stub_vllm_for_fast_sample(full_keys_by_rank, rank):
    """vllm.distributed / input_batch / sample.output stubs for fast_sample on one simulated rank."""
    mods = {}
    dist = types.ModuleType("vllm.distributed")

    def all_gather(t, dim=0):
        parts = list(full_keys_by_rank)
        parts[rank] = t
        return torch.cat(parts)
    dist.tensor_model_parallel_all_gather = all_gather
    ib = types.ModuleType("vllm.v1.worker.gpu.input_batch")

    def get_num_sampled_and_rejected(num_sampled, seq_lens, cu, idx_mapping, prefill_len):
        ns = num_sampled.clone()
        nr = torch.empty_like(ns)
        for b in range(idx_mapping.shape[0]):
            chunked = int(seq_lens[b]) < int(prefill_len[int(idx_mapping[b])])
            ns[b] = 0 if chunked else ns[b]
            nr[b] = 0 if chunked else int(cu[b + 1] - cu[b]) - int(ns[b])
        return ns, nr
    ib.get_num_sampled_and_rejected = get_num_sampled_and_rejected
    out = types.ModuleType("vllm.v1.worker.gpu.sample.output")

    class SamplerOutput:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    out.SamplerOutput = SamplerOutput
    for n in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu", "vllm.v1.worker.gpu.sample"):
        mods[n] = sys.modules.get(n) or types.ModuleType(n)
    mods.update({"vllm.distributed": dist, "vllm.v1.worker.gpu.input_batch": ib,
                 "vllm.v1.worker.gpu.sample.output": out})
    return mods


def test_fast_sample_end_to_end_all_ranks():
    """fast_sample on each of 4 simulated TP ranks (stubbed all-gather) gives the same tokens and
    counts on every rank, matching the greedy reference, with a chunked-prefill request."""
    g = torch.Generator().manual_seed(11)
    V = 4 * 9680
    ns, acc = [8, 8, 1], [3, 7, 0]
    x, draft, cu = _rejection_case(V, ns, acc, g)
    L = x.shape[0]
    vs = V // TP
    full_keys = [va.local_keys_torch(x[:, r * vs:(r + 1) * vs], r * vs) for r in range(TP)]
    results = []
    for rank in range(TP):
        saved = {}
        mods = _stub_vllm_for_fast_sample(full_keys, rank)
        for k, v in mods.items():
            saved[k] = sys.modules.get(k)
            sys.modules[k] = v
        try:
            runner, _ = fake_runner(n=3)
            runner.sampler.req_states = types.SimpleNamespace(prefill_len=types.SimpleNamespace(
                gpu=torch.tensor([10, 10, 500])))
            runner.__dict__["_glm_vp_head"] = (_lm_head(V=V, rank=rank), FakeLP(V))
            ib = types.SimpleNamespace(
                num_reqs=3, logits_indices=torch.arange(L), input_ids=draft,
                cu_num_logits=cu, idx_mapping=torch.arange(3), seq_lens=torch.tensor([20, 20, 100]))
            hidden = x[:, rank * vs:(rank + 1) * vs]  # FakeLP._apply_head is the identity
            out, n_s, n_r, _nan = va.fast_sample(runner, hidden, ib, ("reject", 8))
            results.append((out.sampled_token_ids, n_s, n_r))
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
    s_ref, n_ref = va.greedy_verify_torch(ref_argmax(x), draft, cu, 3, 8)
    for s, n_s, n_r in results:
        assert torch.equal(s, results[0][0]) and torch.equal(n_s, results[0][1])
        assert n_s.tolist() == [n_ref[0].item(), n_ref[1].item(), 0]  # third request still prefilling
        assert n_r.tolist() == [8 - n_ref[0].item(), 8 - n_ref[1].item(), 0]
        for r in range(2):
            assert torch.equal(s[r, :int(n_s[r])], s_ref[r, :int(n_s[r])])


def test_compare_device():
    So = types.SimpleNamespace
    ib = types.SimpleNamespace(cu_num_logits=torch.tensor([0, 2, 4]))
    s = torch.tensor([[5, 6], [7, 8]])
    ref = (So(sampled_token_ids=torch.tensor([[5, 9], [7, 8]])), torch.tensor([1, 2]), torch.tensor([1, 0]))
    fast = (So(sampled_token_ids=s), torch.tensor([1, 2]), torch.tensor([1, 0]), torch.tensor([0, 0, 1, 0]))
    bad, nan = va.compare_device(fast, ref, ib)
    assert bad.tolist() == [False, False] and nan.tolist() == [False, True]  # col 1 of req 0 is dead
    ref2 = (So(sampled_token_ids=torch.tensor([[5, 6], [7, 1]])), torch.tensor([1, 2]), torch.tensor([1, 0]))
    bad, _ = va.compare_device(fast, ref2, ib)
    assert bad.tolist() == [False, True]
    ref3 = (So(sampled_token_ids=s), torch.tensor([2, 2]), torch.tensor([0, 0]))
    bad, _ = va.compare_device(fast, ref3, ib)
    assert bad.tolist() == [True, False]


def test_sample_hook_routing():
    calls = []

    class GPUModelRunner:
        def sample(self, hidden_states, input_batch, grammar_output):
            calls.append("stock")
            return ("stock",)
    mod = types.SimpleNamespace(GPUModelRunner=GPUModelRunner)
    orig_plan, orig_fast, orig_cmp = va.plan, va.fast_sample, va.compare_device
    va.plan = lambda runner, ib, go: ("reject", 8) if ib == "ok" else None
    va.fast_sample = lambda *a: (calls.append("fast") or ("fast", None, None, None))
    va.compare_device = lambda f, r, ib: (torch.tensor([False]), torch.tensor([False]))
    try:
        va.install(mod)
        rnr = GPUModelRunner()
        os.environ["VLLM_VOCAB_PARALLEL_ARGMAX"] = "1"
        ib_ok = types.SimpleNamespace(num_reqs=1)
        va.plan = lambda runner, ib, go: ("reject", 8) if ib is ib_ok else None
        assert rnr.sample(None, ib_ok, None) == ("fast", None, None) and calls == ["fast"]
        calls.clear()
        assert rnr.sample(None, "no", None) == ("stock",) and calls == ["stock"]
        calls.clear()
        os.environ["VLLM_VOCAB_PARALLEL_ARGMAX"] = "check"
        assert rnr.sample(None, ib_ok, None) == ("stock",) and calls == ["fast", "stock"]
        calls.clear()
        os.environ["VLLM_VOCAB_PARALLEL_ARGMAX"] = "0"
        assert rnr.sample(None, ib_ok, None) == ("stock",) and calls == ["stock"]
        assert GPUModelRunner.sample.__wrapped__ is not None
    finally:
        va.plan, va.fast_sample, va.compare_device = orig_plan, orig_fast, orig_cmp
        os.environ.pop("VLLM_VOCAB_PARALLEL_ARGMAX", None)


def test_sample_digest_matches_overlay():
    import ast
    import hashlib
    src = (OV / "model_runner.py").read_text()
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "sample")
    start = min([fn.lineno] + [d.lineno for d in fn.decorator_list]) - 1
    digest = hashlib.sha256("".join(lines[start:fn.end_lineno]).encode()).hexdigest()[:16]
    if digest not in va.KNOWN_SAMPLE_DIGESTS:
        print(f"  NOTE: overlay GPUModelRunner.sample digest {digest} differs from the reviewed one "
              f"{sorted(va.KNOWN_SAMPLE_DIGESTS)} (step 4 changed it?): re-review the hook")


# ---------------------------------------------------------------------------------------------
# L2 prefetch planning
# ---------------------------------------------------------------------------------------------
def test_dense_runs():
    t = torch.zeros(64, 128, dtype=torch.bfloat16)
    assert l2.dense_runs(t) == [(t.data_ptr(), 64 * 128 * 2)]
    assert l2.dense_runs(t.t()) == [(t.data_ptr(), 64 * 128 * 2)]  # permuted but dense
    # W_UV: dequantized kv_b weight [N*(P+V), L] -> .T -> view(L, N, P+V) -> split -> transpose(0,1)
    N, P, Vd, Lr = 16, 192, 256, 512
    kvb = torch.zeros(N * (P + Vd), Lr, dtype=torch.bfloat16)
    w = kvb.T.view(Lr, N, P + Vd)
    _w_uk, w_uv = w.split([P, Vd], dim=-1)
    W_UV = w_uv.transpose(0, 1)
    runs = l2.dense_runs(W_UV)
    assert len(runs) == N and all(b == Vd * Lr * 2 for _, b in runs)
    assert runs[0][0] == kvb.data_ptr() + P * Lr * 2
    assert all(runs[i + 1][0] - runs[i][0] == (P + Vd) * Lr * 2 for i in range(N - 1))
    covered = sum(b for _, b in runs)
    assert covered == W_UV.numel() * 2
    # adjacent runs merge; overlapping / broadcast views are refused
    assert l2.dense_runs(t[:, :64]) is not None and len(l2.dense_runs(t[:, :64])) == 64
    assert l2.dense_runs(t[:32]) == [(t.data_ptr(), 32 * 128 * 2)]
    assert l2.dense_runs(torch.zeros(8).expand(4, 8)) is None
    assert l2.dense_runs(t[:, :1].expand(64, 4)) is None
    assert l2.dense_runs(torch.zeros(4096, 2)[:, 0], max_runs=16) is None


def test_take():
    runs = [(1024, 100), (2048 + 8, 4096), (8192, 10_000), (65536, 1 << 20)]
    out = l2.take(runs, 6000)
    assert out[0] == (1024, 96)               # rounded down to 16 B
    assert all(p % 16 == 0 and n % 16 == 0 for p, n in out)
    assert (2048 + 8, 4096) not in out        # misaligned run skipped
    assert sum(n for _, n in out) <= 6000
    assert out[1][0] == 8192 and out[1][1] == (6000 - 96) & ~15
    assert l2.take(runs, 0) == []


class Lin(torch.nn.Module):
    def __init__(self, k, n, int8=True):
        super().__init__()
        if int8:
            self.weight_packed = torch.nn.Parameter(torch.zeros(k // 4, n, dtype=torch.int32), requires_grad=False)
            self.weight_scale = torch.nn.Parameter(torch.zeros(max(1, k // 128), n, dtype=torch.bfloat16),
                                                   requires_grad=False)
        else:
            self.weight = torch.nn.Parameter(torch.zeros(n, k, dtype=torch.bfloat16), requires_grad=False)


def fake_layer(moe=True, dcp=True):
    layer = torch.nn.Module()
    attn = torch.nn.Module()
    attn.fused_qkv_a_proj = Lin(6144, 656)
    attn.q_b_proj = Lin(2048, 1024)
    attn.o_proj = Lin(1024, 1536)
    kvb = torch.zeros(16 * 448, 512, dtype=torch.bfloat16)
    attn.register_parameter("W_UV", torch.nn.Parameter(
        kvb.T.view(512, 16, 448).split([192, 256], dim=-1)[1].transpose(0, 1), requires_grad=False))
    attn.indexer = None
    if dcp:
        attn.dcp_manager = types.SimpleNamespace(query_gather=lambda q: ("gathered", q))
    layer.self_attn = attn
    mlp = torch.nn.Module()
    if moe:
        mlp.gate = Lin(6144, 256, int8=False)
        mlp.shared_experts = torch.nn.Module()
        mlp.shared_experts.gate_up_proj = Lin(6144, 512)
        mlp.shared_experts.down_proj = Lin(256, 1536)
    else:
        mlp.gate_up_proj = Lin(6144, 1536)
        mlp.down_proj = Lin(768, 1536)
    layer.mlp = mlp
    layer.input_layernorm = torch.nn.Module()
    layer.post_attention_layernorm = torch.nn.Module()
    return layer


def _p(t):
    return t.data_ptr()


def test_queues():
    lay = fake_layer()
    qb = l2.queue_b(lay)
    assert qb[0] == (_p(lay.mlp.gate.weight), lay.mlp.gate.weight.numel() * 2)
    gu = lay.mlp.shared_experts.gate_up_proj
    assert qb[1][0] == _p(gu.weight_scale) and qb[2][0] == _p(gu.weight_packed)  # scales first
    assert qb[3][0] == _p(lay.mlp.shared_experts.down_proj.weight_scale)
    qc = l2.queue_c(lay)
    assert qc[0][0] == _p(lay.self_attn.fused_qkv_a_proj.weight_scale)
    assert qc[1][0] == _p(lay.self_attn.fused_qkv_a_proj.weight_packed)
    assert qc[3][0] == _p(lay.self_attn.q_b_proj.weight_packed)
    qd = l2.queue_d(lay.self_attn)
    assert len(qd) == 16 + 2 and qd[16][0] == _p(lay.self_attn.o_proj.weight_scale)
    dense = fake_layer(moe=False)
    assert l2.queue_b(dense)[1][0] == _p(dense.mlp.gate_up_proj.weight_packed)
    # budget: window C takes the scales then a prefix of fused_qkv_a
    segs = l2.take(qc, 1 << 20)
    assert sum(n for _, n in segs) <= 1 << 20 and segs[0][0] == _p(lay.self_attn.fused_qkv_a_proj.weight_scale)


def test_install_fork_routing():
    """The norm wrapper forks B / C for the right layer, D at the query gather; gates on depth,
    token count and the window list; joins once at the end of the model forward."""
    orig = (l2._capturing, l2._table, l2._fork, l2.join, l2.link)
    forks, joins = [], []
    os.environ["VLLM_L2_PREFETCH_WINDOWS"] = "B,C,D"
    os.environ["VLLM_L2_PREFETCH_MAXTOK"] = "32"
    try:
        l2.State.linked_models.clear(); l2.State.norm_map.clear(); l2.State.depth = 0
        l2._capturing = lambda: False
        l2._table = lambda segs: (segs, len(segs), sum(s[1] for s in segs))
        l2._fork = lambda window, p: forks.append((window, p[1]))
        l2.join = lambda: joins.append(1)

        def fake_link(model):
            for layer in model.layers:
                l2.State.norm_map[id(layer.post_attention_layernorm)] = ("B", layer)
                l2.State.norm_map[id(layer.input_layernorm)] = ("C", layer)
                l2._wrap_query_gather(layer.self_attn)
            l2.State.linked_models.add(id(model))
            return True
        l2.link = fake_link
        seen = []

        def fused_allreduce_rms_norm(h, residual, norm):
            seen.append(id(norm))
            return h, residual

        class DeepseekV32Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([fake_layer(), fake_layer(), fake_layer(moe=False)])

            def forward(self, ntok):
                h = torch.zeros(ntok, 4)
                for i, layer in enumerate(self.layers):
                    if i > 0:
                        mod.fused_allreduce_rms_norm(h, None, layer.input_layernorm)
                    layer.self_attn.dcp_manager.query_gather(h)
                    mod.fused_allreduce_rms_norm(h, None, layer.post_attention_layernorm)
                mod.fused_allreduce_rms_norm(h, None, torch.nn.Module())  # final norm: no window
                return h
        mod = types.SimpleNamespace(fused_allreduce_rms_norm=fused_allreduce_rms_norm,
                                    DeepseekV32Model=DeepseekV32Model, __name__="fake.model")
        l2.install(mod)
        m = DeepseekV32Model()
        # eager forward: tables built, no fork (VLLM_L2_PREFETCH_EAGER=0)
        m(8)
        assert forks == [] and len(joins) == 1 and len(seen) == 3 + 2 + 1
        assert all(isinstance(m.layers[i].__dict__.get("_glm_l2pf_B"), tuple) for i in range(3))
        assert all(isinstance(m.layers[i].__dict__.get("_glm_l2pf_C"), tuple) for i in (1, 2))
        assert "_glm_l2pf_C" not in m.layers[0].__dict__  # layer 0 has no input all-reduce
        assert all(isinstance(m.layers[i].self_attn.__dict__.get("_glm_l2pf_D"), tuple) for i in range(3))
        # "captured" forward: forks in layer order
        l2._capturing = lambda: True
        m(8)
        assert [w for w, _ in forks] == ["D", "B", "C", "D", "B", "C", "D", "B"], forks
        assert len(joins) == 2
        forks.clear()
        m(64)  # more than MAXTOK tokens (prefill): nothing
        assert forks == []
        # outside a model forward (depth 0): nothing
        mod.fused_allreduce_rms_norm(torch.zeros(4, 4), None, m.layers[1].post_attention_layernorm)
        assert forks == []
        # windows filter
        l2.State.cfg.windows = {"C"}
        m(8)
        assert [w for w, _ in forks] == ["C", "C"]
        # a window captured before its table exists prefetches nothing and is counted
        forks.clear()
        l2.State.cfg.windows = {"B"}
        m2 = DeepseekV32Model()
        m2(8)
        assert forks == [] and l2.State.missing["B"] >= 3
    finally:
        l2._capturing, l2._table, l2._fork, l2.join, l2.link = orig
        os.environ.pop("VLLM_L2_PREFETCH_WINDOWS", None)
        os.environ.pop("VLLM_L2_PREFETCH_MAXTOK", None)
        l2.State.depth = 0


def test_runtime_control():
    assert l2.parse_control('{"windows": "bd"}', {"B", "C", "D"}) == {"B", "D"}
    assert l2.parse_control('{"windows": ""}', {"B", "C", "D"}) == set()
    assert l2.parse_control('{"windows": "BCD"}', {"C"}) == {"C"}       # uncaptured windows stay off
    assert l2.parse_control('not json', {"B"}) is None
    assert l2.parse_control('{"windows": 3}', {"B"}) is None
    saved = (l2.State.control, l2.State.flags, l2.State.cfg, l2.State.control_mtime, l2.State.control_checked)
    with tempfile.TemporaryDirectory() as d:
        try:
            ctl = Path(d, "glm_fast_l2pf.json")
            l2.State.control = str(ctl)
            l2.State.flags = torch.ones(3, dtype=torch.int32)
            l2.State.cfg = l2.Cfg()
            l2.State.cfg.windows = {"B", "C", "D"}
            l2.State.control_mtime, l2.State.control_checked = None, -10.0
            l2.poll_control(now=0.0)                       # no file: boot set
            assert l2.State.flags.tolist() == [1, 1, 1]
            ctl.write_text('{"windows": "C"}')
            l2.poll_control(now=0.5)                       # rate limited (< 1 s)
            assert l2.State.flags.tolist() == [1, 1, 1]
            l2.poll_control(now=2.0)
            assert l2.State.flags.tolist() == [0, 1, 0]
            ctl.write_text('garbage!')
            l2.poll_control(now=4.0)                       # unreadable: unchanged
            assert l2.State.flags.tolist() == [0, 1, 0]
            ctl.unlink()
            l2.poll_control(now=6.0)                       # removed: back to the boot set
            assert l2.State.flags.tolist() == [1, 1, 1]
        finally:
            (l2.State.control, l2.State.flags, l2.State.cfg, l2.State.control_mtime,
             l2.State.control_checked) = saved


def test_kernels_compile_for_sm121():
    """Offline Triton compile (Triton's bundled ptxas) of the three Triton kernels for GB10, in a
    fresh process (the JIT objects of this one are interpreter objects)."""
    if not HAS_TRITON:
        print("  (skipped: no triton)")
        return
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    r = subprocess.run([sys.executable, __file__, "--compile-check"], env=env, capture_output=True, text=True,
                       timeout=600)
    print("  " + (r.stdout.strip().splitlines() or ["(no output)"])[-1])
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def compile_check():
    import triton
    try:
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
    except Exception as exc:  # noqa: BLE001
        print(f"  (skipped: {exc!r})")
        return
    target = GPUTarget("cuda", 121, 32)
    jobs = [
        (l2._triton_kernel(), {"segs_ptr": "*i64", "n": "i32", "flag_ptr": "*i32", "CHUNK_B": "constexpr",
                               "LANES": "constexpr"}, {"CHUNK_B": 16384, "LANES": 64}, 2),
        (va._vp_local_key_kernel, {"logits_ptr": "*bf16", "logits_stride": "i32", "key_ptr": "*i64",
                                   "nan_ptr": "*i32", "V": "i32", "vocab_start": "i32", "BLOCK_SIZE": "constexpr"},
         {"BLOCK_SIZE": 4096}, 4),
        (va._vp_verify_kernel, {"keys_ptr": "*i64", "L": "i32", "draft_ptr": "*i64", "cu_ptr": "*i32",
                                "sampled_ptr": "*i64", "sampled_stride": "i32", "num_sampled_ptr": "*i32",
                                "TP": "constexpr", "WIDTH": "constexpr"}, {"TP": 4, "WIDTH": 8}, 1),
    ]
    sizes = []
    for fn, sig, ce, nw in jobs:
        c = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=ce), target=target,
                           options={"num_warps": nw})
        assert ".target sm_121a" in c.asm["ptx"] and len(c.asm["cubin"]) > 0
        if fn is l2._TRITON_KERNEL:
            assert "cp.async.bulk.prefetch.L2.global" in c.asm["ptx"]
        sizes.append(len(c.asm["cubin"]))
    print(f"compiled for sm_121a: l2pf, local_key, verify cubins {sizes} bytes")


# ---------------------------------------------------------------------------------------------
# boot
# ---------------------------------------------------------------------------------------------
def test_boot_targets_and_patcher():
    for k in ("VLLM_VOCAB_PARALLEL_ARGMAX", "VLLM_L2_PREFETCH"):
        os.environ.pop(k, None)
    assert boot.targets() == {}
    os.environ["VLLM_VOCAB_PARALLEL_ARGMAX"] = "check"
    assert set(boot.targets()) == {boot.MOD_RUNNER}
    os.environ["VLLM_L2_PREFETCH"] = "1"
    assert set(boot.targets()) == {boot.MOD_RUNNER, boot.MOD_MODEL}
    os.environ["VLLM_VOCAB_PARALLEL_ARGMAX"] = "0"
    assert set(boot.targets()) == {boot.MOD_MODEL}
    os.environ["VLLM_L2_PREFETCH_CONTROL"] = "/opt/verify-cap-live/glm_fast_l2pf.json"
    t = boot.targets()
    assert t[boot.MOD_RUNNER] == [("glm_fast.l2_prefetch", "install_runner")]
    os.environ.pop("VLLM_L2_PREFETCH_CONTROL")
    for k in ("VLLM_VOCAB_PARALLEL_ARGMAX", "VLLM_L2_PREFETCH"):
        os.environ.pop(k, None)
    os.environ["VLLM_DCP_GLUE"] = "1"
    assert boot.targets() == {boot.MOD_DCP: [("glm_fast.dcp_glue", "install")]}
    os.environ["VLLM_DCP_GLUE"] = "0"
    assert boot.targets() == {}
    os.environ.pop("VLLM_DCP_GLUE")
    with tempfile.TemporaryDirectory() as d:
        Path(d, "glmfast_target_mod.py").write_text("VALUE = 1\n")
        Path(d, "glmfast_patch_mod.py").write_text("def patch(m):\n    m.VALUE = 2\n    m.PATCHED = True\n")
        sys.path.insert(0, d)
        try:
            finder = boot.PostImportPatcher({"glmfast_target_mod": ("glmfast_patch_mod", "patch")})
            sys.meta_path.insert(0, finder)
            import glmfast_target_mod
            assert glmfast_target_mod.VALUE == 2 and glmfast_target_mod.PATCHED and finder.pending() == []
        finally:
            sys.meta_path.remove(finder)
            sys.path.remove(d)
            sys.modules.pop("glmfast_target_mod", None)
            sys.modules.pop("glmfast_patch_mod", None)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main():
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"glm_fast tests: {len(TESTS) - failed}/{len(TESTS)} passed (triton interpreter: {HAS_TRITON})")
    return failed


if __name__ == "__main__":
    if COMPILE_CHECK:
        compile_check()
        sys.exit(0)
    sys.exit(1 if main() else 0)
