"""Bounded-lossy greedy verification: the overlay kernel against the pinned fork
kernel (docs/LOSSY-VERIFICATION-PLAN.md §4). CPU Triton interpreter only."""
import math
import random
from types import SimpleNamespace as NS

import pytest
import torch

from harness import OVERLAY, ROOT, extract
from spec_harness import FIXTURES as RUNTIME

GUMBEL = ROOT / "tests/fixtures/spec_k0/gumbel.py"
NAMES = ["_compute_block_max_and_sumexp", "_compute_global_lse",
         "_compute_block_stats_kernel", "_rejection_kernel", "_resample_kernel",
         "_insert_resampled_kernel", "rejection_sample"]
STOP = [154820, 154827, 154829, 154841, 154842, 154828]
VOCAB = 16389  # three 8192-wide stat blocks (one partial), padded to four


def ints(values):
    return torch.tensor(values, dtype=torch.int32)


def module_constants(path):
    """Top-level constant assignments the extracted kernels reference."""
    import ast
    import triton
    import triton.language as tl
    # The fixtures import these from vllm.triton_utils, which the CPU venv lacks;
    # the interpreter has no libdevice, so log1p (the only call used) is shimmed
    # identically for the fixture and the overlay.
    tldevice = NS(log1p=lambda x: tl.log(1.0 + x))
    out = {"tl": tl, "triton": triton, "tldevice": tldevice, "HAS_TRITON": True}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), out)
            except Exception:
                pass
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = eval(  # noqa: S307 (fixture constants only)
                    compile(ast.Expression(node.value), str(path), "eval"),
                    {"tl": tl, "triton": triton, "HAS_TRITON": True})
            except Exception:
                pass
    return out


def kernels():
    gumbel = extract(GUMBEL, ["tl_rand64", "gumbel_block_argmax"], module_constants(GUMBEL))
    fixture = extract(RUNTIME / "rejection_sampler_utils.py", NAMES, gumbel)
    overlay = extract(OVERLAY / "v1/worker/gpu/spec_decode/rejection_sampler_utils.py",
                      NAMES + ["pad_stop_ids"], {**gumbel, "LOSSY_OFF": -1.0})
    return fixture["rejection_sample"], overlay["rejection_sample"], overlay["pad_stop_ids"]


class Batch:
    """One rejection-sampler call built from per-request (cap, draft ids, logit rows)."""

    def __init__(self, reqs, temps, seeds=None, order=None, max_num_reqs=None):
        self.n = len(reqs)
        order = list(range(self.n)) if order is None else order
        self.max_num_reqs = max_num_reqs or self.n
        sizes = [len(r["rows"]) for r in reqs]
        self.cu = ints([0] + torch.cumsum(ints(sizes), 0).tolist())
        self.logits = torch.cat([r["rows"] for r in reqs]).float()
        self.draft = ints([t for r in reqs for t in [r["anchor"]] + r["draft"]])
        self.mapping = ints([order[i] for i, r in enumerate(reqs) for _ in r["rows"]])
        self.local = ints([j for r in reqs for j in range(len(r["rows"]))])
        self.idx = ints(order)
        self.pos = torch.arange(100055, 100055 + len(self.logits))
        self.temp = torch.zeros(self.max_num_reqs)
        self.seed = torch.full((self.max_num_reqs,), 13, dtype=torch.int64)
        for i, t in enumerate(temps):
            self.temp[order[i]] = t
            if seeds:
                self.seed[order[i]] = seeds[i]

    def run(self, fn, margins=None, min_ps=None, stop=None, **kw):
        args = (self.logits, None, self.draft, self.cu, self.pos, self.idx, self.mapping,
                self.local, self.temp, self.seed, 7)
        if margins is None:
            return fn(*args, **kw)
        margin = torch.full((self.max_num_reqs,), -1.0)
        min_logp = torch.full((self.max_num_reqs,), float("-inf"))
        for i, m in enumerate(margins):
            if m is not None:
                margin[self.idx[i]] = m
        for i, p in enumerate(min_ps or []):
            if p:
                min_logp[self.idx[i]] = math.log(p)
        stop_ids = torch.tensor(STOP if stop is None else stop, dtype=torch.int64)
        return fn(*args, lossy_margin=margin, lossy_min_logp=min_logp, stop_ids=stop_ids, **kw)


def rows(cap, spec, fill=-10.0):
    """`spec[j]` = {token: logit} for row j (cap+1 rows); returns [cap+1, VOCAB]."""
    out = torch.full((cap + 1, VOCAB), fill)
    for j, row in enumerate(spec):
        for tok, val in row.items():
            out[j, tok] = val
    return out


def margin_request(margins, drafts=None, cap=None):
    """Draft j is the runner-up at position j by `margins[j]` nats; row cap is the bonus row."""
    cap = cap or len(margins)
    drafts = drafts or [100 + j for j in range(cap)]
    spec = [{200 + j: 10.0, drafts[j]: 10.0 - margins[j]} for j in range(cap)]
    spec.append({300: 10.0})  # bonus row argmax
    return {"anchor": 2, "draft": drafts, "rows": rows(cap, spec)}


def reference(logits, anchor_draft, margin, min_logp, stop):
    """Python reference of the greedy rule over one request's rows (chain)."""
    cap = len(anchor_draft) - 1
    out, relaxed = [], 0
    for j in range(cap):
        row = logits[j]
        a = int(row.argmax())
        d = int(anchor_draft[j + 1])
        if d == a:
            out.append(d)
            continue
        top2 = torch.topk(row, 2).values
        z1, z2, zd = float(top2[0]), float(top2[1]), float(row[d])
        ok = margin is not None and zd >= z2 and z1 - zd <= margin and d not in stop and a not in stop
        if ok and min_logp > float("-inf"):
            ok = zd - float(torch.logsumexp(row, 0)) >= min_logp
        if ok:
            out.append(d)
            relaxed += 1
        else:
            out.append(a)
            return out, relaxed
    out.append(int(logits[cap].argmax()))
    return out, relaxed


def random_request(rng, cap, near):
    """Random background logits (spread ~ +-4.5) with a planted argmax at 12.0.

    Per position the draft is the argmax, a planted runner-up 0.05-4 nats below
    it, or a planted third-ranked token, with probabilities `near`."""
    g = torch.Generator().manual_seed(rng.randrange(1 << 30))
    logits = torch.randn(cap + 1, VOCAB, generator=g)
    draft = []
    for j in range(cap):
        a, d, third = rng.sample(range(VOCAB), 3)
        logits[j, a] = 12.0
        r = rng.random()
        if r < near[0]:
            draft.append(a)
            continue
        delta = rng.uniform(0.05, 4.0)
        logits[j, d] = 12.0 - delta
        if r >= near[0] + near[1]:
            logits[j, third] = 12.0 - delta + 0.02
        draft.append(d)
    logits[cap, rng.randrange(VOCAB)] = 12.0
    return {"anchor": 2, "draft": draft, "rows": logits}


def test_identity_when_off_greedy_and_sampled_mixed_caps():
    fixture, overlay, _ = kernels()
    rng = random.Random(7)
    reqs = [random_request(rng, cap, (0.6, 0.3, 0.1)) for cap in (1, 3, 5, 7, 7, 0)]
    temps = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    batch = Batch(reqs, temps, seeds=[11, 12, 13, 14, 15, 16])
    want_sampled, want_n = batch.run(fixture)
    got_sampled, got_n, got_relaxed = batch.run(overlay)
    assert torch.equal(got_n, want_n)
    for i in range(batch.n):
        k = int(want_n[i])
        assert torch.equal(got_sampled[i, :k], want_sampled[i, :k])
    assert got_relaxed.tolist() == [0] * batch.n
    # Explicit off tensors are the same as omitting them.
    off_sampled, off_n, off_relaxed = batch.run(overlay, margins=[None] * batch.n)
    assert torch.equal(off_n, want_n) and off_relaxed.tolist() == [0] * batch.n


@pytest.mark.parametrize("margin,accepted", [(1.0, 1), (2.0, 2), (3.5, 3)])
def test_relaxed_accept_fires_exactly_within_margin(margin, accepted):
    _, overlay, _ = kernels()
    req = margin_request([0.3, 1.2, 3.0])
    batch = Batch([req], [0.0])
    sampled, n, relaxed = batch.run(overlay, margins=[margin])
    assert int(n[0]) == accepted + 1 and int(relaxed[0]) == accepted
    expect = req["draft"][:accepted] + ([300] if accepted == 3 else [200 + accepted])
    assert sampled[0, :accepted + 1].tolist() == expect


def test_guards_rank_floor_and_stop_ids():
    _, overlay, _ = kernels()
    # Rank-3 draft within margin: rejected, argmax emitted.
    third = {"anchor": 2, "draft": [100], "rows": rows(1, [{200: 10.0, 201: 9.9, 100: 9.8}, {300: 10.0}])}
    sampled, n, relaxed = Batch([third], [0.0]).run(overlay, margins=[2.0])
    assert n.tolist() == [1] and relaxed.tolist() == [0] and sampled[0, 0].item() == 200
    # Probability floor: p(runner-up) ~ 0.053 with 100 competitors at 8.0.
    spec = [{200: 10.0, 100: 9.8, **{500 + t: 8.0 for t in range(100)}}, {300: 10.0}]
    floor = {"anchor": 2, "draft": [100], "rows": rows(1, spec)}
    p_d = math.exp(9.8) / (math.exp(10.0) + math.exp(9.8) + 100 * math.exp(8.0) + (VOCAB - 102) * math.exp(-10.0))
    assert 0.05 < p_d < 0.06
    for min_p, want in ((0.1, 0), (0.05, 1), (0.0, 1)):
        sampled, n, relaxed = Batch([floor], [0.0]).run(overlay, margins=[1.0], min_ps=[min_p])
        assert relaxed.tolist() == [want], min_p
        assert sampled[0, 0].item() == (100 if want else 200)
    # Stop ids: a stop draft, or a stop argmax, is never relaxed (ids chosen
    # inside the test vocabulary; the served set is passed the same way).
    stop = 900
    for stop_draft, spec in ((True, [{200: 10.0, stop: 9.9}, {300: 10.0}]),
                             (False, [{stop: 10.0, 100: 9.9}, {300: 10.0}])):
        req = {"anchor": 2, "draft": [stop if stop_draft else 100], "rows": rows(1, spec)}
        sampled, n, relaxed = Batch([req], [0.0]).run(overlay, margins=[2.0], stop=[stop, 901])
        assert relaxed.tolist() == [0] and n.tolist() == [1]
        # The same rows with an unrelated stop set relax.
        _, n2, relaxed2 = Batch([req], [0.0]).run(overlay, margins=[2.0], stop=[901])
        assert relaxed2.tolist() == [1] and n2.tolist() == [2]


def test_bookkeeping_full_block_bonus_and_partial_rejection():
    _, overlay, _ = kernels()
    req = margin_request([0.5] * 7)
    sampled, n, relaxed = Batch([req], [0.0]).run(overlay, margins=[1.0])
    assert n.tolist() == [8] and relaxed.tolist() == [7]
    assert sampled[0].tolist() == req["draft"] + [300]
    mixed = margin_request([0.5, 0.5, 2.5, 0.5, 0.5, 0.5, 0.5])
    sampled, n, relaxed = Batch([mixed], [0.0]).run(overlay, margins=[1.0])
    assert n.tolist() == [3] and relaxed.tolist() == [2]
    assert sampled[0, :3].tolist() == mixed["draft"][:2] + [202]


def test_mixed_batch_isolates_lossy_exact_and_sampled_rows():
    fixture, overlay, _ = kernels()
    rng = random.Random(3)
    lossy = margin_request([0.3, 1.2, 3.0])
    exact = random_request(rng, 5, (0.5, 0.4, 0.1))
    sampled_req = random_request(rng, 7, (0.5, 0.4, 0.1))
    batch = Batch([lossy, exact, sampled_req], [0.0, 0.0, 1.0], seeds=[21, 22, 23])
    want_sampled, want_n = batch.run(fixture)
    got_sampled, got_n, got_relaxed = batch.run(overlay, margins=[2.0, None, 2.0])
    assert got_relaxed.tolist()[1:] == [0, 0]
    for i in (1, 2):
        k = int(want_n[i])
        assert int(got_n[i]) == k and torch.equal(got_sampled[i, :k], want_sampled[i, :k])
    assert got_n[0].item() == 3 and got_relaxed[0].item() == 2
    assert got_sampled[0, :3].tolist() == lossy["draft"][:2] + [202]


def test_matches_python_reference_and_is_permutation_invariant():
    _, overlay, _ = kernels()
    src = (OVERLAY / "v1/worker/gpu/spec_decode/rejection_sampler_utils.py").read_text()
    assert "tl.atomic" not in src
    rng = random.Random(11)
    caps = [7, 3, 5, 7, 1, 7]
    reqs = [random_request(rng, cap, (0.3, 0.55, 0.15)) for cap in caps]
    margins = [1.0, 2.5, 0.5, 3.0, 1.5, 2.0]
    min_ps = [0.0, 0.1, 0.0, 0.05, 0.0, 0.2]
    expected = []
    for req, m, p in zip(reqs, margins, min_ps):
        anchor_draft = [req["anchor"]] + req["draft"]
        expected.append(reference(req["rows"], anchor_draft, m, math.log(p) if p else float("-inf"), set(STOP)))
    assert sum(r for _, r in expected) >= 3, "reference should exercise relaxed accepts"
    for order in (None, [5, 2, 0, 4, 1, 3], [3, 4, 5, 0, 1, 2]):
        batch = Batch(reqs, [0.0] * 6, order=order, max_num_reqs=8)
        sampled, n, relaxed = batch.run(overlay, margins=margins, min_ps=min_ps)
        for i, (tokens, r) in enumerate(expected):
            assert n[i].item() == len(tokens), (order, i)
            assert sampled[i, :len(tokens)].tolist() == tokens, (order, i)
            assert relaxed[i].item() == r, (order, i)


def test_stop_id_padding():
    _, _, pad = kernels()
    assert pad(None, torch.device("cpu")).tolist() == [-1] * 8
    assert pad(torch.tensor([5, 9]), torch.device("cpu")).tolist() == [5, 9] + [-1] * 6
    assert pad(torch.arange(9), torch.device("cpu")).shape[0] == 16


def test_request_controls_parse_numerics_and_fail_closed():
    ns = extract(OVERLAY / "v1/worker/gpu/sample/states.py", ["_number", "lossy_controls"],
                 {"math": math, "LOSSY_OFF": -1.0, "LOSSY_MAX_MARGIN": 5.0, "LOSSY_MAX_MIN_P": 0.5})
    controls = ns["lossy_controls"]
    off = (-1.0, float("-inf"))
    def sp(temperature=0.0, **xargs):
        return NS(temperature=temperature, extra_args=xargs or None, structured_outputs=None,
                  logit_bias=None, allowed_token_ids=None, bad_words=None,
                  presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0)
    assert controls(sp(spec_lossy_margin=1)) == (1.0, float("-inf"))
    assert controls(sp(spec_lossy_margin=1.5, spec_lossy_min_p=0.1)) == (1.5, math.log(0.1))
    assert controls(sp(spec_lossy_margin=2.5, spec_lossy_rank=2, spec_lossy_min_p=0)) == (2.5, float("-inf"))
    for bad in (dict(spec_lossy_margin="1"), dict(spec_lossy_margin=True), dict(spec_lossy_margin=0),
                dict(spec_lossy_margin=6), dict(spec_lossy_margin=float("nan")),
                dict(spec_lossy_margin=1, spec_lossy_rank=3), dict(spec_lossy_margin=1, spec_lossy_min_p=0.5),
                dict(spec_lossy_margin=1, spec_lossy_min_p=-0.1), dict()):
        assert controls(sp(**bad)) == off, bad
    assert controls(sp(temperature=1.0, spec_lossy_margin=1)) == off
    bias = sp(spec_lossy_margin=1); bias.logit_bias = {1: 2.0}
    assert controls(bias) == off
    guided = sp(spec_lossy_margin=1); guided.structured_outputs = object()
    assert controls(guided) == off
    pen = sp(spec_lossy_margin=1); pen.repetition_penalty = 1.1
    assert controls(pen) == off
    src = (OVERLAY / "v1/worker/gpu/sample/states.py").read_text()
    assert 'os.environ.get("GLM_SPEC_LOSSY") == "1"' in src
    assert "lossy_controls(sampling_params) if LOSSY_ENABLED" in src
