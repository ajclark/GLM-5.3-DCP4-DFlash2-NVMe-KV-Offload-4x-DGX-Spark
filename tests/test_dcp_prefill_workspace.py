"""Under DCP the indexer's K-gather workspace is sized on each rank's LOCAL
share (cdiv(max_model_len, dcp)) and the splitter checks the workspace
constraint on local lengths, while its logits budget stays on GLOBAL lengths
(the transient profile validated on the cluster). At DCP 1 nothing changes."""
import ast
import random
import types

import pytest
import torch

from harness import ROOT, extract

INDEXER = ROOT / "overlay/vllm/v1/attention/backends/mla/indexer.py"
BUDGET = 256 * 1024 * 1024


def _cfg(max_model_len, dcp):
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(max_model_len=max_model_len),
        parallel_config=types.SimpleNamespace(decode_context_parallel_size=dcp),
    )


@pytest.fixture(scope="module")
def fns():
    return extract(INDEXER, ["get_max_prefill_buffer_size", "split_indexer_prefill_chunks"],
                   extra_globals={"VllmConfig": object})


def _local(seq_lens, dcp):
    return (seq_lens + (dcp - 1)) // dcp


def test_workspace_is_local_under_dcp_and_unchanged_at_dcp1(fns):
    f = fns["get_max_prefill_buffer_size"]
    assert f(_cfg(524288, 1)) == 524288 * 40
    assert f(_cfg(524288, 4)) == 131072 * 40
    assert f(_cfg(100, 3)) == 34 * 40


@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_every_chunk_fits_the_local_workspace_and_global_logits_budget(fns, dcp):
    split = fns["split_indexer_prefill_chunks"]
    ws = fns["get_max_prefill_buffer_size"](_cfg(524288, dcp))
    rng = random.Random(dcp)
    for _ in range(300):
        n = rng.randint(1, 6)
        seq = torch.tensor([rng.choice([rng.randint(1, 524288), 524288, 524287]) for _ in range(n)])
        q = torch.tensor([rng.choice([1, 17, 2048, 2047]) for _ in range(n)])
        kw = {"workspace_seq_lens_cpu": _local(seq, dcp)} if dcp > 1 else {}
        chunks = split(seq, q, ws, BUDGET, **kw)
        covered = [0] * n
        for req_slice, q_slice in chunks:
            assert int(_local(seq, dcp)[req_slice].sum()) <= ws
            m = int(q[req_slice].sum()); n_glob = int(seq[req_slice].sum())
            assert 0 <= q_slice.start < q_slice.stop <= m
            assert (q_slice.stop - q_slice.start) * n_glob <= BUDGET // 4 or (req_slice.stop - req_slice.start == 1 and q_slice.stop - q_slice.start == 1)
            for i in range(req_slice.start, req_slice.stop):
                covered[i] += q_slice.stop - q_slice.start
        assert all(c >= int(q[i]) for i, c in enumerate(covered))


def test_logits_budget_stays_global_so_slices_match_boot2(fns):
    """500k global at DCP4: 16 query slices per 2048-token chunk, exactly as
    the global-length budget gave at boot 2; a local budget would give 4."""
    split = fns["split_indexer_prefill_chunks"]
    seq = torch.tensor([500000]); q = torch.tensor([2048])
    assert len(split(seq, q, 131072 * 40, BUDGET, workspace_seq_lens_cpu=_local(seq, 4))) == 16


def test_dcp1_call_is_unchanged(fns):
    split = fns["split_indexer_prefill_chunks"]
    seq = torch.tensor([40000, 3000]); q = torch.tensor([2048, 700])
    assert split(seq, q, 524288 * 40, BUDGET) == split(seq, q, 524288 * 40, BUDGET, workspace_seq_lens_cpu=seq)


def test_builder_passes_global_for_logits_and_local_for_workspace():
    src = INDEXER.read_text()
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "split_indexer_prefill_chunks"]
    assert len(calls) == 1
    c = calls[0]
    assert isinstance(c.args[0], ast.Name) and c.args[0].id == "prefill_seq_lens_global"
    kws = {k.arg: k.value for k in c.keywords}
    assert isinstance(kws["workspace_seq_lens_cpu"], ast.Name) and kws["workspace_seq_lens_cpu"].id == "workspace_seq_lens"
    tail = src[: src.index("chunk_specs = split_indexer_prefill_chunks(")][-800:]
    assert "(self.dcp_world_size - 1)\n                ) // self.dcp_world_size" in tail
