# SPDX-License-Identifier: Apache-2.0
"""End to end: sharded sparse attention plus the LSE merge equals DCP1.

This exercises the claim the whole design rests on: each rank attends only to
the selected tokens it owns, and the cross-rank log-sum-exp merge reassembles
the exact global softmax, including rows where a rank owns nothing.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from harness import OVERLAY, extract, extract_methods, ref_lse_merge, ref_owner_and_local

SU = extract(
    OVERLAY / "v1/attention/backends/mla/sparse_utils.py",
    [
        "_convert_req_index_to_global_index_kernel",
        "triton_convert_req_index_to_global_index",
        "triton_filter_and_convert_dcp_index",
    ],
)


class _FP8KernelMetadata:
    pass


class _Meta:
    FP8KernelMetadata = _FP8KernelMetadata


FMS = extract_methods(
    OVERLAY / "v1/attention/backends/mla/flashmla_sparse.py",
    "FlashMLASparseImpl",
    ["_forward_fp8_kv_mixed_batch"],
    {
        "FlashMLASparseMetadata": _Meta,
        "triton_filter_and_convert_dcp_index": SU["triton_filter_and_convert_dcp_index"],
        "triton_convert_req_index_to_global_index": SU[
            "triton_convert_req_index_to_global_index"
        ],
    },
)


def partial_attention(q, kv, cols, scale):
    """Normalised softmax over `cols` plus its natural-log sum-exp.

    q: [H, D], kv: [L, D]. Returns (out [H, Dv], lse [H]); an empty selection
    yields the identity element of the merge, (0, -inf).
    """
    H, D = q.shape
    dv = kv.shape[1]
    if not cols:
        return torch.zeros(H, dv), torch.full((H,), -math.inf)
    k = kv[cols]  # [n, D]
    scores = (q.float() @ k.float().T) * scale  # [H, n]
    m = scores.max(dim=-1, keepdim=True).values
    p = torch.exp(scores - m)
    denom = p.sum(dim=-1, keepdim=True)
    out = (p / denom) @ k.float()
    lse = (torch.log(denom) + m).squeeze(-1)
    return out, lse


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("interleave", [1, 2])
def test_sharded_attention_matches_single_rank(world, interleave):
    torch.manual_seed(world * 7 + interleave)
    T, H, D, L, K = 6, 4, 16, 80, 12
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(T, H, D)
    kv = torch.randn(L, D)

    # A shared per-token selection of global positions, as the merged indexer
    # top-k would produce (varying length, causal-ish).
    sel = []
    for t in range(T):
        bound = max(1, min(L, 8 + t * 13))
        picks = torch.randperm(bound)[: min(K, bound)].tolist()
        sel.append(sorted(picks))

    ref = torch.stack(
        [partial_attention(q[t], kv, sel[t], scale)[0] for t in range(T)]
    )

    outs, lses = [], []
    for rank in range(world):
        o, l = [], []
        for t in range(T):
            mine = [
                p
                for p in sel[t]
                if ref_owner_and_local(p, world, interleave)[0] == rank
            ]
            oo, ll = partial_attention(q[t], kv, mine, scale)
            o.append(oo)
            l.append(ll)
        outs.append(torch.stack(o))
        lses.append(torch.stack(l))

    merged = ref_lse_merge(outs, lses)
    torch.testing.assert_close(merged, ref, atol=1e-5, rtol=1e-5)


def test_row_owned_entirely_by_one_rank():
    """The three ranks that own nothing must contribute exactly zero."""
    torch.manual_seed(3)
    world, H, D = 4, 2, 8
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(1, H, D)
    kv = torch.randn(16, D)
    sel = [0, 4, 8, 12]  # all owned by rank 0 at world=4, interleave=1
    assert {ref_owner_and_local(p, world, 1)[0] for p in sel} == {0}

    ref, _ = partial_attention(q[0], kv, sel, scale)
    outs, lses = [], []
    for rank in range(world):
        mine = [p for p in sel if ref_owner_and_local(p, world, 1)[0] == rank]
        o, l = partial_attention(q[0], kv, mine, scale)
        outs.append(o.unsqueeze(0))
        lses.append(l.unsqueeze(0))
    merged = ref_lse_merge(outs, lses)[0]
    torch.testing.assert_close(merged, ref, atol=1e-6, rtol=1e-6)


def _run_mixed_batch(dcp_world_size, topk_local, kernel_out, kernel_lse):
    num_tokens, num_heads = topk_local.shape[0], kernel_out.shape[2]
    fp8_meta = _FP8KernelMetadata()

    def run_kernel(**kwargs):
        return kernel_out, kernel_lse

    meta = SimpleNamespace(
        fp8_extra_metadata=fp8_meta,
        req_id_per_token=torch.zeros(num_tokens, dtype=torch.int32),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        block_size=64,
        cp_kv_cache_interleave_size=1,
    )
    impl = SimpleNamespace(
        dcp_world_size=dcp_world_size,
        dcp_rank=0,
        need_to_return_lse_for_decode=dcp_world_size > 1,
        _fp8_flash_mla_kernel=run_kernel,
    )
    # Bypass the index conversion (covered by test_dcp_index_filter.py) and
    # feed the already-local indices straight in.
    g = FMS["_forward_fp8_kv_mixed_batch"].__globals__
    g["triton_filter_and_convert_dcp_index"] = lambda *a, **k: topk_local
    g["triton_convert_req_index_to_global_index"] = lambda *a, **k: topk_local
    g.setdefault("DCP_COMPACT", False)  # module flag; compaction is covered by test_dcp_compact.py
    g.setdefault("DCP_RS_HEADMAJOR", False)  # module flag; the head-major merge zeroes empty rows itself
    q = torch.zeros(num_tokens, num_heads, 4)
    return FMS["_forward_fp8_kv_mixed_batch"](
        impl, q, torch.zeros(0), topk_local, meta
    )


def test_empty_rows_are_neutralised_to_zero_and_neg_inf():
    """A row whose local shard holds nothing must come back (0, -inf) even if
    the kernel wrote NaN there: 0 * NaN = NaN would poison the merge."""
    num_tokens, num_heads, dv = 3, 2, 1
    topk_local = torch.tensor(
        [[0, 1, -1, -1], [-1, -1, -1, -1], [2, -1, 3, -1]], dtype=torch.int32
    )
    out = torch.full((1, num_tokens, num_heads, dv), float("nan"))
    lse = torch.full((1, num_heads, num_tokens), float("nan"))
    for t in (0, 2):
        out[0, t] = float(t + 1)
        lse[0, :, t] = float(t + 1)

    got_out, got_lse = _run_mixed_batch(4, topk_local, out, lse)

    assert torch.equal(got_out[1], torch.zeros_like(got_out[1]))
    assert torch.isneginf(got_lse[1]).all()
    for t in (0, 2):
        assert torch.equal(got_out[t], torch.full_like(got_out[t], t + 1))
        assert torch.equal(got_lse[t], torch.full_like(got_lse[t], t + 1))
    assert got_out.is_contiguous()
    assert not got_out.isnan().any()
    assert not got_lse.isnan().any()
    # LSE must arrive as [T, H] fp32 for the merge.
    assert got_lse.shape == (num_tokens, num_heads)
    assert got_lse.dtype == torch.float32


def test_no_lse_returned_without_dcp():
    topk_local = torch.tensor([[0, 1]], dtype=torch.int32)
    out = torch.zeros(1, 1, 2, 1)
    lse = torch.zeros(1, 2, 1)
    got_out, got_lse = _run_mixed_batch(1, topk_local, out, lse)
    assert got_lse is None
    assert got_out.shape == (1, 2, 1)


@pytest.mark.parametrize("world", [2, 4])
def test_replicated_query_heads_merge_exactly(world):
    """Robustness: if a drafter layer were ever replicated across the DCP group
    (every rank holding ALL heads, which is NOT the case for this fork's MTP
    head), the DCP head all-gather would carry `world` identical copies. The
    merge must still be exact, and each rank's reduce-scatter chunk (its own
    copy) must equal the single-rank result: redundant work, not wrong output."""
    torch.manual_seed(11 + world)
    T, H, D, L, K = 4, 8, 16, 64, 10
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(T, H, D)  # identical on every rank
    kv = torch.randn(L, D)
    sel = [sorted(torch.randperm(L)[:K].tolist()) for _ in range(T)]
    ref = torch.stack([partial_attention(q[t], kv, sel[t], scale)[0] for t in range(T)])

    # all_gather(dim=heads) of identical q -> world copies; each rank attends
    # over its own KV shard with all world*H heads.
    q_gathered = torch.cat([q] * world, dim=1)  # [T, world*H, D]
    outs, lses = [], []
    for rank in range(world):
        o, l = [], []
        for t in range(T):
            mine = [p for p in sel[t] if ref_owner_and_local(p, world, 1)[0] == rank]
            oo, ll = partial_attention(q_gathered[t], kv, mine, scale)
            o.append(oo)
            l.append(ll)
        outs.append(torch.stack(o))
        lses.append(torch.stack(l))
    merged = ref_lse_merge(outs, lses)  # [T, world*H, Dv], before reduce-scatter
    for rank in range(world):
        chunk = merged[:, rank * H : (rank + 1) * H]  # what reduce_scatter hands rank r
        torch.testing.assert_close(chunk, ref, atol=1e-5, rtol=1e-5)


def test_head_major_merge_keeps_only_the_lse_mask():
    """With GLM_DCP_RS_HEADMAJOR=1 the [T, H, D] masked_fill is skipped: the
    merge kernel writes exact zeros wherever the rescale factor is zero, and
    the -inf LSE is what makes that factor zero (test_dcp_rs_headmajor.py)."""
    num_tokens, num_heads, dv = 3, 2, 1
    topk_local = torch.tensor(
        [[0, 1, -1, -1], [-1, -1, -1, -1], [2, -1, 3, -1]], dtype=torch.int32
    )
    out = torch.full((1, num_tokens, num_heads, dv), float("nan"))
    lse = torch.full((1, num_heads, num_tokens), float("nan"))
    for t in (0, 2):
        out[0, t] = float(t + 1)
        lse[0, :, t] = float(t + 1)
    g = FMS["_forward_fp8_kv_mixed_batch"].__globals__
    g["DCP_RS_HEADMAJOR"] = True
    try:
        got_out, got_lse = _run_mixed_batch(4, topk_local, out, lse)
    finally:
        g["DCP_RS_HEADMAJOR"] = False
    assert got_out[1].isnan().all()  # left for the merge kernel to neutralise
    assert torch.isneginf(got_lse[1]).all()
    for t in (0, 2):
        assert torch.equal(got_out[t], torch.full_like(got_out[t], t + 1))
        assert torch.equal(got_lse[t], torch.full_like(got_lse[t], t + 1))
    assert got_out.is_contiguous() and not got_lse.isnan().any()
