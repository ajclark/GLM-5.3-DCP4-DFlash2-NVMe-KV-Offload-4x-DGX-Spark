# SPDX-License-Identifier: Apache-2.0
"""GLM_DCP_RS_HEADMAJOR: the DCP LSE merge kernel writes its rescaled partial
head-major ([H, T, D]) so the reduce-scatter runs over dim 0 without the
communicator's two relayout copies. Checks the kernel (Triton interpreter on
CPU) against the transcribed stock merge, the zero-factor rows, and the
compaction's empty-row mask that replaces the [T, topk] rescan."""
import torch
from harness import OVERLAY, extract, ref_lse_merge

MLA = extract(
    OVERLAY / "model_executor/layers/attention/mla_attention.py",
    ["_glm_correct_attn_cp_out_hm_kernel", "dcp_lse_ag_out_rs_headmajor"],
)
SU = extract(
    OVERLAY / "v1/attention/backends/mla/sparse_utils.py",
    ["_compact_dcp_candidates_kernel", "compact_dcp_candidates"],
)


class Group:
    """A DCP group of `world` ranks holding every rank's inputs; `rank` is
    the rank whose call is being simulated."""

    def __init__(self, world, rank, lses, partials):
        self.world_size, self.rank_in_group = world, rank
        self._lses, self._partials = lses, partials
        self.scatter_dims = []

    def all_gather(self, t, dim):
        assert dim == 0 and torch.equal(t, self._lses[self.rank_in_group])
        return torch.cat(self._lses, dim=0)

    def reduce_scatter(self, t, dim):
        self.scatter_dims.append(dim)
        assert t.is_contiguous()
        self._partials[self.rank_in_group] = t.clone()
        return t  # replaced by the driver once every rank has run


def merge_all_ranks(outs, lses, world):
    """Run the head-major merge for every rank, then do the reduce-scatter
    over dim 0 the way NCCL would: sum the partials, hand each rank its
    head chunk."""
    partials = [None] * world
    groups = [Group(world, r, lses, partials) for r in range(world)]
    for r in range(world):
        MLA["dcp_lse_ag_out_rs_headmajor"](outs[r], lses[r], groups[r])
    total = sum(p.float() for p in partials)  # [H, T, D]
    heads = total.shape[0] // world
    got = [total[r * heads : (r + 1) * heads] for r in range(world)]
    assert all(g.scatter_dims == [0] for g in groups)
    return got


def rank_inputs(world, tokens=6, heads=8, head_dim=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    outs = [torch.randn(tokens, heads, head_dim, generator=g).to(torch.bfloat16) for _ in range(world)]
    lses = [torch.randn(tokens, heads, generator=g) * 3 for _ in range(world)]
    return outs, lses


def test_matches_stock_merge_for_dcp2_and_dcp4():
    for world in (2, 4):
        outs, lses = rank_inputs(world, seed=world)
        got = merge_all_ranks(outs, lses, world)
        want = ref_lse_merge(outs, lses)  # [T, H, D] summed over ranks
        heads = want.shape[1] // world
        for r in range(world):
            want_r = want[:, r * heads : (r + 1) * heads].transpose(0, 1)  # [H/world, T, D]
            assert got[r].shape == want_r.shape
            assert torch.allclose(got[r], want_r, atol=2e-2, rtol=2e-2)


def test_empty_shard_rows_become_exact_zeros_even_with_nan_output():
    world = 2
    outs, lses = rank_inputs(world, seed=7)
    # rank 1 owns nothing for token 2: its kernel output is garbage/NaN and its
    # LSE is -inf (what the sparse backend stores for an all -1 row).
    outs[1][2] = float("nan")
    lses[1][2] = float("-inf")
    # token 4: no rank owns anything (both -inf, both NaN).
    outs[0][4] = float("nan"); outs[1][4] = float("nan")
    lses[0][4] = float("-inf"); lses[1][4] = float("-inf")
    got = merge_all_ranks(outs, lses, world)
    # the transcribed stock merge multiplies NaN output by a zero factor
    # (NaN); the kernels guard that with where(factor == 0, 0, .), so the
    # reference gets zeros where the kernel input was NaN.
    want = ref_lse_merge([torch.nan_to_num(o) for o in outs], lses)
    heads = want.shape[1] // world
    for r in range(world):
        assert torch.isfinite(got[r]).all()
        want_r = want[:, r * heads : (r + 1) * heads].transpose(0, 1)
        assert torch.allclose(got[r], want_r, atol=2e-2, rtol=2e-2)
        assert got[r][:, 4].abs().max() == 0


def test_head_major_layout_and_dtype():
    world = 2
    outs, lses = rank_inputs(world, tokens=5, heads=4, head_dim=32, seed=3)
    partials = [None] * world
    grp = Group(world, 0, lses, partials)
    out = MLA["dcp_lse_ag_out_rs_headmajor"](outs[0], lses[0], grp)
    assert out.shape == (4, 5, 32) and out.dtype == torch.bfloat16 and out.is_contiguous()
    # the head-major partial is the stock in-place correction, transposed
    want = ref_lse_merge([outs[0], torch.zeros_like(outs[1])], lses)  # rank 1 contributes zeros
    assert torch.allclose(out.float(), want.transpose(0, 1), atol=2e-2, rtol=2e-2)


def test_compaction_reports_empty_rows():
    g = torch.Generator().manual_seed(5)
    idx = torch.randint(0, 999, (6, 64), generator=g, dtype=torch.int32)
    idx[torch.rand(6, 64, generator=g) < 0.5] = -1
    idx[1] = -1
    idx[4] = -1
    out, lengths, empty = SU["compact_dcp_candidates"](idx, return_empty=True)
    assert empty.dtype == torch.bool and empty.tolist() == [False, True, False, False, True, False]
    assert torch.equal(empty, (idx == -1).all(dim=-1))
    assert lengths[1] == 1 and lengths[4] == 1  # clamped for the kernel loop bound
    out2, lengths2 = SU["compact_dcp_candidates"](idx)
    assert torch.equal(out, out2) and torch.equal(lengths, lengths2)


def test_flag_is_plumbed_from_launcher_to_both_overlays():
    root = OVERLAY.parent.parent
    launcher = (root / "launch-glm53big-dcp.sh").read_text()
    assert '-e "GLM_DCP_RS_HEADMAJOR=${DCP_RS_HEADMAJOR:-0}"' in launcher
    assert "DCP_RS_HEADMAJOR=${DCP_RS_HEADMAJOR:-0}" in (root / "rollout_dcp.sh").read_text()
    mla = (OVERLAY / "model_executor/layers/attention/mla_attention.py").read_text()
    assert 'os.environ.get("GLM_DCP_RS_HEADMAJOR") == "1"' in mla
    assert "elif self.dcp_rs_headmajor:" in mla and "head_major=head_major" in mla
    sparse = (OVERLAY / "v1/attention/backends/mla/flashmla_sparse.py").read_text()
    assert 'os.environ.get("GLM_DCP_RS_HEADMAJOR", "0") == "1"' in sparse
    assert "if not DCP_RS_HEADMAJOR:" in sparse and "return_empty=True" in sparse
    node = (root / "bench/spec_node.py").read_text()
    assert "env['DCP_RS_HEADMAJOR'] = '1' if (root/'dcp-rs-headmajor-enabled').exists() else '0'" in node
    ctl = (root / "bench/spec_experiment.py").read_text()
    assert "'--dcp-rs-headmajor'" in ctl and "dcp-rs-headmajor-enabled" in ctl
