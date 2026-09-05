"""GLM_DCP_Q_PREGATHER: gathering the 256-wide pre-expansion query and
expanding every head locally must give the same absorbed query as expanding
the local heads first and gathering the 576-wide result (the stock path)."""
import ast
import pathlib

import torch

from harness import OVERLAY

SRC = OVERLAY / "model_executor/layers/attention/mla_attention.py"


def _load_helper():
    tree = ast.parse(SRC.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "dcp_pregather_expand")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SRC), "exec"), ns)
    return ns["dcp_pregather_expand"]


def _stock_path(q_nope_ranks, q_pe_ranks, w_uk_t_ranks):
    """Per rank: bmm local heads, cat with pe, then all-gather over heads."""
    per_rank = []
    for q_nope, q_pe, w in zip(q_nope_ranks, q_pe_ranks, w_uk_t_ranks):
        ql_nope = torch.bmm(q_nope, w).transpose(0, 1)  # (B, N_local, L)
        per_rank.append(torch.cat((ql_nope, q_pe), dim=-1))  # (B, N_local, L + R)
    return torch.cat(per_rank, dim=1)  # (B, N_total, L + R)


def test_pregather_matches_stock_path():
    torch.manual_seed(0)
    world, n_local, b, p, r, l = 4, 16, 8, 192, 64, 512
    q_nope_ranks = [torch.randn(n_local, b, p, dtype=torch.bfloat16) for _ in range(world)]
    q_pe_ranks = [torch.randn(b, n_local, r, dtype=torch.bfloat16) for _ in range(world)]
    w_uk_t_ranks = [torch.randn(n_local, p, l, dtype=torch.bfloat16) * 0.05 for _ in range(world)]
    expected = _stock_path(q_nope_ranks, q_pe_ranks, w_uk_t_ranks)

    fn = _load_helper()
    w_all = torch.cat(w_uk_t_ranks, dim=0)  # what process_weights_after_loading gathers
    outs = []
    for rank in range(world):
        packed_by_rank = {}

        def gather_dim0(t, rank=rank):
            # a fake all-gather: every rank contributes its own packed tensor
            packed_by_rank[rank] = t
            parts = [torch.cat((q_nope_ranks[i], q_pe_ranks[i].transpose(0, 1)), dim=-1) for i in range(world)]
            parts[rank] = t
            assert t.shape == (n_local, b, p + r)
            return torch.cat(parts, dim=0)

        outs.append(fn(q_nope_ranks[rank], q_pe_ranks[rank], w_all, gather_dim0))
    for out in outs:
        assert out.shape == (b, world * n_local, l + r)
        assert torch.equal(out, expected), "pre-gather path differs from the stock path"
    # the gathered payload is 256 wide, not 576
    assert (p + r) * 2 * 2.25 == (l + r) * 2  # 576 / 256 = 2.25 in bytes per head


def test_flag_and_guards_present():
    src = SRC.read_text()
    assert 'os.environ.get("GLM_DCP_Q_PREGATHER", "0")' in src
    assert "self.W_UK_T_all = None" in src
    assert "mqa_q_pregathered = False" in src
    # the stock gather is skipped only when the query was pre-gathered
    assert "if self.impl.dcp_world_size > 1 and not mqa_q_pregathered:" in src
