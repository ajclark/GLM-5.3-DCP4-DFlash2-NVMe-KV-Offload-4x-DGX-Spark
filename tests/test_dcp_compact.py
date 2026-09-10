# SPDX-License-Identifier: Apache-2.0
"""GLM_DCP_COMPACT: the per-rank top-k list is compacted (valid entries first,
order kept, -1 after) with a per-token length the sparse kernels bound their
loop by. Runs the Triton kernel in interpreter mode on CPU."""
import torch
from harness import OVERLAY, extract

SU = extract(
    OVERLAY / "v1/attention/backends/mla/sparse_utils.py",
    ["_compact_dcp_candidates_kernel", "compact_dcp_candidates"],
)


def _reference(idx):
    out = torch.full_like(idx, -1)
    lengths = []
    for r in range(idx.shape[0]):
        valid = idx[r][idx[r] >= 0]
        out[r, : valid.numel()] = valid
        lengths.append(max(1, valid.numel()))
    return out, torch.tensor(lengths, dtype=torch.int32)


def test_compaction_keeps_order_and_counts():
    g = torch.Generator().manual_seed(1)
    idx = torch.randint(0, 5000, (8, 2048), generator=g, dtype=torch.int32)
    # a DCP4 rank owns ~1/4 of the candidates; the rest are -1 in place
    mask = torch.rand(8, 2048, generator=g) < 0.25
    idx[~mask] = -1
    idx[3] = -1  # one row with nothing local
    idx[5, :] = torch.arange(2048, dtype=torch.int32)  # one row fully local
    out, lengths = SU["compact_dcp_candidates"](idx)
    ref_out, ref_len = _reference(idx)
    assert torch.equal(out, ref_out)
    assert torch.equal(lengths, ref_len)
    assert lengths[3] == 1 and out[3, 0] == -1  # empty row keeps one masked entry
    assert lengths[5] == 2048
    # every valid entry is in front of every -1 in every row
    for r in range(8):
        n = int(lengths[r])
        if int((idx[r] >= 0).sum()) > 0:
            assert (out[r, :n] >= 0).all() and (out[r, n:] == -1).all()


def test_non_power_of_two_width():
    idx = torch.full((3, 1500), -1, dtype=torch.int32)
    idx[0, ::3] = torch.arange(500, dtype=torch.int32)
    idx[1, 1499] = 7
    out, lengths = SU["compact_dcp_candidates"](idx)
    ref_out, ref_len = _reference(idx)
    assert torch.equal(out, ref_out) and torch.equal(lengths, ref_len)


def test_backend_threads_the_length_to_both_kernels():
    src = (OVERLAY / "v1/attention/backends/mla/flashmla_sparse.py").read_text()
    assert 'os.environ.get("GLM_DCP_COMPACT", "0")' in src
    assert "topk_indices, topk_length, empty_rows = compact_dcp_candidates(" in src
    assert "topk_indices, return_empty=True" in src
    assert src.count("topk_length=topk_length,") >= 3  # backend call, b12x helper, Triton fallback
    helper = (OVERLAY / "v1/attention/ops/deepseek_v4_ops/b12x_sparse_helpers.py").read_text()
    assert "DCP overlay: topk_length passthrough" in helper
    assert "topk_length=length_flat," in helper
