# SPDX-License-Identifier: Apache-2.0
"""The indexer's cross-rank top-k merge.

Claim under test: exchanging only each rank's *local* top-K candidates
reproduces the global top-K exactly. A token in the global top-K has at most
K-1 tokens outranking it globally, hence at most K-1 on its own rank, so it is
always in its owner's local top-K.
"""

import pytest
import torch

from harness import OVERLAY, extract, ref_local_seq_lens, ref_owner_and_local


class _FakeDCPGroup:
    """Two-phase stand-in for the NCCL DCP group.

    Phase 'collect' records what each rank would contribute; phase 'replay'
    hands every rank the concatenation, exactly as an all_gather would.
    """

    def __init__(self):
        self.mode = "collect"
        self.collected: list[torch.Tensor] = []
        self.world_size = 0

    def all_gather(self, tensor, dim):
        if self.mode == "collect":
            self.collected.append(tensor.clone())
            return torch.cat([tensor] * max(self.world_size, 1), dim=dim)
        return torch.cat(self.collected, dim=dim)


GROUP = _FakeDCPGroup()
MERGE = extract(
    OVERLAY / "model_executor/layers/sparse_attn_indexer.py",
    ["_pack_dcp_topk_candidates_kernel", "_merge_dcp_topk_global"],
    {"get_dcp_group": lambda: GROUP},
)


def local_view(logits_global, ke_global, rank, world, interleave):
    """This rank's logits row-major over its own shard, plus its causal bound."""
    rows, length = logits_global.shape
    owned = [
        p for p in range(length) if ref_owner_and_local(p, world, interleave)[0] == rank
    ]
    owned.sort(key=lambda p: ref_owner_and_local(p, world, interleave)[1])
    local_logits = logits_global[:, owned].contiguous()
    local_ke = [ref_local_seq_lens(int(k), rank, world, interleave) for k in ke_global]
    return local_logits, local_ke


def local_topk(local_logits, local_ke, k):
    """Emulates top_k_per_row_* : descending, -1 padded, causal-bounded."""
    rows = local_logits.shape[0]
    out = torch.full((rows, k), -1, dtype=torch.int32)
    for r in range(rows):
        n = min(local_ke[r], local_logits.shape[1])
        if n == 0:
            continue
        take = min(k, n)
        _, idx = torch.topk(local_logits[r, :n], take)
        out[r, :take] = idx.to(torch.int32)
    return out


def run_merge(logits_global, ke_global, k, world, interleave, row_starts=None):
    """Drive the real _merge_dcp_topk_global for every rank."""
    per_rank = []
    for rank in range(world):
        ll, lke = local_view(logits_global, ke_global, rank, world, interleave)
        per_rank.append((ll, local_topk(ll, lke, k)))

    results = []
    for phase in ("collect", "replay"):
        GROUP.mode = phase
        GROUP.world_size = world
        if phase == "collect":
            GROUP.collected = []
        results = []
        for rank in range(world):
            ll, ti = per_rank[rank]
            work = ti.clone()
            MERGE["_merge_dcp_topk_global"](
                ll if ll.numel() else None,
                work,
                k,
                rank,
                world,
                interleave,
                row_starts=row_starts,
            )
            results.append(work)
    return results


def reference_global_topk(logits_global, ke_global, k):
    rows = logits_global.shape[0]
    out = torch.full((rows, k), -1, dtype=torch.int32)
    for r in range(rows):
        n = int(ke_global[r])
        if n == 0:
            continue
        take = min(k, n)
        _, idx = torch.topk(logits_global[r, :n], take)
        out[r, :take] = idx.to(torch.int32)
    return out


def scores_of(logits_global, sel):
    """Sorted score multiset of a selection, so ties don't cause false failures."""
    rows = sel.shape[0]
    per_row = []
    for r in range(rows):
        vals = [
            float(logits_global[r, int(p)]) for p in sel[r].tolist() if int(p) >= 0
        ]
        per_row.append(sorted(vals, reverse=True))
    return per_row


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("interleave", [1, 2])
@pytest.mark.parametrize("k", [8, 32])
def test_merge_reproduces_global_topk(world, interleave, k):
    torch.manual_seed(world * 100 + k + interleave)
    rows, length = 5, 96
    logits = torch.randn(rows, length, dtype=torch.float32)
    ke = [length, length - 1, 40, k, 3]

    ref = reference_global_topk(logits, ke, k)
    merged = run_merge(logits, ke, k, world, interleave)

    for rank in range(world):
        assert torch.equal(merged[0], merged[rank]), "ranks disagree on the merge"
    assert scores_of(logits, merged[0]) == scores_of(logits, ref)


@pytest.mark.parametrize("world", [2, 4])
def test_merge_selects_positions_the_owner_can_serve(world):
    """Global ids must round-trip through the owner/local mapping."""
    torch.manual_seed(5)
    rows, length, k = 3, 64, 16
    logits = torch.randn(rows, length, dtype=torch.float32)
    ke = [length] * rows
    merged = run_merge(logits, ke, k, world, 1)[0]
    for r in range(rows):
        for p in merged[r].tolist():
            assert 0 <= p < length
        assert len(set(merged[r].tolist())) == k, "duplicate global ids"


@pytest.mark.parametrize("world", [4])
def test_short_context_leaves_high_ranks_empty(world):
    """A 2-token context lives entirely on ranks 0 and 1; 2 and 3 contribute
    nothing and must not corrupt the merge."""
    torch.manual_seed(9)
    rows, length, k = 2, 2, 8
    logits = torch.randn(rows, length, dtype=torch.float32)
    ke = [2, 1]
    merged = run_merge(logits, ke, k, world, 1)
    ref = reference_global_topk(logits, ke, k)
    for rank in range(world):
        assert torch.equal(merged[rank], merged[0])
    assert scores_of(logits, merged[0]) == scores_of(logits, ref)
    # Row 1 only sees position 0; everything else is padding.
    assert merged[0][1, 0] == 0
    assert (merged[0][1, 1:] == -1).all()


def test_padding_survives_as_negative_one():
    """Fewer than K candidates in total must leave -1 padding, not junk ids."""
    torch.manual_seed(2)
    rows, length, k = 2, 6, 16
    logits = torch.randn(rows, length, dtype=torch.float32)
    merged = run_merge(logits, [6, 6], k, 4, 1)[0]
    for r in range(rows):
        vals = merged[r].tolist()
        assert sorted(v for v in vals if v >= 0) == list(range(6))
        assert vals[6:] == [-1] * (k - 6)


def test_prefill_row_starts_offset_scores():
    """With row_starts the pack kernel must read scores at row_start + local."""
    torch.manual_seed(4)
    world, k = 2, 4
    # Two requests concatenated in one gathered-K buffer: request 0 occupies
    # columns [0, 8), request 1 columns [8, 16) on each rank.
    rows = 2
    local_len = 8
    ll = torch.randn(rows, 2 * local_len, dtype=torch.float32)
    row_starts = torch.tensor([0, local_len], dtype=torch.int32)

    # Rank 0 picks local positions 3 and 1 for both rows.
    ti = torch.tensor([[3, 1, -1, -1], [3, 1, -1, -1]], dtype=torch.int32)
    GROUP.mode = "collect"
    GROUP.world_size = 1
    GROUP.collected = []
    work = ti.clone()
    MERGE["_merge_dcp_topk_global"](ll, work, k, 0, world, 1, row_starts=row_starts)
    packed = GROUP.collected[0]
    # score for row 1, candidate 0 must come from column row_start(=8) + 3.
    assert packed[1, 0, 0] == pytest.approx(float(ll[1, local_len + 3]))
    assert packed[0, 0, 0] == pytest.approx(float(ll[0, 3]))
    # global id = local * world + rank
    assert packed[1, 0, 1] == pytest.approx(3.0 * world + 0)
