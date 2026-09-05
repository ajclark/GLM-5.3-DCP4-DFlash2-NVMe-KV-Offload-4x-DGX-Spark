# SPDX-License-Identifier: Apache-2.0
"""Prefill chunk metadata: per-query causal bounds localized to a DCP rank."""

import pytest
import torch

from harness import BASELINE, OVERLAY, extract, ref_local_seq_lens

K = "_build_prefill_chunk_metadata_kernel"
PATCHED = extract(OVERLAY / "v1/attention/backends/mla/indexer.py", [K])[K]
BASE = extract(BASELINE / "v1/attention/backends/mla/indexer.py", [K])[K]


def build(query_lens, seq_lens, world, rank, interleave, compress_ratio=1):
    num_reqs = len(query_lens)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.tensor(query_lens, dtype=torch.int32).cumsum(0)
    total_query = int(query_start_loc[-1])

    seq = torch.tensor(seq_lens, dtype=torch.int32)
    global_cu = torch.zeros(num_reqs + 1, dtype=torch.int32)
    global_cu[1:] = seq.cumsum(0)

    local_lens = torch.tensor(
        [ref_local_seq_lens(int(s), rank, world, interleave) for s in seq],
        dtype=torch.int32,
    )
    local_cu = torch.zeros(num_reqs + 1, dtype=torch.int32)
    local_cu[1:] = local_lens.cumsum(0)

    token_to_seq = torch.zeros(int(global_cu[-1]), dtype=torch.int32)
    ks = torch.zeros(total_query, dtype=torch.int32)
    ke = torch.zeros(total_query, dtype=torch.int32)

    PATCHED[(num_reqs,)](
        query_start_loc,
        seq,
        global_cu,
        local_cu if world > 1 else global_cu,
        token_to_seq,
        ks,
        ke,
        0,
        total_query,
        DCP_RANK=rank,
        DCP_WORLD=world,
        DCP_INTERLEAVE=interleave,
        BLOCK_SIZE=1024,
        COMPRESS_RATIO=compress_ratio,
    )
    return ks, ke, local_cu, token_to_seq


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("interleave", [1, 2])
def test_bounds_count_exactly_this_ranks_causal_tokens(world, interleave):
    query_lens = [3, 1, 5]
    seq_lens = [17, 9, 40]
    for rank in range(world):
        ks, ke, local_cu, _ = build(query_lens, seq_lens, world, rank, interleave)
        t = 0
        for i, (q, L) in enumerate(zip(query_lens, seq_lens)):
            for j in range(q):
                global_ctx = L - q + j + 1
                expected = ref_local_seq_lens(global_ctx, rank, world, interleave)
                assert int(ks[t]) == int(local_cu[i]), (rank, i, j)
                assert int(ke[t]) - int(ks[t]) == expected, (rank, i, j)
                t += 1


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("interleave", [1, 2])
def test_ranks_partition_the_context(world, interleave):
    """Summed over ranks, the local causal lengths must equal the global one."""
    query_lens = [4, 2]
    seq_lens = [23, 11]
    totals = None
    for rank in range(world):
        ks, ke, _, _ = build(query_lens, seq_lens, world, rank, interleave)
        lens = (ke - ks).long()
        totals = lens if totals is None else totals + lens
    expected = []
    for q, L in zip(query_lens, seq_lens):
        expected.extend(L - q + j + 1 for j in range(q))
    assert totals.tolist() == expected


def test_dcp1_matches_the_baseline_kernel():
    query_lens = [3, 1, 5]
    seq_lens = [17, 9, 40]
    ks, ke, _, token_to_seq = build(query_lens, seq_lens, 1, 0, 1)

    num_reqs = len(query_lens)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.tensor(query_lens, dtype=torch.int32).cumsum(0)
    seq = torch.tensor(seq_lens, dtype=torch.int32)
    cu = torch.zeros(num_reqs + 1, dtype=torch.int32)
    cu[1:] = seq.cumsum(0)
    total_query = int(query_start_loc[-1])
    b_tts = torch.zeros(int(cu[-1]), dtype=torch.int32)
    b_ks = torch.zeros(total_query, dtype=torch.int32)
    b_ke = torch.zeros(total_query, dtype=torch.int32)
    BASE[(num_reqs,)](
        query_start_loc,
        seq,
        cu,
        b_tts,
        b_ks,
        b_ke,
        0,
        total_query,
        BLOCK_SIZE=1024,
        COMPRESS_RATIO=1,
    )
    assert torch.equal(ks, b_ks)
    assert torch.equal(ke, b_ke)
    assert torch.equal(token_to_seq, b_tts)


def test_token_to_seq_stays_global():
    """token_to_seq indexes the global gathered layout; DCP must not move it."""
    query_lens = [2, 2]
    seq_lens = [6, 4]
    _, _, _, tts = build(query_lens, seq_lens, 4, 2, 1)
    assert tts.numel() == 10
    assert tts.tolist() == [0] * 6 + [1] * 4
