# SPDX-License-Identifier: Apache-2.0
"""The top-k index filter: global positions -> this rank's physical slots."""

import pytest
import torch

from harness import (
    BASELINE,
    OVERLAY,
    extract,
    ref_owner_and_local,
    ref_physical_slot,
)

NAMES = [
    "_convert_req_index_to_global_index_kernel",
    "triton_convert_req_index_to_global_index",
    "triton_filter_and_convert_dcp_index",
]
SU = extract(OVERLAY / "v1/attention/backends/mla/sparse_utils.py", NAMES)
SU_BASE = extract(
    BASELINE / "v1/attention/backends/mla/sparse_utils.py",
    NAMES[:2],
)


def make_inputs(num_reqs, tokens_per_req, num_topk, max_blocks, block_size, seed=0):
    g = torch.Generator().manual_seed(seed)
    num_tokens = num_reqs * tokens_per_req
    req_id = torch.arange(num_reqs, dtype=torch.int32).repeat_interleave(tokens_per_req)
    # Distinct, shuffled physical blocks so a wrong index cannot accidentally
    # land on the right slot.
    block_table = torch.stack(
        [
            torch.randperm(64, generator=g)[:max_blocks].to(torch.int32)
            for _ in range(num_reqs)
        ]
    )
    limit = max_blocks * block_size
    token_indices = torch.randint(
        -1, limit, (num_tokens, num_topk), generator=g, dtype=torch.int32
    )
    return req_id, block_table, token_indices


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("interleave", [1, 2])
def test_filter_matches_slot_mapping(world, interleave):
    """Every kept index must be exactly the slot the KV writer used."""
    block_size = 8
    num_reqs, tokens_per_req, num_topk, max_blocks = 3, 2, 128, 6
    req_id, block_table, token_indices = make_inputs(
        num_reqs, tokens_per_req, num_topk, max_blocks, block_size, seed=world * 10
    )
    # Global positions span the whole virtual context: block_size * world
    # tokens per block row.
    virtual_limit = max_blocks * block_size * world
    g = torch.Generator().manual_seed(7)
    token_indices = torch.randint(
        -1, virtual_limit, token_indices.shape, generator=g, dtype=torch.int32
    )

    for rank in range(world):
        out = SU["triton_filter_and_convert_dcp_index"](
            req_id,
            block_table,
            token_indices,
            dcp_size=world,
            dcp_rank=rank,
            cp_kv_cache_interleave_size=interleave,
            BLOCK_SIZE=block_size,
            NUM_TOPK_TOKENS=num_topk,
            BLOCK_N=64,
        )
        for t in range(token_indices.shape[0]):
            row = block_table[req_id[t]]
            for j in range(num_topk):
                pos = int(token_indices[t, j])
                if pos < 0:
                    assert out[t, j] == -1
                    continue
                _, local = ref_owner_and_local(pos, world, interleave)
                if local // block_size >= max_blocks:
                    assert out[t, j] == -1  # out of range guard
                    continue
                expected = ref_physical_slot(
                    row, pos, block_size, world, interleave, rank
                )
                assert int(out[t, j]) == (-1 if expected is None else expected), (
                    rank,
                    t,
                    j,
                    pos,
                )


@pytest.mark.parametrize("world", [2, 4])
def test_every_valid_position_kept_exactly_once(world):
    """Across the group each in-range global position survives on one rank."""
    block_size = 8
    max_blocks = 4
    req_id = torch.zeros(1, dtype=torch.int32)
    block_table = torch.arange(max_blocks, dtype=torch.int32).reshape(1, max_blocks)
    positions = torch.arange(
        max_blocks * block_size * world, dtype=torch.int32
    ).reshape(1, -1)
    num_topk = positions.shape[1]

    kept = torch.zeros(num_topk, dtype=torch.int64)
    for rank in range(world):
        out = SU["triton_filter_and_convert_dcp_index"](
            req_id,
            block_table,
            positions,
            dcp_size=world,
            dcp_rank=rank,
            BLOCK_SIZE=block_size,
            NUM_TOPK_TOKENS=num_topk,
            BLOCK_N=num_topk,
        )
        kept += (out[0] >= 0).long()
    assert torch.equal(kept, torch.ones(num_topk, dtype=torch.int64))


def test_dcp1_is_bit_identical_to_baseline():
    """The patched kernel with DCP_SIZE=1 must not move a single index."""
    block_size = 8
    req_id, block_table, token_indices = make_inputs(4, 3, 256, 6, block_size, seed=3)

    base = SU_BASE["triton_convert_req_index_to_global_index"](
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=256,
        BLOCK_N=128,
    )
    patched = SU["triton_convert_req_index_to_global_index"](
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=256,
        BLOCK_N=128,
    )
    via_dcp1 = SU["triton_filter_and_convert_dcp_index"](
        req_id,
        block_table,
        token_indices,
        dcp_size=1,
        dcp_rank=0,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=256,
        BLOCK_N=128,
    )
    assert torch.equal(base, patched)
    assert torch.equal(base, via_dcp1)


def test_valid_counts_agree_with_output():
    block_size = 8
    req_id, block_table, token_indices = make_inputs(2, 2, 128, 4, block_size, seed=11)
    out, counts = SU["triton_filter_and_convert_dcp_index"](
        req_id,
        block_table,
        token_indices,
        dcp_size=2,
        dcp_rank=1,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=128,
        BLOCK_N=64,
        return_valid_counts=True,
    )
    assert torch.equal(counts.long(), (out >= 0).sum(dim=1))
