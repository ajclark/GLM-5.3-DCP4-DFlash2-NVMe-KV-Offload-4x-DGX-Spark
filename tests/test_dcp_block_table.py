# SPDX-License-Identifier: Apache-2.0
"""The patched block tables: per-group CP world size.

Drives the real `BlockTable` / `MultiGroupBlockTable` classes and the real
slot-mapping Triton kernel from the overlay, with the process DCP group faked
at world size 4. Group 0 (the target's full-attention cache) must be
token-sharded; group 1 (the DFlash drafter's sliding-window cache, CP size 1)
must map every position on every rank and size its table for the full
context.
"""

import numpy as np
import pytest
import torch

from harness import OVERLAY, cdiv, extract, ref_owner_and_local

WORLD = 4


class _FakeGroup:
    def __init__(self, world, rank):
        self.world_size = world
        self.rank_in_group = rank


class _CpuGpuBuffer:
    """Just enough of vllm.v1.utils.CpuGpuBuffer for the block table."""

    def __init__(self, *size, dtype, device, pin_memory):
        self.cpu = torch.zeros(*size, dtype=dtype)
        self.gpu = torch.zeros(*size, dtype=dtype)
        self.np = self.cpu.numpy()

    def copy_to_gpu(self, n=None):
        if n is None:
            self.gpu.copy_(self.cpu)
        else:
            self.gpu[:n].copy_(self.cpu[:n])
        return self.gpu


def _load(rank):
    return extract(
        OVERLAY / "v1/worker/block_table.py",
        ["_compute_slot_mapping_kernel", "BlockTable", "MultiGroupBlockTable"],
        {
            "np": np,
            "get_dcp_group": lambda: _FakeGroup(WORLD, rank),
            "get_pcp_group": lambda: _FakeGroup(1, 0),
            "get_total_cp_world_size": lambda: WORLD,
            "CpuGpuBuffer": _CpuGpuBuffer,
            "PAD_SLOT_ID": -1,
            "logger": None,
        },
    )


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_replicated_group_maps_every_position_sharded_group_a_quarter(rank):
    ns = _load(rank)
    block_size, max_len, max_reqs, max_tokens = 8, 512, 2, 64
    mg = ns["MultiGroupBlockTable"](
        max_num_reqs=max_reqs,
        max_model_len=max_len,
        max_num_batched_tokens=max_tokens,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[block_size, block_size],
        kernel_block_sizes=[block_size, block_size],
        cp_world_sizes=[WORLD, 1],
    )
    sharded, replicated = mg.block_tables

    # Table geometry: sharded rows cover max_len // WORLD positions.
    assert sharded.max_num_blocks_per_req == cdiv(max_len, block_size * WORLD)
    assert replicated.max_num_blocks_per_req == cdiv(max_len, block_size)
    assert sharded.dcp_world_size == WORLD and sharded.dcp_rank == rank
    assert replicated.dcp_world_size == 1 and replicated.dcp_rank == 0

    # One request holding blocks [5, 6, 7, ...]; 40 consecutive positions.
    num_pos = 40
    for bt in (sharded, replicated):
        bt.add_row(list(range(5, 5 + bt.max_num_blocks_per_req)), 0)
        bt.commit_block_table(1)
    positions = torch.arange(num_pos, dtype=torch.int64)
    qsl = torch.tensor([0, num_pos], dtype=torch.int32)
    sharded.compute_slot_mapping(1, qsl, positions)
    replicated.compute_slot_mapping(1, qsl, positions)
    s_slots = sharded.slot_mapping.gpu[:num_pos].tolist()
    r_slots = replicated.slot_mapping.gpu[:num_pos].tolist()

    # Replicated: every position has a slot, at block[pos // bs] * bs + pos % bs.
    for pos, slot in enumerate(r_slots):
        assert slot == (5 + pos // block_size) * block_size + pos % block_size

    # Sharded: exactly the positions this rank owns, at the DCP-local slot.
    for pos, slot in enumerate(s_slots):
        owner, local = ref_owner_and_local(pos, WORLD, 1)
        if owner != rank:
            assert slot == -1, (pos, slot)
        else:
            assert slot == (5 + local // block_size) * block_size + local % block_size
    assert sum(s == -1 for s in s_slots) == num_pos - sum(
        ref_owner_and_local(p, WORLD, 1)[0] == rank for p in range(num_pos)
    )


def test_default_cp_world_sizes_is_the_old_behaviour():
    """Without cp_world_sizes every group takes the process CP size."""
    ns = _load(rank=2)
    mg = ns["MultiGroupBlockTable"](
        max_num_reqs=1,
        max_model_len=1024,
        max_num_batched_tokens=16,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[16, 16],
        kernel_block_sizes=[16, 16],
    )
    for bt in mg.block_tables:
        assert bt.dcp_world_size == WORLD
        assert bt.max_num_blocks_per_req == cdiv(1024, 16 * WORLD)


def test_only_one_or_process_cp_size_is_accepted():
    ns = _load(rank=0)
    with pytest.raises(AssertionError):
        ns["BlockTable"](8, 1, 4, 8, False, torch.device("cpu"), 8, 1, cp_world_size=2)
