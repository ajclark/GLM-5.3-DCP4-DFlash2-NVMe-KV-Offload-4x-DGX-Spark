"""Run the real V2 table and slot kernels against CPU allocations.

Only buffer allocation/staging is replaced. Addressing, request permutation,
padding, and host bounds validation execute the patched production source.
"""
from collections.abc import Iterable

import pytest
import torch

from harness import OVERLAY, cdiv, extract, ref_owner_and_local, tl, triton


# Unchanged pointer primitive from vLLM v1/worker/gpu/buffer_utils.py.
@triton.jit
def _load_ptr(ptr_to_ptr, elem_dtype):
    ptr = tl.load(ptr_to_ptr)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)


class Staged:
    def __init__(self, size, dtype, device):
        self.gpu = torch.zeros(size, dtype=dtype)
        self.pending = []

    def stage_write(self, row, start, values):
        self.pending.append((row, start, list(values)))

    def apply_write(self):
        for row, start, values in self.pending:
            self.gpu[row, start:start + len(values)] = torch.tensor(values)
        self.pending.clear()


class Uva:
    def __init__(self, size, dtype):
        self.gpu = torch.zeros(size, dtype=dtype)
        self.np = self.gpu.numpy()

    def copy_to_uva(self):
        pass


class Writer:
    def __init__(self, *args):
        pass

    def apply(self, tables, *args):
        for table in tables:
            table.apply_write()


def tables(cp=2, rank=0, interleave=1, block_size=64, kernel_block_size=64,
           maxlen=180224, cp_sizes=None):
    cls = extract(OVERLAY / 'v1/worker/gpu/block_table.py', [
        '_gather_block_tables_kernel', '_compute_slot_mappings_kernel', 'BlockTables'], {
        'Iterable': Iterable, '_load_ptr': _load_ptr, 'PAD_SLOT_ID': -1,
        'StagedWriteTensor': Staged, 'UvaBackedTensor': Uva,
        'FusedStagedWriter': Writer,
    })['BlockTables']
    sizes = [cp, 1] if cp_sizes is None else cp_sizes
    return cls([block_size] * 2, 3, 128,
               [cdiv(maxlen, block_size * size) for size in sizes],
               torch.device('cpu'), [kernel_block_size] * 2,
               cp_size=cp, cp_rank=rank, cp_interleave=interleave, cp_sizes=sizes)


@pytest.mark.parametrize('cp,rank', [(1, 0), (2, 0), (2, 1), (4, 0), (4, 1), (4, 3)])
@pytest.mark.parametrize('interleave', [1, 4])
@pytest.mark.parametrize('kernel_block_size', [16, 64])
def test_long_slots_match_independent_owner_reference(cp, rank, interleave, kernel_block_size):
    bt = tables(cp, rank, interleave, kernel_block_size=kernel_block_size)
    for row in (0, 2):
        groups = []
        for size in bt.cp_sizes:
            groups.append([500 + row * 4000 + 3 * i for i in range(cdiv(180224, 64 * size))])
        bt.append_block_ids(row, tuple(groups), overwrite=True)
    bt.apply_staged_writes()
    positions = list(range(90108, 90120)) + list(range(170120, 170132))
    mapping = torch.tensor([2, 0], dtype=torch.int32)
    actual = bt.compute_slot_mappings(mapping, torch.tensor([0, 12, 24], dtype=torch.int32),
                                     torch.tensor(positions, dtype=torch.int64), 32)
    for group, size in enumerate(bt.cp_sizes):
        for i, pos in enumerate(positions):
            owner, local = ref_owner_and_local(pos, size, interleave)
            expected = -1
            if owner == (rank if size > 1 else 0):
                row = 2 if i < 12 else 0
                physical = int(bt.block_tables[group].gpu[row, local // kernel_block_size])
                expected = physical * kernel_block_size + local % kernel_block_size
            assert actual[group, i].item() == expected
        assert actual[group, 24:].tolist() == [-1] * 8
        assert bt.slot_mappings[group, 32:].tolist() == [-1] * 96


def test_large_physical_slots_and_invalid_logical_positions():
    bt = tables(cp=1, maxlen=256)
    bt.append_block_ids(0, ([1 << 25] * 4, [1 << 25] * 4), overwrite=True)
    bt.apply_staged_writes()
    slots = bt.compute_slot_mappings(torch.tensor([0], dtype=torch.int32),
        torch.tensor([0, 4], dtype=torch.int32), torch.tensor([-1, 0, 255, 256]), 4)
    assert slots.tolist() == [[-1, 1 << 31, (1 << 31) + 63, -1]] * 2


def test_gather_permutation_padding_and_malformed_count_stay_inside_rows():
    bt = tables(maxlen=512)
    for i, table in enumerate(bt.block_tables):
        table.gpu.copy_(torch.arange(table.gpu.numel()).reshape(table.gpu.shape) + 100 * (i + 1))
        bt.num_blocks.np[i, :] = table.gpu.shape[1]
        # Deliberately invalid metadata remains within bounded CPU allocations.
        bt.num_blocks.np[i, 2] = table.gpu.shape[1] + 3
        bt.input_block_tables[i].fill_(-999)
    result = bt.gather_block_tables(torch.tensor([2, 0], dtype=torch.int32), 3)
    for group, gathered in enumerate(result):
        torch.testing.assert_close(gathered[0], bt.block_tables[group].gpu[2])
        torch.testing.assert_close(gathered[1], bt.block_tables[group].gpu[0])
        assert gathered[2].tolist() == [0] * gathered.shape[1]


def test_append_rejects_entire_update_before_staging_any_group():
    bt = tables(maxlen=256, kernel_block_size=16)
    with pytest.raises(ValueError, match='overflow'):
        bt.append_block_ids(0, ([2], [3] * 5), overwrite=True)
    assert all(not table.pending for table in bt.block_tables)
    assert not bt.num_blocks.gpu.any()
    bt.append_block_ids(0, ([2], [3, 4]), overwrite=True)
    bt.append_block_ids(0, ([5], [6, 7]), overwrite=False)
    bt.apply_staged_writes()
    assert bt.block_tables[1].gpu[0].tolist() == list(range(12, 20)) + list(range(24, 32))
    previous = bt.num_blocks.gpu.clone()
    with pytest.raises(ValueError, match='overflow'):
        bt.append_block_ids(0, ([], [8]), overwrite=False)
    torch.testing.assert_close(bt.num_blocks.gpu, previous)
    bt.append_block_ids(0, ([9], [10]), overwrite=True)
    assert bt.num_blocks.gpu[:, 0].tolist() == [4, 4]


@pytest.mark.parametrize('row,groups', [(-1, ([], [])), (3, ([], [])), (0, ([],))])
def test_invalid_request_and_group_count(row, groups):
    bt = tables()
    with pytest.raises(ValueError, match='invalid request'):
        bt.append_block_ids(row, groups, overwrite=True)
    assert all(not table.pending for table in bt.block_tables)


def test_layout_reinitialization_restores_per_group_geometry():
    bt = tables()
    assert [t.gpu.shape[1] for t in bt.block_tables] == [1408, 2816]
    bt.cp_sizes_tensor.zero_()
    bt.init_block_table_layout_tensors()
    assert bt.cp_sizes_tensor.tolist() == [2, 1]
    assert bt.block_table_ptrs.tolist() == [t.gpu.data_ptr() for t in bt.block_tables]


def test_invalid_cp_size_rejected():
    with pytest.raises(ValueError, match='KV group CP sizes'):
        tables(cp=4, cp_sizes=[4, 2])
