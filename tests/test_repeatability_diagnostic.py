"""Real sparse kernels' causal contracts, using the CPU Triton interpreter.

These tests establish what the kernels do with supplied metadata. They do
not reproduce the live target-logit variation or GPU arithmetic ordering.
"""

import numpy as np
import pytest
import torch

from harness import ROOT, extract, tl
# Import after harness has enabled TRITON_INTERPRET, including tl standard ops.
from triton.runtime import interpreter


STAGE = ROOT / "stage/glm-triton"
INDEXER = extract(STAGE / "sm12x_mqa.py", ["_fp8_paged_mqa_logits_rowwise_kernel"])[
    "_fp8_paged_mqa_logits_rowwise_kernel"
]
ATTEND = extract(
    STAGE / "sm12x_sparse_mla_attn.py", ["_fused_gather_dequant_attn_kernel"]
)["_fused_gather_dequant_attn_kernel"]


def _index_case(start):
    # The requested 48-token chunk is backed by three permuted physical
    # pages. Earlier logical pages are outside this launch and unallocated.
    # Values use FP32 storage to isolate address/mask semantics from FP8
    # conversion; the actual kernel casts both inputs to FP32 before dot.
    q = torch.full((1, 8, 32, 128), 1 / 128, dtype=torch.float32)
    kv = torch.empty((3, 16, 128), dtype=torch.float32)
    scale = torch.ones((3, 16), dtype=torch.float32)
    weights = torch.full((8, 32), 1 / 32, dtype=torch.float32)
    table = torch.full((1, start // 16 + 3), -1, dtype=torch.int32)
    table[0, start // 16:] = torch.tensor([2, 0, 1])
    for offset in range(48):
        page = int(table[0, (start + offset) // 16])
        kv[page, offset % 16] = offset + 1
    bounds = torch.arange(start + 3, start + 11, dtype=torch.int32)[None]
    return q, kv, scale, weights, bounds, table


def _index(case, start):
    q, kv, scale, weights, bounds, table = case
    out = torch.full((8, 48), float("nan"))
    INDEXER[(8, 3)](
        q, kv, scale, weights, bounds, table, out,
        start, 8, 48, 8, 32, 128, 16,
        *q.stride(), *kv.stride(), *scale.stride(), *weights.stride(),
        *bounds.stride(), *table.stride(), *out.stride(),
        BLOCK_N=16, BLOCK_D=64, BLOCK_H=8,
    )
    return out


@pytest.mark.parametrize("start", [0, 100096, 170016])
@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
def test_actual_indexer_masks_future_cache_and_entire_future_tiles(start, poison):
    case = _index_case(start)
    before = _index(case, start)
    _, kv, scale, _, _, table = case
    # No query is allowed to inspect offsets >= 10. Poison both key and
    # scale storage, including the entirely future second/third tiles.
    for offset in range(10, 48):
        page = int(table[0, (start + offset) // 16])
        kv[page, offset % 16] = poison
        scale[page, offset % 16] = poison
    after = _index(case, start)
    assert torch.equal(before, after)
    for row in range(8):
        bound = row + 3
        assert torch.equal(after[row, :bound], torch.arange(1, bound + 1).float())
        assert torch.isneginf(after[row, bound:]).all()


@pytest.mark.parametrize("start", [0, 100096, 170016])
def test_actual_indexer_a_later_token_changes_only_rows_allowed_to_see_it(start):
    case = _index_case(start)
    before = _index(case, start)
    page = int(case[-1][0, (start + 5) // 16])
    case[1][page, 5] = 80
    after = _index(case, start)
    assert torch.equal(before[:3], after[:3])
    assert torch.equal(after[3:, 5], torch.full((5,), 80.0))
    mask = torch.ones_like(before, dtype=torch.bool)
    mask[3:, 5] = False
    assert torch.equal(before[mask], after[mask])


def test_actual_indexer_query_rows_do_not_mix():
    case = _index_case(100096)
    before = _index(case, 100096)
    case[0][0, 4] *= 2
    after = _index(case, 100096)
    assert torch.equal(before[:4], after[:4])
    assert torch.equal(before[5:], after[5:])
    assert torch.equal(after[4, :7], 2 * before[4, :7])
    assert torch.isneginf(after[4, 7:]).all()


def test_actual_indexer_cannot_correct_an_incorrectly_broadcast_final_bound():
    case = _index_case(100096)
    correct = _index(case, 100096)
    case[4].fill_(100106)
    wrong = _index(case, 100096)
    assert torch.isneginf(correct[0, 3:10]).all()
    assert torch.isfinite(wrong[0, 3:10]).all()
    # Positive control: a metadata fault really can expose future tokens.
    # This intentionally wrong input is not evidence of a live fault.
    assert torch.equal(correct[-1], wrong[-1])


def _attend(cache, indices, length):
    q = torch.zeros((1, 16, 576), dtype=torch.bfloat16)
    indices = torch.tensor([indices], dtype=torch.int32)
    lens = torch.tensor([length], dtype=torch.int32)
    maximum = torch.full((1, 16), -float("inf"))
    denom = torch.zeros((1, 16))
    acc = torch.zeros((1, 16, 576))
    ATTEND[(1, 1)](
        q, cache, indices, lens, maximum, denom, acc,
        *q.stride(), cache.stride(0), *indices.stride(),
        *maximum.stride(), *acc.stride(), len(cache), 16,
        512, 64, 128, 512, 528, indices.shape[1], 1.0,
        HEAD_BLOCK=16, BLOCK_N=16, BLOCK_NOPE=512, BLOCK_ROPE=64,
    )
    return acc[..., :512] / denom[..., None], denom


def _packed_cache():
    # Actual fp8_ds_mla layout: 512 FP8 bytes, four FP32 scales, 64 BF16
    # RoPE values. FP8 e4m3 encodings 0x38/0x40/0x44 are exactly 1/2/3.
    cache = torch.zeros((3, 656), dtype=torch.uint8)
    cache[:, :512] = torch.tensor([0x38, 0x40, 0x44], dtype=torch.uint8)[:, None]
    cache[:, 512:528] = torch.ones((3, 4)).view(torch.uint8)
    return cache


@pytest.fixture
def bf16_interpreter_dot(monkeypatch):
    # This installed interpreter stores BF16 as uint16 and create_dot only
    # decodes FP8, so an unadapted BF16 dot multiplies the storage bits.
    # Decode BF16 operands for NumPy's FP32 dot. This is a CPU semantics
    # adapter, not a GPU reduction-order or tensor-core accuracy simulation.
    original = interpreter.InterpreterBuilder.create_dot

    def dot(builder, a, b, d, input_precision, max_num_imprecise_acc):
        def decode(value):
            if value.dtype == tl.bfloat16:
                data = (value.data.astype(np.uint32) << 16).view(np.float32)
                return interpreter.TensorHandle(data, tl.float32)
            return value
        return original(builder, decode(a), decode(b), d,
                        input_precision, max_num_imprecise_acc)

    monkeypatch.setattr(interpreter.InterpreterBuilder, "create_dot", dot)


def test_actual_sparse_consumer_trusts_supplied_candidates_for_causality(bf16_interpreter_dot):
    cache = _packed_cache()
    only_past, _ = _attend(cache, [0, 2, -1, 999], 1)
    includes_future, _ = _attend(cache, [0, 2, -1, 999], 2)
    assert torch.equal(only_past, torch.ones_like(only_past))
    assert torch.equal(includes_future, torch.full_like(includes_future, 2.0))
    # The consumer has no absolute query position argument. Slot 2 is
    # intentionally designated "future" in this test; it is valid physical
    # storage and is attended whenever upstream includes it within length.


def test_actual_sparse_consumer_masks_sentinel_oob_and_length(bf16_interpreter_dot):
    cache = _packed_cache()
    expected, _ = _attend(cache, [0], 1)
    for indices, length in [([0, -1, 999], 3), ([0, 2], 1)]:
        actual, denom = _attend(cache, indices, length)
        assert torch.equal(actual, expected)
        assert torch.equal(denom, torch.ones_like(denom))
