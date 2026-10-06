# SPDX-License-Identifier: Apache-2.0
"""DCP combine glue without layout copies (VLLM_DCP_GLUE=1; results/prefill-item2-20261006/REPORT.md, E1).

The stock ``cp_lse_ag_out_rs`` corrects the attention output in place ([B, H, D], token-major) and then
reduce-scatters it over heads. The reduce-scatter (vLLM's NCCL path and glm_roce's RoCE path alike) first
does ``movedim(0, 1).contiguous()``: a full copy of the output into head-major layout (~0.6 ms a layer at a
2048-token chunk). Here the correction kernel writes its result straight into a head-major buffer and hands
the reduce-scatter a [B, H, D] *view* of it, so that ``contiguous()`` is a no-op.

Bit-identical: the kernel body below is the stock ``_correct_attn_cp_out_kernel`` statement for statement;
only the store addresses differ (separate output strides). The reduce-scatter receives the same values.
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

_STOCK: dict = {}


@triton.jit
def _correct_attn_cp_out_hm_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    new_stride_B,
    new_stride_H,
    new_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    N_ROUNDED: tl.constexpr,
    IS_BASE_E: tl.constexpr,
):
    # Same arithmetic as vllm.v1.attention.ops.dcp._correct_attn_cp_out_kernel (vLLM 0.29.0).
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)
    num_n_offsets = tl.arange(0, N_ROUNDED)

    lse_offsets = (
        num_n_offsets * lses_stride_N
        + batch_idx * lses_stride_B
        + head_idx * lses_stride_H
    )

    lse = tl.load(lses_ptr + lse_offsets).to(tl.float32)
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
    lse -= lse_max
    if IS_BASE_E:
        lse_exp = tl.exp(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log(lse_acc)
    else:
        lse_exp = tl.exp2(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log2(lse_acc)
    lse += lse_max

    lse_offsets = batch_idx * lses_stride_B + head_idx * lses_stride_H
    tl.store(vlse_ptr + lse_offsets, lse)

    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )
    new_offsets = (
        batch_idx * new_stride_B
        + head_idx * new_stride_H
        + d_offsets * new_stride_D
    )

    lse_offset = (
        lse_idx * lses_stride_N + batch_idx * lses_stride_B + head_idx * lses_stride_H
    )
    lse_tmp = tl.load(lses_ptr + lse_offset).to(tl.float32)
    lse_finally = lse_tmp - lse
    lse_finally = tl.where(
        (lse_finally != lse_finally) | (lse_finally == float("inf")),
        -float("inf"),
        lse_finally,
    )
    factor = tl.exp(lse_finally) if IS_BASE_E else tl.exp2(lse_finally)
    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor
    output = tl.where(factor == 0.0, 0.0, output)

    tl.store(new_output_ptr + new_offsets, output)


def correct_attn_out_head_major(out: torch.Tensor, lses: torch.Tensor, cp_rank: int,
                                is_lse_base_on_e: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """``dcp.correct_attn_out`` with the corrected output in a fresh head-major buffer.

    Returns ``(new, lse)`` where ``new`` is a [B, H, D] view of a contiguous [H, B, D] tensor.
    """
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    assert out.ndim == 3, f"expected out [B,H,D] or [B,1,H,D], got {tuple(out.shape)}"
    if lses.ndim == 4 and lses.shape[-1] == 1:
        lses = lses.squeeze(-1)
    if lses.ndim == 4 and lses.shape[1] == 1:
        lses = lses.squeeze(1)
    assert lses.ndim == 3, f"expected lses [N,B,H], got {tuple(lses.shape)}"
    B, H, D = out.shape
    N = lses.shape[0]
    o_sB, o_sH, o_sD = out.stride()
    l_sN, l_sB, l_sH = lses.stride()
    head_major = torch.empty((H, B, D), dtype=out.dtype, device=out.device)
    new = head_major.permute(1, 0, 2)
    n_sB, n_sH, n_sD = new.stride()
    lse = torch.empty_strided((B, H), (l_sB, l_sH), device=lses.device, dtype=lses.dtype)
    _correct_attn_cp_out_hm_kernel[(B, H, 1)](
        out, new, lses, lse,
        o_sB, o_sH, o_sD, n_sB, n_sH, n_sD, l_sN, l_sB, l_sH, cp_rank,
        HEAD_DIM=D, N_ROUNDED=N, IS_BASE_E=is_lse_base_on_e,
    )
    return new, lse


def make_cp_lse_ag_out_rs(dcp_module):
    stock = dcp_module.cp_lse_ag_out_rs

    def cp_lse_ag_out_rs(cp_attn_out, cp_attn_lse, cp_group, ctx=None, return_lse=False,
                         is_lse_base_on_e=True, seq_lens=None, query_start_loc=None):
        if cp_group.world_size == 1:
            return stock(cp_attn_out, cp_attn_lse, cp_group, ctx=ctx, return_lse=return_lse,
                         is_lse_base_on_e=is_lse_base_on_e, seq_lens=seq_lens,
                         query_start_loc=query_start_loc)
        # = dcp._cp_lse_common, with the head-major correction.
        cp_attn_lse = cp_attn_lse.contiguous()
        dcp_module.mask_dcp_empty_shards_(cp_attn_lse, seq_lens, query_start_loc)
        lses = cp_group.all_gather(cp_attn_lse, dim=0).reshape(
            (cp_group.world_size,) + cp_attn_lse.shape
        )
        out, lse = correct_attn_out_head_major(cp_attn_out, lses, cp_group.rank_in_group,
                                               is_lse_base_on_e=is_lse_base_on_e)
        out = cp_group.reduce_scatter(out, dim=1)
        if return_lse:
            cp_num_heads = lse.shape[1] // cp_group.world_size
            cp_rank = cp_group.rank_in_group
            lse = lse[:, cp_num_heads * cp_rank : cp_num_heads * (cp_rank + 1)]
            return out, lse
        return out

    cp_lse_ag_out_rs.__wrapped__ = stock
    cp_lse_ag_out_rs._glm_dcp_glue = True
    return cp_lse_ag_out_rs


def install(module) -> None:
    """Post-import patch of ``vllm.v1.attention.ops.dcp`` (before any MLADCPManager binds the combine)."""
    if getattr(module.cp_lse_ag_out_rs, "_glm_dcp_glue", False):
        return
    _STOCK["cp_lse_ag_out_rs"] = module.cp_lse_ag_out_rs
    module.cp_lse_ag_out_rs = make_cp_lse_ag_out_rs(module)
