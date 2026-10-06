# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from .utils import maybe_prefix


# Drafter-diet overlay: an fp8 copy of the (shared, BF16) target lm_head for the drafter's
# candidate top-k (VLLM_DFLASH_HEAD_FP8=1). Port of knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4
# overlay/glm_ds_draft.py (MIT; from their DeepSeek-V4.1 draft_head_fp8.py): e4m3 weights with
# one power-of-two exponent per 32x32 block and a Triton GEMM, so the draft reads half the
# bytes. Only draft proposals change; the target's logits and every committed token do not.
_HEAD_FP8 = os.environ.get("VLLM_DFLASH_HEAD_FP8", "0") == "1"
_HEAD_FP8_MAX_M = 128
_HEAD_TILES = ((16, 16, 256, 4, 3), (64, 64, 128, 4, 3), (128, 128, 64, 8, 3))  # max M, BM, BK, warps, stages
_HEAD_KERNEL = None

# Rank-consistent drafter (VLLM_DFLASH_DET_CONV, default on). The conv kernel_projection is a
# ReplicatedLinear whose output enters the drafter's residual stream with no all-reduce after
# it. In the int8 drafter it is a Marlin W8A16 GEMM with n=1536 < 2048 and k=6144 >= 2048, so
# VLLM_MARLIN_USE_ATOMIC_ADD=1 picks Marlin's atomic-add split-K, whose summation order varies
# from run to run: every TP rank would then carry a slightly different residual, and on a
# near-tie the candidate selector could draft different tokens on different ranks. Running
# these GEMMs with Marlin's deterministic global reduce keeps every rank bit-identical, as
# the BF16 drafter (cuBLAS) was. Found by the step-4 review (results/step4-hostgap-20261001).
_DET_CONV = os.environ.get("VLLM_DFLASH_DET_CONV", "1") == "1"


def _marlin_scheme(linear: nn.Module):
    """The compressed-tensors scheme of `linear` when it runs on the Marlin kernel, else None."""
    scheme = getattr(linear, "scheme", None)
    kernel = getattr(scheme, "kernel", None)
    if kernel is None or type(kernel).__name__ != "MarlinLinearKernel":
        return None
    if getattr(kernel.config, "act_type", None) in (torch.int8, torch.float8_e4m3fn):
        return None
    return scheme


def _marlin_apply_deterministic(scheme, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """`apply_gptq_marlin_linear` for a W*A16 layer with use_atomic_add=False (no bias)."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import marlin_utils as mu

    kernel = scheme.kernel
    c = kernel.config
    w_q, w_s, w_zp, w_gidx = kernel._get_weight_params(layer)
    k_in, n_out = c.partition_weight_shape
    reshaped_x = x.reshape(-1, x.shape[-1])
    out_shape = x.shape[:-1] + (n_out,)
    padded_n, padded_k = mu.marlin_repacked_nk(w_q, c.weight_type.size_bits)
    reshaped_x = mu.marlin_pad_dim(reshaped_x, k_in, padded_k)
    output = ops.marlin_gemm(
        reshaped_x, None, w_q, None, w_s, None, None, w_zp, w_gidx,
        layer.g_idx_sort_indices, kernel.workspace, c.weight_type,
        size_m=reshaped_x.shape[0], size_n=padded_n, size_k=padded_k,
        is_k_full=kernel.is_k_full, use_atomic_add=False,
        use_fp32_reduce=mu.USE_FP32_REDUCE_DEFAULT, is_zp_float=False,
    )
    output = mu.marlin_unpad_output(output, n_out, padded_n)
    return output.reshape(out_shape)


def _make_head_twin(w: torch.Tensor, chunk_rows: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [N, K] -> (e4m3 [N, K], exponent uint8 [ceil(N/32), K/32]); built in row chunks
    so the fp32 temporaries stay small on unified memory."""
    n, k = w.shape
    assert k % 32 == 0, k
    q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=w.device)
    s = torch.empty(((n + 31) // 32, k // 32), dtype=torch.uint8, device=w.device)
    for r0 in range(0, n, chunk_rows):
        r1 = min(n, r0 + chunk_rows)
        rows = r1 - r0
        pad = (-rows) % 32
        wf = F.pad(w[r0:r1].float(), (0, 0, 0, pad)).view((rows + pad) // 32, 32, k // 32, 32)
        amax = wf.abs().amax(dim=(1, 3)).clamp_min(2.0**-126)
        e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
        qq = (wf / torch.exp2(e)[:, None, :, None]).to(torch.float8_e4m3fn)
        q[r0:r1] = qq.view(rows + pad, k)[:rows]
        s[r0 // 32 : r0 // 32 + (rows + pad) // 32] = (e + 127).to(torch.uint8)
    return q, s


def _head_kernel():
    global _HEAD_KERNEL
    if _HEAD_KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _head_fp8_kernel(X, W8, S, Y, M, N, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                             BK: tl.constexpr):
            pid = tl.program_id(0)
            m = tl.program_id(1) * BM + tl.arange(0, BM)
            n = pid * BN + tl.arange(0, BN)
            nmask = n < N
            acc = tl.zeros((BM, BN), tl.float32)
            for k0 in range(0, K, BK):
                k = k0 + tl.arange(0, BK)
                x = tl.load(X + m[:, None] * K + k[None, :], m[:, None] < M, 0.0)
                w8 = tl.load(W8 + n[None, :] * K + k[:, None], nmask[None, :], 0.0)
                e = tl.load(S + (n[None, :] // 32) * (K // 32) + k[:, None] // 32, nmask[None, :], 127)
                w = (w8.to(tl.float32) * tl.exp2(e.to(tl.float32) - 127.0)).to(tl.bfloat16)
                acc += tl.dot(x, w)
            tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & nmask[None, :])

        _HEAD_KERNEL = _head_fp8_kernel
    return _HEAD_KERNEL


def _head_fp8(x: torch.Tensor, twin: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """x [M, K] bf16 (M <= 128) -> bf16 local logits [M, N], fp32 accumulation."""
    import triton

    w8, s = twin
    m, k = x.shape
    n = w8.shape[0]
    y = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _, bm, bk, warps, stages = next(t for t in _HEAD_TILES if m <= t[0])
    _head_kernel()[(triton.cdiv(n, 32), triton.cdiv(m, bm))](
        x.contiguous(), w8, s, y, m, n, k, bm, 32, bk, num_warps=warps, num_stages=stages
    )
    return y


class _HeadFp8Method:
    def __init__(self, twin):
        self.twin = twin

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise RuntimeError("DFlash2 fp8 draft head: unexpected embedding bias")
        return _head_fp8(x, self.twin)


class _HeadProxy:
    """Stands in for the vocab-parallel lm_head inside LogitsProcessor.get_top_k_tokens,
    which reads only .quant_method.apply, .shard_indices and .tp_size."""

    __slots__ = ("quant_method", "shard_indices", "tp_size")

    def __init__(self, head, twin):
        self.quant_method = _HeadFp8Method(twin)
        self.shard_indices = head.shard_indices
        self.tp_size = head.tp_size


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


class DFlashGroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        params_dtype: torch.dtype,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.kernel_projection = ReplicatedLinear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            # Drafter-diet overlay: quantized when the draft checkpoint's config
            # targets it (a BF16 checkpoint has no quant config, so nothing changes).
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "kernel_projection"),
            return_bias=False,
        )
        self._det_scheme = _marlin_scheme(self.kernel_projection) if _DET_CONV else None

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._det_scheme is not None:
            projected = _marlin_apply_deterministic(
                self._det_scheme, self.kernel_projection, hidden_states
            )
        else:
            projected = self.kernel_projection(hidden_states)
        coefficients = projected.reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, 1)


class DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config,
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        draft_config = config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            # Query tokens per request: the bonus token plus the mask tokens.
            block_size=1 + speculative_config.num_speculative_tokens,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=quant_config,
        )
        self.attention_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "attention_conv")
        )
        self.mlp_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "mlp_conv")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )


@support_torch_compile
class CandidateSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.hidden_projection(hidden_states)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
        )


class DFlash2Qwen3Model(DFlashQwen3Model):
    decoder_layer_cls = DFlash2Qwen3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        draft_config = self.config.dflash_config
        self.input_embedding_scale = float(
            draft_config.get("input_embedding_scale", 1.0)
        )
        # Without its own tag the selector shares the draft head's compile cache.
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(draft_config["selector_rank"]),
                top_k=int(draft_config["selector_top_k"]),
                params_dtype=vllm_config.model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DFlash2Qwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        draft_config = self.config.dflash_config
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        self.candidate_logits_processor = LogitsProcessor(
            vllm_config.model_config.get_vocab_size(),
            scale=float(draft_config.get("output_multiplier", 1.0)),
            soft_cap=softcap if softcap > 0 else None,
        )

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head = self.lm_head
        if _HEAD_FP8 and hidden_states.shape[0] <= _HEAD_FP8_MAX_M:
            twin = getattr(self, "_head_twin", None)
            if twin is None and not torch.cuda.is_current_stream_capturing():
                # First eager call (profiling / warm-up): the shared target head is loaded.
                twin = self._head_twin = _make_head_twin(self.lm_head.weight.data)
            if twin is not None:
                head = _HeadProxy(self.lm_head, twin)
        return self.candidate_logits_processor.get_top_k_tokens(
            head, hidden_states, self.model.candidate_selector.top_k
        )


EntryClass = DFlash2Qwen3ForCausalLM
