# SPDX-License-Identifier: Apache-2.0
# Exact source excerpt; see manifest.json.
class DFlashQwen3Model:
    def _build_fused_kv_buffers(self) -> None:
        """Build fused weight buffers for precompute_and_store_context_kv.

        Must be called after weights are loaded. Stacks the KV-projection
        weights, K-norm weights, and RoPE parameters from every attention
        layer so that precompute_and_store_context_kv can run one fused
        GEMM for all layers at once. Also aliases the weight of the hidden_norm.
        """
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        has_bias = attn0.qkv_proj.bias is not None

        self._hidden_norm_weight = self.hidden_norm.weight.data

        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
        if has_bias:
            kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
            self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)
        else:
            self._fused_kv_bias = None

        # K-norm weights: list of [head_dim] tensors, one per layer.
        self._k_norm_weights = [a.k_norm.weight.data for a in layers_attn]

        # RoPE parameters
        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        self._rope_is_neox = attn0.rotary_emb.is_neox_style
        # Validation that RoPE params are the same across all layers
        for attn in layers_attn[1:]:
            assert (
                attn.rotary_emb.head_size == self._rope_head_size
                and attn.rotary_emb.is_neox_style == self._rope_is_neox
            ), "All layers must have the same RoPE parameters for DFlash precomputation"

        # Layer metadata
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.q_norm.variance_epsilon
        # Validation that all layers have the same attention config
        for attn in layers_attn[1:]:
            assert (
                attn.kv_size == self._kv_size
                and attn.head_dim == self._head_dim
                and attn.num_kv_heads == self._num_kv_heads
                and attn.q_norm.variance_epsilon == self._rms_norm_eps
            ), "All layers must have the same attn config for DFlash precomputation"

        # References to inner Attention layers for direct cache writes
        self._attn_layers = [layer.self_attn.attn for layer in self.layers]
