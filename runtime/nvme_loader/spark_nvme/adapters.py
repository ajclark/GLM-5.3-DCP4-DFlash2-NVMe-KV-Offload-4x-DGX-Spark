"""Explicit model-side effects; the transport and artifact format stay generic.

Native postprocessing always runs. These adapters cover only side effects of
model.load_weights that a direct pre-kernel storage restore would otherwise miss.
"""


class NativePreKernelAdapter:
    def allows_change(self, module_name, attribute, value):
        return False

    def after_restore(self, model):
        pass


class DeepseekAdapter(NativePreKernelAdapter):
    def allows_change(self, module_name, attribute, value):
        # Loader-only queue; any unconsumed tensor requires native handling.
        return attribute == "_pending_indexer_wk_fp8" and value == ["value", {}]


class DFlashAdapter(NativePreKernelAdapter):
    derived_attributes = frozenset({
        "_hidden_norm_weight", "_fused_kv_weight", "_fused_kv_bias",
        "_k_norm_weights", "_rope_head_size", "_rope_cos_sin_cache",
        "_rope_is_neox", "_num_attn_layers", "_kv_size", "_head_dim",
        "_num_kv_heads", "_rms_norm_eps", "_attn_layers",
        "_fused_hidden_norm_weight", "_context_kv_weight", "_context_kv_bias",
    })

    def allows_change(self, module_name, attribute, value):
        return module_name == "model" and attribute in self.derived_attributes

    def after_restore(self, model):
        model.model._build_fused_kv_buffers()


ADAPTERS = {
    **dict.fromkeys(("LlamaForCausalLM", "GPT2LMHeadModel", "Qwen2ForCausalLM",
                     "Qwen3ForCausalLM"), NativePreKernelAdapter),
    **dict.fromkeys(("DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM",
                     "DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"), DeepseekAdapter),
    **dict.fromkeys(("DFlashQwen3ForCausalLM", "DFlash2Qwen3ForCausalLM"), DFlashAdapter),
}


def adapter_for(model, *, required=True):
    cls = ADAPTERS.get(type(model).__name__)
    if cls is None or not type(model).__module__.startswith("vllm."):
        if required:
            raise ValueError(f"no prepared adapter for {type(model).__name__}")
        return NativePreKernelAdapter()
    return cls()
