#!/usr/bin/env python3
"""Exercise release allocation and offload normalization without loading weights."""
from types import SimpleNamespace as NS
import torch
from vllm.config import CacheConfig, VllmConfig
from vllm.platforms.interface import Platform
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_groups, get_kv_cache_config_from_groups,
    generate_scheduler_kv_cache_config, resolve_kv_cache_block_sizes,
    dcp_world_size_for_kv_cache_spec, _max_memory_usage_bytes_from_groups,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowSpec, iter_layer_specs
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import build_offloading_config

cache = CacheConfig(block_size=64, cache_dtype='fp8', enable_prefix_caching=True)
cache.kv_cache_layout = 'BLHNC'
parallel = NS(decode_context_parallel_size=2, prefill_context_parallel_size=1,
    tensor_parallel_size=4, pipeline_parallel_size=1, world_size=4, rank=0,
    data_parallel_index=0, data_parallel_size=1, data_parallel_rank_local=None,
    cp_kv_cache_interleave_size=1, dcp_kv_cache_interleave_size=1)
config = NS(cache_config=cache, parallel_config=parallel, use_v2_model_runner=True,
    model_config=NS(model='glm-regression', dtype=torch.bfloat16, hf_config=NS(model_type='glm_moe_dsa'),
                    max_model_len=180224, is_deepseek_mla=True, use_mla=True),
    speculative_config=NS(use_eagle=lambda:True, method='dflash'),
    scheduler_config=NS(disable_hybrid_kv_cache_manager=False),
    kv_transfer_config=NS(kv_connector='MultiNodeSlabConnector', engine_id='regression',
                          kv_connector_extra_config={}), kv_events_config=None,
    max_in_flight_tokens=4096)
specs = {}
for i in range(78):
    specs[f'target.{i}'] = MLAAttentionSpec(block_size=64, num_kv_heads=1,
        head_size=576, dtype=torch.uint8, state_content_bytes=656, cache_dtype_str='fp8_ds_mla')
    specs[f'indexer.{i}'] = MLAAttentionSpec(block_size=64, num_kv_heads=1,
        head_size=128, dtype=torch.uint8, state_content_bytes=132)
for i in range(6):
    specs[f'draft.{i}'] = SlidingWindowSpec(block_size=64, num_kv_heads=8,
        head_size=128, dtype=torch.bfloat16, sliding_window=2048)
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.attention.backend import AttentionType, MultipleOf
attention = NS(attn_type=AttentionType.DECODER, kv_cache_dtype='auto', sliding_window=2048,
    attn_backend=NS(is_mla=lambda:False, customize_spec=lambda spec:spec,
                   get_supported_kernel_block_sizes=lambda:[MultipleOf(16)]),
    num_kv_heads=8, head_size=128, head_size_v=128, kv_cache_torch_dtype=torch.bfloat16)
assert Attention.get_kv_cache_spec(attention,config).block_size == 64
# Ordinary full-attention dtype ratios must not inflate MLA block sizes.
Platform._align_heterogeneous_kv_block_size(config, NS(is_mla=lambda:False))
assert cache.block_size == 64 and cache.skip_page_size_padded is None
for dcp in (1,2,4):
    parallel.decode_context_parallel_size = dcp
    groups = get_kv_cache_groups(config, specs)
    assert len(groups) == 2, groups
    assert all(isinstance(s, SlidingWindowSpec) for s in iter_layer_specs(groups[1].kv_cache_spec))
    assert [dcp_world_size_for_kv_cache_spec(g.kv_cache_spec, dcp) for g in groups] == [dcp,1]
    allocated = get_kv_cache_config_from_groups(config, groups, 6_000_000_000)
    scheduler = generate_scheduler_kv_cache_config([allocated]*4)
    worker_offload = build_offloading_config(config, allocated)
    scheduler_offload = build_offloading_config(config, scheduler)
    assert [g.tokens_per_block for g in worker_offload.groups] == [64*dcp,64]
    assert worker_offload.groups == scheduler_offload.groups
    assert resolve_kv_cache_block_sizes(allocated,config) == (64*dcp,64)
    assert resolve_kv_cache_block_sizes(scheduler,config) == (64*dcp,64)
    assert worker_offload.cache.tokens_per_hash == 64
    assert allocated.num_blocks > 1408
    if dcp > 1:
        needed = _max_memory_usage_bytes_from_groups(config,groups)
        assert needed + worker_offload.worker_kv_bytes_per_block <= 6_000_000_000, needed
    VllmConfig.adjust_dcp_kv_cache_interleave_size(config,allocated)
    assert parallel.cp_kv_cache_interleave_size == 1
    print(f'DCP{dcp}: native BLHNC allocation, draft ownership, scheduler/worker hashes and offload agree',flush=True)
print('Release hybrid cache configuration passed',flush=True)

from unittest.mock import patch
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility
layers = {'target': NS(impl=NS(dcp_world_size=4, need_to_return_lse_for_decode=True)),
          'draft': NS(impl=NS(dcp_world_size=1, need_to_return_lse_for_decode=False))}
with patch('vllm.v1.worker.cp_utils.get_layers_from_vllm_config', return_value=layers):
    check_attention_cp_compatibility(config)
    layers['target'].impl.need_to_return_lse_for_decode=False
    try:
        check_attention_cp_compatibility(config)
    except AssertionError:
        pass
    else:
        raise AssertionError('a sharded target without LSE was accepted')
print('CP startup compatibility accepts replicated draft and rejects sharded target without LSE',flush=True)

from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_offload.tiering.multinode import _content_identity, _slab_cache_dtype
config.attention_config=NS(backend=AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120)
cache.cache_dtype='fp8'
scheduler_identity=_content_identity(config,None,{})
assert _slab_cache_dtype(config)=='fp8_ds_mla'
cache.cache_dtype='fp8_ds_mla'
assert _content_identity(config,None,{})==scheduler_identity
cache.cache_dtype='fp8'
print('Scheduler and worker sparse FP8 alias normalization agree',flush=True)
