#!/usr/bin/env python3
"""Validate real checkpoint configuration and cache admission without loading weights."""
from pathlib import Path
import runpy
import json
from types import SimpleNamespace as NS
from vllm.engine.arg_utils import EngineArgs
from vllm.config import KVTransferConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.platforms.interface import Platform
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups, _max_memory_usage_bytes_from_groups
from vllm.model_executor.layers.attention.attention import Attention

fixture = runpy.run_path(str(Path(__file__).with_name('config_regression.py')))
config = EngineArgs(model='/models/glm-5.3', trust_remote_code=True,
    tensor_parallel_size=4, decode_context_parallel_size=2, dcp_comm_backend='ag_rs',
    dcp_q_replicate=False, nnodes=4, node_rank=3, distributed_executor_backend='mp',
    max_model_len=180224, max_num_batched_tokens=2048, max_num_seqs=12,
    kv_cache_dtype='fp8', kv_cache_dtype_skip_layers=['sliding_window'],
    kv_cache_memory_bytes=6_000_000_000, enable_prefix_caching=True, async_scheduling=True,
    attention_backend=AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
    speculative_config={'method':'dflash','model':'/models/dflash2-draft',
        'num_speculative_tokens':7,'draft_tensor_parallel_size':1,'attention_backend':'FLASH_ATTN'},
    kv_transfer_config=KVTransferConfig(kv_connector='MultiNodeSlabConnector',
        kv_connector_module_path='vllm.v1.kv_offload.tiering.multinode',kv_role='kv_both')
).create_engine_config()
config.cache_config.block_size=64
config.cache_config.kv_cache_layout='BLHNC'
from vllm.v1.kv_offload.tiering.multinode import _content_identity, _slab_cache_dtype
before_dtype=config.cache_config.cache_dtype
identity=_content_identity(config,config.speculative_config.model,{})
assert _slab_cache_dtype(config)=='fp8_ds_mla'
config.cache_config.cache_dtype='fp8_ds_mla'
assert _content_identity(config,config.speculative_config.model,{})==identity
config.cache_config.cache_dtype=before_dtype
print('Scheduler and constructed-worker sparse FP8 content identities agree',flush=True)
Platform._align_heterogeneous_kv_block_size(config,NS(is_mla=lambda:False))
assert config.cache_config.block_size==64
assert config.cache_config.skip_page_size_padded is None
assert Attention.get_kv_cache_spec(fixture['attention'],config).block_size==64
assert config.max_in_flight_tokens==4096
assert config.model_config.use_mla
needed=_max_memory_usage_bytes_from_groups(config,get_kv_cache_groups(config,fixture['specs']))
assert needed < 6_000_000_000,needed
print(json.dumps({'ok':True,'model_type':config.model_config.hf_config.model_type,
    'use_mla':config.model_config.use_mla,'in_flight_tokens':config.max_in_flight_tokens,
    'worst_case_kv_bytes':needed,'block_size':config.cache_config.block_size}),flush=True)
