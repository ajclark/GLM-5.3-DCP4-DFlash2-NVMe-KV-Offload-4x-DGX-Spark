"""Distinguish prompt-cache continuation from generated-KV disk reload."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
from adaptive_spec import request_body
from spec_durable_check import BOUNDARY,PREFIX_OUTPUT,audit_reload,reload_body


def fixture():
    ids=[4]*(BOUNDARY-40)
    return {'chunks':[{'data':{'prompt_token_ids':ids}}],
            'usage':{'prompt_tokens':len(ids)},'token_ids':list(range(220))}


def test_raw_restart_prefix_includes_committed_generated_tokens():
    first=fixture(); original=request_body('test',7,'first',256)
    original['vllm_xargs'].update(spec_policy='adaptive',spec_cost_table='{}')
    body=reload_body(first,original,'reload')
    assert len(body['prompt'])==BOUNDARY+40
    assert body['prompt'][-PREFIX_OUTPUT:]==first['token_ids'][:PREFIX_OUTPUT]
    assert not body['add_special_tokens']
    assert 'messages' not in body
    assert body['vllm_xargs']['spec_label']=='reload'
    assert original['vllm_xargs']['spec_label']=='first'
    pydantic=pytest.importorskip('pydantic')
    pydantic.TypeAdapter(dict[str,str|int|float]|None).validate_python(body['vllm_xargs'])


def test_prefill_only_cache_hit_is_insufficient_for_generated_reload():
    first=fixture(); result={'token_ids':first['token_ids'][PREFIX_OUTPUT:]}
    delta={'vllm:external_prefix_cache_hits_total{engine="0"}':BOUNDARY-256,
           'vllm:prefix_cache_hits_total{engine="0"}':0,
           'vllm:kv_offload_load_bytes_total{engine="0"}':1000000}
    assert not audit_reload(first,result,delta)['generated_block_reload_passed']
    assert audit_reload(first,result,delta)['prompt_cache_continuation_passed']
    delta['vllm:external_prefix_cache_hits_total{engine="0"}']=BOUNDARY
    assert audit_reload(first,result,delta)['generated_block_reload_passed']
    assert not audit_reload(first,result,delta)['prompt_cache_continuation_passed']
    result['token_ids'][0]=-1
    assert not audit_reload(first,result,delta)['generated_block_reload_passed']


def test_restart_requires_boundary_crossing_and_exact_prompt_ids():
    first=fixture(); first['token_ids']=list(range(70))
    with pytest.raises(RuntimeError,match='boundary'):
        reload_body(first,request_body('test',7,'test',256),'reload')
    first=fixture(); first['chunks']=[]
    with pytest.raises(RuntimeError,match='prompt token IDs'):
        reload_body(first,request_body('test',7,'test',256),'reload')
