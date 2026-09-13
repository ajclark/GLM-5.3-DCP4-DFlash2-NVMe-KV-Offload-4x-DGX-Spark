"""Regression gates for the release port and deployment preparation."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'runtime/vllm029'


def test_release_manifest_matches_all_shipped_sources():
    manifest=json.loads((RUNTIME/'manifest.json').read_text())
    assert manifest['version']=='0.29.0'
    shipped={str(p.relative_to(RUNTIME/'overlay/vllm')) for p in (RUNTIME/'overlay/vllm').rglob('*.py')}
    assert shipped==set(manifest['files'])
    for name, row in manifest['files'].items():
        source=(RUNTIME/'overlay/vllm'/name).read_bytes()
        assert hashlib.sha256(source).hexdigest()==row['sha256']
        ast.parse(source)
    dependencies={str(p.relative_to(RUNTIME/'overlay')) for p in (RUNTIME/'overlay/flashinfer').rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts}
    assert dependencies==set(manifest['dependency_files'])
    for name,row in manifest['dependency_files'].items():
        assert hashlib.sha256((RUNTIME/'overlay'/name).read_bytes()).hexdigest()==row['sha256']


def test_replicated_draft_table_regressions_against_release(monkeypatch):
    # Reuse the independent slot-ownership oracle and allocation doubles from
    # the existing suite, executing the actual ported release kernels.
    import test_spec_v2_block_tables as t
    monkeypatch.setattr(t,'OVERLAY',RUNTIME/'overlay/vllm')
    for cp in (1,2,4):
        for rank in range(cp):
            for interleave in (1,4):
                for kernel_block_size in (16,64):
                    t.test_long_slots_match_independent_owner_reference(cp,rank,interleave,kernel_block_size)
    t.test_large_physical_slots_and_invalid_logical_positions()
    t.test_gather_permutation_padding_and_malformed_count_stay_inside_rows()
    t.test_append_rejects_entire_update_before_staging_any_group()
    t.test_layout_reinitialization_restores_per_group_geometry()
    t.test_invalid_cp_size_rejected()
    for row,groups in [(-1,([],[])),(3,([],[])),(0,([],))]:
        t.test_invalid_request_and_group_count(row,groups)


def test_sliding_window_width_is_not_divided_by_dcp():
    from harness import extract_methods,cdiv
    from types import SimpleNamespace as NS
    method=extract_methods(RUNTIME/'overlay/vllm/v1/kv_cache_interface.py','SlidingWindowSpec',
                           ['max_num_blocks_per_req'],{'VllmConfig':object})
    config=NS(parallel_config=NS(decode_context_parallel_size=4))
    assert method['max_num_blocks_per_req'](NS(block_size=64),config,180224)==2816


def test_release_launcher_uses_image_code_and_ring_collectives():
    text=(RUNTIME/'launch.sh').read_text()
    assert '--dcp-comm-backend ag_rs' in text
    assert '-e VLLM_ALLREDUCE_USE_FLASHINFER=0' in text
    assert '--entrypoint vllm' in text
    assert ':/usr/local/lib/python3.12/dist-packages/vllm/' not in text
    assert '/var/tmp/kvcache-vllm029' in text


def test_cp_validation_checks_only_sharded_implementations():
    from types import SimpleNamespace as NS
    from typing import Any, cast
    source = ast.parse((RUNTIME/'overlay/vllm/v1/worker/cp_utils.py').read_text())
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'check_attention_cp_compatibility')
    layers = {'target': NS(impl=NS(dcp_world_size=2, need_to_return_lse_for_decode=True)),
              'draft': NS(impl=NS(dcp_world_size=1, need_to_return_lse_for_decode=False))}
    scope = {'VllmConfig': object, 'AttentionLayerBase': object, 'Any': Any, 'cast': cast,
             'get_layers_from_vllm_config': lambda *_: layers}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<release-cp-check>', 'exec'), scope)
    config = NS(parallel_config=NS(prefill_context_parallel_size=1,
                decode_context_parallel_size=2, cp_kv_cache_interleave_size=1),
                speculative_config=object())
    check = scope['check_attention_cp_compatibility']
    check(config)
    layers['target'].impl.need_to_return_lse_for_decode = False
    with pytest.raises(AssertionError, match='softmax LSE'):
        check(config)
    # Implementations without an explicit ownership override remain subject
    # to the process's DCP requirements.
    del layers['target'].impl.dcp_world_size
    with pytest.raises(AssertionError, match='softmax LSE'):
        check(config)
