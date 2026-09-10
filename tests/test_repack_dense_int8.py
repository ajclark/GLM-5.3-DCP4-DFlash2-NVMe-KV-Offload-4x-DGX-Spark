"""CPU checkpoint overlays, pinned loader/dequant contracts, and byte gates."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace as NS
from typing import Generator, Iterable, Mapping

import numpy as np
import pytest
import torch

from harness import ROOT, extract, extract_methods

sys.path.insert(0, str(ROOT / 'bench'))
import repack_dense_int8 as repack
from dense_indexed_loader import indexed_weights
from dense_bytes_inventory import METADATA, summarize
from dense_indexer_gate import score

FIXTURE = ROOT / 'tests/fixtures/dense_int8'
NAME = 'model.layers.0.self_attn.indexer.wq_b.weight'


@pytest.fixture
def checkpoint(tmp_path):
    directory = tmp_path / 'source'
    directory.mkdir()
    generator = torch.Generator().manual_seed(74)
    tensors = {NAME: torch.randn(7, 256, generator=generator).bfloat16(),
               'model.layers.0.self_attn.indexer.wk.weight': torch.randn(4, 256, generator=generator).bfloat16(),
               'model.layers.0.input_layernorm.weight': torch.ones(256, dtype=torch.bfloat16)}
    repack.write_tensors(directory / 'original.safetensors', tensors)
    config = {'dtype': 'bfloat16', 'quantization_config': {'quant_method': 'compressed-tensors', 'format': 'pack-quantized',
              'ignore': [r're:model[.]layers[.]0[.].*', r're:model[.]layers[.][0-9]+[.]self_attn[.]indexer(?:$|[.].*)'],
              'config_groups': {'existing': {'targets': ['Linear'], 'weights': {'num_bits': 4, 'symmetric': True, 'strategy': 'group', 'group_size': 128}}}}}
    (directory / 'config.json').write_text(json.dumps(config))
    (directory / 'tokenizer_config.json').write_text('{"test": true}')
    (directory / repack.INDEX).write_text(json.dumps({'metadata': {'total_size': sum(t.numel() * t.element_size() for t in tensors.values())},
                                                    'weight_map': {name: 'original.safetensors' for name in tensors}}))
    return directory, tensors


def safe_open_fake(path, framework='pt', device='cpu'):
    assert framework == 'pt' and device == 'cpu'
    class Handle:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def keys(self): return repack.header(path)[0].keys()
        def get_tensor(self, name): return repack.load_tensor(path, name)
    return Handle()


def fork_dequant():
    return extract(FIXTURE / 'compressed_tensors_embedding.py', ['_dequant_gather_kernel', '_dequant_gather_triton'])['_dequant_gather_triton']


@pytest.mark.parametrize('scheme', ['per-channel', 'group-128'])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
def test_pack_roundtrips_through_fork_dequant_within_one_rtn_step(scheme, dtype):
    weight = torch.randn(5, 256, generator=torch.Generator().manual_seed(23)).to(dtype)
    weight[0] = 0
    result = repack.quantize(weight, scheme, row_block=2)
    assert result['weight_packed'].shape == (5, 64) and result['weight_packed'].dtype == torch.int32
    assert result['weight_shape'].tolist() == [5, 256] and result['weight_shape'].dtype == torch.int64
    step = result['weight_scale'].float().repeat_interleave(256 if scheme == 'per-channel' else 128, dim=1)
    expected = (weight.float() / step).round().clamp(-127, 127) * step
    actual = fork_dequant()(torch.arange(5), result['weight_packed'], result['weight_scale'], 256, 8)
    assert torch.all((actual.float() - expected).abs() <= step), (actual.float() - expected).abs().max()
    assert torch.all((actual.float() - weight.float()).abs() <= step * 1.02)
    assert torch.count_nonzero(actual[0]) == 0


def test_signed_bias_little_endian_lanes_and_round_to_even():
    weight = torch.tensor([[-127., -1., 0., 127., .5, 1.5, 2.5, -2.5]])
    result = repack.quantize(weight, 'per-channel')
    assert result['weight_scale'].item() == 1
    # q+128, low K lane first. High lane sets the int32 sign bit.
    assert result['weight_packed'][0, 0].item() == 0xFF807F01 - 2**32
    actual = fork_dequant()(torch.tensor([0]), result['weight_packed'], result['weight_scale'], 8, 8)
    assert actual.tolist() == [[-127., -1., 0., 127., 0., 2., 2., -2.]]


@pytest.mark.parametrize('scheme', ['per-channel', 'group-128'])
def test_repack_writes_only_new_tensors_and_symlinks_unmodified_shards(checkpoint, tmp_path, scheme):
    directory, tensors = checkpoint
    original = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()}
    out = tmp_path / 'repacked'
    report = repack.repack(directory, [re.escape(NAME)], scheme, out)
    assert (out / 'original.safetensors').is_symlink()
    assert (out / 'tokenizer_config.json').is_symlink()
    assert not (out / 'config.json').is_symlink()
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()} == original
    new = repack.read_json(out / repack.INDEX)
    stem = NAME[:-6]
    assert NAME not in new['weight_map']
    assert set(repack.header(out / report['selected'][0]['output_shard'])[0]) == {stem + key for key in ('weight_packed', 'weight_scale', 'weight_shape')}
    assert report['new_bytes'] == 7 * 256 + 7 * (1 if scheme == 'per-channel' else 2) * 2 + 16
    assert new['metadata']['total_size'] == report['original_logical_bytes'] - report['saved_bytes']
    loaded = dict(indexed_weights(out, safe_open_fn=safe_open_fake))
    assert loaded.keys() == new['weight_map'].keys()
    assert torch.equal(loaded['model.layers.0.input_layernorm.weight'], tensors['model.layers.0.input_layernorm.weight'])
    decoded = fork_dequant()(torch.arange(7), loaded[stem + 'weight_packed'], loaded[stem + 'weight_scale'], 256, 8)
    step = loaded[stem + 'weight_scale'].float().repeat_interleave(256 if scheme == 'per-channel' else 128, 1)
    assert torch.all((decoded.float() - tensors[NAME].float()).abs() <= step * 1.02)


def test_ignore_regex_subtraction_is_checked_by_real_fork_matching(checkpoint):
    directory, _ = checkpoint
    report, config, _ = repack.plan(directory, [re.escape(NAME)], 'per-channel')
    quant = config['quantization_config']
    ns = extract(FIXTURE / 'compressed_tensors_utils.py',
                 ['_is_equal_or_regex_match', 'check_equal_or_regex_match', 'should_ignore_layer'],
                 {'re': re, 'Iterable': Iterable, 'Mapping': Mapping, 'MappingProxyType': MappingProxyType})
    ignored = lambda module: ns['should_ignore_layer'](module, quant['ignore'])
    assert not ignored(NAME[:-7])
    assert ignored('model.layers.0.self_attn.indexer.wk')
    assert ignored('model.layers.0.input_layernorm')
    assert ignored('model.layers.78.self_attn.indexer.wq_b')
    assert not ignored('lm_head')
    assert next(iter(quant['config_groups'])) == 'dense_repack_w8a16'
    assert quant['config_groups']['dense_repack_w8a16']['weights']['group_size'] == -1


def test_default_fork_iterator_really_reads_retired_names_and_adapter_does_not(checkpoint, tmp_path):
    directory, _ = checkpoint
    out = tmp_path / 'packed'
    repack.repack(directory, [re.escape(NAME)], 'per-channel', out)
    touched = []
    def opening(path, framework='pt', **kwargs):
        handle = safe_open_fake(path, framework, **kwargs)
        get = handle.get_tensor
        def record(name):
            touched.append(name)
            return get(name)
        handle.get_tensor = record
        return handle
    ns = extract(FIXTURE / 'default_iterator.py', ['safetensors_weights_iterator'], {
        'Generator': Generator, 'DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS': 1,
        'DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE': 4096, '_natural_sort_key': str,
        '_get_fs_type': lambda _: 'ext4', '_get_checkpoints_size_bytes': lambda _: 100,
        '_get_available_ram_bytes': lambda: 1000, 'logger': NS(info_once=lambda *a: None),
        'tqdm': lambda values, **kwargs: values, 'enable_tqdm': bool, '_BAR_FORMAT': '', 'safe_open': opening,
        'should_skip_weight': lambda *args: False})
    raw = dict(ns['safetensors_weights_iterator']([str(path) for path in out.glob('*.safetensors')], False, 'lazy'))
    assert NAME in raw and NAME in touched
    touched.clear()
    active = dict(indexed_weights(out, safe_open_fn=opening))
    assert NAME not in active and NAME not in touched


def test_dry_run_reads_no_tensor_payloads_and_creates_nothing(checkpoint, tmp_path, monkeypatch):
    directory, _ = checkpoint
    monkeypatch.setattr(repack, 'load_tensor', lambda *a: pytest.fail('dry run read tensor data'))
    out = tmp_path / 'not-created'
    report = repack.repack(directory, [re.escape(NAME)], 'per-channel', out, dry_run=True)
    assert not out.exists() and report['selected_tensors'] == 1
    command = [sys.executable, str(ROOT / 'bench/repack_dense_int8.py'), str(directory), '--tensor-regex', re.escape(NAME),
               '--scheme', 'group-128', '--out', str(out), '--dry-run']
    result = subprocess.run(command, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['original_shards_copied'] == 0
    assert not out.exists()


@pytest.mark.parametrize('patterns', [[r'no.match'], [r'.*'], [re.escape(NAME), r'no.match']])
def test_invalid_or_overbroad_selection_fails_before_output(checkpoint, tmp_path, patterns):
    directory, _ = checkpoint
    with pytest.raises(ValueError):
        repack.repack(directory, patterns, 'per-channel', tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_input_cleans_only_the_new_output(checkpoint, tmp_path, bad):
    directory, tensors = checkpoint
    tensors[NAME][0, 0] = bad
    (directory / 'original.safetensors').unlink()
    repack.write_tensors(directory / 'original.safetensors', tensors)
    with pytest.raises(ValueError, match='nonfinite'):
        repack.repack(directory, [re.escape(NAME)], 'per-channel', tmp_path / 'bad')
    assert directory.is_dir() and not (tmp_path / 'bad').exists()


def test_overwrite_and_in_source_outputs_are_refused(checkpoint, tmp_path):
    directory, _ = checkpoint
    existing = tmp_path / 'existing'
    existing.mkdir()
    (existing / 'keep').write_text('unchanged')
    for out in (directory, directory / 'nested', existing):
        with pytest.raises(ValueError):
            repack.repack(directory, [re.escape(NAME)], 'per-channel', out)
    assert (existing / 'keep').read_text() == 'unchanged'
    alias = tmp_path / 'source-alias'
    alias.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError):
        repack.repack(directory, [re.escape(NAME)], 'per-channel', alias / 'nested')
    assert not (directory / 'nested').exists()


def test_memory_bound_packing_shape_and_scale_underflow_fail_closed(checkpoint):
    directory, _ = checkpoint
    with pytest.raises(ValueError, match='max-tensor-mib'):
        repack.plan(directory, [re.escape(NAME)], 'per-channel', .0001)
    for shape, scheme in [((3, 6), 'per-channel'), ((3, 132), 'group-128')]:
        with pytest.raises(ValueError):
            repack.quantize(torch.ones(shape), scheme)
    with pytest.raises(ValueError, match='underflow'):
        repack.quantize(torch.full((2, 8), 1e-20), 'per-channel', torch.float16)
    with pytest.raises(ValueError, match='scratch width'):
        repack.block_rows(1, repack.ROW_VALUES + 4)
    assert repack.block_rows(100, 32768) == 8


def test_plugin_registration_preserves_config_prefix_filter_and_draft_fallback(checkpoint, tmp_path, monkeypatch):
    import types
    import dense_indexed_loader as adapter
    directory, _ = checkpoint
    out = tmp_path / 'packed'
    repack.repack(directory, [re.escape(NAME)], 'per-channel', out)
    registry = {}
    def register(name):
        def wrap(cls):
            registry[name] = cls
            return cls
        return wrap
    class Default:
        counter_before_loading_weights = 0.0
        def __init__(self, config):
            self.config, self.local_expert_ids = config, {7}
        def _get_weights_iterator(self, source):
            return iter([('unchanged-draft', None)])
    for name, attrs in {
        'vllm.model_executor.model_loader': {'register_model_loader': register},
        'vllm.model_executor.model_loader.default_loader': {'DefaultModelLoader': Default},
        'vllm.model_executor.model_loader.weight_utils': {'should_skip_weight': lambda name, ids: ids == {7} and name.endswith('.wk.weight')},
        'safetensors': {'safe_open': safe_open_fake},
    }.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    config = NS(load_format='dense-indexed')
    loader = adapter.register()(config)
    assert registry['dense-indexed'] is type(loader)
    assert config.load_format == 'dense-indexed' and loader.config.load_format == 'safetensors'
    loaded = dict(loader._get_weights_iterator(NS(model_or_path=out, subfolder=None, prefix='target.')))
    assert loaded and all(name.startswith('target.') for name in loaded)
    assert all(not name.endswith('.wk.weight') for name in loaded)
    assert loader.counter_before_loading_weights > 0
    assert list(loader._get_weights_iterator(NS(model_or_path=directory, subfolder=None, prefix=''))) == [('unchanged-draft', None)]


def test_corrupt_header_index_traversal_and_destination_collision_fail(checkpoint):
    directory, _ = checkpoint
    index = repack.read_json(directory / repack.INDEX)
    index['weight_map'][NAME] = '../outside.safetensors'
    (directory / repack.INDEX).write_text(json.dumps(index))
    with pytest.raises(ValueError, match='unsafe'):
        repack.plan(directory, [re.escape(NAME)], 'per-channel')
    with (directory / 'original.safetensors').open('ab') as stream:
        stream.write(b'x')
    with pytest.raises(ValueError, match='length mismatch'):
        repack.header(directory / 'original.safetensors')


def test_runtime_host_guard_has_no_override(monkeypatch):
    monkeypatch.setattr(repack.socket, 'gethostname', lambda: 'spark-06c4')
    with pytest.raises(RuntimeError, match='sandbox-only'):
        repack.sandbox_only()
    monkeypatch.setattr(repack.socket, 'gethostname', lambda: 'renamed')
    monkeypatch.setattr(repack.platform, 'machine', lambda: 'aarch64')
    with pytest.raises(RuntimeError, match='sandbox-only'):
        repack.sandbox_only()


def test_saved_real_headers_correct_memo_counts_and_quantization_classes():
    report = summarize(json.loads(METADATA.read_text()))
    assert report['indexer_wq_b']['active_layers'] == 21
    assert report['target_families']['indexer_bf16']['stream_bytes_per_rank_cycle'] == 393619968
    assert report['target_families']['lm_head_bf16']['stream_bytes_per_rank_cycle'] == 475791360
    assert report['target_families']['kv_b_checkpoint']['stream_bytes_per_rank_cycle'] == 0
    assert report['target_families']['kv_b_derived_bf16']['stream_bytes_per_rank_cycle'] == 572522496
    assert report['target_families']['mtp_not_loaded']['stream_bytes_per_rank_cycle'] == 0
    assert report['draft_tp']['4']['stream_bytes_per_rank_cycle'] == 2216143872
    assert report['draft_tp']['1']['stream_bytes_per_rank_cycle'] == 6814411776


def test_pinned_dflash_load_keeps_target_tp_config_and_shares_target_head():
    from contextlib import nullcontext
    import types
    import sys
    tag_module = types.ModuleType('vllm.compilation.backends')
    tag_module.set_model_tag = lambda tag: nullcontext()
    target = NS(model=NS(embed_tokens='target-embedding'), lm_head='target-head')
    draft = NS(model=NS(embed_tokens='draft-embedding'), lm_head='draft-head')
    observed = []
    def modified(obj, **kwargs):
        return NS(**{**vars(obj), **kwargs})
    def get_model(**kwargs):
        observed.append(kwargs['vllm_config'])
        return draft
    ns = extract(FIXTURE / 'dflash_load.py', ['load_dflash_model'], {'nn': NS(Module=object), 'VllmConfig': object,
        'get_dflash_causal': lambda _: False, 'replace': modified, 'get_model': get_model,
        'get_pp_group': lambda: NS(world_size=1), '_should_share': lambda *args: True})
    config = NS(parallel_config=NS(tensor_parallel_size=4), attention_config=NS(),
                speculative_config=NS(draft_model_config=NS(), draft_tensor_parallel_size=1))
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, 'vllm.compilation.backends', tag_module)
        loaded = ns['load_dflash_model'](target, config)
    assert observed[0].parallel_config is config.parallel_config
    assert observed[0].parallel_config.tensor_parallel_size == 4
    assert loaded.lm_head == target.lm_head


def test_raw_draft_context_builder_requires_untransposed_bf16_qkv():
    method = extract_methods(FIXTURE / 'draft_kv.py', 'DFlashQwen3Model', ['_build_fused_kv_buffers'])['_build_fused_kv_buffers']
    def model(transpose):
        weight = torch.zeros(12, 16, dtype=torch.bfloat16)
        if transpose:
            weight = weight.t().contiguous()
        attn = NS(qkv_proj=NS(weight=weight, bias=None), q_size=8, kv_size=2, head_dim=2,
                  num_kv_heads=1, k_norm=NS(weight=torch.ones(2)), q_norm=NS(variance_epsilon=1e-5),
                  rotary_emb=NS(head_size=2, cos_sin_cache=torch.zeros(1), is_neox_style=False), attn=None)
        return NS(layers=[NS(self_attn=attn)], hidden_norm=NS(weight=torch.ones(16)))
    baseline, quantized_layout = model(False), model(True)
    method(baseline)
    method(quantized_layout)
    assert baseline._fused_kv_weight.shape == (4, 16)
    assert quantized_layout._fused_kv_weight.shape != baseline._fused_kv_weight.shape
    with pytest.raises(RuntimeError):
        torch.nn.functional.linear(torch.ones(1, 16, dtype=torch.bfloat16), quantized_layout._fused_kv_weight)


def test_indexer_gate_requires_real_selection_and_rejects_topk_drift():
    baseline = np.tile(np.arange(32, dtype=np.float32), (8, 1))
    assert score(baseline, baseline.copy(), topk=4, min_rows=8)['passed']
    changed = baseline[:, ::-1].copy()
    failed = score(baseline, changed, topk=4, min_rows=8)
    assert not failed['passed'] and failed['metrics']['mean_topk_overlap'] == 0
    with pytest.raises(ValueError, match='more eligible keys'):
        score(baseline, baseline, topk=32)
    masked = baseline.copy()
    masked[:, 0] = -np.inf
    with pytest.raises(ValueError, match='causal masks'):
        score(baseline, masked, topk=4)
    assert not score(baseline, baseline, topk=4)['checks']['coverage']


def test_pinned_fixtures_match_their_source_manifest():
    manifest = json.loads((FIXTURE / 'manifest.json').read_text())
    for name, row in manifest.items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == row['sha256']
