"""Opt-in loader for symlinked repack overlays; no registration at import time.

Stage register() as a vllm.general_plugins entry point ONLY in a future guarded
experiment. There are no network calls, model construction, or CUDA operations
in this module. Filtering happens before get_tensor, not after materialization.
"""
from collections import defaultdict
import json
from pathlib import Path
import time


def indexed_weights(model, *, safe_open_fn=None, skip_weight=None):
    if safe_open_fn is None:
        from safetensors import safe_open
        safe_open_fn = safe_open
    model = Path(model)
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    groups = defaultdict(list)
    for name, shard in index['weight_map'].items():
        relative = Path(shard)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe shard path')
        groups[shard].append(name)
    for shard, names in sorted(groups.items()):
        with safe_open_fn(model / shard, framework='pt', device='cpu') as handle:
            available = set(handle.keys())
            for name in sorted(names):
                if name not in available:
                    raise ValueError('index points to missing tensor: ' + name)
                if skip_weight is None or not skip_weight(name):
                    yield name, handle.get_tensor(name)


def register():
    """vLLM general-plugin entry point; importing this file stays CPU/offline."""
    from copy import copy
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.model_executor.model_loader.weight_utils import should_skip_weight

    @register_model_loader('dense-indexed')
    class DenseIndexedLoader(DefaultModelLoader):
        def __init__(self, load_config):
            # The unchanged default preparation path also loads the draft.
            local_config = copy(load_config)
            local_config.load_format = 'safetensors'
            super().__init__(local_config)

        def _get_weights_iterator(self, source):
            model = Path(source.model_or_path)
            if source.subfolder:
                model /= source.subfolder
            if not (model / 'dense-repack-manifest.json').is_file():
                return super()._get_weights_iterator(source)
            if self.counter_before_loading_weights == 0.0:
                self.counter_before_loading_weights = time.perf_counter()
            iterator = indexed_weights(model, skip_weight=lambda name: should_skip_weight(name, self.local_expert_ids))
            return ((source.prefix + name, tensor) for name, tensor in iterator)

    return DenseIndexedLoader
