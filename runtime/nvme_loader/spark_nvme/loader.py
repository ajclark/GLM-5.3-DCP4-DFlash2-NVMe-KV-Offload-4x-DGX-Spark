"""vLLM native pre-kernel artifacts with bounded, local concurrent restore."""
from copy import copy
import json
import logging
import os
from pathlib import Path
import time
import uuid

import torch

from .artifact import canonical, digest, inspect, inventory, publish, restore
from .identity import verify_source
from .adapters import adapter_for

log = logging.getLogger(__name__)


def simple(v):
    if v is None or type(v) in (bool, int, float, str):
        return True
    if type(v) is list:
        return all(simple(x) for x in v)
    if type(v) is dict:
        return all(type(k) is str and simple(x) for k, x in v.items())
    return False


def scalar_state(model):
    # Copy containers: loaders can mutate flags and lists in place.
    return json.loads(canonical({name: {k: v for k, v in vars(module).items()
        if not k.startswith("_") and simple(v)} for name, module in model.named_modules()}))


def scalar_delta(before, after):
    delta = {}
    for name in before:
        if name not in after or before[name].keys() - after[name].keys():
            raise ValueError("native load removed module/attributes; needs adapter")
    for name, attrs in after.items():
        for k, v in attrs.items():
            if name not in before or k not in before[name] or before[name][k] != v:
                delta.setdefault(name, {})[k] = v
    return delta


def object_state(model):
    """Detect non-registered tensor/runtime attributes introduced by a loader."""
    def tag(v):
        if simple(v):
            return ["value", json.loads(canonical(v))]
        if isinstance(v, torch.Tensor):
            return ["tensor", id(v), list(v.shape), str(v.dtype)]
        if isinstance(v, (list, tuple)):
            return [type(v).__name__, [tag(x) for x in v]]
        if isinstance(v, set):
            return ["set", sorted(repr(x) for x in v)]
        if isinstance(v, dict):
            return ["dict", sorted((str(k), tag(x)) for k, x in v.items())]
        return [type(v).__qualname__, id(v)]
    return {name: {k: tag(v) for k, v in vars(m).items()
                  if k not in ("_parameters", "_buffers", "_modules")}
            for name, m in model.named_modules()}


def check_side_effects(before, after, model):
    adapter = adapter_for(model, required=False)
    for name in before.keys() | after.keys():
        for key in before.get(name, {}).keys() | after.get(name, {}).keys():
            a, b = before.get(name, {}).get(key), after.get(name, {}).get(key)
            if a == b:
                continue
            # Public JSON flags are explicitly serialized. Empty loader-only
            # bookkeeping is not consulted by inference.
            if not key.startswith("_") and b and b[0] == "value":
                continue
            if adapter.allows_change(name, key, b):
                continue
            raise ValueError(f"load_weights changed {name}.{key}; prepared adapter required")


def plan(model, model_config):
    import vllm
    from vllm.config import get_current_vllm_config
    from vllm.distributed import get_world_group
    adapter_for(model)
    if any(getattr(getattr(m, "quant_method", None), "uses_meta_device", False)
           for m in model.modules()):
        raise ValueError("online quantization requires native loading")
    root = Path(os.environ["NVME_ARTIFACT_ROOT"])
    source = root / "sources" / (Path(model_config.model).name + ".json")
    source_id = verify_source(model_config.model, source)
    cfg = get_current_vllm_config()
    pc, cc = cfg.parallel_config, cfg.cache_config
    if pc.data_parallel_size != 1:
        raise ValueError("DP placement requires a prepared-rank namespace adapter")
    if getattr(model_config, "quantization_config", None) is not None:
        raise ValueError("explicit quantization overrides require a prepared contract adapter")
    if getattr(model_config, "model_weights", None):
        raise ValueError("separate model_weights requires a source-identity adapter")
    spec = cfg.speculative_config
    contract = {
        "adapter": "native-prekernel-v2", "content_id": source_id,
        "model_class": type(model).__module__ + "." + type(model).__qualname__,
        "hf_config": model_config.hf_config.to_dict(), "dtype": str(model_config.dtype),
        "quantization": model_config.quantization, "vllm": vllm.__version__,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "capability": list(torch.cuda.get_device_capability()),
        "runtime": os.getenv("NVME_RUNTIME_ID", "unversioned"),
        "max_model_len": model_config.max_model_len,
        "kv_cache_dtype": cc.cache_dtype,
        "kv_cache_dtype_skip_layers": getattr(cc, "kv_cache_dtype_skip_layers", None),
        "attention_backend": str(cfg.attention_config.backend),
        "spec_method": getattr(spec, "method", None),
        "spec_tokens": getattr(spec, "num_speculative_tokens", None),
        "topology": {k: getattr(pc, k, None) for k in (
            "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
            "prefill_context_parallel_size", "decode_context_parallel_size",
            "enable_expert_parallel", "expert_placement_strategy")},
    }
    contract = json.loads(canonical(contract))
    common_id = digest(contract)
    contract["rank"] = get_world_group().rank
    path = root / "artifacts" / common_id / f"rank-{contract['rank']}"
    schema, refs = inventory(model)
    del refs
    return contract, common_id, path, schema


def activation_gate(receipt, group):
    receipts = [receipt]
    if torch.distributed.is_initialized():
        receipts = [None] * torch.distributed.get_world_size(group)
        torch.distributed.all_gather_object(receipts, receipt, group=group)
    errors = [r["error"] for r in receipts if r["error"]]
    if errors:
        raise ValueError("NVME activation failed: " + "; ".join(errors))
    for r in receipts:
        if any(r.get(k) != receipt.get(k) for k in
               ("content_id", "source_metadata_id", "compatibility_id", "generation")):
            raise ValueError("NVME all-rank activation identity mismatch")


def register_loader():
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    @register_model_loader("nvme")
    class NvmeModelLoader(DefaultModelLoader):
        def __init__(self, load_config):
            native = copy(load_config)
            native.load_format = "auto"
            super().__init__(native)
            self._constructing = False
            self._stream_enabled = False
            self._stream_receipts = []

        def _get_weights_iterator(self, source):
            if not self._stream_enabled:
                yield from super()._get_weights_iterator(source)
                return
            from .streaming import checkpoint_metadata, stream_weights, prefetch_weights, UnsupportedStream
            from vllm.model_executor.model_loader.ep_weight_filter import should_skip_weight
            from vllm.model_executor.model_loader.weight_utils import _natural_sort_key
            _, files, safe = self._prepare_weights(source.model_or_path, source.subfolder,
                source.revision, source.fall_back_to_pt, source.allow_patterns_overrides)
            if not safe or self.load_config.safetensors_load_strategy == "torchao":
                self._stream_receipts.append({"native_fallback": True, "prefix": source.prefix})
                yield from super()._get_weights_iterator(source)
                return
            files = sorted(files, key=_natural_sort_key)
            memory_limit = int(os.getenv("NVME_STREAM_MEMORY_BYTES", str(2 << 30)))
            owned_limit = int(os.getenv("NVME_STREAM_OWNED_BYTES", str(6 << 30)))
            backend = os.getenv("NVME_STREAM_BACKEND", "runai")
            if backend not in ("runai", "coalesced"):
                raise ValueError("unknown NVME streaming backend")
            batch_bytes = int(os.getenv("NVME_STREAM_BATCH_BYTES", str(64 << 20)))
            try:
                metadata = checkpoint_metadata(files)
                if backend == "coalesced":
                    from .coalesced import plan_batches, check_direct_support
                    if any(b.bytes > owned_limit for b in plan_batches(metadata, batch_bytes)):
                        raise UnsupportedStream("largest batch exceeds configured owned budget")
                    check_direct_support(metadata)
                else:
                    import runai_model_streamer
                    if max(t.bytes for t in metadata[0].values()) > min(memory_limit, owned_limit):
                        raise UnsupportedStream("largest tensor exceeds configured streaming memory budget")
            except (UnsupportedStream, ImportError):
                log.exception("unsupported stream encoding; using native reconstruction")
                self._stream_receipts.append({"native_fallback": True, "prefix": source.prefix})
                yield from super()._get_weights_iterator(source)
                return
            metrics = {"prefix": source.prefix}
            self._stream_receipts.append(metrics)
            if self.counter_before_loading_weights == 0.0:
                self.counter_before_loading_weights = time.perf_counter()
            device = os.getenv("NVME_STREAM_DEVICE", "cpu")
            options = dict(
                concurrency=int(os.getenv("NVME_STREAM_CONCURRENCY", "32")),
                memory_limit=memory_limit,
                owned_limit=owned_limit,
                device=device,
                skip=lambda n: should_skip_weight(n, self.local_expert_ids), metrics=metrics)
            if backend == "coalesced":
                from .coalesced import coalesced_weights
                iterator = coalesced_weights(files, metadata, batch_bytes=batch_bytes, **options)
            else:
                iterator = prefetch_weights(stream_weights(files, metadata, **options), device=device)
            try:
                for name, tensor in iterator:
                    yield source.prefix + name, tensor
            finally:
                iterator.close()

        def load_model(self, vllm_config, model_config, prefix=""):
            self._constructing = True
            try:
                return super().load_model(vllm_config, model_config, prefix)
            finally:
                self._constructing = False

        def load_weights(self, model, model_config):
            if not self._constructing:
                raise ValueError("NVME artifacts require a fresh worker; in-place reload is unsupported")
            from vllm.distributed import get_world_group
            world = get_world_group()
            receipt = {"rank": world.rank, "generation": os.getenv("NVME_GENERATION", "default")}
            try:
                receipt.update(self._load(model, model_config))
                receipt["error"] = None
            except Exception as exc:
                log.exception("NVME rank %d failed before activation", world.rank)
                causes = []
                while exc is not None and len(causes) < 4:
                    causes.append(f"{type(exc).__name__}: {exc}")
                    exc = exc.__cause__ or exc.__context__
                receipt["error"] = f"rank {world.rank}: " + " <- ".join(causes)
            activation_gate(receipt, world.cpu_group)
            log.warning("NVME_LOAD %s", json.dumps(receipt, sort_keys=True))

        def _load(self, model, model_config):
            mode = os.getenv("NVME_LOADER_MODE", "auto")
            if mode not in ("auto", "prepare", "restore", "native", "stream"):
                raise ValueError("invalid NVME_LOADER_MODE")
            if mode == "stream":
                started = time.monotonic()
                self._stream_receipts = []
                # Native online quantization can require special layerwise
                # ordering/reconstruction. Keep its established path.
                self._stream_enabled = not any(
                    getattr(getattr(m, "quant_method", None), "uses_meta_device", False)
                    for m in model.modules())
                try:
                    super().load_weights(model, model_config)
                finally:
                    self._stream_enabled = False
                if any(not row.get("native_fallback") and not row.get("complete")
                       for row in self._stream_receipts):
                    raise ValueError("native model did not consume the complete checkpoint stream")
                sources = [{k: row[k] for k in ("prefix", "source_metadata_id", "native_fallback")
                            if k in row} for row in self._stream_receipts]
                return {"restored": False,
                        "streamed": any(not r.get("native_fallback") for r in self._stream_receipts),
                        "sources": self._stream_receipts,
                        "source_metadata_id": digest(sources), "content_verified": False,
                        "identity_kind": "source-metadata-only",
                        "compatibility_id": digest({"class": type(model).__qualname__,
                            "config": json.loads(json.dumps(model_config.hf_config.to_dict(), default=str)),
                            "dtype": str(model_config.dtype)}),
                        "loader_seconds": time.monotonic() - started}
            if mode == "native":
                super().load_weights(model, model_config)
                return {"restored": False, "native_bypass": True}
            started = time.monotonic()
            try:
                contract, common_id, path, before_schema = plan(model, model_config)
                before_scalars = scalar_state(model)
                before_objects = object_state(model)
            except Exception:
                if mode != "auto":
                    raise
                log.exception("NVME plan unavailable; using native checkpoint loader")
                super().load_weights(model, model_config)
                return {"restored": False, "native_bypass": True}
            restored, metrics = False, {}
            if path.exists() and mode != "prepare":
                try:
                    manifest = inspect(path, contract)
                    if manifest["schema"] != before_schema:
                        raise ValueError("constructor schema differs from artifact")
                except Exception:
                    if mode == "restore":
                        raise
                    log.exception("NVME metadata rejected; native load will prepare a replacement")
                    path.rename(path.with_name(path.name + ".rejected-" + uuid.uuid4().hex))
                else:
                    try:
                        metrics = restore(model, path, contract,
                            depth=int(os.getenv("NVME_READ_DEPTH", "32")),
                            direct=os.getenv("NVME_DIRECT", "1") == "1")
                        modules = dict(model.named_modules())
                        for name, attrs in manifest["loader_state"]["scalar_delta"].items():
                            for key, value in attrs.items():
                                setattr(modules[name], key, value)
                        adapter_for(model).after_restore(model)
                    except Exception:
                        # Once writes have begun, exit this worker; no in-place
                        # native retry against partially changed runtime state.
                        if mode == "auto":
                            path.rename(path.with_name(path.name + ".rejected-" + uuid.uuid4().hex))
                        raise
                    restored = True
            if not restored:
                if mode == "restore":
                    raise FileNotFoundError(f"prepared rank artifact missing: {path}")
                super().load_weights(model, model_config)
                if not path.exists():
                    try:
                        check_side_effects(before_objects, object_state(model), model)
                        changes = scalar_delta(before_scalars, scalar_state(model))
                        manifest = publish(model, path, contract, expected_schema=before_schema,
                                           loader_state={"scalar_delta": changes})
                        metrics["artifact_id"] = manifest["artifact_id"]
                    except Exception:
                        log.exception("NVME preparation unavailable; native model remains valid")
                        if mode == "prepare":
                            raise
            return {"content_id": contract["content_id"], "compatibility_id": common_id,
                    "restored": restored, "loader_seconds": time.monotonic() - started, **metrics}
