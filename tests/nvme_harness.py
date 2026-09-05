"""Build a minimal `vllm` package for testing the multi-node tier: the
dependency-light fork modules are used for real (offload base, CPU manager
and policies, tiering manager/spec/base, async lookup, fs thread pool, file
mapper, worker dispatch, connector common types); the heavy ones are stubbed
(logger, config, platform, kv-cache interface, connector scheduler/worker
bases, CUDA handlers)."""
import importlib
import pathlib
import shutil
import sys
import tempfile
import textwrap
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = pathlib.Path.home() / "lmcache-mg/spark-src/vllm"
OVERLAY = ROOT / "overlay/vllm/v1/kv_offload/tiering/multinode.py"

REAL = [
    "v1/kv_offload/base.py", "v1/kv_offload/factory.py", "v1/kv_offload/file_mapper.py",
    "v1/kv_offload/cpu/common.py", "v1/kv_offload/cpu/manager.py", "v1/kv_offload/cpu/spec.py",
    "v1/kv_offload/cpu/shared_offload_region.py",
    "v1/kv_offload/cpu/policies/base.py", "v1/kv_offload/cpu/policies/lru.py", "v1/kv_offload/cpu/policies/arc.py",
    "v1/kv_offload/tiering/base.py", "v1/kv_offload/tiering/async_lookup.py", "v1/kv_offload/tiering/manager.py",
    "v1/kv_offload/tiering/spec.py", "v1/kv_offload/tiering/factory.py",
    "v1/kv_offload/tiering/fs/thread_pool.py", "v1/kv_offload/tiering/fs/io.py",
    "v1/kv_offload/worker/worker.py",
    "distributed/kv_transfer/kv_connector/v1/offloading/common.py",
]
PACKAGES = [
    "", "config", "utils", "platforms", "v1", "v1/core", "v1/core/sched", "v1/attention",
    "v1/kv_offload", "v1/kv_offload/cpu", "v1/kv_offload/cpu/policies", "v1/kv_offload/tiering",
    "v1/kv_offload/tiering/fs", "v1/kv_offload/worker", "distributed", "distributed/kv_transfer",
    "distributed/kv_transfer/kv_connector", "distributed/kv_transfer/kv_connector/v1",
    "distributed/kv_transfer/kv_connector/v1/offloading",
]
STUBS = {
    "logger.py": "import logging\ndef init_logger(name):\n    return logging.getLogger(name)\n",
    "config/__init__.py": "class VllmConfig:\n    pass\n",
    "utils/math_utils.py": "def cdiv(a, b):\n    return -(-a // b)\ndef round_up(x, m):\n    return -(-x // m) * m\n",
    "utils/platform_utils.py": "def is_pin_memory_available():\n    return False\n",
    "platforms/__init__.py": "class _P:\n    def is_cuda_alike(self):\n        return True\n    def is_xpu(self):\n        return False\ncurrent_platform = _P()\n",
    "v1/core/kv_cache_utils.py": textwrap.dedent('''
        import math
        def resolve_kv_cache_block_sizes(kv_cache_config, vllm_config):
            sizes = [g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups]
            return sizes, math.gcd(*sizes)
    '''),
    "v1/core/sched/output.py": "class SchedulerOutput:\n    pass\n",
    "v1/attention/backend.py": "class AttentionBackend:\n    pass\n",
    "v1/kv_cache_interface.py": textwrap.dedent('''
        from dataclasses import dataclass, field
        @dataclass
        class KVCacheSpec:
            block_size: int
        @dataclass
        class FullAttentionSpec(KVCacheSpec):
            pass
        @dataclass
        class MLAAttentionSpec(FullAttentionSpec):
            pass
        @dataclass
        class SlidingWindowSpec(KVCacheSpec):
            sliding_window: int = 2048
        @dataclass
        class KVCacheTensor:
            size: int
        @dataclass
        class KVCacheGroupSpec:
            layer_names: list
            kv_cache_spec: object
            is_eagle_group: bool = False
        @dataclass
        class KVCacheConfig:
            num_blocks: int
            kv_cache_tensors: list
            kv_cache_groups: list
        def cp_world_size_for_kv_cache_spec(spec, cp_world_size):
            return cp_world_size if isinstance(spec, FullAttentionSpec) else 1
    '''),
    "v1/outputs.py": textwrap.dedent('''
        from dataclasses import dataclass, field
        @dataclass
        class KVConnectorOutput:
            finished_sending: set | None = None
            finished_recving: set | None = None
            kv_connector_stats: object = None
            kv_cache_events: object = None
            kv_connector_worker_meta: object = None
            invalid_block_ids: set = field(default_factory=set)
            expected_finished_count: int = 0
    '''),
    "v1/kv_offload/cpu/gpu_worker.py": textwrap.dedent('''
        import torch
        class _NoopHandler:
            def __init__(self, dst_tensors=None):
                self.dst_tensors = dst_tensors
            def transfer_async(self, job_id, spec):
                return True
            def get_finished(self):
                return []
            def wait(self, job_ids):
                pass
            def shutdown(self):
                pass
        from vllm.v1.kv_offload.worker.worker import TransferResult
        class SingleDirectionOffloadingHandler:
            """Test stand-in for the fork's CUDA copy handler: same spec
            contract (GPU spec group-major with group_sizes; CPU spec parallel
            block ids), copies rows synchronously, completes on the next
            get_finished()."""
            def __init__(self, gpu_tensors, cpu_tensors, block_size_factor, kv_cache_groups_data_refs, gpu_to_cpu, mmap_region=None):
                assert block_size_factor == 1
                self.gpu_tensors, self.cpu_tensors, self.refs, self.gpu_to_cpu = gpu_tensors, cpu_tensors, kv_cache_groups_data_refs, gpu_to_cpu
                self.dst_tensors = cpu_tensors if gpu_to_cpu else gpu_tensors
                self.transfer_type = ("GPU", "CPU") if gpu_to_cpu else ("CPU", "GPU")
                self._done = []
                self.calls = []
            def transfer_async(self, job_id, spec):
                src, dst = spec
                gpu_spec = src if self.gpu_to_cpu else dst
                cpu_spec = dst if self.gpu_to_cpu else src
                gpu_ids, cpu_ids = list(gpu_spec.block_ids), list(cpu_spec.block_ids)
                assert len(gpu_spec.group_sizes) == len(self.refs) and sum(gpu_spec.group_sizes) == len(gpu_ids) == len(cpu_ids)
                i = nbytes = 0
                for g, size in enumerate(gpu_spec.group_sizes):
                    for _ in range(size):
                        gb, cb = int(gpu_ids[i]), int(cpu_ids[i]); i += 1
                        for ref in self.refs[g]:
                            n = ref.page_size_bytes
                            if self.gpu_to_cpu:
                                self.cpu_tensors[ref.tensor_idx][cb, :n] = self.gpu_tensors[ref.tensor_idx][gb, :n]
                            else:
                                self.gpu_tensors[ref.tensor_idx][gb, :n] = self.cpu_tensors[ref.tensor_idx][cb, :n]
                            nbytes += n
                self.calls.append((job_id, list(gpu_spec.group_sizes)))
                self._done.append(TransferResult(job_id=job_id, success=True, transfer_size=nbytes, transfer_time=0.001, transfer_type=self.transfer_type))
                return True
            def get_finished(self):
                out, self._done = self._done, []
                return out
            def wait(self, job_ids):
                pass
            def shutdown(self):
                pass
        class CpuGpuOffloadingHandlers:
            """Test stand-in: allocates the per-worker CPU tensors like the real
            (non-mmap) path and exposes them as gpu_to_cpu_handler.dst_tensors."""
            def __init__(self, kv_caches, block_size_factor, num_cpu_blocks, mmap_region=None):
                assert mmap_region is None
                self.cpu_tensors = [
                    torch.zeros((num_cpu_blocks, t.page_size_bytes * block_size_factor), dtype=torch.int8)
                    for t in kv_caches.tensors
                ]
                self.gpu_to_cpu_handler = _NoopHandler(self.cpu_tensors)
                self.cpu_to_gpu_handler = _NoopHandler()
    '''),
    "distributed/kv_transfer/kv_connector/v1/base.py": textwrap.dedent('''
        import enum
        from abc import ABC, abstractmethod
        class KVConnectorRole(enum.Enum):
            SCHEDULER = 0
            WORKER = 1
        class KVConnectorMetadata(ABC):
            pass
        class KVConnectorWorkerMetadata(ABC):
            @abstractmethod
            def aggregate(self, other):
                ...
        class KVConnectorStats:
            pass
        class KVConnectorBase_V1(ABC):
            def __init__(self, vllm_config, role, kv_cache_config):
                self._vllm_config = vllm_config
                self._role = role
                self._kv_cache_config = kv_cache_config
                self._connector_metadata = None
    '''),
    "distributed/kv_transfer/kv_connector/v1/offloading/metrics.py": textwrap.dedent('''
        class OffloadingConnectorStats:
            def __init__(self):
                self.data = {}
            def increase_counter(self, name, value=1):
                self.data[name] = self.data.get(name, 0) + value
            def observe_histogram(self, name, value):
                self.data.setdefault(name, []).append(value)
            def aggregate(self, other):
                return self
    '''),
    # Fakes of the connector bases: only the attributes/methods the
    # multinode subclasses rely on (test_nvme_invariants checks the real
    # classes still have them).
    "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py": textwrap.dedent('''
        from types import SimpleNamespace
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
            OffloadingConnectorMetadata, OffloadingWorkerMetadata)
        class OffloadingConnectorScheduler:
            def __init__(self, spec):
                self.spec = spec
                self.config = SimpleNamespace(num_workers=spec.vllm_config.parallel_config.world_size)
                self.manager = spec.get_manager()
                self._job_counter = 0
                self._jobs = {}
                self.seen_completed = {}
            def _generate_job_id(self):
                job_id = self._job_counter
                self._job_counter += 1
                return job_id
            def build_connector_meta(self, scheduler_output):
                self.manager.on_schedule_end()
                return OffloadingConnectorMetadata(load_jobs={}, store_jobs={}, jobs_to_flush=set())
            def update_connector_output(self, connector_output):
                meta = connector_output.kv_connector_worker_meta
                if meta is None:
                    return
                assert isinstance(meta, OffloadingWorkerMetadata), type(meta)  # the fork uses isinstance too
                for job_id, count in meta.completed_jobs.items():
                    self.seen_completed[job_id] = self.seen_completed.get(job_id, 0) + count
            def has_pending_push_work(self):
                return bool(self._jobs) or self.manager.has_pending_work()
            def shutdown(self):
                self.manager.shutdown()
    '''),
    "distributed/kv_transfer/kv_connector/v1/offloading/worker.py": textwrap.dedent('''
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingWorkerMetadata
        from vllm.v1.kv_offload.worker.worker import OffloadingWorker
        class OffloadingConnectorWorker:
            def __init__(self, spec):
                self.spec = spec
                self.worker = OffloadingWorker()
                self._load_jobs = {}
                self._unsubmitted_store_jobs = []
                self._connector_worker_meta = OffloadingWorkerMetadata()
            def _register_handlers(self, kv_caches):
                for src_cls, dst_cls, handler in self.spec.get_handlers(kv_caches):
                    self.worker.register_handler(src_cls, dst_cls, handler)
            def start_kv_transfers(self, metadata):
                for job_id, entry in self._unsubmitted_store_jobs:
                    assert self.worker.transfer_async(job_id, entry)
                self._unsubmitted_store_jobs.clear()
                for job_id, entry in metadata.load_jobs.items():
                    self._load_jobs[job_id] = entry.req_id
                    assert self.worker.transfer_async(job_id, entry.transfer_spec)
            def prepare_store_kv(self, metadata):
                for job_id, entry in metadata.store_jobs.items():
                    self._unsubmitted_store_jobs.append((job_id, entry.transfer_spec))
            def build_connector_worker_meta(self):
                if not self._connector_worker_meta.completed_jobs:
                    return None
                meta = self._connector_worker_meta
                self._connector_worker_meta = OffloadingWorkerMetadata()
                return meta
            def shutdown(self):
                self.worker.shutdown()
    '''),
    "distributed/kv_transfer/kv_connector/v1/offloading_connector.py": textwrap.dedent('''
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
        class OffloadingConnector(KVConnectorBase_V1):
            pass
    '''),
}

_BUILT: dict = {}


def build_vllm():
    """Materialize the stub package once per session and import multinode."""
    if "module" in _BUILT:
        return _BUILT["module"]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="vllm-stub-"))
    pkg = tmp / "vllm"
    for p in PACKAGES:
        d = pkg / p
        d.mkdir(parents=True, exist_ok=True)
        init = d / "__init__.py"
        if not init.exists():
            init.write_text("")
    for rel in REAL:
        src = SRC / rel
        assert src.exists(), src
        shutil.copy(src, pkg / rel)
    for rel, body in STUBS.items():
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text(body)
    shutil.copy(OVERLAY, pkg / "v1/kv_offload/tiering/multinode.py")
    for name in [m for m in list(sys.modules) if m == "vllm" or m.startswith("vllm.")]:
        del sys.modules[name]
    sys.path.insert(0, str(tmp))
    mod = importlib.import_module("vllm.v1.kv_offload.tiering.multinode")
    _BUILT["module"] = mod
    _BUILT["dir"] = tmp
    return mod


# --- fake engine configuration -----------------------------------------------

MLA_PAGE = 64 * 656       # fp8_ds_mla, 64 slots per physical block
INDEXER_PAGE = 64 * 132   # indexer k-cache
DRAFT_PAGE = 64 * 2 * 128 * 2 * 2   # drafter SWA block: 2 heads x 128 x K,V x bf16


def make_config(root_dir, rank, world_size=4, dcp=4, num_gpu_blocks=32, cpu_bytes=None, extra=None, direct=False, slab=None):
    """A VllmConfig/KVCacheConfig pair shaped like the deployed GLM-5.3 stack:
    group 0 = MLA target (block 64, sharded x dcp -> 256 tokens), group 1 =
    DFlash drafter sliding window (block 64, replicated)."""
    from vllm.v1.kv_cache_interface import (KVCacheConfig, KVCacheGroupSpec, KVCacheTensor,
                                            MLAAttentionSpec, SlidingWindowSpec)
    tensors = [KVCacheTensor(size=num_gpu_blocks * MLA_PAGE),
               KVCacheTensor(size=num_gpu_blocks * INDEXER_PAGE),
               KVCacheTensor(size=num_gpu_blocks * DRAFT_PAGE)]
    groups = [KVCacheGroupSpec(layer_names=["L0.attn", "L0.indexer"], kv_cache_spec=MLAAttentionSpec(block_size=64)),
              KVCacheGroupSpec(layer_names=["D0.attn"], kv_cache_spec=SlidingWindowSpec(block_size=64, sliding_window=2048))]
    kv_cache_config = KVCacheConfig(num_blocks=num_gpu_blocks, kv_cache_tensors=tensors, kv_cache_groups=groups)
    per_block_cluster = (MLA_PAGE + INDEXER_PAGE + DRAFT_PAGE) * world_size
    if cpu_bytes is None:
        cpu_bytes = per_block_cluster * 8   # 8 CPU blocks
    extra_config = {
        "spec_name": "MultiNodeTieringOffloadingSpec",
        "spec_module_path": "vllm.v1.kv_offload.tiering.multinode",
        "cpu_bytes_to_use": cpu_bytes,
        "secondary_tiers": [{"type": "fs_worker", "root_dir": str(root_dir), "n_read_threads": 2, "n_write_threads": 2}],
    }
    if direct:
        extra_config = {"spec_name": "MultiNodeDirectFsOffloadingSpec",
                        "spec_module_path": "vllm.v1.kv_offload.tiering.multinode",
                        "root_dir": str(root_dir), "bounce_blocks": 3, "n_read_threads": 2, "n_write_threads": 2}
    if slab is not None:   # slab = disk_bytes_per_rank
        extra_config = {"spec_name": "MultiNodeSlabOffloadingSpec",
                        "spec_module_path": "vllm.v1.kv_offload.tiering.multinode",
                        "root_dir": str(root_dir), "disk_bytes_per_rank": int(slab), "bounce_blocks": 3,
                        "n_read_threads": 2, "n_write_threads": 2}
    if extra:
        extra_config.update(extra)
    vllm_config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(kv_connector_extra_config=extra_config),
        parallel_config=types.SimpleNamespace(
            decode_context_parallel_size=dcp, prefill_context_parallel_size=1,
            tensor_parallel_size=world_size, pipeline_parallel_size=1, world_size=world_size, rank=rank),
        model_config=types.SimpleNamespace(model="/models/glm-5.3"),
        cache_config=types.SimpleNamespace(block_size=64, cache_dtype="fp8_ds_mla", enable_prefix_caching=True),
        speculative_config=None, kv_events_config=None, use_v2_model_runner=False, instance_id="test",
    )
    return vllm_config, kv_cache_config


def make_canonical_kv_caches(num_gpu_blocks=32):
    import torch
    from vllm.v1.kv_offload.base import CanonicalKVCacheRef, CanonicalKVCaches, CanonicalKVCacheTensor
    tensors = [CanonicalKVCacheTensor(tensor=torch.zeros((num_gpu_blocks, p), dtype=torch.int8), page_size_bytes=p)
               for p in (MLA_PAGE, INDEXER_PAGE, DRAFT_PAGE)]
    refs = [[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=MLA_PAGE), CanonicalKVCacheRef(tensor_idx=1, page_size_bytes=INDEXER_PAGE)],
            [CanonicalKVCacheRef(tensor_idx=2, page_size_bytes=DRAFT_PAGE)]]
    return CanonicalKVCaches(tensors=tensors, group_data_refs=refs)
