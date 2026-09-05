"""The multi-node tier subclasses the fork's OffloadingConnector classes and
relies on a handful of their attributes and methods. Pin them against the
image's real source so a fork update cannot silently break the overlay."""
import ast
import pathlib
import re

SRC = pathlib.Path.home() / "lmcache-mg/spark-src/vllm"
K = SRC / "distributed/kv_transfer/kv_connector/v1"
OVERLAY = pathlib.Path(__file__).resolve().parents[1] / "overlay/vllm/v1/kv_offload/tiering/multinode.py"


def methods(path, cls):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    raise AssertionError(f"{path}: class {cls} not found")


def test_scheduler_base_surface():
    src = (K / "offloading/scheduler.py").read_text()
    m = methods(K / "offloading/scheduler.py", "OffloadingConnectorScheduler")
    assert {"_generate_job_id", "build_connector_meta", "update_connector_output", "has_pending_push_work"} <= m
    assert "self.config = SchedulerOffloadConfig.from_spec(spec)" in src
    assert "num_workers=spec.vllm_config.parallel_config.world_size" in src
    assert "self.manager: OffloadingManager = spec.get_manager()" in src
    # completions are counted per worker until num_workers have reported
    assert "job_status.pending_count -= count" in src
    # keys per group come from every hash_block_size_factor-th request hash
    assert "group_config.hash_block_size_factor" in src and "make_offload_key(req_block_hash, group_config.group_idx)" in src


def test_worker_base_surface():
    src = (K / "offloading/worker.py").read_text()
    m = methods(K / "offloading/worker.py", "OffloadingConnectorWorker")
    assert {"start_kv_transfers", "get_finished", "build_connector_worker_meta", "_register_handlers", "shutdown"} <= m
    for attr in ("self.worker = OffloadingWorker()", "self._load_jobs", "self._connector_worker_meta = OffloadingWorkerMetadata()"):
        assert attr in src
    assert "assert transfer_result.success" in src  # the base still asserts; ours tolerates tier failures


def test_common_metadata_surface():
    src = (K / "offloading/common.py").read_text()
    assert "completed_jobs: dict[int, int]" in src and "def mark_completed" in src and "def aggregate" in src
    assert "load_jobs: dict[int, TransferJob]" in src and "store_jobs: dict[int, TransferJob]" in src


def test_connector_and_factories_surface():
    src = (K / "offloading_connector.py").read_text()
    assert "self.connector_scheduler = OffloadingConnectorScheduler(spec)" in src
    assert "self.connector_worker = OffloadingConnectorWorker(spec)" in src
    assert "class OffloadingConnector(KVConnectorBase_V1, SupportsHMA)" in src
    fac = (SRC / "distributed/kv_transfer/kv_connector/factory.py").read_text()
    assert "kv_connector_module_path" in fac
    spec_fac = (SRC / "v1/kv_offload/factory.py").read_text()
    assert 'extra_config.get("spec_module_path")' in spec_fac


def test_tiering_surfaces():
    spec = (SRC / "v1/kv_offload/tiering/spec.py").read_text()
    for attr in ("self.secondary_tier_configs", "self.eviction_policy", "self._manager", "def create_handlers", "def get_handlers"):
        assert attr in spec or attr in (SRC / "v1/kv_offload/cpu/spec.py").read_text()
    gw = (SRC / "v1/kv_offload/cpu/gpu_worker.py").read_text()
    assert "self.gpu_to_cpu_handler = SingleDirectionOffloadingHandler(" in gw and "self.dst_tensors: list[torch.Tensor]" in gw
    mgr = (SRC / "v1/kv_offload/tiering/manager.py").read_text()
    for call in ("tier.submit_store(job_metadata)", "tier.submit_load(job_metadata)", "tier.get_finished_jobs()", "self.primary_tier.prepare_read(", "self.primary_tier.complete_write("):
        assert call in mgr
    base = (SRC / "v1/kv_offload/tiering/base.py").read_text()
    for name in ("def lookup", "def submit_store", "def submit_load", "def get_finished_jobs", "def has_pending_work", "def drain_jobs", "def on_new_request", "def on_request_finished", "def on_schedule_end", "def shutdown"):
        assert name in base
    fm = (SRC / "v1/kv_offload/file_mapper.py").read_text()
    assert "rank=parallel_config.rank" in fm and 'f"{self.base_path}_r{self.rank}"' in fm
    al = (SRC / "v1/kv_offload/tiering/async_lookup.py").read_text()
    for name in ("def batch_lookup", "def lookup", "def flush", "def cleanup", "def shutdown"):
        assert name in al


def test_single_host_assumption_still_present_in_fork():
    """The reason this overlay exists: the fork's tiering shares one /dev/shm
    region across workers and runs fs I/O in the scheduler. If that changes
    upstream, revisit whether the overlay is still needed."""
    sor = (SRC / "v1/kv_offload/cpu/shared_offload_region.py").read_text()
    assert '/dev/shm/vllm_offload_' in sor
    spec = (SRC / "v1/kv_offload/tiering/spec.py").read_text()
    assert "torch.accelerator.current_device_index()" in spec


def test_overlay_uses_only_pinned_surfaces():
    src = OVERLAY.read_text()
    for needed in ("KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)",
                   "self._handlers.gpu_to_cpu_handler.dst_tensors", "self.config.num_workers",
                   "self._generate_job_id()", "parallel_agnostic=False", "BLOCK_SIZE_ALIGNMENT = 1", "def make_file_mapper"):
        assert needed in src, needed
    assert "os.O_DIRECT" not in src  # indexer page (8448 B) is not 512-aligned
    assert "from_offloading_spec(" not in src  # digest must be rank-independent


def test_surfaces_used_by_the_direct_tier():
    gw = (SRC / "v1/kv_offload/cpu/gpu_worker.py").read_text()
    m = methods(SRC / "v1/kv_offload/cpu/gpu_worker.py", "SingleDirectionOffloadingHandler")
    assert {"transfer_async", "get_finished", "wait", "shutdown"} <= m
    for kw in ("gpu_tensors: list[torch.Tensor]", "cpu_tensors: list[torch.Tensor]", "block_size_factor: int",
               "kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]]", "gpu_to_cpu: bool"):
        assert kw in gw, kw
    # stores wait for the calling thread's current stream: pump() must run on the main thread
    assert "stream.wait_stream(current_platform.current_stream())" in gw
    base = (SRC / "v1/kv_offload/base.py").read_text()
    assert "group_sizes: Sequence[int]," in base and "block_indices: Sequence[int]," in base
    assert "class PrepareStoreOutput" in base and "keys_to_store: list[OffloadKey]" in base
    wk = (SRC / "distributed/kv_transfer/kv_connector/v1/offloading/worker.py").read_text()
    assert "def prepare_store_kv" in wk and "self._unsubmitted_store_jobs" in wk
    cb = (SRC / "distributed/kv_transfer/kv_connector/v1/base.py").read_text()
    assert "def get_block_ids_with_load_errors(self) -> set[int]:" in cb
    sched = (SRC / "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py").read_text()
    # keys handed to prepare_store / prepare_load are group-major and parallel to the GPU block ids
    assert "src_spec = self.manager.prepare_load(keys_to_load, req_status.req_context)" in sched
    assert "dst_spec = store_output.store_spec" in sched and "keys_to_store = set(store_output.keys_to_store)" in sched


def test_surfaces_used_by_the_slab_store():
    sched = (SRC / "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py").read_text()
    # _FailureAwareScheduler reads self._jobs[job_id].keys / .is_store before the base completion path
    assert "self._jobs: dict[int, TransferJobStatus] = {}" in sched
    assert "keys: set[OffloadKey]" in sched and "is_store: bool" in sched
    assert "if job_status.is_store:\n                self.manager.complete_store(job_status.keys, req_status.req_context)" in sched
    base = (SRC / "distributed/kv_transfer/kv_connector/v1/base.py").read_text()
    assert "def get_block_ids_with_load_errors(self) -> set[int]:" in base
    src = OVERLAY.read_text()
    for needed in ("SLAB_HEADER_BYTES = 128", 'struct.Struct("<8sIIIIQ36s60x")', "def slab_geometry", "class SlabIO",
                   "self._pwrite_all(fd, b\"\\0\" * SLAB_HEADER_BYTES, off)", "class _FailureAwareScheduler",
                   "self.condemned", "def _evictable"):
        assert needed in src, needed
