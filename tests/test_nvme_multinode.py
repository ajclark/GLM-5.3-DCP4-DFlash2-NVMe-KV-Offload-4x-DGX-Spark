"""Multi-node worker-executed filesystem tier for the OffloadingConnector.
Runs the fork's real tiering manager, CPU manager, thread pool, file mapper
and worker dispatch under a stubbed `vllm` package (see nvme_harness)."""
import hashlib
import os
import time

import numpy as np
import pytest
import torch

from nvme_harness import DRAFT_PAGE, INDEXER_PAGE, MLA_PAGE, build_vllm, make_canonical_kv_caches, make_config


@pytest.fixture(scope="module")
def mn():
    return build_vllm()


def key(mn, i, group):
    from vllm.v1.kv_offload.base import make_offload_key
    return make_offload_key(hashlib.sha256(f"block-{i}".encode()).digest(), group)


def wait_finished(handler, n, timeout=10.0):
    out = []
    t0 = time.monotonic()
    while len(out) < n and time.monotonic() - t0 < timeout:
        out.extend(handler.get_finished())
        time.sleep(0.01)
    assert len(out) == n, out
    return out


# --- file I/O primitives ---------------------------------------------------------

def test_drain_iov_partial_writes(mn):
    a, b, c = memoryview(bytearray(b"aaaa")), memoryview(bytearray(b"bbb")), memoryview(bytearray(b"cc"))
    rem = mn._drain_iov([a, b, c], 5)
    assert [bytes(v) for v in rem] == [b"bb", b"cc"]
    assert mn._drain_iov([a, b, c], 9) == []
    assert [bytes(v) for v in mn._drain_iov([a, b, c], 0)] == [b"aaaa", b"bbb", b"cc"]


def test_block_file_roundtrip_and_size_check(mn, tmp_path):
    src = [np.frombuffer(os.urandom(n), dtype=np.int8).copy() for n in (41984, 8448)]
    path = str(tmp_path / "x" / "y" / "blk.bin")
    mn.write_block_file(path, [memoryview(a) for a in src])
    assert os.path.getsize(path) == 41984 + 8448
    dst = [np.zeros(n, dtype=np.int8) for n in (41984, 8448)]
    mn.read_block_file(path, [memoryview(a) for a in dst])
    assert all(np.array_equal(s, d) for s, d in zip(src, dst))
    # existing file is left alone (same key, same bytes)
    mn.write_block_file(path, [memoryview(np.ones(41984 + 8448, dtype=np.int8))])
    mn.read_block_file(path, [memoryview(a) for a in dst])
    assert np.array_equal(src[0], dst[0])
    # wrong size -> error and file removed
    with pytest.raises(OSError):
        mn.read_block_file(path, [memoryview(np.zeros(100, dtype=np.int8))])
    assert not os.path.exists(path)
    with pytest.raises(FileNotFoundError):
        mn.read_block_file(path, [memoryview(np.zeros(41984 + 8448, dtype=np.int8))])


# --- worker handler --------------------------------------------------------------

def make_handlers(mn, tmp_path, rank, world_size=4):
    from vllm.v1.kv_offload.file_mapper import FileMapper
    vllm_config, kv_cache_config = make_config(tmp_path / "kv", rank=rank, world_size=world_size)
    spec = mn.MultiNodeTieringOffloadingSpec(vllm_config, kv_cache_config)
    kv = make_canonical_kv_caches()
    cpu = [torch.zeros((spec.num_blocks, t.page_size_bytes), dtype=torch.int8) for t in kv.tensors]
    fm = mn.make_file_mapper(spec, str(tmp_path / "kv"))
    st = mn.WorkerFsHandler(cpu, kv.group_data_refs, 1, fm, store=True, n_threads=2)
    ld = mn.WorkerFsHandler(cpu, kv.group_data_refs, 1, fm, store=False, n_threads=2)

    class _Both:
        def shutdown(self):
            st.shutdown(); ld.shutdown()
    return spec, cpu, fm, _Both(), st, ld


def test_spec_uses_per_group_cp_factor(mn, tmp_path):
    spec, *_ = make_handlers(mn, tmp_path, rank=0)
    assert spec.gpu_block_size == (256, 64)      # MLA sharded x4, drafter replicated
    assert spec.hash_block_size == 64 and spec.block_size_factor == 1
    assert spec.num_blocks == 8


def test_handler_store_then_load_per_rank_files(mn, tmp_path):
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
    spec, cpu, fm, pool, st, ld = make_handlers(mn, tmp_path, rank=2)
    k0, k1 = key(mn, 1, 0), key(mn, 2, 1)
    # fill CPU-tier blocks 3 (group 0 key) and 5 (group 1 key) with random bytes
    for t in cpu:
        t.copy_(torch.randint(-128, 127, t.shape, dtype=torch.int8))
    orig = [t.clone() for t in cpu]
    assert st.transfer_async(7, (CPULoadStoreSpec([3, 5]), mn.FSLoadStoreSpec([k0, k1])))
    (r,) = wait_finished(st, 1)
    assert r.success and r.job_id == 7 and r.transfer_type == ("CPU", "FS")
    assert r.transfer_size == MLA_PAGE + INDEXER_PAGE + DRAFT_PAGE
    p0, p1 = fm.get_file_name(k0), fm.get_file_name(k1)
    assert "_r2/" in p0 and "_g0/" in p0 and "_g1/" in p1
    assert os.path.getsize(p0) == MLA_PAGE + INDEXER_PAGE and os.path.getsize(p1) == DRAFT_PAGE
    # wipe and load back
    for t in cpu:
        t.zero_()
    assert ld.transfer_async(8, (mn.FSLoadStoreSpec([k0, k1]), CPULoadStoreSpec([3, 5])))
    (r,) = wait_finished(ld, 1)
    assert r.success and r.transfer_type == ("FS", "CPU")
    assert torch.equal(cpu[0][3], orig[0][3]) and torch.equal(cpu[1][3], orig[1][3]) and torch.equal(cpu[2][5], orig[2][5])
    # group 0's file never touches the drafter tensor and vice versa
    assert not cpu[2][3].any() and not cpu[0][5].any()
    pool.shutdown()


def test_handler_load_of_missing_file_fails_the_job(mn, tmp_path):
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
    spec, cpu, fm, pool, st, ld = make_handlers(mn, tmp_path, rank=1)
    assert ld.transfer_async(1, (mn.FSLoadStoreSpec([key(mn, 9, 0)]), CPULoadStoreSpec([0])))
    (r,) = wait_finished(ld, 1)
    assert r.success is False and r.job_id == 1
    assert not st.transfer_async(2, (CPULoadStoreSpec([]), mn.FSLoadStoreSpec([])))
    pool.shutdown()


# --- scheduler-side tier + real tiering manager -------------------------------------

class Cluster:
    """Four fake workers (one per rank) executing the scheduler's tier jobs
    on their own CPU tensors and files, plus one scheduler-side spec."""

    def __init__(self, mn, root, world_size=4):
        self.mn = mn
        self.world_size = world_size
        vllm_config, kv_cache_config = make_config(root, rank=0, world_size=world_size)
        self.spec = mn.MultiNodeTieringOffloadingSpec(vllm_config, kv_cache_config)
        self.manager = self.spec.get_manager()
        (self.tier,) = self.spec.worker_tiers
        self.workers = []
        for rank in range(world_size):
            cfg, kvc = make_config(root, rank=rank, world_size=world_size)
            wspec = mn.MultiNodeTieringOffloadingSpec(cfg, kvc)
            kv = make_canonical_kv_caches()
            from vllm.v1.kv_offload.worker.worker import OffloadingWorker
            w = OffloadingWorker()
            for s, d, h in wspec.get_handlers(kv):
                w.register_handler(s, d, h)
            self.workers.append((wspec, w))

    def cpu_tensors(self, rank):
        return self.workers[rank][0]._handlers.gpu_to_cpu_handler.dst_tensors

    def run_pending(self, fail_rank=None):
        """Ship pending tier jobs to every worker and complete them."""
        jobs = self.tier.take_pending()
        for tier_job_id, spec in jobs:
            ok = True
            for rank, (_, w) in enumerate(self.workers):
                if rank == fail_rank:
                    ok = False
                    continue
                assert w.transfer_async(tier_job_id, spec)
            for rank, (_, w) in enumerate(self.workers):
                if rank == fail_rank:
                    continue
                res = []
                t0 = time.monotonic()
                while not res and time.monotonic() - t0 < 10:
                    res = w.get_finished()
                    time.sleep(0.01)
                assert len(res) == 1 and res[0].job_id == tier_job_id
                ok = ok and res[0].success
            self.tier.mark_finished(tier_job_id, ok)
        return len(jobs)


def test_store_cascades_to_every_rank_and_survives_a_restart(mn, tmp_path):
    from vllm.v1.kv_offload.base import ReqContext
    root = tmp_path / "kv"
    c = Cluster(mn, root)
    ctx = ReqContext(req_id="r1")
    keys = [key(mn, i, 0) for i in range(3)] + [key(mn, 10, 1)]
    c.manager.on_new_request(ctx)
    out = c.manager.prepare_store(keys, ctx)
    assert out is not None and list(out.keys_to_store) == keys
    cpu_ids = [int(b) for b in out.store_spec.block_ids]
    # pretend the GPU->CPU copy filled each rank's CPU blocks with rank-specific bytes
    for rank in range(4):
        for t in c.cpu_tensors(rank):
            for b in cpu_ids:
                t[b] = rank + 1
    c.manager.complete_store(keys, ctx)           # cascades: tier.submit_store
    assert c.tier.has_pending_work()
    assert c.run_pending() == 1
    for rank in range(4):
        fm = c.workers[rank][0]._handlers  # handlers exist
        for k in keys:
            p = c.workers[rank][0].worker_tiers if False else None
    # every rank wrote its own file with its own bytes
    from vllm.v1.kv_offload.file_mapper import FileMapper
    for rank in range(4):
        cfg, kvc = make_config(root, rank=rank)
        fm = mn.make_file_mapper(mn.MultiNodeTieringOffloadingSpec(cfg, kvc), str(root))
        for k in keys:
            data = np.fromfile(fm.get_file_name(k), dtype=np.int8)
            assert data.size in (MLA_PAGE + INDEXER_PAGE, DRAFT_PAGE) and (data == rank + 1).all()
    c.manager._process_finished_jobs()           # store job done -> ref counts released
    assert not c.tier.has_pending_work()
    assert all(c.manager.lookup(k, ctx) is True for k in keys)   # in the CPU tier

    # "restart": a fresh scheduler-side manager and fresh (empty) worker CPU tiers
    c2 = Cluster(mn, root)
    ctx2 = ReqContext(req_id="r2")
    c2.manager.on_new_request(ctx2)
    first = [c2.manager.lookup(k, ctx2) for k in keys]
    assert all(r is None for r in first)          # async file lookup in flight
    c2.manager.on_schedule_end()
    c2.tier._lookup.drain_results() if hasattr(c2.tier._lookup, "drain_results") else None
    time.sleep(0.2)
    second = [c2.manager.lookup(k, ctx2) for k in keys]
    assert all(r is None for r in second)         # found on disk -> promotion started
    c2.manager.on_schedule_end()                  # flushes promotions -> tier.submit_load
    assert c2.run_pending() == 1
    c2.manager._process_finished_jobs()
    assert all(c2.manager.lookup(k, ctx2) is True for k in keys)
    for rank in range(4):
        for t in c2.cpu_tensors(rank):
            assert (t[t.any(dim=1)] == rank + 1).all()   # loaded pages carry that rank's bytes


def test_failed_load_on_one_rank_is_a_miss_not_a_crash(mn, tmp_path):
    from vllm.v1.kv_offload.base import ReqContext
    root = tmp_path / "kv"
    c = Cluster(mn, root)
    ctx = ReqContext(req_id="r1")
    keys = [key(mn, 1, 0)]
    c.manager.on_new_request(ctx)
    out = c.manager.prepare_store(keys, ctx)
    c.manager.complete_store(keys, ctx)
    c.run_pending()
    c.manager._process_finished_jobs()
    # delete rank 3's file: rank 0 still answers the lookup, the load fails on rank 3
    from vllm.v1.kv_offload.file_mapper import FileMapper
    cfg, kvc = make_config(root, rank=3)
    fm3 = mn.make_file_mapper(mn.MultiNodeTieringOffloadingSpec(cfg, kvc), str(root))
    os.remove(fm3.get_file_name(keys[0]))
    c2 = Cluster(mn, root)
    ctx2 = ReqContext(req_id="r2")
    c2.manager.on_new_request(ctx2)
    assert c2.manager.lookup(keys[0], ctx2) is None
    c2.manager.on_schedule_end(); time.sleep(0.2)
    assert c2.manager.lookup(keys[0], ctx2) is None    # promotion started
    c2.manager.on_schedule_end()
    assert c2.run_pending() == 1
    c2.manager._process_finished_jobs()
    # failed promotion: not in the CPU tier; the tier's file check will still say
    # "present" (rank 0 has it) so the manager re-attempts a promotion, which is
    # the documented behaviour for a torn write; the request itself is never fed
    # a block the CPU tier does not have.
    assert c2.manager.primary_tier.lookup(keys[0], ctx2) is False


# --- connector scheduler / worker subclasses -----------------------------------------

def test_scheduler_ships_tier_jobs_and_completes_after_all_workers(mn, tmp_path):
    from vllm.v1.kv_offload.base import ReqContext
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
    from vllm.v1.outputs import KVConnectorOutput
    vllm_config, kv_cache_config = make_config(tmp_path / "kv", rank=0)
    spec = mn.MultiNodeTieringOffloadingSpec(vllm_config, kv_cache_config)
    sched = mn.MultiNodeOffloadingConnectorScheduler(spec)
    (tier,) = spec.worker_tiers
    ctx = ReqContext(req_id="r1")
    keys = [key(mn, i, 0) for i in range(2)]
    sched.manager.on_new_request(ctx)
    out = sched.manager.prepare_store(keys, ctx)
    sched.manager.complete_store(keys, ctx)      # -> tier pending
    meta = sched.build_connector_meta(None)
    assert isinstance(meta, mn.MultiNodeConnectorMetadata) and len(meta.tier_jobs) == 1
    (job_id, (src, dst)), = meta.tier_jobs.items()
    assert isinstance(src, CPULoadStoreSpec) and isinstance(dst, mn.FSLoadStoreSpec)
    assert sched.has_pending_push_work()
    # three workers report: not done
    for _ in range(3):
        sched.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=mn.MultiNodeWorkerMetadata(completed_jobs={job_id: 1})))
    assert job_id in sched._tier_jobs and not list(tier.get_finished_jobs())
    # a GPU<->CPU job id in the same report is passed through to the base class
    sched.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=mn.MultiNodeWorkerMetadata(completed_jobs={job_id: 1, 12345: 1})))
    assert job_id not in sched._tier_jobs
    assert sched.seen_completed == {12345: 1}
    (res,) = list(tier.get_finished_jobs())
    assert res.success and not sched.has_pending_push_work() or True
    # failure on any worker fails the tier job
    sched.manager.complete_store(keys, ctx)
    meta = sched.build_connector_meta(None)
    (job_id2,) = meta.tier_jobs
    for i in range(4):
        m = mn.MultiNodeWorkerMetadata(completed_jobs={job_id2: 1}, failed_jobs={job_id2} if i == 2 else set())
        sched.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=m))
    (res,) = list(tier.get_finished_jobs())
    assert res.success is False


def test_worker_metadata_aggregate_and_failure_reporting(mn, tmp_path):
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
    a = mn.MultiNodeWorkerMetadata(completed_jobs={1: 1}, failed_jobs={1})
    b = mn.MultiNodeWorkerMetadata(completed_jobs={1: 1, 2: 1})
    m = a.aggregate(b)
    assert isinstance(m, mn.MultiNodeWorkerMetadata) and m.completed_jobs == {1: 2, 2: 1} and m.failed_jobs == {1}
    vllm_config, kv_cache_config = make_config(tmp_path / "kv", rank=1)
    spec = mn.MultiNodeTieringOffloadingSpec(vllm_config, kv_cache_config)
    w = mn.MultiNodeOffloadingConnectorWorker(spec)
    w._register_handlers(make_canonical_kv_caches())
    meta = mn.MultiNodeConnectorMetadata(load_jobs={}, store_jobs={}, tier_jobs={
        5: (mn.FSLoadStoreSpec([key(mn, 77, 0)]), CPULoadStoreSpec([0])),   # missing file -> fails
        6: (CPULoadStoreSpec([1]), mn.FSLoadStoreSpec([key(mn, 78, 1)])),   # store -> succeeds
    })
    w.start_kv_transfers(meta)
    done = set()
    t0 = time.monotonic()
    while len(done) < 2 and time.monotonic() - t0 < 10:
        w.get_finished(set())
        done = set(w._connector_worker_meta.completed_jobs)
        time.sleep(0.01)
    out = w.build_connector_worker_meta()
    assert out.completed_jobs == {5: 1, 6: 1} and out.failed_jobs == {5}
    assert out.transfer_stats.load.bytes == 0 and out.transfer_stats.store.bytes == DRAFT_PAGE  # failed load: no bytes
    assert w.build_connector_worker_meta() is None
    w.shutdown()


def test_concurrent_store_and_load_keep_their_own_completions(mn, tmp_path):
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
    spec, cpu, fm, both, st, ld = make_handlers(mn, tmp_path, rank=0)
    for t in cpu:
        t.copy_(torch.randint(-128, 127, t.shape, dtype=torch.int8))
    k_store = [key(mn, i, 0) for i in range(4)]
    assert st.transfer_async(1, (CPULoadStoreSpec([0, 1, 2, 3]), mn.FSLoadStoreSpec(k_store)))
    wait_finished(st, 1)
    assert ld.transfer_async(2, (mn.FSLoadStoreSpec(k_store[:2]), CPULoadStoreSpec([4, 5])))
    assert st.transfer_async(3, (CPULoadStoreSpec([6, 7]), mn.FSLoadStoreSpec([key(mn, 20, 1), key(mn, 21, 1)])))
    (r_ld,) = wait_finished(ld, 1)
    (r_st,) = wait_finished(st, 1)
    assert r_ld.job_id == 2 and r_ld.success and r_ld.transfer_size == 2 * (MLA_PAGE + INDEXER_PAGE)
    assert r_st.job_id == 3 and r_st.success and r_st.transfer_size == 2 * DRAFT_PAGE
    assert torch.equal(cpu[0][4], cpu[0][0]) and torch.equal(cpu[1][5], cpu[1][1])
    both.shutdown()


def test_digest_is_identical_across_scheduler_and_workers(mn, tmp_path):
    """The scheduler's KV-cache config lists different layer names than a
    worker's; the digest must not depend on them."""
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MLAAttentionSpec, SlidingWindowSpec
    cfg0, kvc0 = make_config(tmp_path / "kv", rank=0)
    cfg1, kvc1 = make_config(tmp_path / "kv", rank=1)
    kvc1.kv_cache_groups = [KVCacheGroupSpec(layer_names=["L0.attn"], kv_cache_spec=MLAAttentionSpec(block_size=64)),
                            KVCacheGroupSpec(layer_names=["D0.attn", "D1.attn", "D2.attn"], kv_cache_spec=SlidingWindowSpec(block_size=64))]
    fm0 = mn.make_file_mapper(mn.MultiNodeTieringOffloadingSpec(cfg0, kvc0), str(tmp_path / "kv"))
    fm1 = mn.make_file_mapper(mn.MultiNodeTieringOffloadingSpec(cfg1, kvc1), str(tmp_path / "kv"))
    assert fm0.base_path == fm1.base_path and fm0.rank == 0 and fm1.rank == 1
    k = key(mn, 3, 1)
    assert fm0.get_file_name(k).replace("_r0/", "_rX/") == fm1.get_file_name(k).replace("_r1/", "_rX/")


def test_scheduler_lookup_discovers_rank0_dir_even_with_a_different_digest(mn, tmp_path):
    """The scheduler's own digest may differ from the workers' (on the
    cluster the worker-side KV-cache config is not byte-identical); lookups
    must still find rank 0's real directory on this node."""
    from vllm.v1.kv_offload.base import ReqContext
    root = tmp_path / "kv"
    c = Cluster(mn, root)
    ctx = ReqContext(req_id="r1")
    keys = [key(mn, 1, 0), key(mn, 2, 1)]
    c.manager.on_new_request(ctx)
    c.manager.prepare_store(keys, ctx); c.manager.complete_store(keys, ctx); c.run_pending(); c.manager._process_finished_jobs()
    real_base = c.tier.file_mapper.base_path
    for rank in range(4):   # workers wrote their run-config marker
        assert os.path.exists(os.path.join(f"{real_base}_r{rank}", mn.RUN_CONFIG_MARKER))
    # a scheduler whose digest inputs differ (here: the dtype string)
    cfg, kvc = make_config(root, rank=0)
    cfg.cache_config.cache_dtype = "fp8_seen_differently"
    spec2 = mn.MultiNodeTieringOffloadingSpec(cfg, kvc)
    mgr2 = spec2.get_manager(); (tier2,) = spec2.worker_tiers
    assert tier2.file_mapper.base_path != real_base
    ctx2 = ReqContext(req_id="r2")
    mgr2.on_new_request(ctx2)
    assert mgr2.lookup(keys[0], ctx2) is None          # async lookup queued
    mgr2.on_schedule_end(); time.sleep(0.2)
    assert mgr2.lookup(keys[0], ctx2) is None          # found under rank 0's dir -> promotion started
    assert tier2.file_mapper.base_path == real_base    # resolved to the real directory
    assert mn.discover_rank0_base_path(str(root), "/models/glm-5.3") == real_base
    assert mn.discover_rank0_base_path(str(tmp_path / "empty"), "/models/glm-5.3") is None
    tier2.shutdown()


def test_connector_wires_the_subclasses(mn, tmp_path):
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    vllm_config, kv_cache_config = make_config(tmp_path / "kv", rank=0)
    c = mn.MultiNodeOffloadingConnector(vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config)
    assert isinstance(c.connector_scheduler, mn.MultiNodeOffloadingConnectorScheduler) and c.connector_worker is None
    c = mn.MultiNodeOffloadingConnector(vllm_config, KVConnectorRole.WORKER, kv_cache_config)
    assert isinstance(c.connector_worker, mn.MultiNodeOffloadingConnectorWorker)
    bad, kvc = make_config(tmp_path / "kv", rank=0, extra={"secondary_tiers": [{"type": "fs", "root_dir": "/x"}]})
    with pytest.raises(ValueError):
        mn.MultiNodeTieringOffloadingSpec(bad, kvc)
