"""Direct NVMe tier: disk-resident offload manager (scheduler side) and the
bounce-buffer GPU<->NVMe handlers (worker side), against the fork's real
worker dispatch and file mapper, with the CUDA copy handler stubbed as a
synchronous row copy that honours the same spec contract."""
import hashlib
import os
import time

import pytest
import torch

from nvme_harness import DRAFT_PAGE, INDEXER_PAGE, MLA_PAGE, build_vllm, make_canonical_kv_caches, make_config


@pytest.fixture(scope="module")
def mn():
    return build_vllm()


def key(mn, i, group):
    from vllm.v1.kv_offload.base import make_offload_key
    return make_offload_key(hashlib.sha256(f"blk-{i}".encode()).digest(), group)


def gpu_spec(mn, keys, gpu_ids):
    from vllm.v1.kv_offload.base import GPULoadStoreSpec, get_offload_group_idx
    sizes = [0, 0]
    for k in keys:
        sizes[get_offload_group_idx(k)] += 1
    return GPULoadStoreSpec(gpu_ids, group_sizes=sizes, block_indices=[0, 0])


def drive(handler, want, timeout=10.0):
    out = []
    t0 = time.monotonic()
    while len(out) < want and time.monotonic() - t0 < timeout:
        out.extend(handler.get_finished())
        time.sleep(0.005)
    assert len(out) == want, out
    return out


def make_worker(mn, root, rank):
    from vllm.v1.kv_offload.worker.worker import OffloadingWorker
    cfg, kvc = make_config(root, rank=rank, direct=True)
    spec = mn.MultiNodeDirectFsOffloadingSpec(cfg, kvc)
    kv = make_canonical_kv_caches()
    w = OffloadingWorker()
    handlers = {}
    for s, d, hnd in spec.get_handlers(kv):
        w.register_handler(s, d, hnd)
        handlers[(s.medium(), d.medium())] = hnd
    return spec, kv, w, handlers


def test_spec_and_manager_basics(mn, tmp_path):
    from vllm.v1.kv_offload.base import ReqContext
    cfg, kvc = make_config(tmp_path / "kv", rank=0, direct=True)
    spec = mn.MultiNodeDirectFsOffloadingSpec(cfg, kvc)
    assert spec.gpu_block_size == (256, 64) and spec.block_size_factor == 1
    m = spec.get_manager()
    assert isinstance(m, mn.FSOffloadingManager)
    assert os.path.exists(mn.make_file_mapper(spec, str(tmp_path / "kv")).get_config_file_path())
    ctx = ReqContext(req_id="r")
    ks = [key(mn, i, 0) for i in range(3)]
    out = m.prepare_store(ks, ctx)
    assert out.keys_to_store == ks and isinstance(out.store_spec, mn.FSLoadStoreSpec) and out.store_spec.keys == ks
    assert m.has_pending_work()
    assert m.lookup(ks[0], ctx) is None                     # store in flight
    assert m.prepare_store(ks, ctx).keys_to_store == []     # not twice
    m.complete_store(ks[:2], ctx, success=True); m.complete_store(ks[2:], ctx, success=False)
    assert not m.has_pending_work()
    assert m.lookup(ks[0], ctx) is True and m.lookup(ks[2], ctx) is None  # known / async file check
    m.on_schedule_end(); time.sleep(0.2)
    assert m.lookup(ks[2], ctx) is False                    # nothing on disk
    with pytest.raises(ValueError):
        mn.MultiNodeDirectFsOffloadingSpec(*make_config(tmp_path / "kv", rank=0, direct=True, extra={"root_dir": ""}))
    m.shutdown()


def test_store_and_load_roundtrip_through_bounce_chunks(mn, tmp_path):
    from vllm.v1.kv_offload.base import ReqContext
    root = tmp_path / "kv"
    spec, kv, w, handlers = make_worker(mn, root, rank=2)
    ctrl = spec.controller
    assert ctrl.n_bounce == 3
    gpu = [t.tensor for t in kv.tensors]
    for t in gpu:
        t.copy_(torch.randint(-128, 127, t.shape, dtype=torch.int8))
    orig = [t.clone() for t in gpu]
    # 5 target blocks + 3 drafter blocks, group-major, parallel gpu block ids
    keys = [key(mn, i, 0) for i in range(5)] + [key(mn, 10 + i, 1) for i in range(3)]
    gpu_ids = [1, 2, 3, 4, 5, 20, 21, 22]
    assert w.transfer_async(100, (gpu_spec(mn, keys, gpu_ids), mn.FSLoadStoreSpec(keys)))
    (res,) = drive(handlers[("GPU", "FS")], 1)
    assert res.job_id == 100 and res.success and res.transfer_type == ("GPU", "FS")
    assert res.transfer_size == 5 * (MLA_PAGE + INDEXER_PAGE) + 3 * DRAFT_PAGE
    assert len(ctrl.free) == 3 and not ctrl.jobs               # slots recycled
    assert len(ctrl.g2b.calls) >= 3                            # chunked through 3 slots
    fm = mn.make_file_mapper(spec, str(root))
    for k in keys:
        assert os.path.getsize(fm.get_file_name(k)) == ctrl.files.block_nbytes(k)
    # wipe the GPU blocks and load them back
    for t in gpu:
        t.zero_()
    assert w.transfer_async(101, (mn.FSLoadStoreSpec(keys), gpu_spec(mn, keys, gpu_ids)))
    (res,) = drive(handlers[("FS", "GPU")], 1)
    assert res.job_id == 101 and res.success and res.transfer_type == ("FS", "GPU")
    for gb in (1, 2, 3, 4, 5):
        assert torch.equal(gpu[0][gb], orig[0][gb]) and torch.equal(gpu[1][gb], orig[1][gb])
        assert not gpu[2][gb].any()                            # drafter tensor untouched for target keys
    for gb in (20, 21, 22):
        assert torch.equal(gpu[2][gb], orig[2][gb]) and not gpu[0][gb].any()
    assert len(ctrl.free) == 3 and not ctrl.jobs and not ctrl.take_failed_gpu_blocks()
    # a re-store of existing keys is a cheap no-op on disk (files kept)
    mt = os.path.getmtime(fm.get_file_name(keys[0]))
    assert w.transfer_async(102, (gpu_spec(mn, keys[:2], gpu_ids[:2]), mn.FSLoadStoreSpec(keys[:2])))
    drive(handlers[("GPU", "FS")], 1)
    assert os.path.getmtime(fm.get_file_name(keys[0])) == mt
    w.shutdown()


def test_load_with_a_missing_file_fails_only_those_blocks(mn, tmp_path):
    root = tmp_path / "kv"
    spec, kv, w, handlers = make_worker(mn, root, rank=1)
    gpu = [t.tensor for t in kv.tensors]
    for t in gpu:
        t.copy_(torch.randint(-128, 127, t.shape, dtype=torch.int8))
    orig = [t.clone() for t in gpu]
    stored = [key(mn, i, 0) for i in range(4)]
    assert w.transfer_async(1, (gpu_spec(mn, stored, [1, 2, 3, 4]), mn.FSLoadStoreSpec(stored)))
    drive(handlers[("GPU", "FS")], 1)
    for t in gpu:
        t.zero_()
    keys = stored[:2] + [key(mn, 99, 0)] + stored[2:]      # one key never stored
    ids = [1, 2, 7, 3, 4]
    assert w.transfer_async(2, (mn.FSLoadStoreSpec(keys), gpu_spec(mn, keys, ids)))
    (res,) = drive(handlers[("FS", "GPU")], 1)
    assert res.job_id == 2 and res.success is False
    failed = spec.controller.take_failed_gpu_blocks()
    assert 7 in failed and failed <= {7, 1, 2, 3, 4}         # its chunk-mates may be reported too
    # every block whose file existed and was not in the failed chunk is restored
    for gb in (1, 2, 3, 4):
        if gb not in failed:
            assert torch.equal(gpu[0][gb], orig[0][gb])
    assert len(spec.controller.free) == 3 and not spec.controller.jobs
    w.shutdown()


def test_direct_worker_reports_finished_recving_and_invalid_blocks(mn, tmp_path):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingConnectorMetadata, TransferJob
    root = tmp_path / "kv"
    cfg, kvc = make_config(root, rank=3, direct=True)
    spec = mn.MultiNodeDirectFsOffloadingSpec(cfg, kvc)
    w = mn.MultiNodeDirectConnectorWorker(spec)
    kv = make_canonical_kv_caches()
    w._register_handlers(kv)
    keys = [key(mn, 5, 0)]
    meta = OffloadingConnectorMetadata(
        load_jobs={9: TransferJob(req_id="req-A", transfer_spec=(mn.FSLoadStoreSpec(keys), gpu_spec(mn, keys, [3])))},
        store_jobs={})
    w.start_kv_transfers(meta)
    recv = set()
    t0 = time.monotonic()
    while not recv and time.monotonic() - t0 < 10:
        _, recv = w.get_finished(set())
        time.sleep(0.005)
    assert recv == {"req-A"}                                  # the request is released even on failure
    assert w.take_invalid_block_ids() == {3}                  # and its block is recomputed
    out = w.build_connector_worker_meta()
    assert out.completed_jobs == {9: 1} and out.transfer_stats.load.bytes == 0
    w.shutdown()


def test_direct_connector_wiring(mn, tmp_path):
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    cfg, kvc = make_config(tmp_path / "kv", rank=0, direct=True)
    c = mn.MultiNodeDirectConnector(cfg, KVConnectorRole.SCHEDULER, kvc)
    assert c.connector_scheduler is not None and isinstance(c.connector_scheduler.manager, mn.FSOffloadingManager)
    assert c.get_block_ids_with_load_errors() == set()
    c = mn.MultiNodeDirectConnector(cfg, KVConnectorRole.WORKER, kvc)
    assert isinstance(c.connector_worker, mn.MultiNodeDirectConnectorWorker)
    bad, kvc2 = make_config(tmp_path / "kv", rank=0)          # tiered spec -> refused
    with pytest.raises(ValueError):
        mn.MultiNodeDirectConnector(bad, KVConnectorRole.WORKER, kvc2)


def test_scheduler_side_flow_store_then_lookup_from_a_fresh_manager(mn, tmp_path):
    """Store through one worker, then a fresh scheduler-side manager (a restart)
    must see the files and hand out an FS load spec; the same keys are not
    stored twice."""
    from vllm.v1.kv_offload.base import ReqContext
    root = tmp_path / "kv"
    spec, kv, w, handlers = make_worker(mn, root, rank=0)
    keys = [key(mn, i, 0) for i in range(3)] + [key(mn, 30, 1)]
    assert w.transfer_async(1, (gpu_spec(mn, keys, [1, 2, 3, 9]), mn.FSLoadStoreSpec(keys)))
    drive(handlers[("GPU", "FS")], 1)
    cfg, kvc = make_config(root, rank=0, direct=True)
    cfg.cache_config.cache_dtype = "fp8_seen_differently"   # scheduler digest differs, as on the cluster
    m = mn.MultiNodeDirectFsOffloadingSpec(cfg, kvc).get_manager()
    ctx = ReqContext(req_id="r")
    assert [m.lookup(k, ctx) for k in keys] == [None] * 4
    m.on_schedule_end(); time.sleep(0.2)
    assert all(m.lookup(k, ctx) is True for k in keys)
    assert m.prepare_store(keys, ctx).keys_to_store == []
    spec_ = m.prepare_load(keys, ctx)
    assert isinstance(spec_, mn.FSLoadStoreSpec) and spec_.keys == keys and m.has_pending_work()
    m.complete_load(keys, ctx)
    assert not m.has_pending_work()
    m.shutdown(); w.shutdown()
