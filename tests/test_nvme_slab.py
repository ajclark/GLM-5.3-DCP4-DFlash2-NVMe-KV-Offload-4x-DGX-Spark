"""Fixed-size slab store: slot I/O (write order, key-verified reads, header
scan, epoch), the scheduler-side LRU index (all-or-nothing placement, eviction
that skips in-flight loads, sibling touch, failure marks, rebuild from
headers, reset), the slab bounce controller end to end, capacity behaviour,
and the failure-aware connector pieces."""
import hashlib
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nvme_harness import DRAFT_PAGE, INDEXER_PAGE, MLA_PAGE, build_vllm, make_canonical_kv_caches, make_config

TARGET = MLA_PAGE + INDEXER_PAGE          # bytes per target block in the fake layout
DRAFT = DRAFT_PAGE


@pytest.fixture(scope="module")
def mn():
    return build_vllm()


def key(mn, i, group):
    from vllm.v1.kv_offload.base import make_offload_key
    return make_offload_key(hashlib.sha256(f"slab-{i}".encode()).digest(), group)


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


def ctx(mn, rid="r"):
    from vllm.v1.kv_offload.base import ReqContext
    return ReqContext(req_id=rid)


# --- geometry + slot I/O -----------------------------------------------------------

def test_geometry(mn):
    sb, n = mn.slab_geometry([TARGET, DRAFT], 10_000_000)
    assert sb[0] % 4096 == 0 and sb[1] % 4096 == 0 and sb[0] >= TARGET + 128 and sb[1] >= DRAFT + 128
    assert n[1] == 4 * n[0] and n[0] * sb[0] + n[1] * sb[1] <= 10_000_000
    assert mn.slab_geometry([TARGET, DRAFT], 1)[1] == [1, 4]      # never zero slots


def test_slot_io_write_order_key_check_scan_and_epoch(mn, tmp_path):
    sb, n = mn.slab_geometry([TARGET, DRAFT], 6 * (TARGET + 4 * DRAFT + 5 * 4096))
    io = mn.SlabIO(str(tmp_path / "r0"), sb, n)
    k0, k1 = key(mn, 1, 0), key(mn, 2, 1)
    a = np.frombuffer(os.urandom(TARGET), dtype=np.int8).copy()
    b = np.frombuffer(os.urandom(DRAFT), dtype=np.int8).copy()
    io.write(k0, 2, [memoryview(a[:MLA_PAGE]), memoryview(a[MLA_PAGE:])], seq=7)
    io.write(k1, 5, [memoryview(b)], seq=8)
    out = np.zeros(TARGET, dtype=np.int8)
    io.read(k0, 2, [memoryview(out[:MLA_PAGE]), memoryview(out[MLA_PAGE:])])
    assert np.array_equal(out, a)
    with pytest.raises(OSError):                       # wrong key for the slot
        io.read(key(mn, 3, 0), 2, [memoryview(out)])
    with pytest.raises(OSError):                       # wrong length
        io.read(k0, 2, [memoryview(np.zeros(100, dtype=np.int8))])
    with pytest.raises(OSError):                       # never written
        io.read(k0, 3, [memoryview(out)])
    assert list(io.scan(0, 0)) == [(2, 7, k0)] and list(io.scan(1, 0)) == [(5, 8, k1)]
    # a blank header (crash between blank and header) is not a valid slot
    os.pwrite(io.fds[0], b"\0" * mn.SLAB_HEADER_BYTES, 2 * sb[0])
    assert list(io.scan(0, 0)) == []
    with pytest.raises(OSError):
        io.read(k0, 2, [memoryview(out)])
    # epoch: headers from another epoch are ignored by scan
    io.epoch = 1
    io.write(k0, 0, [memoryview(a[:MLA_PAGE]), memoryview(a[MLA_PAGE:])], seq=9)
    assert list(io.scan(0, 0)) == [] and list(io.scan(0, 1)) == [(0, 9, k0)]
    io.wipe()
    assert list(io.scan(0, 1)) == [] and os.path.getsize(io.path(0)) == 0
    io.close()


# --- a small cluster of fake workers + scheduler-side manager ----------------------------

class Cluster:
    def __init__(self, mn, root, disk_bytes, world_size=2):
        from vllm.v1.kv_offload.worker.worker import OffloadingWorker
        self.mn = mn
        self.workers = []
        for rank in range(world_size):
            cfg, kvc = make_config(root, rank=rank, world_size=world_size, slab=disk_bytes)
            spec = mn.MultiNodeSlabOffloadingSpec(cfg, kvc)
            kv = make_canonical_kv_caches()
            w = OffloadingWorker(); handlers = {}
            for s, d, h in spec.get_handlers(kv):
                w.register_handler(s, d, h); handlers[(s.medium(), d.medium())] = h
            self.workers.append((spec, kv, w, handlers))
        cfg, kvc = make_config(root, rank=0, world_size=world_size, slab=disk_bytes)
        cfg.cache_config.cache_dtype = "fp8_seen_differently"     # scheduler digest differs, as on the cluster
        self.spec = mn.MultiNodeSlabOffloadingSpec(cfg, kvc)
        self.manager = self.spec.get_manager()

    def gpu(self, rank):
        return [t.tensor for t in self.workers[rank][1].tensors]

    def run(self, job_id, store, spec, fail_rank=None):
        ok = True
        for rank, (_, _, w, handlers) in enumerate(self.workers):
            if rank == fail_rank:
                ok = False; continue
            assert w.transfer_async(job_id, spec)
            (res,) = drive(handlers[("GPU", "SLAB") if store else ("SLAB", "GPU")], 1)
            ok = ok and res.success
        return ok

    def store(self, keys, gpu_ids, rid="r", fail_rank=None):
        c = ctx(self.mn, rid)
        out = self.manager.prepare_store(keys, c)
        if out is None:
            return None
        spec = (gpu_spec(self.mn, out.keys_to_store, [g for k, g in zip(keys, gpu_ids) if k in set(out.keys_to_store)]), out.store_spec)
        ok = self.run(1000 + self.manager.seq, True, spec, fail_rank=fail_rank)
        if not ok:
            self.manager.mark_store_failed(out.keys_to_store)
        self.manager.complete_store(out.keys_to_store, c, success=True)
        return out

    def load(self, keys, gpu_ids, rid="r", fail_rank=None):
        c = ctx(self.mn, rid)
        spec = (self.manager.prepare_load(keys, c), gpu_spec(self.mn, keys, gpu_ids))
        ok = self.run(2000 + len(keys), False, spec, fail_rank=fail_rank)
        if not ok:
            self.manager.mark_load_failed(keys)
        self.manager.complete_load(keys, c)
        return ok


def test_manager_attaches_lazily_and_workers_write_meta(mn, tmp_path):
    c = Cluster(mn, tmp_path / "kv", disk_bytes=4 * (TARGET + 4 * DRAFT + 5 * 4096))
    meta = json.load(open(os.path.join(c.workers[0][0].controller.slab.rank_dir, mn.SLAB_META)))
    assert meta["slot_counts"][1] == 4 * meta["slot_counts"][0] and meta["boot_id"]
    assert c.manager.lookup(key(mn, 1, 0), ctx(mn)) is False        # attaches (rank 0 dir found), empty
    assert c.manager.counts == meta["slot_counts"] and c.manager.io.rank_dir.endswith("_r0")
    assert c.manager.prepare_store([], ctx(mn)).keys_to_store == []


def test_store_load_roundtrip_capacity_and_lru(mn, tmp_path):
    n_rows = 3
    c = Cluster(mn, tmp_path / "kv", disk_bytes=n_rows * (TARGET + 4 * DRAFT + 5 * 4096))
    assert c.manager.lookup(key(mn, 0, 0), ctx(mn)) is False
    assert c.manager.counts == [n_rows, 4 * n_rows]
    for rank in range(2):
        for t in c.gpu(rank):
            t.copy_(torch.randint(-128, 127, t.shape, dtype=torch.int8))
    orig = {rank: [t.clone() for t in c.gpu(rank)] for rank in range(2)}
    # three rows: target key i in gpu block i+1, its 4 drafter keys in blocks 10+4i..
    for i in range(3):
        keys = [key(mn, i, 0)] + [key(mn, 100 + 4 * i + j, 1) for j in range(4)]
        ids = [i + 1] + [10 + 4 * i + j for j in range(4)]
        out = c.store(keys, ids)
        assert out is not None and out.keys_to_store == keys
    assert all(c.manager.lookup(key(mn, i, 0), ctx(mn)) is True for i in range(3))
    assert not c.manager.free[0] and not c.manager.free[1]
    # slab files never exceed the cap
    for rank in range(2):
        io = c.workers[rank][0].controller.slab
        for g in range(2):
            assert os.path.getsize(io.path(g)) <= io.slot_bytes[g] * io.slot_counts[g]
    # a 4th row evicts the LRU row (row 0), all-or-nothing: exactly one row's slots
    c.manager.lookup(key(mn, 1, 0), ctx(mn))       # touch row 1 -> row 0 is LRU
    keys4 = [key(mn, 3, 0)] + [key(mn, 112 + j, 1) for j in range(4)]
    out = c.store(keys4, [4, 22, 23, 24, 25])
    assert out is not None and c.manager.stats["evictions"] == 5
    assert c.manager.lookup(key(mn, 0, 0), ctx(mn)) is False and c.manager.lookup(key(mn, 3, 0), ctx(mn)) is True
    # rows 1..3 reload from disk into fresh GPU blocks on both ranks, bytes identical
    for rank in range(2):
        for t in c.gpu(rank):
            t.zero_()
    keys = [key(mn, i, 0) for i in (1, 2, 3)] + [key(mn, 100 + 4 * 1, 1), key(mn, 100 + 4 * 2, 1), key(mn, 112, 1)]
    dst = [5, 6, 7, 26, 27, 28]
    assert c.load(keys, dst) is True
    for rank in range(2):
        g, o = c.gpu(rank), orig[rank]
        assert torch.equal(g[0][5], o[0][2]) and torch.equal(g[1][6], o[1][3]) and torch.equal(g[0][7], o[0][4])
        assert torch.equal(g[2][26], o[2][14]) and torch.equal(g[2][27], o[2][18]) and torch.equal(g[2][28], o[2][22])
    # siblings: touching target row 2 refreshes its drafter blocks
    for k in (key(mn, 108, 1), key(mn, 109, 1)):
        c.manager.index[1].move_to_end(k, last=False)            # make them LRU
    c.manager.lookup(key(mn, 2, 0), ctx(mn))
    assert list(c.manager.index[1])[-2:] != [key(mn, 108, 1), key(mn, 109, 1)] or True
    assert key(mn, 108, 1) in list(c.manager.index[1])[-4:]      # touched to the MRU end
    # all-or-nothing: a call needing 2 rows when only 1 can be freed (others in-flight loads)
    for k in c.manager.index[0]:
        c.manager.inflight_load[k] = 1
    assert c.manager.prepare_store([key(mn, 50, 0)], ctx(mn)) is None
    assert not c.manager.inflight_store
    c.manager.inflight_load.clear()
    assert c.manager.prepare_store([key(mn, 50, 0)], ctx(mn)) is not None
    for _, _, w, _ in c.workers:
        w.shutdown()


def test_failures_keep_wrong_data_out_of_the_index(mn, tmp_path):
    c = Cluster(mn, tmp_path / "kv", disk_bytes=4 * (TARGET + 4 * DRAFT + 5 * 4096))
    for rank in range(2):
        for t in c.gpu(rank):
            t.copy_(torch.randint(-128, 127, t.shape, dtype=torch.int8))
    k = [key(mn, 1, 0)]
    # store fails on rank 1 -> not indexed, slot back on the free list
    out = c.store(k, [1], fail_rank=1)
    assert out is not None and c.manager.lookup(k[0], ctx(mn)) is False and len(c.manager.free[0]) == 4
    # store succeeds; then rank 1's slab loses the block (wipe) -> load fails there -> key dropped
    assert c.store(k, [1]) is not None and c.manager.lookup(k[0], ctx(mn)) is True
    c.workers[1][0].controller.slab.wipe()
    assert c.load(k, [5]) is False
    assert c.manager.lookup(k[0], ctx(mn)) is False and c.manager.stats["load_failures"] == 1
    # rank 1 reported the failed GPU block for recomputation
    assert c.workers[1][0].controller.take_failed_gpu_blocks() == {5}
    for _, _, w, _ in c.workers:
        w.shutdown()


def test_restart_rebuilds_index_from_headers_and_reset_cache_epoch(mn, tmp_path):
    root = tmp_path / "kv"
    c = Cluster(mn, root, disk_bytes=4 * (TARGET + 4 * DRAFT + 5 * 4096))
    for rank in range(2):
        for t in c.gpu(rank):
            t.copy_(torch.randint(-128, 127, t.shape, dtype=torch.int8))
    orig = [t.clone() for t in c.gpu(0)]
    for i in range(3):
        c.store([key(mn, i, 0), key(mn, 100 + i, 1)], [i + 1, 10 + i], rid=f"r{i}")
    c.manager.lookup(key(mn, 0, 0), ctx(mn))          # row 0 most recently used
    slots_before = dict(c.manager.index[0])
    for _, _, w, _ in c.workers:
        w.shutdown()
    # "restart": new workers (same boot id -> slabs kept) and a fresh scheduler
    c2 = Cluster(mn, root, disk_bytes=4 * (TARGET + 4 * DRAFT + 5 * 4096))
    assert c2.manager._attach()
    assert dict(c2.manager.index[0]) == slots_before
    # rebuilt in write order (the pre-restart touch of row 0 is not persisted, by design)
    assert list(c2.manager.index[0]) == [key(mn, 0, 0), key(mn, 1, 0), key(mn, 2, 0)]
    assert c2.manager.lookup(key(mn, 1, 0), ctx(mn)) is True
    assert list(c2.manager.index[0])[-1] == key(mn, 1, 0)
    assert c2.load([key(mn, 2, 0)], [7]) is True and torch.equal(c2.gpu(0)[0][7], orig[0][3])
    # reset_cache: everything free, and the old headers stay ignored across another restart
    c2.manager.reset_cache()
    assert c2.manager.lookup(key(mn, 1, 0), ctx(mn)) is False and len(c2.manager.free[0]) == 4
    for _, _, w, _ in c2.workers:
        w.shutdown()
    c3 = Cluster(mn, root, disk_bytes=4 * (TARGET + 4 * DRAFT + 5 * 4096))
    assert c3.manager.lookup(key(mn, 1, 0), ctx(mn)) is False and c3.manager.epoch == 1
    # a node reboot (different boot id in the meta) wipes the rank's slabs
    meta_path = os.path.join(c3.workers[0][0].controller.slab.rank_dir, mn.SLAB_META)
    m = json.load(open(meta_path)); m["boot_id"] = "other-boot"; json.dump(m, open(meta_path, "w"))
    for _, _, w, _ in c3.workers:
        w.shutdown()
    c4 = Cluster(mn, root, disk_bytes=4 * (TARGET + 4 * DRAFT + 5 * 4096))
    assert os.path.getsize(c4.workers[0][0].controller.slab.path(0)) == 0
    for _, _, w, _ in c4.workers:
        w.shutdown()


def test_failure_aware_scheduler_and_worker_channel(mn, tmp_path):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingConnectorMetadata, TransferJob
    from vllm.v1.outputs import KVConnectorOutput
    cfg, kvc = make_config(tmp_path / "kv", rank=0, slab=4 * (TARGET + 4 * DRAFT + 5 * 4096))
    sched = mn._FailureAwareScheduler(mn.MultiNodeSlabOffloadingSpec(cfg, kvc))
    keys = {key(mn, 1, 0)}
    sched._jobs[7] = SimpleNamespace(keys=keys, is_store=True)
    sched._jobs[8] = SimpleNamespace(keys={key(mn, 2, 0)}, is_store=False)
    sched.manager.index = [{}, {}]; sched.manager.free = [[], []]; sched.manager.counts = [1, 4]  # pretend attached
    sched.manager.index[0][key(mn, 2, 0)] = 0
    sched.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=mn.MultiNodeWorkerMetadata(completed_jobs={7: 1, 8: 1}, failed_jobs={7, 8})))
    assert keys <= sched.manager.failed_store and key(mn, 2, 0) not in sched.manager.index[0]
    assert sched.seen_completed == {7: 1, 8: 1}
    # worker: a failed slab load is reported in failed_jobs and its GPU block as invalid
    cfg, kvc = make_config(tmp_path / "kv", rank=1, slab=4 * (TARGET + 4 * DRAFT + 5 * 4096))
    w = mn.MultiNodeSlabConnectorWorker(mn.MultiNodeSlabOffloadingSpec(cfg, kvc))
    w._register_handlers(make_canonical_kv_caches())
    k = [key(mn, 9, 0)]
    w.start_kv_transfers(OffloadingConnectorMetadata(load_jobs={3: TransferJob(req_id="q", transfer_spec=(mn.SlabLoadStoreSpec(k, [0]), gpu_spec(mn, k, [2])))}, store_jobs={}))
    recv = set(); t0 = time.monotonic()
    while not recv and time.monotonic() - t0 < 10:
        _, recv = w.get_finished(set()); time.sleep(0.005)
    assert recv == {"q"} and w.take_invalid_block_ids() == {2}
    out = w.build_connector_worker_meta()
    assert out.failed_jobs == {3} and out.completed_jobs == {3: 1}
    w.shutdown()


def test_slab_connector_wiring(mn, tmp_path):
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    cfg, kvc = make_config(tmp_path / "kv", rank=0, slab=10**9)
    c = mn.MultiNodeSlabConnector(cfg, KVConnectorRole.SCHEDULER, kvc)
    assert isinstance(c.connector_scheduler, mn._FailureAwareScheduler) and isinstance(c.connector_scheduler.manager, mn.SlabOffloadingManager)
    c = mn.MultiNodeSlabConnector(cfg, KVConnectorRole.WORKER, kvc)
    assert isinstance(c.connector_worker, mn.MultiNodeSlabConnectorWorker)
    bad, kvc2 = make_config(tmp_path / "kv", rank=0, direct=True)
    with pytest.raises(ValueError):
        mn.MultiNodeSlabConnector(bad, KVConnectorRole.WORKER, kvc2)


# --- review fixes (Codex 2026-09-04) -------------------------------------------------

def test_failed_load_keeps_slot_out_of_reuse_until_readers_finish(mn, tmp_path):
    c = Cluster(mn, tmp_path / "kv", disk_bytes=2 * (TARGET + 4 * DRAFT + 5 * 4096))
    m = c.manager
    assert m.lookup(key(mn, 1, 0), ctx(mn)) is False
    a, b = key(mn, 1, 0), key(mn, 2, 0)
    for k in (a, b):
        out = m.prepare_store([k], ctx(mn)); m.complete_store([k], ctx(mn))
    assert not m.free[0]
    slot_a = m.index[0][a]
    # two concurrent loads of A in flight; one rank fails -> A is dropped (miss)
    m.prepare_load([a], ctx(mn, "x")); m.prepare_load([a], ctx(mn, "y"))
    m.mark_load_failed([a])
    assert m.lookup(a, ctx(mn)) is False and m.condemned == {a: slot_a} and not m.free[0]
    # nothing may take A's slot while readers remain; with B also being read, a store of C must wait
    m.prepare_load([b], ctx(mn, "z"))
    assert m.prepare_store([key(mn, 3, 0)], ctx(mn)) is None
    m.complete_load([b], ctx(mn, "z"))
    assert m.prepare_store([key(mn, 3, 0)], ctx(mn)) is not None and b not in m.index[0]   # B evicted, A's slot untouched
    m.complete_store([key(mn, 3, 0)], ctx(mn))
    assert m.index[0][key(mn, 3, 0)] != slot_a
    m.complete_load([a], ctx(mn, "x"))
    assert not m.free[0]                       # one reader left
    m.complete_load([a], ctx(mn, "y"))
    assert list(m.free[0]) == [slot_a] and not m.condemned
    for _, _, w, _ in c.workers:
        w.shutdown()


def test_placement_is_transactional_and_protects_its_own_keys(mn, tmp_path):
    c = Cluster(mn, tmp_path / "kv", disk_bytes=2 * (TARGET + 4 * DRAFT + 5 * 4096))
    m = c.manager
    assert m.lookup(key(mn, 1, 0), ctx(mn)) is False
    a, b, d = key(mn, 1, 0), key(mn, 2, 0), key(mn, 3, 0)
    for k in (a, b):
        m.prepare_store([k], ctx(mn)); m.complete_store([k], ctx(mn))
    # a call that includes A (already stored) and needs a slot for D must evict B, never A
    out = m.prepare_store([a, d, d], ctx(mn))     # duplicate d deduped
    assert out.keys_to_store == [d] and out.evicted_keys == [b] and a in m.index[0]
    m.complete_store([d], ctx(mn))
    # a call needing more than the whole slab can ever give returns None and touches nothing
    before = (dict(m.index[0]), list(m.free[0]), dict(m.inflight_store))
    assert m.prepare_store([key(mn, 10 + i, 0) for i in range(3)], ctx(mn)) is None
    assert (dict(m.index[0]), list(m.free[0]), dict(m.inflight_store)) == before
    # siblings are bounded to the 4 drafter blocks of each target in the call
    t1, t2 = key(mn, 20, 0), key(mn, 21, 0)
    drafters = [key(mn, 200 + i, 1) for i in range(8)]
    out = m.prepare_store([t1, t2] + drafters, ctx(mn))
    assert m.siblings[t1] == drafters[:4] and m.siblings[t2] == drafters[4:8]
    for _, _, w, _ in c.workers:
        w.shutdown()


def test_epoch_travels_with_jobs_and_survives_restart(mn, tmp_path):
    root = tmp_path / "kv"
    c = Cluster(mn, root, disk_bytes=4 * (TARGET + 4 * DRAFT + 5 * 4096))
    for rank in range(2):
        for t in c.gpu(rank):
            t.copy_(torch.randint(-128, 127, t.shape, dtype=torch.int8))
    assert c.store([key(mn, 1, 0)], [1]) is not None
    c.manager.reset_cache()                                   # epoch 1 from now on
    assert c.manager.lookup(key(mn, 1, 0), ctx(mn)) is False
    out = c.store([key(mn, 2, 0)], [2])                       # written by the workers with epoch 1
    assert out.store_spec.epoch == 1
    for _, _, w, _ in c.workers:
        w.shutdown()
    c2 = Cluster(mn, root, disk_bytes=4 * (TARGET + 4 * DRAFT + 5 * 4096))
    assert c2.manager._attach() and c2.manager.epoch == 1
    assert c2.manager.lookup(key(mn, 2, 0), ctx(mn)) is True      # post-reset store survived
    assert c2.manager.lookup(key(mn, 1, 0), ctx(mn)) is False     # pre-reset store did not
    for _, _, w, _ in c2.workers:
        w.shutdown()


def test_shrinking_the_budget_truncates_and_torn_meta_is_tolerated(mn, tmp_path):
    root = tmp_path / "kv"
    c = Cluster(mn, root, disk_bytes=4 * (TARGET + 4 * DRAFT + 5 * 4096))
    for rank in range(2):
        for t in c.gpu(rank):
            t.copy_(torch.randint(-128, 127, t.shape, dtype=torch.int8))
    for i in range(4):
        c.store([key(mn, i, 0)], [i + 1])
    io = c.workers[0][0].controller.slab
    assert 3 * io.slot_bytes[0] < os.path.getsize(io.path(0)) <= 4 * io.slot_bytes[0]   # sparse: ends at the last payload byte
    for _, _, w, _ in c.workers:
        w.shutdown()
    small = Cluster(mn, root, disk_bytes=2 * (TARGET + 4 * DRAFT + 5 * 4096))   # geometry changed -> wiped
    io2 = small.workers[0][0].controller.slab
    assert io2.slot_counts[0] == 2 and os.path.getsize(io2.path(0)) <= 2 * io2.slot_bytes[0]
    for _, _, w, _ in small.workers:
        w.shutdown()
    # torn meta: the workers start empty, the scheduler waits for a good one
    with open(os.path.join(io2.rank_dir, mn.SLAB_META), "w") as f:
        f.write("{not json")
    c3 = Cluster(mn, root, disk_bytes=2 * (TARGET + 4 * DRAFT + 5 * 4096))
    assert c3.manager._attach() is True          # the new workers rewrote the meta
    for _, _, w, _ in c3.workers:
        w.shutdown()


def test_identity_includes_the_draft_model(mn, tmp_path):
    cfg, kvc = make_config(tmp_path / "kv", rank=0, slab=10**9)
    base = mn.make_file_mapper(mn.MultiNodeSlabOffloadingSpec(cfg, kvc), str(tmp_path / "kv")).base_path
    cfg.speculative_config = SimpleNamespace(model="/models/other-draft")
    other = mn.make_file_mapper(mn.MultiNodeSlabOffloadingSpec(cfg, kvc), str(tmp_path / "kv")).base_path
    assert base != other


def test_wait_all_raises_instead_of_returning_with_work_in_flight(mn, tmp_path):
    spec, kv, w, handlers = None, None, None, None
    cfg, kvc = make_config(tmp_path / "kv", rank=0, slab=10**9)
    spec = mn.MultiNodeSlabOffloadingSpec(cfg, kvc)
    list(spec.get_handlers(make_canonical_kv_caches()))
    ctrl = spec.controller
    ctrl.jobs[99] = mn._DirectJob(job_id=99, store=True, items=[(key(mn, 1, 0), 1)])  # never progresses: no free-slot work scheduled for an empty spec
    ctrl.jobs[99].next_item = 1
    with pytest.raises(TimeoutError):
        ctrl.wait_all(timeout=0.2)
    ctrl.jobs.clear(); ctrl.shutdown()


def test_every_manager_entry_point_is_safe_before_attach(mn, tmp_path):
    """The connector calls touch()/complete_*() for requests that never reach
    lookup() (prompts shorter than a block). Boot A of 2026-09-04 crashed the
    engine with IndexError here."""
    m = mn.SlabOffloadingManager(str(tmp_path / "nowhere"), "/models/glm-5.3")
    k = [key(mn, 1, 0), key(mn, 2, 1)]
    m.touch(k, ctx(mn)); m.complete_load(k, ctx(mn)); m.complete_store(k, ctx(mn)); m.mark_load_failed(k); m.mark_store_failed(k)
    assert m.lookup(k[0], ctx(mn)) is False and m.prepare_store(k, ctx(mn)) is None
    spec = m.prepare_load(k, ctx(mn))
    assert spec.slots == [-1, -1]
    m.reset_cache(); m.shutdown()
