#!/usr/bin/env python3
"""Unit tests for the slab tier's threaded O_DIRECT engine, geometry and manager.

Runs anywhere with torch + numpy. Without vLLM installed, the vLLM symbols the
overlay imports are stubbed (the real OffloadKey / spec helpers are reproduced),
and GPU copies go through a host memmove backend; inside the serving image with
a GPU, pass --cuda to exercise the real CUDA backend (streams, events and
swap_blocks_batch) against device tensors.

    python3 test_slab_threaded.py [--dir DIR] [--cuda]
"""
from __future__ import annotations

import argparse
import ctypes
import enum
import importlib.util
import os
import random
import shutil
import sys
import tempfile
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OVERLAY = HERE / "overlay/vllm/v1/kv_offload/tiering/multinode.py"


def install_vllm_stubs() -> None:
    try:
        import vllm.v1.kv_offload.base  # noqa: F401
        return  # real vLLM available
    except Exception:  # noqa: BLE001
        pass

    class _Mod(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            cls = type(name, (), {"__init__": lambda self, *a, **k: None})
            setattr(self, name, cls)
            return cls

    def mod(name):
        m = sys.modules.get(name)
        if m is None:
            m = _Mod(name)
            sys.modules[name] = m
            parent, _, child = name.rpartition(".")
            if parent:
                setattr(mod(parent), child, m)
        return m

    import logging
    mod("vllm.logger").init_logger = logging.getLogger
    mod("vllm.utils.platform_utils").is_pin_memory_available = lambda: False
    mod("vllm.version").__version__ = "0.29.0"
    base = mod("vllm.v1.kv_offload.base")

    def make_offload_key(block_hash: bytes, group_idx: int) -> bytes:
        return block_hash + group_idx.to_bytes(4, "big", signed=False)

    def get_offload_group_idx(key: bytes) -> int:
        return int.from_bytes(key[-4:], "big", signed=False)

    class LookupResult(enum.Enum):
        MISS = enum.auto()
        HIT = enum.auto()
        HIT_PENDING = enum.auto()
        RETRY = enum.auto()

    @dataclass
    class PrepareStoreOutput:
        keys_to_store: list
        store_spec: object
        evicted_keys: list

    class LoadStoreSpec:
        pass

    class GPULoadStoreSpec(LoadStoreSpec):
        def __init__(self, block_ids, group_sizes, block_indices):
            self.block_ids = np.array(block_ids, dtype=np.int64)
            self.group_sizes = list(group_sizes)
            self.block_indices = list(block_indices)

    @dataclass
    class CanonicalKVCacheTensor:
        tensor: torch.Tensor
        page_size_bytes: int

    @dataclass
    class CanonicalKVCacheRef:
        tensor_idx: int
        page_size_bytes: int
        mapping: object = None

    @dataclass
    class CanonicalKVCaches:
        tensors: list
        group_data_refs: list

    @dataclass
    class TransferResult:
        job_id: int
        success: bool
        transfer_size: int | None = None
        transfer_time: float | None = None

    class OffloadingSpec:
        def __init__(self, config):
            self.config = config
            self.extra_config = getattr(config, "extra_config", {})

    for k, v in dict(OffloadKey=bytes, make_offload_key=make_offload_key,
                     get_offload_group_idx=get_offload_group_idx, LookupResult=LookupResult,
                     PrepareStoreOutput=PrepareStoreOutput, LoadStoreSpec=LoadStoreSpec,
                     GPULoadStoreSpec=GPULoadStoreSpec, CanonicalKVCacheTensor=CanonicalKVCacheTensor,
                     CanonicalKVCacheRef=CanonicalKVCacheRef, CanonicalKVCaches=CanonicalKVCaches,
                     TransferResult=TransferResult, OffloadingSpec=OffloadingSpec).items():
        setattr(base, k, v)
    for name in ["vllm.distributed.kv_transfer.kv_connector.v1.base",
                 "vllm.distributed.kv_transfer.kv_connector.v1.offloading.common",
                 "vllm.distributed.kv_transfer.kv_connector.v1.offloading.config",
                 "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler",
                 "vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker",
                 "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector",
                 "vllm.v1.kv_offload.cpu.common", "vllm.v1.kv_offload.factory",
                 "vllm.v1.kv_offload.file_mapper", "vllm.v1.kv_offload.tiering.fs.thread_pool"]:
        mod(name)


def load_overlay():
    install_vllm_stubs()
    spec = importlib.util.spec_from_file_location("multinode_under_test", OVERLAY)
    m = importlib.util.module_from_spec(spec)
    sys.modules["multinode_under_test"] = m
    spec.loader.exec_module(m)
    return m


class HostCopyBackend:
    """Stand-in for the CUDA backend: 'GPU' tensors live in host memory."""

    def thread_init(self):
        return None

    def submit_marker(self):
        return None

    def new_event(self):
        return None

    def copy(self, stream, marker, src, dst, sizes, done_event, host_src):
        for s, d, n in zip(src.tolist(), dst.tolist(), sizes.tolist()):
            ctypes.memmove(d, s, n)

    @staticmethod
    def wait(event):
        return None


def key(i: int, g: int) -> bytes:
    return random.Random(i).randbytes(32) + g.to_bytes(4, "big")


def build(m, root: Path, cuda: bool, num_blocks=64, pages=((200_000, 60_000), (150_000,))):
    """Two tensors; group 0 spans both (two refs), group 1 one ref: exercises
    multi-segment payloads and per-group slot sizes."""
    dev = "cuda" if cuda else "cpu"
    t0 = torch.zeros((num_blocks, 262_144), dtype=torch.int8, device=dev)
    t1 = torch.zeros((num_blocks, 65_536), dtype=torch.int8, device=dev)
    B = sys.modules["vllm.v1.kv_offload.base"]
    kv = B.CanonicalKVCaches(
        tensors=[B.CanonicalKVCacheTensor(t0, 262_144), B.CanonicalKVCacheTensor(t1, 65_536)],
        group_data_refs=[[B.CanonicalKVCacheRef(0, pages[0][0]), B.CanonicalKVCacheRef(1, pages[0][1])],
                         [B.CanonicalKVCacheRef(0, pages[1][0])]])
    group_bytes = [sum(pages[0]), sum(pages[1])]
    slot_bytes, counts = m.slab_geometry(group_bytes, 40 * 300_000, m.group_row_ratios([128, 64]))
    io = m.SlabIO(str(root / "r0"), slot_bytes, counts, preallocate=True)
    return kv, io, (t0, t1), group_bytes


def gpu_spec(m, ids, groups):
    B = sys.modules["vllm.v1.kv_offload.base"]
    sizes = [sum(1 for g in groups if g == gi) for gi in range(2)]
    return B.GPULoadStoreSpec(ids, group_sizes=sizes, block_indices=[0, 0])


def run_job(ctl, job_id, store, spec_pair, timeout=60):
    assert ctl.submit(job_id, store, spec_pair)
    t0 = time.time()
    while True:
        ctl.pump()
        res = ctl.take_results(store)
        if res:
            assert len(res) == 1 and res[0].job_id == job_id
            return res[0]
        assert time.time() - t0 < timeout, "job timed out"
        time.sleep(0.001)


def test_geometry(m):
    assert m.group_row_ratios([128, 64]) == [1, 2]
    assert m.group_row_ratios([256, 64]) == [1, 4]
    assert m.group_row_ratios([64]) == [1]
    sb, counts = m.slab_geometry([3452160, 3452160], 400_000_000_000, [1, 2])
    assert sb == [3452928, 3452928] and counts[1] == 2 * counts[0]
    assert counts[0] == 400_000_000_000 // (3 * 3452928), counts
    # legacy default unchanged
    sb, counts = m.slab_geometry([3452160, 3452160], 30_000_000_000)
    assert counts == [1737, 6948], counts
    print("geometry ok", counts)


def test_roundtrip(m, root: Path, cuda: bool, direct: bool):
    kv, io, (t0, t1), gbytes = build(m, root, cuda)
    backend = None if cuda else HostCopyBackend()
    ctl = m.ThreadedSlabController(kv, io, n_read_threads=4, n_write_threads=3, rows_per_thread=2,
                                   direct_io=direct, backend=backend)
    gen = torch.Generator().manual_seed(0)
    src_blocks = list(range(0, 24))
    t0[:24] = torch.randint(-128, 127, (24, t0.shape[1]), dtype=torch.int8, generator=gen).to(t0.device)
    t1[:24] = torch.randint(-128, 127, (24, t1.shape[1]), dtype=torch.int8, generator=gen).to(t1.device)
    groups = [0] * 12 + [1] * 12
    keys = [key(i, g) for i, g in enumerate(groups)]
    slots = [i if g == 0 else i - 12 for i, g in enumerate(groups)]
    epoch, seq = 3, 7
    S = m.SlabLoadStoreSpec
    r = run_job(ctl, 1, True, (gpu_spec(m, src_blocks, groups), S(keys, slots, epoch=epoch, seq=seq)))
    assert r.success and r.transfer_size == 12 * gbytes[0] + 12 * gbytes[1], r
    # the index rebuild sees every slot with its seq
    found = {g: sorted((s, q) for s, q, _ in io.scan(g, epoch)) for g in (0, 1)}
    assert [s for s, _ in found[0]] == list(range(12)) and all(q == seq for _, q in found[0]), found[0]
    assert len(found[1]) == 12
    assert list(io.scan(0, epoch + 1)) == []
    # load into different GPU blocks
    dst_blocks = list(range(30, 54))
    r = run_job(ctl, 2, False, (S(keys, slots, epoch=epoch, seq=seq), gpu_spec(m, dst_blocks, groups)))
    assert r.success, r
    for i, g in enumerate(groups):
        s_, d_ = src_blocks[i], dst_blocks[i]
        if g == 0:
            assert torch.equal(t0[d_, :200_000], t0[s_, :200_000]), i
            assert torch.equal(t1[d_, :60_000], t1[s_, :60_000]), i
            assert int(t0[d_, 200_000:].abs().sum()) == 0  # outside the ref: untouched
        else:
            assert torch.equal(t0[d_, :150_000], t0[s_, :150_000]), i
    assert not ctl.take_failed_gpu_blocks()
    # wrong epoch -> every block fails and is reported
    r = run_job(ctl, 3, False, (S(keys[:3], slots[:3], epoch=epoch + 1, seq=seq),
                                gpu_spec(m, [60, 61, 62], groups[:3])))
    assert not r.success and ctl.take_failed_gpu_blocks() == {60, 61, 62}
    # corrupt one payload byte on disk -> CRC failure on that block only
    fd = os.open(io.path(0), os.O_RDWR)
    off = 5 * io.slot_bytes[0] + m.SLAB_HEADER_BYTES + 1234
    b = os.pread(fd, 1, off)
    os.pwrite(fd, bytes([b[0] ^ 0xFF]), off)
    os.close(fd)
    r = run_job(ctl, 4, False, (S(keys[4:7], slots[4:7], epoch=epoch, seq=seq),
                                gpu_spec(m, [40, 41, 42], groups[4:7])))
    assert not r.success and ctl.take_failed_gpu_blocks() == {41}
    # a slot holding another key -> failure
    r = run_job(ctl, 5, False, (S([keys[0]], [1], epoch=epoch, seq=seq), gpu_spec(m, [43], [0])))
    assert not r.success and ctl.take_failed_gpu_blocks() == {43}
    # many concurrent jobs, interleaved loads and stores, wait_all drains
    for j in range(20):
        ctl.submit(100 + j, j % 2 == 0,
                   (gpu_spec(m, src_blocks[:4], groups[:4]), S(keys[:4], slots[:4], epoch=epoch, seq=seq + j))
                   if j % 2 == 0 else
                   (S(keys[8:12], slots[8:12], epoch=epoch, seq=seq), gpu_spec(m, [44, 45, 46, 47], groups[8:12])))
    ctl.wait_all(timeout=60)
    assert not ctl.jobs
    res = ctl.take_results(True) + ctl.take_results(False)
    assert len(res) == 20 and all(x.success for x in res), [x for x in res if not x.success]
    ctl.shutdown()
    io.close()
    print(f"roundtrip ok (direct={ctl.direct}, cuda={cuda})")


def test_manager(m, root: Path):
    """Scheduler index: store/lookup/evict with the bounded preflight, sibling ratio
    from the geometry, rebuild from headers."""
    rank = root / "mgr_r0"
    rank.mkdir()
    slot_bytes, counts = [8192, 8192], [4, 8]
    import json
    (rank / m.SLAB_META).write_text(json.dumps({"slot_counts": counts, "slot_bytes": slot_bytes}))
    mgr = m.SlabOffloadingManager(str(root), "x", expected_base=str(root / "mgr"))
    assert mgr._attach() and mgr.row_ratio == 2
    LR = sys.modules["vllm.v1.kv_offload.base"].LookupResult
    t = [key(1000 + i, 0) for i in range(6)]
    d = [key(2000 + i, 1) for i in range(12)]
    out = mgr.prepare_store(t[:2] + d[:4], None)
    assert out is not None and mgr.siblings[t[0]] == d[0:2] and mgr.siblings[t[1]] == d[2:4]
    mgr.complete_store(out.keys_to_store, None)
    out = mgr.prepare_store(t[2:4] + d[4:8], None)
    mgr.complete_store(out.keys_to_store, None)
    assert mgr.lookup(t[0], None) == LR.HIT          # t0 (+ d0,d1 via siblings) now most recent
    out = mgr.prepare_store(t[4:6] + d[8:12], None)   # needs 2 target + 4 drafter evictions
    assert out is not None
    assert set(out.evicted_keys) == {t[1], t[2], d[2], d[3], d[4], d[5]}, out.evicted_keys
    mgr.complete_store(out.keys_to_store, None)
    # everything pinned by in-flight loads -> the transactional preflight refuses
    mgr.prepare_load(t[:1] + t[3:6], None)
    assert mgr.prepare_store([key(3000, 0)], None) is None
    print("manager ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="scratch dir (use an ext4 NVMe path for O_DIRECT)")
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()
    m = load_overlay()
    test_geometry(m)
    root = Path(tempfile.mkdtemp(prefix="slabtest-", dir=a.dir))
    try:
        test_roundtrip(m, root / "direct", a.cuda, direct=True)
        test_roundtrip(m, root / "buffered", a.cuda, direct=False)
        test_manager(m, root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("ALL OK")


if __name__ == "__main__":
    main()
