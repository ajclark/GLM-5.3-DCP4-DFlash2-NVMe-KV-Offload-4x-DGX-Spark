#!/usr/bin/env python3
"""GPU microbenchmark of the slab tier's threaded engine vs the raw NVMe.

Runs inside the serving image on one Spark while vLLM is stopped (needs the GPU
and ~8 GB of free memory):

    docker run --rm --gpus all --ipc host --ulimit memlock=-1:-1 \
      -v /var/tmp/kvtier-bench:/bench -v $PWD:/src:ro --entrypoint python3 IMAGE \
      /src/kvtier_engine_bench.py --dir /bench

Uses production geometry (3,452,160-byte pages, 128 B header, 4 KB-rounded
slots). Reports: raw O_DIRECT read GB/s (the device ceiling for this access
pattern), engine store GB/s, and engine load GB/s (NVMe -> GPU, CRC verified,
content checked) for a sweep of read threads x rows per thread.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_slab_threaded import load_overlay  # noqa: E402

PAGE = 3_452_160


def raw_direct_read(path: str, slot_bytes: int, nslots: int, threads: int, nreads: int) -> float:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    slots = [random.Random(7).randrange(nslots) for _ in range(nreads)]
    tl = threading.local()

    def rd(slot):
        b = getattr(tl, "b", None)
        if b is None:
            b = tl.b = mmap.mmap(-1, slot_bytes)
        os.preadv(fd, [b], slot * slot_bytes)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(rd, slots))
    dt = time.perf_counter() - t0
    os.close(fd)
    return nreads * slot_bytes / dt / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--blocks", type=int, default=1536, help="GPU blocks (x3.45 MB)")
    ap.add_argument("--slots", type=int, default=6000, help="slab slots (x3.45 MB on disk)")
    ap.add_argument("--load-blocks", type=int, default=1200)
    ap.add_argument("--threads", default="8,16,24,32")
    ap.add_argument("--rows", default="2,4")
    ap.add_argument("--write-threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    m = load_overlay()
    B = sys.modules["vllm.v1.kv_offload.base"]
    root = Path(a.dir) / f"engine-bench-{os.getpid()}"
    root.mkdir(parents=True)
    res: dict = {"host": os.uname().nodename, "page": PAGE, "blocks": a.blocks, "slots": a.slots}
    try:
        gpu = torch.empty((a.blocks, PAGE), dtype=torch.int8, device="cuda")
        g = torch.Generator(device="cuda").manual_seed(1)
        gpu.copy_(torch.randint(-128, 127, gpu.shape, dtype=torch.int8, device="cuda", generator=g))
        torch.cuda.synchronize()
        # reference snapshot (head and tail of every block) taken before any load overwrites blocks
        ref_head, ref_tail = gpu[:, :4096].cpu(), gpu[:, -4096:].cpu()
        kv = B.CanonicalKVCaches(tensors=[B.CanonicalKVCacheTensor(gpu, PAGE)],
                                 group_data_refs=[[B.CanonicalKVCacheRef(0, PAGE)]])
        slot_bytes = [m._round_up(m.SLAB_HEADER_BYTES + PAGE, 4096)]
        io = m.SlabIO(str(root / "r0"), slot_bytes, [a.slots], preallocate=True)
        S = m.SlabLoadStoreSpec

        def spec(ids):
            return B.GPULoadStoreSpec(ids, group_sizes=[len(ids)], block_indices=[0])

        def run(ctl, job, store, pair):
            t0 = time.perf_counter()
            assert ctl.submit(job, store, pair)
            ctl.wait_all(timeout=600)
            dt = time.perf_counter() - t0
            r = ctl.take_results(store)
            assert len(r) == 1, r
            return r[0], dt

        # ---- store: fill every slot (GPU block i % blocks -> slot i)
        ctl = m.ThreadedSlabController(kv, io, n_read_threads=8, n_write_threads=a.write_threads,
                                       rows_per_thread=2)
        keys = [random.Random(i).randbytes(32) + (0).to_bytes(4, "big") for i in range(a.slots)]
        ids = [i % a.blocks for i in range(a.slots)]
        r, dt = run(ctl, 1, True, (spec(ids), S(keys, list(range(a.slots)), epoch=0, seq=1)))
        assert r.success
        res["store_GBps"] = a.slots * PAGE / dt / 1e9
        res["store_seconds"] = dt
        print(f"store {a.slots} slots: {res['store_GBps']:.2f} GB/s ({dt:.1f}s, {a.write_threads} writers)", flush=True)
        ctl.shutdown()
        os.sync()

        # ---- raw device ceiling for this access pattern
        res["raw_direct_read_GBps"] = {}
        for t in (8, 16, 32, 64):
            v = raw_direct_read(io.path(0), slot_bytes[0], a.slots, t, 1200)
            res["raw_direct_read_GBps"][t] = v
            print(f"raw O_DIRECT slot reads, {t} threads: {v:.2f} GB/s", flush=True)

        # ---- engine loads: random slots -> GPU blocks, verify content
        res["load"] = []
        rng = random.Random(3)
        job = 10
        for rows in [int(x) for x in a.rows.split(",")]:
            for t in [int(x) for x in a.threads.split(",")]:
                ctl = m.ThreadedSlabController(kv, io, n_read_threads=t, n_write_threads=2, rows_per_thread=rows)
                n = min(a.load_blocks, a.blocks)
                pick = rng.sample(range(a.slots), n)
                dst = rng.sample(range(a.blocks), n)   # distinct destinations
                job += 1
                r, dt = run(ctl, job, False, (S([keys[s] for s in pick], pick, epoch=0, seq=1), spec(dst)))
                assert r.success
                torch.cuda.synchronize()
                # slot s holds GPU block (s % blocks) as it was before any load
                ok = all(torch.equal(gpu[d, :4096].cpu(), ref_head[s % a.blocks]) and
                         torch.equal(gpu[d, -4096:].cpu(), ref_tail[s % a.blocks])
                         for s, d in zip(pick, dst))
                gbps = len(pick) * PAGE / dt / 1e9
                res["load"].append({"threads": t, "rows": rows, "GBps": gbps, "seconds": dt, "content_ok": ok})
                print(f"engine load {len(pick)} blocks, {t} threads x {rows} rows: {gbps:.2f} GB/s "
                      f"({dt:.2f}s) content_ok={ok}", flush=True)
                ctl.shutdown()
        io.close()
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
