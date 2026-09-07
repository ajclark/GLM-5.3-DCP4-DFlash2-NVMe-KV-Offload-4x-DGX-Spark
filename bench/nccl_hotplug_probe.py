#!/usr/bin/env python3
"""On-cluster probe for the hot-plug-aware NCCL net plugin, run inside the serving image
on all four ranks (see bench/run-nccl-hotplug-probe.sh).

Real NCCL 2.31 core + real ConnectX-7 + the plugin, no vLLM:
  1. gloo control plane over the LAN, one NCCL communicator over the ring (vLLM's raw
     PyNccl wrapper, same as bench/nccl_multicomm.py), all-reduce + all-gather, checked;
  2. write "ready-for-cycle" and wait for the operator to run
        spark-idle.sh --down   (plugin suspend on every node, adapters off)
        spark-idle.sh --up     (adapters on, ring verified, plugin resume)
     which is detected through the plugin's own status file in NCCL_HOTPLUG_CTL_DIR
     (state suspended -> active, generation increased);
  3. the SAME communicator all-reduces and all-gathers again, checked, with per-op timing
     before and after so performance parity is measured on the same run;
  4. optionally a second cycle (NCCL_HOTPLUG_PROBE_CYCLES).
Prints one JSON line per phase on rank 0. Exit 0 only if every check passed.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    cudaStream_t,
    ncclDataTypeEnum,
    ncclRedOpTypeEnum,
)

DTYPE = torch.bfloat16
NCCL_DTYPE = ncclDataTypeEnum.ncclBfloat16
CTL = os.environ.get("NCCL_HOTPLUG_CTL_DIR", "/var/tmp/nccl-hotplug")


def log(rank, **kw):
    if rank == 0:
        print(json.dumps(kw), flush=True)


def plugin_status():
    """(gen, state) from this process's status file, or (None, None)."""
    path = os.path.join(CTL, f"status.{os.getpid()}")
    try:
        with open(path) as f:
            parts = f.read().split()
        return int(parts[0]), parts[1]
    except Exception:
        return None, None


def timed(fn, iters=50):
    torch.cuda.synchronize()
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    return round(statistics.median(samples), 1)


def main():
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    cycles = int(os.environ.get("NCCL_HOTPLUG_PROBE_CYCLES", "1"))
    torch.cuda.set_device(0)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world, timeout=timedelta(minutes=30))
    lib = NCCLLibrary()
    log(rank, phase="init", nccl=lib.ncclGetVersion(), torch=torch.__version__, world=world,
        plugin=os.environ.get("NCCL_NET_PLUGIN", ""), ctl=CTL)

    obj = [bytes(lib.ncclGetUniqueId().internal)] if rank == 0 else [None]
    dist.broadcast_object_list(obj, src=0)
    uid = lib.unique_id_from_bytes(obj[0])
    comm = lib.ncclCommInitRank(world, uid, rank)
    stream = torch.cuda.Stream()
    s = cudaStream_t(stream.cuda_stream)

    n_ar = 98_304 // 2          # TP all-reduce payload of the 8-token verify pass
    n_ag = 147_456 // 2         # query all-gather payload per rank
    ar_buf = torch.full((n_ar,), float(rank + 1), device="cuda", dtype=DTYPE)
    ag_src = torch.full((n_ag,), float(rank + 1), device="cuda", dtype=DTYPE)
    ag_dst = torch.zeros((world, n_ag), device="cuda", dtype=DTYPE)
    expected_sum = world * (world + 1) / 2

    def all_reduce():
        ar_buf.fill_(float(rank + 1))
        lib.ncclAllReduce(ar_buf.data_ptr(), ar_buf.data_ptr(), ar_buf.numel(), NCCL_DTYPE, ncclRedOpTypeEnum.ncclSum, comm, s)

    def all_gather():
        lib.ncclAllGather(ag_src.data_ptr(), ag_dst.data_ptr(), ag_src.numel(), NCCL_DTYPE, comm, s)

    def check(tag):
        with torch.cuda.stream(stream):
            all_reduce(); all_gather()
        torch.cuda.synchronize()
        err_ar = float((ar_buf - expected_sum).abs().max())
        exp = torch.tensor([j + 1 for j in range(world)], device="cuda", dtype=DTYPE).view(world, 1)
        err_ag = float((ag_dst - exp).abs().max())
        t = torch.tensor([err_ar, err_ag], dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        ok = bool(t.max().item() == 0.0)
        with torch.cuda.stream(stream):
            us_ar = timed(all_reduce); us_ag = timed(all_gather)
        log(rank, phase=tag, correct=ok, max_err=float(t.max().item()), allreduce_us=us_ar, allgather_us=us_ag)
        return ok

    ok_all = check("baseline")
    gen0, st0 = plugin_status()
    log(rank, phase="plugin", status_gen=gen0, status_state=st0)

    for cyc in range(1, cycles + 1):
        # hand over to the operator: suspend must see NO outstanding requests, so we do nothing on the comm
        dist.barrier()
        log(rank, phase="ready-for-cycle", cycle=cyc, note="run spark-idle.sh --down then --up now")
        # wait for the plugin to report suspended, then active again
        t0 = time.time(); seen_suspended = False
        while True:
            gen, st = plugin_status()
            if st == "suspended":
                seen_suspended = True
            if seen_suspended and st == "active":
                break
            if time.time() - t0 > 1800:
                log(rank, phase="timeout", cycle=cyc, waited_s=round(time.time() - t0)); ok_all = False; break
            time.sleep(0.5)
        # everyone must be back before the first collective touches the ring again
        t = torch.tensor([1.0 if seen_suspended else 0.0]); dist.all_reduce(t, op=dist.ReduceOp.MIN)
        if t.item() < 1.0:
            ok_all = False
        log(rank, phase="resumed", cycle=cyc, wall_s=round(time.time() - t0, 1))
        ok_all = check(f"after-cycle-{cyc}") and ok_all
        settle = int(os.environ.get("NCCL_HOTPLUG_PROBE_SETTLE_S", "0"))
        if settle > 0:   # does a slow first measurement after the cycle recover by itself?
            time.sleep(settle)
            ok_all = check(f"after-cycle-{cyc}-settled-{settle}s") and ok_all

    lib.ncclCommDestroy(comm)
    dist.barrier()
    log(rank, phase="done", ok=ok_all)
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
