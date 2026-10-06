#!/usr/bin/env python3
"""E3 measurement (results/prefill-item2-20261006/REPORT.md): prefill-size TP all-reduce on one NCCL ring vs two
opposite rings, each carrying half the tensor (run inside the serving image on all four ranks, serving stopped).

Adapted from spark-cluster-mla-dcp/bench/nccl_multicomm.py: communicators are created through vLLM's raw NCCL
wrapper so the comm rank order can be chosen; with one channel NCCL rings the comm ranks in order, so comm ranks
[0,3,2,1] run the physical ring 06c4 -> 365c -> ddbf -> a218 backwards.

Configs (NCCL_BENCH_CONFIGS, comma separated): ``1fwd`` (one ring, the serving layout), ``2fwdrev`` (half on a
forward ring, half on a reversed ring, issued in one NCCL group on two streams). The NIC list, merging and channel
count come from the container env, so each env is a separate run.

Per config and size: eager CUDA-event time of one all-reduce (slowest rank, median of repeats), max |error| against
an fp32 reference of the four inputs and against the ``1fwd`` result, and the MemAvailable / CUDA free-memory drop
caused by creating and first using the config's communicators. One JSON line per result on rank 0.
"""
from __future__ import annotations

import json
import os
import statistics
from datetime import timedelta

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    cudaStream_t,
    ncclDataTypeEnum,
    ncclRedOpTypeEnum,
)

PERMS = {"fwd": [0, 1, 2, 3], "rev": [0, 3, 2, 1]}
SIZES = [12_582_912, 25_165_824]  # bytes per rank: [1024, 6144] and [2048, 6144] bf16


def mem_available_mb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    return float("nan")


class Comm:
    def __init__(self, lib, perm, world, rank):
        self.lib, self.perm = lib, perm
        self.comm_rank = perm.index(rank)
        obj = [bytes(lib.ncclGetUniqueId().internal)] if rank == perm[0] else [None]
        dist.broadcast_object_list(obj, src=perm[0])
        self.handle = lib.ncclCommInitRank(world, lib.unique_id_from_bytes(obj[0]), self.comm_rank)
        self.stream = torch.cuda.Stream()

    def destroy(self):
        self.lib.ncclCommDestroy(self.handle)


def all_reduce(lib, comms, src, dst, launch):
    """Split src/dst evenly over the comms; one NCCL group, one stream per comm, joined back to ``launch``."""
    n = len(comms)
    parts_s, parts_d = src.chunk(n), dst.chunk(n)
    fork = torch.cuda.Event()
    fork.record(launch)
    for c in comms:
        c.stream.wait_event(fork)
    lib.ncclGroupStart()
    for c, s, d in zip(comms, parts_s, parts_d):
        lib.ncclAllReduce(s.data_ptr(), d.data_ptr(), s.numel(), ncclDataTypeEnum.ncclBfloat16,
                          ncclRedOpTypeEnum.ncclSum, c.handle, cudaStream_t(c.stream.cuda_stream))
    lib.ncclGroupEnd()
    for c in comms:
        j = torch.cuda.Event()
        j.record(c.stream)
        launch.wait_event(j)


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    configs = os.environ.get("NCCL_BENCH_CONFIGS", "1fwd,2fwdrev").split(",")
    repeats = int(os.environ.get("NCCL_BENCH_REPEATS", "7"))
    iters = int(os.environ.get("NCCL_BENCH_ITERS", "20"))
    label = os.environ.get("NCCL_BENCH_LABEL", "")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world, timeout=timedelta(minutes=10))
    lib = NCCLLibrary()
    env = {k: os.environ.get(k) for k in ("NCCL_IB_HCA", "NCCL_IB_MERGE_NICS", "NCCL_MAX_NCHANNELS")}
    if rank == 0:
        print(json.dumps({"type": "metadata", "label": label, "nccl": lib.ncclGetVersion(), "env": env,
                          "configs": configs, "sizes": SIZES}), flush=True)
    g = torch.Generator(device="cpu").manual_seed(1234 + rank)
    inputs = {sz: torch.randn(sz // 2, generator=g).to(torch.bfloat16) for sz in SIZES}
    refs = {}
    for sz, x in inputs.items():  # fp32 reference of the four ranks' inputs (gloo, CPU)
        parts = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(parts, x)
        refs[sz] = sum(p.float() for p in parts)
    baseline = {}
    with torch.inference_mode():
        for cfg in configs:
            torch.cuda.synchronize()
            mem0, free0 = mem_available_mb(), torch.cuda.mem_get_info()[0] / 2**20
            comms = [Comm(lib, PERMS["fwd"], world, rank)] if cfg == "1fwd" else \
                [Comm(lib, PERMS["fwd"], world, rank), Comm(lib, PERMS["rev"], world, rank)]
            launch = torch.cuda.Stream()
            for sz in SIZES:
                src = inputs[sz].cuda()
                dst = torch.empty_like(src)
                with torch.cuda.stream(launch):
                    all_reduce(lib, comms, src, dst, launch)
                torch.cuda.synchronize()
                if sz == SIZES[0]:
                    dist.barrier()
                    mem1, free1 = mem_available_mb(), torch.cuda.mem_get_info()[0] / 2**20
                out = dst.float().cpu()
                err_ref = float((out - refs[sz]).abs().max())
                if cfg == "1fwd":
                    baseline[sz] = dst.cpu()
                diff_base = None
                if sz in baseline and cfg != "1fwd":
                    b = baseline[sz]
                    diff_base = {"max_abs": float((dst.cpu().float() - b.float()).abs().max()),
                                 "frac_bits_differ": float((dst.cpu().view(torch.int16) != b.view(torch.int16))
                                                           .float().mean())}
                with torch.cuda.stream(launch):
                    for _ in range(5):
                        all_reduce(lib, comms, src, dst, launch)
                torch.cuda.synchronize()
                samples = []
                for _ in range(repeats):
                    dist.barrier()
                    torch.cuda.synchronize()
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(launch):
                        e0.record(launch)
                        for _ in range(iters):
                            all_reduce(lib, comms, src, dst, launch)
                        e1.record(launch)
                    torch.cuda.synchronize()
                    t = torch.tensor([e0.elapsed_time(e1) * 1e3 / iters], dtype=torch.float64)
                    dist.all_reduce(t, op=dist.ReduceOp.MAX)
                    samples.append(float(t.item()))
                samples.sort()
                stress = int(os.environ.get("NCCL_BENCH_STRESS", "0"))
                stress_res = None
                if stress:
                    # NCCL is deterministic for a fixed config: every repeat must equal the first result bit for bit.
                    with torch.cuda.stream(launch):
                        all_reduce(lib, comms, src, dst, launch)
                    torch.cuda.synchronize()
                    ref_bits = dst.clone()
                    bad = 0
                    worst = 0.0
                    for i in range(stress):
                        with torch.cuda.stream(launch):
                            all_reduce(lib, comms, src, dst, launch)
                        torch.cuda.synchronize()
                        if not torch.equal(dst.view(torch.int16), ref_bits.view(torch.int16)):
                            bad += 1
                            worst = max(worst, float((dst.float() - ref_bits.float()).abs().max()))
                    t = torch.tensor([bad, worst], dtype=torch.float64)
                    dist.all_reduce(t, op=dist.ReduceOp.MAX)
                    stress_res = {"iters": stress, "mismatching_iters_max_rank": int(t[0]), "worst_abs": float(t[1])}
                mem = torch.tensor([mem0 - mem1, free0 - free1], dtype=torch.float64)
                dist.all_reduce(mem, op=dist.ReduceOp.MAX)
                if rank == 0:
                    print(json.dumps({
                        "type": "result", "label": label, "config": cfg, "bytes": sz,
                        "p50_us": round(statistics.median(samples), 1), "min_us": round(samples[0], 1),
                        "max_us": round(samples[-1], 1),
                        "GBps_per_rank_bus": round(2 * (world - 1) / world * sz / (statistics.median(samples) * 1e3), 2),
                        "max_err_vs_fp32": err_ref, "vs_1fwd": diff_base,
                        "mem_drop_mb_max": {"MemAvailable": round(float(mem[0]), 1), "cuda_free": round(float(mem[1]), 1)},
                        "stress": stress_res,
                        "out_sha256": __import__("hashlib").sha256(dst.cpu().view(torch.int16).numpy().tobytes()).hexdigest()[:16],
                    }), flush=True)
            for c in comms:
                c.destroy()
            torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(json.dumps({"type": "done", "label": label}), flush=True)


if __name__ == "__main__":
    main()
