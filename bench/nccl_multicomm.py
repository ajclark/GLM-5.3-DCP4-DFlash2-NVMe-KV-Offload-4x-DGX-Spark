#!/usr/bin/env python3
"""Idea 1 microbenchmark: split each decode collective across N one-channel
NCCL communicators with permuted rank order (forward ring, reversed ring),
and compare against the single communicator the serving stack uses.

Runs inside the serving image on each of the four ranks (see
run-nccl-multicomm.sh). Control plane is a gloo process group over the LAN;
every NCCL communicator is created through vLLM's raw wrapper so the comm
rank order can be chosen: with one channel NCCL rings the comm ranks in
order, so comm ranks [0,3,2,1] give the physical ring reversed, which still
only touches adjacent links.

Sizes are the 8-token verify pass of GLM-5.3 at TP4:
  q all-gather      147,456 B per rank (64 heads x 576 bf16 x 8 tokens total)
  LSE all-gather        512 B per rank
  indexer merge AG   32,768 B per rank
  output reduce-scatter 524,288 B in per rank (131,072 B out)
  TP all-reduce      98,304 B (8 x 6144 bf16)

Env: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, NCCL_BENCH_VARIANT,
NCCL_BENCH_NS (default "1,2,4,8"), NCCL_BENCH_REPEATS (7), NCCL_BENCH_ITERS
(100), NCCL_BENCH_WARMUP (50). Output: one JSON line per configuration on
rank 0.
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
ELEM = 2

COLLECTIVES = [
    # name, kind, bytes per rank of the *send* side
    ("q_allgather", "ag", 147_456),
    ("lse_allgather", "ag", 512),
    ("indexer_allgather", "ag", 32_768),
    ("out_reducescatter", "rs", 524_288),
    ("tp_allreduce", "ar", 98_304),
]

PERMS = {"fwd": [0, 1, 2, 3], "rev": [0, 3, 2, 1]}


class Comm:
    def __init__(self, lib: NCCLLibrary, perm: list[str], world: int, rank: int):
        self.lib = lib
        self.perm = perm
        self.comm_rank = perm.index(rank)
        # comm rank 0 (global rank perm[0]) mints the id; everyone gets it via gloo
        obj = [bytes(lib.ncclGetUniqueId().internal)] if rank == perm[0] else [None]
        dist.broadcast_object_list(obj, src=perm[0])
        uid = lib.unique_id_from_bytes(obj[0])
        self.handle = lib.ncclCommInitRank(world, uid, self.comm_rank)
        self.stream = torch.cuda.Stream()

    def destroy(self):
        self.lib.ncclCommDestroy(self.handle)


def make_comms(lib, n: int, order: str, world: int, rank: int) -> list[Comm]:
    comms = []
    for i in range(n):
        name = "fwd" if order == "fwd" else ("fwd" if i % 2 == 0 else "rev")
        comms.append(Comm(lib, PERMS[name], world, rank))
    return comms


def run_collective(lib, kind: str, comms: list[Comm], bufs: list[tuple[torch.Tensor, torch.Tensor]],
                   launch_stream: torch.cuda.Stream):
    """Fork the N comm streams off `launch_stream`, issue the N chunks in one
    NCCL group (one per communicator on its own stream), join back."""
    fork = torch.cuda.Event()
    fork.record(launch_stream)
    for comm in comms:
        comm.stream.wait_event(fork)
    lib.ncclGroupStart()
    for comm, (src, dst) in zip(comms, bufs):
        s = cudaStream_t(comm.stream.cuda_stream)
        if kind == "ag":
            lib.ncclAllGather(src.data_ptr(), dst.data_ptr(), src.numel(), NCCL_DTYPE, comm.handle, s)
        elif kind == "rs":
            lib.ncclReduceScatter(src.data_ptr(), dst.data_ptr(), dst.numel(), NCCL_DTYPE,
                                  ncclRedOpTypeEnum.ncclSum, comm.handle, s)
        else:
            lib.ncclAllReduce(src.data_ptr(), dst.data_ptr(), src.numel(), NCCL_DTYPE,
                              ncclRedOpTypeEnum.ncclSum, comm.handle, s)
    lib.ncclGroupEnd()
    for comm in comms:
        join = torch.cuda.Event()
        join.record(comm.stream)
        launch_stream.wait_event(join)


def make_buffers(kind: str, send_bytes: int, n: int, world: int, rank: int):
    """Per-chunk (src, dst) pairs. AG: src [count/n] -> dst [world, count/n].
    RS: src [world, count/n] -> dst [count/n]. AR: src [count/n] -> dst [count/n]."""
    count = send_bytes // ELEM
    chunk = max(1, count // n)
    bufs = []
    for i in range(n):
        if kind == "ag":
            src = torch.full((chunk,), float(rank + 1), device="cuda", dtype=DTYPE)
            dst = torch.zeros((world, chunk), device="cuda", dtype=DTYPE)
        elif kind == "rs":
            src = torch.full((world, chunk // world if chunk >= world else 1), float(rank + 1), device="cuda", dtype=DTYPE)
            dst = torch.zeros((src.shape[1],), device="cuda", dtype=DTYPE)
        else:
            src = torch.full((chunk,), float(rank + 1), device="cuda", dtype=DTYPE)
            dst = torch.zeros((chunk,), device="cuda", dtype=DTYPE)
        bufs.append((src, dst))
    return bufs


def check(kind: str, bufs, comms, world: int) -> float:
    """Max abs error of the result against the closed form (values are small
    integers, exact in bf16)."""
    err = 0.0
    expected_sum = world * (world + 1) / 2
    for comm, (src, dst) in zip(comms, bufs):
        if kind == "ag":
            # row j of dst holds comm-rank j's data = global rank perm[j] + 1
            exp = torch.tensor([comm.perm[j] + 1 for j in range(world)], device="cuda", dtype=DTYPE).view(world, 1)
            err = max(err, float((dst - exp).abs().max()))
        else:
            err = max(err, float((dst - expected_sum).abs().max()))
    return err


def main():
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    variant = os.environ.get("NCCL_BENCH_VARIANT", "unknown")
    ns = [int(x) for x in os.environ.get("NCCL_BENCH_NS", "1,2,4,8").split(",")]
    repeats = int(os.environ.get("NCCL_BENCH_REPEATS", "7"))
    iters = int(os.environ.get("NCCL_BENCH_ITERS", "100"))
    warmup = int(os.environ.get("NCCL_BENCH_WARMUP", "50"))
    torch.cuda.set_device(0)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world, timeout=timedelta(minutes=10))
    lib = NCCLLibrary()
    if rank == 0:
        print(json.dumps({"type": "metadata", "variant": variant, "nccl": lib.ncclGetVersion(),
                          "torch": torch.__version__, "ns": ns, "repeats": repeats, "iters": iters,
                          "proto_env": os.environ.get("NCCL_PROTO", ""), "world": world}), flush=True)

    configs = [(n, order) for n in ns for order in (["fwd"] if n == 1 else ["fwd", "alt"])]
    with torch.inference_mode():
        for n, order in configs:
            comms = make_comms(lib, n, order, world, rank)
            for name, kind, send_bytes in COLLECTIVES:
                bufs = make_buffers(kind, send_bytes, n, world, rank)
                launch = torch.cuda.Stream()
                # correctness once (eager), then eager warmup (lazy connect)
                with torch.cuda.stream(launch):
                    run_collective(lib, kind, comms, bufs, launch)
                torch.cuda.synchronize()
                err = check(kind, bufs, comms, world)
                err_t = torch.tensor([err], dtype=torch.float64)
                dist.all_reduce(err_t, op=dist.ReduceOp.MAX)
                with torch.cuda.stream(launch):
                    for _ in range(warmup):
                        run_collective(lib, kind, comms, bufs, launch)
                torch.cuda.synchronize()
                # time two ways: eager launches (as the earlier sweep did) and
                # replays of one captured iteration (as vLLM runs it)
                results = {}
                for mode in ("eager", "graph"):
                    graph = None
                    if mode == "graph":
                        try:
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, stream=launch):
                                run_collective(lib, kind, comms, bufs, launch)
                            torch.cuda.synchronize()
                            for _ in range(5):
                                graph.replay()
                            torch.cuda.synchronize()
                        except Exception as exc:
                            results[mode] = f"capture failed: {type(exc).__name__}"; continue
                    samples = []; issue = []
                    for _ in range(repeats):
                        dist.barrier(); torch.cuda.synchronize()
                        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                        with torch.cuda.stream(launch):
                            e0.record(launch)
                            c0 = time.perf_counter()
                            for _ in range(iters):
                                if graph is not None:
                                    graph.replay()
                                else:
                                    run_collective(lib, kind, comms, bufs, launch)
                            issue.append((time.perf_counter() - c0) * 1e6 / iters)  # CPU cost to enqueue one iteration
                            e1.record(launch)
                        torch.cuda.synchronize()
                        us = e0.elapsed_time(e1) * 1e3 / iters
                        t = torch.tensor([us], dtype=torch.float64)
                        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # slowest rank
                        samples.append(float(t.item()))
                    samples.sort()
                    results[mode] = {"p50_us": round(statistics.median(samples), 1), "min_us": round(samples[0], 1),
                                     "max_us": round(samples[-1], 1), "cpu_issue_us": round(statistics.median(issue), 1)}
                    del graph
                if rank == 0:
                    print(json.dumps({"type": "result", "variant": variant, "collective": name, "kind": kind,
                                      "send_bytes_per_rank": send_bytes, "n_comms": n, "order": order,
                                      "eager": results.get("eager"), "graph": results.get("graph"),
                                      "max_err": float(err_t.item())}), flush=True)
                del bufs
            for c in comms:
                c.destroy()
            torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(json.dumps({"type": "done", "variant": variant}), flush=True)


if __name__ == "__main__":
    main()
