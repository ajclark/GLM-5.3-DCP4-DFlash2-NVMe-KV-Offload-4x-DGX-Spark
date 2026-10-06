"""GPU side of a one-shot RDMA all-reduce on GB10, without the network.

The b12x / ring kernel does two memory-bound passes per op:
  stage:  read the input from device memory, store it into this rank's pinned send slot
          (plain v4 stores to cudaHostAlloc'ed memory);
  reduce: read the three peers' pinned receive slots with ld.relaxed.sys.v4 (system scope)
          plus the own input, sum 8 bf16 lanes per 16 bytes in fp32, store the output.
This script times both passes alone (CUDA graphs, 100 launches per replay, cold buffers rotated
past the 24 MB L2), at the payloads of T = 1..32 tokens x 6144 bf16, and compares memory kinds
and load flavours to find what the per-byte cost is made of.

Run on one Spark with the model stopped:
  docker run --rm --gpus all --ipc=host --entrypoint python3 -v $HOME/ring-bench:/w <image> /w/bench_gpu_pinned.py
"""

from __future__ import annotations

import json
import statistics
import sys

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

__device__ __forceinline__ uint4 ld_sys(const uint4* p) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_vol(const uint4* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void acc(float* a, uint4 w, bool first) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&w);
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    float2 f = __bfloat1622float2(h[j]);
    if (first) { a[2*j] = f.x; a[2*j+1] = f.y; } else { a[2*j] += f.x; a[2*j+1] += f.y; }
  }
}
__device__ __forceinline__ uint4 pack(const float* a) {
  uint4 w; __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&w);
  #pragma unroll
  for (int j = 0; j < 4; j++) h[j] = __floats2bfloat162_rn(a[2*j], a[2*j+1]);
  return w;
}
__global__ void k_copy(const uint4* src, uint4* dst, long n) {
  long st = (long)gridDim.x * blockDim.x;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += st) dst[i] = src[i];
}
// mode 0: peers via ld.relaxed.sys (b12x), 1: plain ld.global, 2: ld.volatile
__global__ void k_reduce(const uint4* own, const uint4* p0, const uint4* p1, const uint4* p2,
                         uint4* out, long n, int mode) {
  long st = (long)gridDim.x * blockDim.x;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += st) {
    float a[8];
    uint4 w0, w1, w2;
    if (mode == 0) { w0 = ld_sys(p0 + i); w1 = ld_sys(p1 + i); w2 = ld_sys(p2 + i); }
    else if (mode == 1) { w0 = p0[i]; w1 = p1[i]; w2 = p2[i]; }
    else { w0 = ld_vol(p0 + i); w1 = ld_vol(p1 + i); w2 = ld_vol(p2 + i); }
    acc(a, own[i], true); acc(a, w0, false); acc(a, w1, false); acc(a, w2, false);
    out[i] = pack(a);
  }
}
static uint4* P(int64_t addr) { return reinterpret_cast<uint4*>(addr); }
void copy(int64_t src, int64_t dst, int64_t nbytes, int64_t blocks) {
  k_copy<<<blocks, 512, 0, c10::cuda::getCurrentCUDAStream()>>>(P(src), P(dst), nbytes / 16);
}
void reduce(int64_t own, int64_t p0, int64_t p1, int64_t p2, int64_t out, int64_t nbytes,
            int64_t blocks, int64_t mode) {
  k_reduce<<<blocks, 512, 0, c10::cuda::getCurrentCUDAStream()>>>(P(own), P(p0), P(p1), P(p2), P(out),
                                                                nbytes / 16, (int)mode);
}
"""
CPP = "void copy(int64_t, int64_t, int64_t, int64_t);\nvoid reduce(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);"

SIZES_T = (1, 2, 4, 8, 16, 32)
ROW = 6144 * 2
ROT = 24  # buffer sets rotated per graph (> L2 at the large sizes)
LAUNCHES = 96


def main() -> int:
    ext = load_inline("ringbench", cpp_sources=CPP, cuda_sources=SRC, functions=["copy", "reduce"],
                      extra_cuda_cflags=["-O3", "-arch=sm_121a"], verbose=False)
    dev = torch.device("cuda", 0)
    smax = max(SIZES_T) * ROW

    def alloc(kind: str):
        if kind == "dev":
            return [torch.empty(smax, dtype=torch.uint8, device=dev) for _ in range(ROT)]
        return [torch.empty(smax, dtype=torch.uint8, pin_memory=True) for _ in range(ROT)]

    bufs = {k: {"a": alloc(k), "b": alloc(k), "c": alloc(k), "d": alloc(k)} for k in ("dev", "pin")}
    for k in bufs:
        for role in bufs[k]:
            for t in bufs[k][role]:
                t.copy_(torch.randint(0, 255, (smax,), dtype=torch.uint8))
    out = [torch.empty(smax, dtype=torch.uint8, device=dev) for _ in range(ROT)]
    torch.cuda.synchronize()

    def timed(fn) -> float:
        """us per launch: CUDA graph of LAUNCHES launches over rotating buffer sets."""
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for i in range(4):
                fn(i)
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            for i in range(LAUNCHES):
                fn(i % ROT)
        times = []
        for _ in range(12):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1) * 1000 / LAUNCHES)
        return statistics.median(times[2:])

    res = {"device": torch.cuda.get_device_name(0), "rows": []}

    def row(name, T, blocks, us, nbytes_moved):
        r = {"case": name, "T": T, "KiB": T * ROW // 1024, "blocks": blocks, "us": round(us, 2),
             "GBps": round(nbytes_moved / us / 1e3, 1)}
        res["rows"].append(r)
        print(json.dumps(r), flush=True)

    # empty-kernel floor
    for blocks in (8,):
        us = timed(lambda i: ext.copy(bufs["dev"]["a"][i].data_ptr(), out[i].data_ptr(), 16, blocks))
        row("empty (16 B copy)", 0, blocks, us, 16)
    for T in SIZES_T:
        n = T * ROW
        for blocks in (8, 16, 48):
            # stage: device -> pinned (b12x stage), and references
            for name, src, dst in (("stage dev->pin", "dev", "pin"), ("copy dev->dev", "dev", "dev"),
                                   ("copy pin->dev", "pin", "dev")):
                S, D = bufs[src]["a"], (bufs[dst]["b"] if dst != "dev" else out)
                us = timed(lambda i, S=S, D=D: ext.copy(S[i].data_ptr(), D[i].data_ptr(), n, blocks))
                row(name, T, blocks, us, 2 * n)
            # reduce: own (device) + 3 peers
            for name, kind, mode in (("reduce peers=pin ld.sys (b12x)", "pin", 0),
                                     ("reduce peers=pin ld.global", "pin", 1),
                                     ("reduce peers=pin ld.volatile", "pin", 2),
                                     ("reduce peers=dev ld.sys", "dev", 0),
                                     ("reduce peers=dev ld.global", "dev", 1)):
                B = bufs[kind]
                own = bufs["dev"]["a"]
                us = timed(lambda i, B=B, own=own, mode=mode: ext.reduce(
                    own[i].data_ptr(), B["b"][i].data_ptr(), B["c"][i].data_ptr(), B["d"][i].data_ptr(),
                    out[i].data_ptr(), n, blocks, mode))
                row(name, T, blocks, us, 5 * n)
    print("RESULT " + json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
