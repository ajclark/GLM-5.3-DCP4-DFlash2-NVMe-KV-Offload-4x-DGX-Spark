// SPDX-License-Identifier: Apache-2.0
// glm_fast L2 prefetch kernel (VLLM_L2_PREFETCH). Exact: a prefetch is only a cache hint.
//
// Re-implementation, for the GLM-5.3 serving repo, of the side-stream
// cp.async.bulk.prefetch.L2 kernel in knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4 @770d115
// (overlay/glm_l2_prefetch.py, Apache-2.0 per its SPDX header; originally from knapcio's
// DeepSeek-V4.1 stack). PTX cp.async.bulk.prefetch.L2: NVIDIA, sm_90+.
//
// segs: [n][2] int64 {global address, bytes}; addresses and sizes are 16-byte aligned by
// the host planner. Each CTA takes segments round-robin; each thread issues one bulk
// prefetch of up to `chunk` bytes per step. The instruction is fire-and-forget, so the
// kernel finishes in microseconds and the DRAM traffic continues in the background.
// flag: optional device int; 0 = do nothing. It lets a captured graph keep its prefetch
// nodes while the host turns a window off at runtime (in-boot A/B without recapture).
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void glm_l2pf_kernel(const long long* __restrict__ segs, int n, long long chunk,
                                const int* __restrict__ flag) {
  if (flag != nullptr && *flag == 0) return;
  for (int s = blockIdx.x; s < n; s += gridDim.x) {
    const long long addr = segs[2 * s];
    const long long bytes = segs[2 * s + 1];
    for (long long off = (long long)threadIdx.x * chunk; off < bytes;
         off += (long long)blockDim.x * chunk) {
      long long sz = bytes - off < chunk ? bytes - off : chunk;
      sz &= ~15LL;
      if (sz > 0) {
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                     :: "l"(addr + off), "r"((unsigned)sz) : "memory");
      }
    }
  }
}

extern "C" int glm_l2pf_launch(const void* segs, int n, long long chunk, int ctas, const void* flag,
                               void* stream) {
  if (n <= 0) return 0;
  int grid = n < ctas ? n : ctas;
  if (grid < 1) grid = 1;
  glm_l2pf_kernel<<<grid, 64, 0, (cudaStream_t)stream>>>((const long long*)segs, n, chunk,
                                                         (const int*)flag);
  return (int)cudaGetLastError();
}
