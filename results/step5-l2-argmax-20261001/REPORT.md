# Step 5: L2 prefetch in collective windows, vocab-parallel target argmax, drafter-start gap (2026-10-01)

Overview: `docs/DECODE-SPEEDUPS.md`, step 5. CPU tests: 19/19 pass.

Measured on an abliterated graft of GLM-5.3 with the same quantization envelope as GLM-5.3 Int4-Int8Mix (Int4
group-128 routed experts, Int8 group-128 elsewhere, identical shapes).

## Design

Both parts are exact. They live in `runtime/vllm029/verify_cap_overlay/glm_fast/` and are installed by
`glm_fast.pth` → `glm_fast.boot`, as import-time monkeypatches in the glm_roce style. No vLLM file is edited, and
`model_runner.py` and `verify_cap.py` are untouched. With both switches off, nothing is installed.

### Part 1: vocab-parallel target argmax (`VLLM_VOCAB_PARALLEL_ARGMAX=0|1|check`, default 0)
**Today:** every rank all-gathers `[L, 154880]` bf16 target logits, with L = n·(K+1). That is 1.9 MB per rank and
184 µs at L = 8. It then runs the rejection sampler's block-stats, reject, resample and insert kernels, about 270 µs in
all after the lm_head GEMM.

**New, for all-greedy batches only** (hooked at `GPUModelRunner.sample`):
- the same shard GEMM, so the shard logits are bit-identical;
- a Triton kernel that turns each row's first shard maximum into one int64 key, `[ordered fp32 value | 0xFFFFFFFF −
  global id]`;
- an 8-byte-per-row TP all-gather;
- a per-request kernel that takes the max key (largest value, lowest id) and runs the greedy branch of the stock
  rejection, resample and insert kernels on ids.

**Exactness:**
- Stock greedy takes the first maximal index. Shards are contiguous id ranges and the key's low word is reversed, so
  ties go to the lowest id, as in stock.
- −0.0 is folded into +0.0.
- NaN is treated as −inf, so the id is always in range. The stock NaN result depends on the reduction tree, and the two
  stock paths already disagree.

**Eligibility** comes from host state identical on every TP rank, so the collectives stay aligned. The fast path runs
only when:
- every request is greedy;
- no request needs logits processing, logprobs or a grammar;
- the stock rejection sampler is in use;
- the draft-logits cache is at least as wide as the vocab;
- the lm_head is plain.

**Check mode** runs both paths and returns the stock result. Device-side counters record mismatches, and rank 0 logs
them every 500 steps.

### Part 2: L2 prefetch in collective windows (`VLLM_L2_PREFETCH=1`, default 0)
A side stream forks before each collective and runs a ≤8-CTA kernel that issues `cp.async.bulk.prefetch.L2` for the
weights read next. The budget is 12 MiB per window, scales before packed weights.

| window | fires at | prefetches |
|---|---|---|
| B | post-attention all-reduce + norm | router → shared gate_up → shared down |
| C | MoE all-reduce, fused into the next layer's input norm | the next layer's `fused_qkv_a` |
| D | DCP query gather | W_UV (16 per-head runs) → o_proj |

- **Hooks:** they wrap the module-level `fused_allreduce_rms_norm` name, so they sit above NCCL, the step-7 ring and
  flashinfer alike. Window D wraps `dcp_manager.query_gather`.
- **Graphs:** the fork and join are capture-legal.
  - The join is at the end of `DeepseekV32Model.forward`.
  - Only forwards of ≤128 tokens fork; nothing forks under Dynamo.
  - Tables are built during the eager warm-up before capture.
- **Runtime A/B:** each window has a device flag, set from a control file (`{"windows":"BD"}`) polled at most once a
  second. Windows turn on or off without recapture.
- **Kernel:** `l2pf.cu` is built by nvcc at image build and called via ctypes, with a Triton inline-asm fallback.
- **Credits:** a re-implementation of knapcio @770d115 (`glm_l2_prefetch*.py`, `glm_target_argmax.py`). Those files are
  **Apache-2.0** per their SPDX headers and NOTICE. All glm_fast files carry Apache-2.0 headers and attribution.

### Drafter-start gap (investigation)
**Finding.** The gap comes with each graph launch, not with the drafter.
- The first collective of every target and draft graph starts 210–580 µs late (median), up to 990 µs, on every rank,
  and then runs 300–700 µs instead of 40–90 µs.
- Every later collective starts 3–7 µs after its predecessor.
- The cost is about 1.5–2 ms per cycle.

**Cause (supported).** In captured graphs, NCCL adds a host task per collective. The chain of host tasks runs at graph
start, so only the first collective waits. It waits on the wake-up of CUDA's callback thread, then of NCCL's proxy
thread, from the Grace cores' deep idle states: LPI-2 and LPI-3 exit in 231 µs and 433 µs.

**Next.** With `ROCE_TP=1` (step 7), the decode graphs hold no NCCL ops at ≤1 MiB, because the ring and RoCEnante
proxies busy-poll instead of using graph host nodes. The gap should vanish there; a profile will confirm.

## Expected value
- **Argmax:** −0.2 ms per pass at C1 K=7, −0.8 ms at C4.
- **L2 prefetch:** −2 to −5 ms per pass (planning range). The GPU test's section E measures it per layer.

## Keep criteria
- **Argmax:** check mode shows `mismatches=0` on the probes and C4; then count100 byte parity and C4 ≥ +0.3%.
- **L2 prefetch:** GPU test D/E pass. Then an in-boot ABBA per window at fixed K=7 and K=1 keeps windows that are
  ≥ 0.5 ms better, with byte parity, CLEAN probes and rank-0 memory ≥ 0.8 GB.

## Results

### GPU test (`gpu-test.log`; spark-365c, model stopped, verifycap9-ring image): 14/14 PASS
- **A1–A4, argmax:** 0 mismatches against `torch.argmax` on 3,100 random and 18 adversarial rows. Triton keys equal the
  torch keys bitwise. NaN rows stay in range.
- **B1–B3, verify path:**
  - B1: 0 bad batches against the image's `rejection_sample` (300 batches, 1,124 requests).
  - B2: 0 mismatches against `gumbel_sample` at temperature 0.
  - B3: NaN steps stay in range.
- **C, sampling work after the target head:**

  | L | stock (gather+copy+sampler) | fast |
  |---:|---:|---:|
  | 8 | 170 µs | 34 µs |
  | 32 | 318 µs | 33 µs |

  The NCCL gather shrinks from 1.86 MB per rank to 192 B.
- **D1–D2, prefetch kernel:** the CUDA and Triton kernels build. Captured fork → prefetch → join graphs replay
  bit-identically with the flag on and off.
- **E, value after an 80 µs window** (cold → prefetched):

  | weights | budget | cold | prefetched | change |
  |---|---:|---:|---:|---:|
  | int8 Marlin fused_qkv_a, 16.1 MB | 16 MiB | 133 µs | 77 µs | −42% |
  | int8 Marlin o_proj, 25.2 MB | 12 MiB | 178 µs | 130 µs | −27% |
  | int8 Marlin shared gate_up | 8 MiB | 61 µs | 37 µs | −40% |
  | bf16 GEMV, 12.6 MB | 12 MiB | 115 µs | 67 µs | −42% |

  With the flag off, the captured graph runs like the cold one (E3).

### Live, in-boot ABBA (boot A, `../step456-integration-20261001/bootA-l2.log`)
Image verifycap10-stack, `L2_PREFETCH=1`, windows switched through the control file, fixed K, live periods:

| requests × K | off (2 arms) | **B+C+D on (2 arms)** | change |
|---|---:|---:|---:|
| 1 × 7 | 124.37 / 123.55 ms | **117.83 / 117.01** | **−6.5 ms (−5.3%)** |
| 1 × 1 | 79.44 / 79.49 | **74.10 / 73.88** | **−5.5 ms (−6.9%)** |
| 4 × 7 | 263.53 / 264.51 | **259.59 / 261.46** | −3.5 ms (−1.3%) |
| 4 × 1 | 141.54 / 141.86 | **136.26 / 136.47** | −5.3 ms (−3.8%) |

The arms reproduce within ±1 ms. No contamination, no engine errors; rank-0 MemAvailable stays at 3.3 GB.

The live gain at C1 (6.5 ms per cycle, two passes) matches the in-situ Marlin gap measured in step 6 (24.0 → 21.6 ms
per pass). Most of it is recovered.

### Argmax check mode (boot A)
`VOCAB_ARGMAX=check` over the A/B traffic:
- 7,494 fast steps checked, 17,397 requests, **0 mismatches** (0 NaN requests).
- 6 steps took the stock path: one grammar request, five with sampled rows.

### Decision
- **L2 prefetch:** keep, windows B+C+D at 12 MiB each. A follow-up could try `L2_PREFETCH_MB_C=16`, since the GPU test
  found fused_qkv_a best at 16 MiB.
- **Vocab-parallel argmax:** keep (`VOCAB_ARGMAX=1`).
- **Drafter-start gap:** with the ring (step 7), the GPU idle gap from draft end to verify start reads 0.31 ms (max
  0.86) in the step-4 gap timer. The NCCL host-node stall described above is gone from the decode graphs.

Production numbers with everything on: `../step456-integration-20261001/` (boot B).
