# Where the DCP4 verify-pass penalty goes (torch profiler, 2026-09-05)

Traces: `results/dcp4-dflash-300k-prof/` (serving config, DCP=4, 307k window,
slab tier) and `results/dcp1-dflash-80k-prof/` (same image and overlays with
`DCP_SIZE=1`, 82k window, no tier). One count100 request each (100 greedy
tokens, thinking off), all four ranks, `analyze_trace.py` per rank. Ranks are
symmetric to within 0.5 ms; numbers below are rank 0 (DCP=4) and rank 1
(DCP=1), median over the 12-13 steady-state target verify passes (8 tokens).
At DCP=4 the verify pass is one CUDA-graph replay (173 ms); at DCP=1 it is
not graph-replayed as a unit and was cut as the gap between consecutive
drafter graphs (129 ms, which also contains scheduler time).

| kernel bucket per verify pass | DCP=1 | DCP=4 | delta |
|---|---:|---:|---:|
| MoE experts (marlin_moe) + marlin GEMM | 84.7 ms | 85.0 ms | 0 |
| b12x sparse MLA attention (`UnifiedPrefillMG`) | 12.3 ms (78 x 158 us) | 23.8 ms (77 x 309 us) | **+11.5** |
| NCCL all-gather (query 78, LSE 78, indexer merge ~20) | 0.4 ms | 12.3 ms (LSE ones 1.8) | **+11.9** |
| NCCL reduce-scatter (attention output) | 0 | 7.5 ms (76 x 99 us) | **+7.5** |
| dense GEMM (cutlass) | 16.2 ms | 21.2 ms | +5.0 |
| NCCL all-reduce (TP) | 13.0 ms | 14.6 ms | +1.6 |
| other kernels + top-k | 3.0 ms | 6.7 ms | +3.7 |
| GPU idle inside the pass | 0.7 ms | 4.5 ms | +3.8 |
| **wall** | **129.3 ms** | **172.9 ms** | **+43.6** (measured cycle: 138 vs 173) |

What this says:

- The GPU is busy 97% of the pass on both sides; nothing is overlapped, so
  every collective and every extra kernel adds to the cycle one for one.
- The collectives are ~19.4 ms of the penalty, not the ~33 ms the all-reduce
  latency table predicted: the big all-gathers run at ~105 us and the
  reduce-scatters at ~99 us. Halving their payload (pre-expansion query
  gather, projection before merge) is worth ~2-3 ms each, not 5-6.
- The attention kernel doubles (158 -> 309 us per layer). Under DCP the
  filter leaves the other ranks' 3/4 of the 2,048 top-k slots in place as -1
  and the kernel masks them per cell but still walks them, for 64 gathered
  heads instead of 16. Compacting each token's owned candidates to the front
  and passing the count (`GLM_DCP_COMPACT=1`, `compact_dcp_candidates`,
  b12x `topk_length`) lets the kernel walk ~1/4 of the slots: expected
  ~-11 ms per pass, no memory cost.
- Same-clock DCP=1 baseline (production launcher at the 2000 MHz lock,
  `results/baseline-dcp1-prod-2000mhz`): count100 56.5 tok/s at 138.9 ms,
  so the residual DCP4 cost after compaction is 16.3 ms per cycle.
- Measured levers (boots `dcp4-dflash-300k-compact`, `-compact-pregather`):
  compaction verify pass 172.9 -> 150.7 ms (attention 24.1 -> 2.9 ms);
  pre-gather on top 150.7 -> 150.1 ms (all-gather 12.1 -> 12.0). The big
  all-gathers are at their latency floor (~88-105 us) at either payload.
- The +5 ms of dense GEMM is not DCP. Diffing GEMM launch shapes between
  the two passes shows the extra launches come in multiples of the draft
  steps (grid (8,48,1): 6 -> 21, (8,12,1): 1 -> 12, with the drafter's
  flash kernel 1 -> 6): the DCP=4 verify graph replays the drafter inside
  it, while at DCP=1 the drafter is the separate 7 ms graph the gap cut
  excluded. With that, the DCP=1 cycle is 129 + 7 = 136 ms (measured 138)
  and the DCP penalty decomposes as attention +11.5, all-gather +11.9,
  reduce-scatter +7.5, TP all-reduce +1.6, idle +3.8, which is the 36 ms.
