# Step 7: ring-aware TP all-reduce over RDMA (2026-10-01)

Overview: `docs/DECODE-SPEEDUPS.md`, step 7 ("ring 2-hop RDMA all-reduce for TP4"), done before steps 4–6 because it
was the largest remaining lever. In the September trace the NCCL TP all-reduce cost 14.75 ms per verify pass (166 × ~89 µs).

**Image:** `spark-vllm:0.29.0-nvme4-stridefix-kvtier-verifycap9-ring-20261001`. It is step 3 plus the ring transport.

**Launch:** `ROCE_TP=1` (new). It sets:
- `GLM_ROCE_GROUPS=dcp,tp`, `GLM_ROCE_TP_RING=1`;
- `GLM_ROCE_TP_MAX_SIZE=1MiB`, so all-reduces up to 85 tokens use the ring and prefill stays on NCCL.

The DCP pairs keep RoCEnante (`ROCE_DCP=1`). Code paths below are relative to `runtime/vllm029/`
(`roce/…`) or `runtime/vllm029/roce/` (`glm_roce/…`).

Measured on an abliterated graft of GLM-5.3 with the same quantization envelope as GLM-5.3 Int4-Int8Mix (Int4
group-128 routed experts, Int8 group-128 elsewhere, identical shapes).

## Design

**The problem.** b12x RoCEnante (step 3) needs a link to every peer. The cluster is a switchless ring (06c4 → 365c → ddbf
→ a218 → 06c4, ring order = TP rank order), so the opposite rank (r+2) is not directly reachable. Step 3 could only use
RoCEnante for the DCP pairs, which are direct links.

**The ring transport** (`roce/glm_roce/_ring_proxy.c`, `roce/glm_roce/ring.py`) keeps b12x's pinned region layout,
doorbell and CuTe kernels unchanged, and replaces only the C proxy that moves bytes.
- **Per op s, each rank's proxy runs, strictly in turn:**
  - `own(s)`: write the staged payload to both ring neighbours, each write followed on the same RC queue pair by the
    4-byte flag `s` in `flag[r][slot]`.
  - `forward(s)`: once the ccw neighbour's payload has landed, write it on to the cw neighbour with its flag.
  - Every rank therefore receives all three peers' payloads, one flag per source. This is exactly the layout the b12x
    one-shot kernel waits on when compiled with `hca_count = 1`.
- **Reduction:** fp32 in fixed rank order inside the kernel. It is identical on every rank and deterministic.
- **Not NCCL's bits:** NCCL's ring adds partial sums in bf16 in a different order. The ring sum is rounded once from
  fp32, so it is closer to the exact sum.
- **Safety with two slots:**
  - `own(s+1)` waits until `forward(s)` has completed.
  - The proof that no slot is rewritten while it is still being read, and that at most two doorbells are pending, is in
    the header of `_ring_proxy.c`.
  - The 4-ring's own dependency chain already enforces the forward ordering. Waiting for completion is belt and braces
    and costs nothing measurable, because the next doorbell is always tens of µs later.
- **Links are discovered, not configured.**
  - Every rank publishes the IPv4 GIDs of its active RDMA devices. Each ring edge uses the device pair on a shared /24.
  - The DCP device (`rocep1s0f1`) is avoided, so the ring runs on roceP2p1s0f0/f1, the ports NCCL uses, and the DCP
    pairs keep rocep1s0f1.
  - Discovered links (`[cw, ccw]` per rank):

    | node | cw | ccw |
    |---|---|---|
    | 06c4 | roceP2p1s0f1 (.100) | roceP2p1s0f0 (.104) |
    | 365c | f0 (.102) | f1 (.100) |
    | ddbf | f1 (.103) | f0 (.102) |
    | a218 | f0 (.104) | f1 (.103) |
- **Integration:** in the shim (`glm_roce/install.py`, `adapter.py`), the TP communicator builds the ring runtime when
  `GLM_ROCE_TP_RING=1`, with its own size limits. TP all-gathers stay on NCCL (`GLM_ROCE_TP_GATHER_MAX_SIZE=0`). Graph
  capture, the fail-stop health check and the vote/REQUIRE semantics are unchanged from step 3.
- **Build:** the proxy is compiled at image build, hardened like the b12x one, into the root-owned proxy cache.
- **Idle behaviour:** the same idle backoff as the patched b12x proxy: hot for 50 ms after activity, then naps up to
  5 ms.

## Validation

### Protocol test (no RDMA)
`roce/tests/test_ring_proxy.c`:
- **Setup:** four ranks in one process run the production proxy loop against a fake NIC. Each queue pair executes its
  writes in order after random delays, with occasional 2–4 ms stalls of one queue pair while the others run ahead.
  Payloads land in two halves.
- **Kernel stand-in:** a "kernel" thread per rank stages, rings, waits for the flags and checks every peer's bytes after
  a random delay.
- **Result:** 20,000 ops × 4 ranks, sizes 16 B – 64 KiB: **0 errors** in both modes. The slot bound fires.
- **Mutation check:** dropping the "own posted" condition deadlocks and is caught. Relaxing the forward-completion wait
  is not caught, because it is safe on a 4-ring, as argued above.

### Link discovery
`roce/tests/test_ring_links.py`, on the real GID table: 5/5 pass.

### Four-rank hardware test, model stopped
`roce/test_tp_ring.py` (`ringtest/ring-*.log`) runs through vLLM's TP=4 / DCP=2 groups with `GLM_ROCE_GROUPS=tp,dcp`:
- **Wiring:** the TP adapter runs the ring, the DCP adapter is enabled, and `verify_targets` passes.
- **Eager:** `[T, 6144]` bf16 for T = 1…85, plus fp32 and odd sizes.
  - Every routed case equals the fixed-order fp32 reference `((x0 + x1) + x2) + x3` rounded once, and is
    **byte-identical on all four ranks.**
  - Versus NCCL, about 33% of bf16 elements differ by one bf16 ulp (max 0.25 at magnitude ~16). That is NCCL's rounding,
    not the ring's.
  - Over-size (`[200, 6144]`) and non-16-byte (`[7, 3]`) cases correctly stay on NCCL.
- **CUDA graphs:** 78 layers × (TP all-reduce + DCP all-gather) with compute between, at T = 8 and 32.
  - 300 replays each with fresh inputs and eager all-reduces interleaved.
  - 13 checkpoints per T, **all exact.**
- **Totals:** 61,537 ring ops per rank, every one forwarded, error word 0.
- **Latency:** 156 back-to-back all-reduces in one graph, median per all-reduce over 4 ranks:

  | T (tokens) | NCCL Ring | RDMA ring | saved |
  |---:|---:|---:|---:|
  | 1 | 52 µs | **21 µs** | −31 µs |
  | 8 | 95 µs | **49 µs** | −46 µs |
  | 32 | 184 µs | **126 µs** | −58 µs |

  T=8 (one request at K=7) is the common case: 156 all-reduces per verify pass × 46 µs ≈ **−7 ms per pass**, plus the
  drafter's all-reduces.

### Live
- **Startup:** `GLM_ROCE_RING ready` on all four ranks with the links above. TP all-reduces are routed from the first
  step. The DCP pairs are still on RoCEnante, and the first guarded step is healthy.
- **Smoke test:** count-to-30 is correct.
- **Memory:** rank-0 MemAvailable is 3.5 GB at idle.

## Cycle time (live periods, fixed K, short context; `costgrid/periods.json`)

| requests × K | step 3 | **step 7 (ring)** | change |
|---|---:|---:|---:|
| 1 × 1 | 85.0 ms | **79.5** | −5.5 |
| 1 × 3 | 102.2 | **95.6** | −6.6 |
| 1 × 5 | 117.5 | **111.0** | −6.5 |
| 1 × 7 | 130.7 | **124.4** | −6.3 (−4.8%) |
| 2 × 7 | 186.3 | **179.9** | −6.4 |
| 3 × 7 | 234.7 | **227.3** | −7.4 |
| 4 × 1 | 148.5 | **140.9** | −7.6 |
| 4 × 7 | 272.7 | **264.2** | −8.5 |

This matches the microbenchmark: about −46 µs × ~156 TP all-reduces per verify pass, plus the drafter's. `costs-step7.json`,
built from this grid, is loaded through the control file.

## Acceptance at fixed K=7 (`c1-fixed7/`)
Tokens per cycle:

| | step 3 | step 7 |
|---|---:|---:|
| prose | 2.40 | 2.41 |
| code | 4.76 | 4.83 |

The ring's different rounding does not move acceptance.

Decode tok/s at fixed K=7:

| | step 3 | step 7 |
|---|---:|---:|
| prose | 18.6 | 20.6 |
| code | 35.9 | 37.3 |

## C1 with the verification cap (`c1-auto-full/`; 0 of 104 requests contaminated, 0 errors)
Paired with step 3 on the same prompts:

| set | stock (09-29) | step 3 | **step 7** | vs step 3 (paired) | vs stock |
|---|---:|---:|---:|---:|---:|
| prose (15), mean tok/s | 17.92 | 23.07 | **24.54** | **+6.5%** (14/15 faster) | **+37%** |
| code (15) | 34.21 | 37.75 | **39.53** | **+4.7%** (12/15) | **+16%** |
| prose+think (5) | 16.43 | 22.26 | **23.52** | +5.7% (4/5) | +43% |
| code+think (5) | 31.58 | 35.08 | **36.87** | +5.4% (5/5) | +17% |
| pi agent turns (63), median | 22.72 | 27.70 | **29.99** | **+6.8%** (49/63) | +32% |

Agent tokens per cycle are 3.36, against 3.40 at step 3 on these turns. The step-2 dip is still open. The step-4 review
found a candidate cause: the int8 drafter's conv projections are not rank-deterministic. The fix, `DFLASH_DET_CONV`, ships
with the next image.

## C2–C4 aggregate (`conc/`; 3 clean rounds each)

| n | set | base (09-30) | step 3 | **step 7** | vs step 3 | vs base |
|---|---|---:|---:|---:|---:|---:|
| 2 | prose | 24.1 | 33.0 | **35.2** | +6.7% | **+46%** |
| 2 | code | 46.5 | 50.5 | **52.4** | +3.8% | +13% |
| 2 | mix | 31.7 | 41.4 | **42.9** | +3.6% | +35% |
| 3 | prose | 28.8 | 40.6 | **43.5** | +7.1% | **+51%** |
| 3 | code | 56.2 | 62.0 (1 round) | **60.6** | −2.3% | +8% |
| 3 | mix | 36.6 | 44.1 | **46.5** | +5.4% | +27% |
| 4 | prose | 33.8 | 48.1 (2 rounds) | **49.5** | +2.9% | **+46%** |
| 4 | code | 65.1 | 66.8 | **71.3** | +6.7% | +10% |
| 4 | mix | 43.3 | 53.0 | **52.8** | −0.4% | +22% |

The step-3 cells built from 1–2 rounds are noisier.

## Decision

**Keep. It is the production default (`ROCE_TP=1`).**
- Results are deterministic and identical on every rank. They are not NCCL's bits, because the ring's sum is rounded
  once.
- Cycles are 5.5–8.5 ms shorter.
- C1 is +4.7–6.8% over step 3: prose 24.5 tok/s (+37% over stock), agents +32% over stock.
- `ROCE_TP=0` reverts to NCCL with no rebuild.

**Power:** each node now runs two hot proxy threads during decode, the ring's and the DCP pair's. Both back off when
idle. The operator checks power.

## Follow-up: ring variants (maintenance window 1, model stopped; `ab/`, `run_ring_ab.sh`)
Four-rank test, 60 replays per variant, all exact. Median µs per all-reduce, mean of the four ranks:

| variant | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---:|---:|---:|---:|---:|---:|
| NCCL Ring (same runs) | 51–55 | 60–61 | 71–73 | 96–97 | 133–136 | 184–187 |
| ring, grid 8 (production) | 21.1 | 25.9 | 31.4 | 49.0 | 81.4 | 127.6 |
| ring, grid 16 | 21.0 | 25.6 | 32.5 | 49.1 | 85.7 | 135.2 |
| split, grid 8 | 20.2 | 24.0 | 31.1 | 49.5 | **74.2** | **121.9** |
| split, grid 16 | 20.3 | 25.0 | 32.5 | 48.5 | 77.3 | 124.9 |

**Split mode** (`GLM_ROCE_RING_SPLIT=1`) sends stripe 0 clockwise and stripe 1 counter-clockwise.
- It passes the protocol test in both modes, and the hardware test exactly.
- It gains only at ≥16 tokens (about −6 µs), so it is off. Link bytes are not the limit at decode sizes.
- A bigger kernel grid does not help either.

What does limit T=8 (49 µs vs 21 µs at T=1) remains open. The candidates are the NIC's host-memory DMA per hop and the
kernel's system-scope loads.
