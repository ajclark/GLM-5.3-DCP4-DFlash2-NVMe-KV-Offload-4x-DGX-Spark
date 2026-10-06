# Step 3: RoCEnante one-shot collectives for the DCP pairs (2026-10-01)

Overview: `docs/DECODE-SPEEDUPS.md`, step 3. Review of the vendored code, and the conditions for integrating it:
`docs/ROCENANTE-REVIEW.md`.

**Image:** `spark-vllm:0.29.0-nvme4-stridefix-kvtier-verifycap8-roce-20261001`. It is steps 1 and 2 plus
`runtime/vllm029/roce/` and per-re-preparation timing in the verify-cap log. Code paths below (`roce/…`) are relative
to `runtime/vllm029/`.

Measured on an abliterated graft of GLM-5.3 with the same quantization envelope as GLM-5.3 Int4-Int8Mix (Int4
group-128 routed experts, Int8 group-128 elsewhere, identical shapes).

**Launch:** `ROCE_DCP=1`, now the default. That sets:
- `GLM_ROCE_ALLREDUCE=1`, `GLM_ROCE_GROUPS=dcp`
- `B12X_ROCE_HCA=rocep1s0f1`, `B12X_ROCE_GID_INDEX=3`
- `B12X_ROCE_SPIN_LIMIT=90000000`, `B12X_ROCE_IDLE_MAX_NAP_US=5000`
- `B12X_DISABLE_CUTLASS_RUNTIME_PATCHES=1`
- `GLM_ROCE_MAX_SIZE=1MiB`, `GLM_ROCE_GATHER_MAX_SIZE=4MiB`

## Design

**What is routed.** Each DCP group is a pair of directly cabled ring neighbours: 06c4–365c and ddbf–a218. These
collectives of the DCP groups go through b12x RoCEnante's one-shot RDMA collectives:
- the query all-gather;
- the LSE all-gather;
- the indexer merge all-gather;
- the attention-output reduce-scatter.

TP all-reduces stay on NCCL Ring.

**The vendored code:** 30 files, byte-identical to upstream b12x `b58f34ea` except `_roce_proxy.c`. That file carries
two reviewed patches:
- an idle backoff: hot for 50 ms, then naps that double up to 5 ms;
- a check that a payload fits its slot.

`roce/b12x/PROVENANCE.json` records this, and `roce/tests/test_vendored_b12x.py` checks it.

**knapcio's shim, adapted:**
- routes the groups named in `GLM_ROCE_GROUPS`;
- does middle-dim gathers as a dim-0 gather plus `movedim`, as vLLM's own NCCL path does;
- does the world-2 reduce-scatter as an exchange of halves plus one add per element;
- enters each routed communicator's capture context from the module `graph_capture()`;
- runs the health check for every routed group;
- prepares the padded-gather scratch before capture.

**The proxy build:** it is compiled at image build with `-Wall -Wextra -Werror -fstack-protector-strong
-D_FORTIFY_SOURCE=2` into a root-owned path. The `.so` sha256 is `865ed704…`. vLLM hook targets are verified at
build time.

## Validation

### Proxy idle behaviour

`roce/tests/test_proxy_idle.c` ran on a Spark with no RDMA device:
- **Idle CPU of the proxy thread:** 0.2–1% of a core with the patch, against 1.7–2.4% upstream. Wake-ups fall from
  tens of thousands a second to at most a few hundred.
- **Latency:** hot doorbells are posted within 50–140 µs in this test harness, and the first doorbell after idle
  within 0.2–1.9 ms.
- **Correctness:** catch-up and the slot bound pass.

### Two-rank DCP pair test

`roce/test_dcp_pair.py`, model stopped, both pairs (`pair-*.log`):
- **Wiring:** the DCP communicator has the adapter, the TP communicator does not, and `verify_targets` passes.
- **Eager:** the shapes tested were the query gather [T,16,576] bf16 on dim 1, the LSE gather [T,32] fp32 on dim 0,
  the indexer merge [T,2048,2] fp32 on dim 1 and the reduce-scatter [T,32,512] bf16 on dim 1, for T = 2, 8, 16, 32,
  48 and 96. Every case is **byte-identical to the NCCL path and to a CPU reference.**
- **CUDA graphs:** a 4-layer decode-like sequence at T = 8 and T = 32, with 300 replays each and fresh inputs, and
  eager collectives interleaved. Every 25th replay was checked: **all exact.** Totals: 24,224 RDMA ops, 0 errors.
- **Latency,** for 78 layers × (query gather + LSE gather + reduce-scatter) in one graph:

| | NCCL ring | RoCEnante |
|---|---:|---:|
| T=8 (1 request) | 11.0 ms (47 µs per collective) | **5.8 ms (25 µs)** |
| T=32 (4 requests) | 20.2 ms (87 µs) | **12.6 ms (54 µs)** |

### Live

- **Startup:** `GLM_ROCE_READY` on all four ranks, routing confirmed, the first guarded step healthy.
- **Smoke test:** count-to-100 is correct.
- **Memory:** rank-0 MemAvailable is 2.9 GB at idle.
- **Power:** the operator's check was nominal ("no power anomalies so far").

## Cycle time (live periods, fixed K, short context; `costgrid/periods.json`)

| requests × K | step 1 | step 2 | **step 3** | step 3 vs step 2 | total vs step 1 |
|---|---:|---:|---:|---:|---:|
| 1 × 1 | 94.3 ms | 89.2 | **85.0** | −4.2 ms | −10% |
| 1 × 3 | 114.2 | 108.4 | **102.2** | −6.2 | −11% |
| 1 × 5 | 127.4 | 120.7 | **117.5** | −3.2 | −8% |
| 1 × 7 | 142.3 | 135.2 | **130.7** | −4.5 | −8% |
| 2 × 7 | 201.6 | 191.7 | **186.3** | −5.4 | −8% |
| 3 × 7 | 246.5 | 241.6 | **234.7** | −6.9 | −5% |
| 4 × 7 | 284.8 | 278.0 | **272.7** | −5.3 | −4% |

`costs-step3.json`, built from this grid, is loaded through the control file
(`~/verify-cap-live/costs-step3.json` on spark-06c4).

## Acceptance at fixed K=7 (`c1-fixed7/`)

Tokens/cycle: prose 2.40 and code 4.76, against 2.39 / 4.80 for the BF16 drafter and 2.38 / 4.85 at step 2.
Collectives do not change acceptance, as expected.

Decode tok/s at fixed K=7 (cycle time only):

| set | stock | step 2 | step 3 |
|---|---:|---:|---:|
| prose | 17.4 | 18.0 | 18.6 (+7% vs stock) |
| code | 33.4 | 35.3 | 35.9 (+7.5% vs stock) |

7 of 40 requests overlapped outside traffic and were excluded.

## C1 with the verification cap (`c1-auto-full/`; 0 of 103 requests contaminated)

| set | stock (09-29) | verify cap v3 (09-30) | step 2 | **step 3** | vs v3 | vs stock |
|---|---:|---:|---:|---:|---:|---:|
| prose (15), mean tok/s | 17.92 | 20.78 | 21.91 | **23.07** | +11% | **+29%** |
| code (15) | 34.21 | 34.77 | (3 prompts) | **37.75** | +9% | **+10%** |
| prose+think (5) | 16.43 | 20.02 | — | **22.26** | +11% | **+35%** |
| code+think (5) | 31.58 | 32.56 | — | **35.08** | +8% | **+11%** |
| pi agent turns (63), median | 22.72 | 26.90 | 26.69 | **27.70** | paired **+6.0%** (faster in 48 of 63) | paired **+22%** |

The agent turns gained +5.4% paired against step 2. Their tokens per cycle (2.91) are still below v3's 3.01; that
open item is carried over from step 2.

## C2–C4 aggregate (`conc/`; means of clean rounds, rounds in brackets)

| n | set | base (09-30 production) | step 2 | **step 3** | vs step 2 | vs base |
|---|---|---:|---:|---:|---:|---:|
| 2 | prose | 24.1 | 31.3 | 33.0 (3) | +5.4% | **+37%** |
| 2 | code | 46.5 | 47.4 | 50.5 (3) | +6.5% | **+9%** |
| 2 | mix | 31.7 | 34.7 | 41.4 (3) | +19.3% | **+31%** |
| 3 | prose | 28.8 | 39.5 | 40.6 (3) | +2.6% | **+41%** |
| 3 | code | 56.2 | 56.3 | 62.0 (1) | +10.0% | **+10%** |
| 3 | mix | 36.6 | 44.3 | 44.1 (3) | −0.4% | **+20%** |
| 4 | prose | 33.8 | 46.0 | 48.1 (2) | +4.6% | **+42%** |
| 4 | code | 65.1 | 65.1 | 66.8 (3) | +2.8% | **+3%** |
| 4 | mix | 43.3 | 49.6 | 53.0 (3) | +6.8% | **+22%** |

Operator traffic overlapped part of this run, and those rounds are excluded; cells with 1–2 rounds are noisier. The
n=2 mix jump also reflects step 2's 2-round cell.

## Decision

**Keep. It is the production default** (`ROCE_DCP=1`).
- Collectives are exact.
- Cycles are 3–7 ms shorter.
- C1 prose is now +29% over stock and code +10%.
- C2–C4 prose is +37–42% over the September 30 base, and code gains under concurrency for the first time (+3–10%).

**Power:** the operator checks it, and `ROCE_DCP=0` reverts to NCCL with no rebuild.

## Notes

- **Re-preparation is cheap:** the verifycap8 log shows 0.55–0.66 ms per re-preparation, at 35–50% of decisions at
  batch sizes. The step-4 estimate (6–7 ms each) was wrong; the larger host costs are the ~1 ms K broadcast and the
  host-bound gap before each step ("wait 0.00 ms"). Step 4 is re-scoped accordingly.
- **Run bookkeeping:** an idle-gated resume script and a manual stop overlapped. The run finished correctly, and the
  logs show the duplicate waiter being stopped.
