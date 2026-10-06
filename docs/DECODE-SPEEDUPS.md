# Decode speedups (2026-09-29 → 10-01)

Measured on an abliterated graft of GLM-5.3 with the same quantization envelope as GLM-5.3 Int4-Int8Mix (Int4
group-128 routed experts, Int8 group-128 elsewhere, identical shapes).

Starting point: the [draft-aware verification cap](DFLASH2-VERIFY-CAP-RESULTS.md) (09-30). Each step below was
designed, implemented behind its own launcher switch, measured, and then kept or reverted. Step 7 ran before steps
4–5 because it was the largest lever. Several ideas come from knapcio's GLM-5.3-Flash stack; see the README's Credits.

| step | status | result | report |
|---|---|---|---|
| 1 batch verify cap + small fixes | **kept** | C2–C4 aggregate: prose +24–34%, mix +11–16%, code flat; C1 unchanged; prefix hit +128–256 tokens | [step 1](../results/step1-batchcap-20260930/REPORT.md) |
| 2 drafter diet: int8 drafter, `fc` split over TP, FP8 draft head | **kept** | cycles −5 to −10 ms; C1 prose +5.4%; C2–C4 prose/mix +3–6% more; ~0.8 GB/rank freed | [step 2](../results/step2-drafter-20260930/REPORT.md) |
| 3 RoCEnante for the DCP pairs | **kept** (`ROCE_DCP=1`) | byte-identical to NCCL; cycles −4 to −6 ms more; C1 prose 23.1 (+29% vs stock), code 37.8 (+10%), agents +22% paired; C2–C4 prose +37–42% vs base | [step 3](../results/step3-roce-20261001/REPORT.md), [review](ROCENANTE-REVIEW.md) |
| 7 ring 2-hop TP all-reduce | **kept** (`ROCE_TP=1`) | new ring transport for RoCEnante's kernels: exact and rank-identical (not NCCL's bits); all-reduce 95 → 49 µs at 8 tokens (39 µs in split mode); cycles −5.5 to −8.5 ms; C1 prose 24.5, code 39.5, agents 30.0 (+4.7–6.8% paired vs step 3) | [step 7](../results/step7-ring-allreduce-20261001/REPORT.md), [latency](RDMA-COLLECTIVES-LATENCY.md) |
| 4 C1 verify-cap refinements | **kept** (`VERIFY_CAP_LOCAL_DECIDE=1`) | rank-local K decision (GPU broadcast of rank 0's confidences, no gloo per step): GPU gap before each verify 1.4 → 0.4 ms; C1 prose +2.0%, code +3.9% paired. Also fixed: the int8 drafter's conv projections used non-deterministic Marlin atomic-add, so ranks could draft different tokens (`DFLASH_DET_CONV=1`) | [step 4](../results/step4-hostgap-20261001/REPORT.md) |
| 5 L2 prefetch + vocab-parallel argmax | **kept** (`L2_PREFETCH=1`, `VOCAB_ARGMAX=1`) | exact; prefetch −6.5 ms per C1 cycle (in-boot ABBA); argmax 0 mismatches in 17k checked requests, ~−1.5 ms | [step 5](../results/step5-l2-argmax-20261001/REPORT.md) |
| **stack (steps 1–5, 7)** | **production** | C1 prose **26.7 tok/s (+49% vs stock)**, code **42.8 (+25%)**, agents **33.0 (+45%)**; C2–C4 prose +54–62% vs the 09-30 base; cycles 72 / 116 ms at 1 × K=1 / K=7 | [integration](../results/step456-integration-20261001/REPORT.md) |

Not adopted: a hand-written int8 decode kernel for the dense layers (step 6). Stock Marlin already runs at 92% of the
read roofline in isolation, so the kernel could save at most ~0.6 ms per pass; the in-situ gap was environmental and
step 5's L2 prefetch addressed it.

## How every step was measured

- **Fixed cost per cycle:** `runtime/vllm029/cycle_bench.py` at forced K=7 and K=1, 2K/32K/100K context. Baseline
  `results/verify-cost-screen-20260929` (K=7 141.6 ms, K=1 93.9 ms at 4K). Noise is about ±0.5 ms.
- **Single request (C1):** `runtime/vllm029/spec_accept_probe.py`: 15 prose and 15 code prompts, 5 + 5 with thinking
  on, and replayed coding-agent turns of up to ~90K context. Baseline
  `results/dflash-acceptance-20260930-verifycap3` and `…-verifycap3-agent`.
- **2–4 concurrent requests:** `runtime/vllm029/conc_bench.py`: aggregate tok/s over the window where all n streams
  decode. The base was measured in the same boot as step 1 with the batch logic off.
- **Gates on every step:** count100 byte parity for the exact changes, clean probe outputs, no preemptions or engine
  errors, rank-0 MemAvailable ≥ 0.8 GB throughout. The endpoint was idle during every measurement; the probes flag
  any request that overlapped other traffic.

All code lives in the overlay layer `runtime/vllm029/verify_cap_overlay/` (plus `runtime/vllm029/roce/`), which is
outside `manifest.json`, so the NVMe slab salt and a warm slab survive every rebuild.
