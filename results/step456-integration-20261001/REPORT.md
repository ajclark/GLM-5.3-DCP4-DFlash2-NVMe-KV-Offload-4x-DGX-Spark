# Steps 4–6 on top of the ring: integration and production measurement (2026-10-01)

**Image:** `spark-vllm:0.29.0-nvme4-stridefix-kvtier-verifycap10-stack-20261001`. It contains:
- step 7, the TP ring (`ROCE_TP=1`);
- step 4, the rank-local K decision, the early decision and the GPU gap timer;
- the drafter determinism fix `DFLASH_DET_CONV`;
- step 5, `glm_fast`: the vocab-parallel argmax and L2 prefetch;
- a `periods_epoch` reset in the control file, for in-boot A/B arms;
- `costs.json` = `costs-step7.json`.

Step 6 (a dense int8 decode kernel) was stopped and is not in the image; see `docs/DECODE-SPEEDUPS.md`.

Measured on an abliterated graft of GLM-5.3 with the same quantization envelope as GLM-5.3 Int4-Int8Mix (Int4
group-128 routed experts, Int8 group-128 elsewhere, identical shapes).

Per-step designs:
- `../step4-hostgap-20261001/REPORT.md`
- `../step5-l2-argmax-20261001/REPORT.md`
- `../step7-ring-allreduce-20261001/REPORT.md`

## Boot A: in-boot A/B of L2 prefetch, argmax check (`bootA-l2.log`, `bootA-l2/`)
**Switches:** `VERIFY_CAP_LOCAL_DECIDE=1 VERIFY_CAP_EARLY=1 VERIFY_CAP_GAP_EVENTS=1 VOCAB_ARGMAX=check L2_PREFETCH=1`.
The prefetch windows were switched through `~/verify-cap-live/glm_fast_l2pf.json`, with a fresh `periods_epoch` per arm.
Arms run off / B+C+D / B+C+D / off at fixed K, one conc_bench round per arm (`arms.sh`).

| requests × K | off | B+C+D | change |
|---|---:|---:|---:|
| 1 × 7 | 124.37 / 123.55 ms | 117.83 / 117.01 | **−6.5 ms** |
| 1 × 1 | 79.44 / 79.49 | 74.10 / 73.88 | **−5.5 ms** |
| 4 × 7 | 263.53 / 264.51 | 259.59 / 261.46 | −3.5 ms |
| 4 × 1 | 141.54 / 141.86 | 136.26 / 136.47 | −5.3 ms |

- **Step 4 at fixed K:** the off arms match step 7 (124.4 ms at 1 × 7), so step 4 and the determinism fix cost nothing
  at fixed K.
- **Argmax check mode:** 7,494 fast steps and 17,397 requests checked, **0 mismatches**.

## Boot B: production candidate (`bootB/`)
**Switches:** `VERIFY_CAP_LOCAL_DECIDE=1 VERIFY_CAP_EARLY=1 VERIFY_CAP_GAP_EVENTS=1 VOCAB_ARGMAX=1 L2_PREFETCH=1`
(windows B+C+D, 12 MiB each), plus the defaults (`ROCE_TP=1`, `ROCE_DCP=1`, `DFLASH_DET_CONV=1`). Series: `run_bootB.sh`.

### Cycle time (live periods, fixed K, short context; `bootB/costgrid/periods.json`)

| requests × K | step 3 | step 7 (ring) | **stack** | stack vs step 7 | stack vs step 3 |
|---|---:|---:|---:|---:|---:|
| 1 × 1 | 85.0 ms | 79.5 | **72.0** | −7.5 | −15% |
| 1 × 3 | 102.2 | 95.6 | **87.6** | −8.0 | −14% |
| 1 × 5 | 117.5 | 111.0 | **102.7** | −8.3 | −13% |
| 1 × 7 | 130.7 | 124.4 | **116.0** | −8.4 | −11% |
| 2 × 7 | 186.3 | 179.9 | **170.2** | −9.7 | −9% |
| 3 × 7 | 234.7 | 227.3 | **222.4** | −4.9 | −5% |
| 4 × 1 | 148.5 | 140.9 | **130.9** | −10.0 | −12% |
| 4 × 7 | 272.7 | 264.2 | **259.5** | −4.7 | −5% |

The C1 change (−8 ms) is larger than L2 prefetch alone (−6.5 ms). The remaining ~1.5 ms is the vocab-parallel argmax:
the stock path's eager 1.9 MB NCCL logits gather plus sampler work.

### C1 with the verification cap (`bootB/c1-auto-full/`; 0 of 104 contaminated, 0 errors; agent-turn text blanked)
Paired on the same prompts:

| set | stock (09-29) | step 3 | step 7 | **stack** | vs step 7 (paired) | vs stock |
|---|---:|---:|---:|---:|---:|---:|
| prose (15), mean tok/s | 17.92 | 23.07 | 24.54 | **26.66** | **+8.7%** (15/15 faster) | **+49%** |
| code (15) | 34.21 | 37.75 | 39.53 | **42.82** | **+8.5%** (14/15) | **+25%** |
| prose+think (5) | 16.43 | 22.26 | 23.52 | **25.20** | +7.3% (5/5) | +53% |
| code+think (5) | 31.58 | 35.08 | 36.87 | **41.25** | +11.5% (5/5) | +31% |
| pi agent turns (63), median | 22.72 | 27.70 | 29.99 | **32.98** | **+7.1%** (55/63) | **+45%** |

Agent tokens per cycle on the paired turns: 3.357 at step 7 → 3.395 here (+1.1%). This boot is the first with
`DFLASH_DET_CONV=1`.

### C2–C4 aggregate (`bootB/conc/`; 3 clean rounds each)

| n | set | base (09-30) | step 3 | step 7 | **stack** | vs step 7 | vs base |
|---|---|---:|---:|---:|---:|---:|---:|
| 2 | prose | 24.1 | 33.0 | 35.2 | **37.2** | +5.7% | **+54%** |
| 3 | prose | 28.8 | 40.6 | 43.5 | **46.7** | +7.4% | **+62%** |
| 4 | prose | 33.8 | 48.1 | 49.5 | **53.7** | +8.5% | **+59%** |
| 2 | mix | 31.7 | 41.4 | 42.9 | **44.8** | +4.4% | +41% |
| 3 | mix | 36.6 | 44.1 | 46.5 | **48.1** | +3.4% | +31% |
| 4 | mix | 43.3 | 53.0 | 52.8 | **54.7** | +3.6% | +26% |
| 2 | code | 46.5 | 50.5 | 52.4 | **53.0** | +1.1% | +14% |
| 3 | code | 56.2 | 62.0 | 60.6 | **62.7** | +3.5% | +12% |
| 4 | code | 65.1 | 66.8 | 71.3 | **73.3** | +2.8% | +13% |

### Gates
- **Probes:** CLEAN, no engine errors.
- **Argmax:** 40,494 fast steps; 6 stock steps (one grammar request, five with sampled rows).
- **Rank check:** ok 1258/1258 sync points. Selector confidences differed from rank 0's on 0 of 59,280 drafted rows per
  rank (`DFLASH_DET_CONV=1`).
- **Memory:** rank-0 MemAvailable minimum over the series was 2.86 GB (2.9 GB at the end).

### What step 4 sees live (`bootB/verifycap-log.txt`)
Under async scheduling the host is 80–100 ms ahead of the GPU: "draft launch → verify launch" is a host time of 82–98
ms. As a result:
- the early decision never triggers (0%);
- re-preparations (25–32% of decisions, 0.39 ms each) happen off the GPU's critical path;
- the GPU's draft end → verify start gap is 0.38–0.40 ms (max 0.75).

The early decision and re-preparation do not matter. Boot C shows that the gloo K broadcast did.

## Boot C: the stack with step 4 off (`bootC/`)
**Switches:** as boot B but `VERIFY_CAP_LOCAL_DECIDE=0 VERIFY_CAP_EARLY=0` (rank-0 decide + gloo broadcast of K), with
`GAP_EVENTS=1`. C1 prose/code probe (`--max-context-chars 0`), paired with boot B.

**GPU idle gap, draft end → verify start** (rank-0 CUDA events, 60 s windows of ~600 steps):

| | median | max |
|---|---:|---:|
| step 4 off | **1.34–1.47 ms** | 2.9 |
| step 4 on (boot B) | **0.38–0.40 ms** | 0.75 |

| set | step 4 off (C) | step 4 on (B) | B vs C, paired |
|---|---:|---:|---:|
| prose (15) | 26.13 | 26.66 | **+2.0%** (13/15 faster) |
| code (15) | 41.21 | 42.82 | **+3.9%** (13/15) |
| prose+think (5) | 24.94 | 25.20 | +1.0% (3/5) |
| code+think (5) | 38.60 | 41.25 | +6.9% (5/5) |

Acceptance differs a little between boots (code tokens/cycle 4.33 vs 4.45), so the code gain is partly noise. Prose
gained with lower acceptance (2.27 vs 2.24), so its +2% is cycle time. That matches the ~1 ms gap reduction in a ~50 ms
average prose cycle. **Step 4 is kept** with local decide on. The early decision never triggers under async
scheduling, so it is off.

## Production defaults (launcher, 2026-10-01 ~10:00)
- Image `verifycap10-stack`.
- `ROCE_DCP=1 ROCE_TP=1 DFLASH_DET_CONV=1 VERIFY_CAP_LOCAL_DECIDE=1 VOCAB_ARGMAX=1 L2_PREFETCH=1` (windows B+C+D); the
  early decision and gap timer are off.
- Prefill cadence stays off (`PREFILL_CADENCE=1`), as asked.
- The control file names `costs-stack.json`, measured in boot B.
- Verified with a plain `start-glm53.sh`: the container env and a count test are correct.
