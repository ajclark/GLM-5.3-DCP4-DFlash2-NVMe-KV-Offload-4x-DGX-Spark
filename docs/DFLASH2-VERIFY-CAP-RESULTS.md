# Draft-aware verification cap for DFlash2 — results (2026-09-29/30)

A rewrite of September's adaptive-verification experiment (legacy runtime, never promoted;
`results/adaptive-spec/`) for today's vLLM 0.29 stack. Code: `runtime/vllm029/verify_cap_overlay/`.
Evidence: `results/verify-cost-screen-20260929/`, `results/dflash-acceptance-20260930-verifycap3/`
and `results/dflash-acceptance-20260930-verifycap3-agent/`. The offline tools that recorded the
acceptance traces, fitted the calibration and replayed policies are not published.

Measured on an abliterated graft of GLM-5.3 with the same quantization envelope as GLM-5.3
Int4-Int8Mix (Int4 group-128 routed experts, Int8 group-128 elsewhere, identical shapes).

## Outcome

| Set (C1, greedy, same prompts as the stock acceptance probe) | Stock tok/s | Verify cap (v3) | Change |
|---|---|---|---|
| Prose (15) | 17.5 | 20.7 | **+18.1%** (every prompt faster) |
| Prose, thinking on (5) | 16.4 | 19.9 | **+21.1%** |
| Code (15) | 33.5 | 34.3 | +2.5% |
| Code, thinking on (5) | 31.4 | 32.4 | +3.0% |
| **pi agent turns** (63 clean of 64; real sessions, thinking high, tools, up to ~90K context) | 21.4 | 25.0 | **+16.8%** (faster in 55 of 63) |
| ↳ reasoning-heavy turns | 18.8 | 22.2 | +18.1% |
| ↳ tool-call-heavy turns | 30.1 | 32.0 | +6.4% |

Decode tok/s = generated tokens ÷ decode time, one request at a time on the idle endpoint.
The September controller, on its own held-out set, reached prose +15.8% / code +1.0%, and
was slightly negative (−1.2%) on real pi sessions; this version is +15–17% on them.
Agent baseline: the stock replay of the same turns (run not published); verify-cap run:
`results/dflash-acceptance-20260930-verifycap3-agent` (median turn 22.7 → 26.9 tok/s,
tokens/cycle 3.09 → 3.01). Agent-turn output text is blanked in the published rows.

## What it does

The DFlash2 drafter always proposes 7 tokens, and the target verifies all 7 every cycle.
For prose, drafts 4–7 are rarely accepted (acceptance by position 0.66 / 0.37 / 0.19 /
0.10 / …), yet verifying them costs expert-weight reads on every cycle. Measured on this
runtime, verifying fewer drafts is much cheaper:

| Drafts verified (K) | 4K ctx | 32K | 100K |
|---|---|---|---|
| 7 | 141.6 ms | 142.1 | 143.8 |
| 5 | 132.3 | 128.7 | 121.5 |
| 3 | 106.7 | 114.1 | 109.9 |
| 1 | 93.9 | 93.7 | 95.9 |

Each cycle, at concurrency 1, the worker verifies only the first K ∈ {1, 3, 5, 7} drafts:

1. **Draft-aware decision.** Right after drafting, it reads the selector's confidence in each
   drafted token (already computed by the path walk), maps it through a calibration table
   (position × confidence bin, fitted from exact offline acceptance traces), and picks the K
   that maximizes expected tokens per millisecond against the measured cost table.
2. **Online correction.** A per-request and a slow global bias in log-odds space learn from
   the drafts that were verified, so a request whose drafts keep being accepted (code) earns
   longer verification. Unverified positions are never counted (no censoring bias).
3. **Speculative preparation.** The host prepares the next step with the request's most
   frequent recent cap while the GPU is still busy, then decides and re-prepares only if
   the decision differs. Without this, waiting for the draft left the GPU idle during step
   preparation and cost 5–8% (measured with a forced-K=7 control).
4. **Correctness by construction.** Unverified drafts count as rejected; the scheduler's
   normal rollback applies. A shorter verification checks a prefix of the same greedy draft,
   so greedy output is unchanged up to this stack's numerical nondeterminism (the stock
   configuration itself reproduces only 11 of 40 outputs byte-for-byte across runs).
   Rank 0 decides and broadcasts K over the TP CPU group (~0.9 ms) so every rank verifies
   the same tokens. Anything but one greedy decode request (concurrency ≥ 2, sampling,
   structured output, penalties) runs at K=7, unchanged.

## How it compares with the September controller (v4)

| | September v4 | This version |
|---|---|---|
| Decision input | the request's own acceptance history | the current draft's confidence + history-based correction |
| Warm-up | 8 cycles at K=7 per request; short turns never adapt | adapts from the first cycle |
| Exploration | forced K=7 probe every 16 steps | none needed |
| Cost table | offline, legacy runtime (long-context rows contaminated by the 90k drafter bug) | measured on this runtime |
| Integration | scheduler patch | worker-side hook; scheduler untouched |

## Iterations (live, same 40 prompts, decode tok/s vs stock)

| Version | Prose | Code | Prose+think | Code+think | Change |
|---|---|---|---|---|---|
| control (same code, forced K=7) | −5.4% | −6.3% | −1.0% | −7.7% | sync overhead alone |
| v1 | +13.6% | +0.7% | +13.8% | −1.7% | first working version |
| v2 | +16.3% | +0.9% | +17.2% | +2.2% | calibration refit on more traces + online correction |
| **v3** | **+18.1%** | **+2.5%** | **+21.1%** | **+3.0%** | speculative preparation (host no longer waits) |
| v4 (caps {3,5,7}) | +15.2% (8 prompts) | — | — | — | dropping K=1 was worse live; stopped early |

## State

- **Default since 2026-09-30 ~01:45:** `start-glm53.sh` and the per-node
  `runtime/vllm029/launch.sh` default to the verify-cap image with `VERIFY_CAP=1`; production runs it with the
  warm NVMe slab reused. `VERIFY_CAP=0 bash start-glm53.sh` turns it off
  without changing the image; `DCP_IMAGE=spark-vllm:0.29.0-nvme4-stridefix-kvtier-20260928`
  returns to the previous image.
- Test image: `spark-vllm:0.29.0-nvme4-stridefix-kvtier-verifycap4-20260930` on all four
  Sparks (v3 behaviour with the default caps). Since 2026-09-30 ~16:00 production runs
  `…-verifycap5-voicefirst-20260930`: the same verification cap plus the voice-first
  scheduler ([VOICE-FIRST-SCHEDULING-RESULTS.md](VOICE-FIRST-SCHEDULING-RESULTS.md)). It is layered on the production image and is
  not in `manifest.json`, so the NVMe slab salt — and the warm slab — are unchanged.
- Memory: three extra single-request CUDA graphs; rank 0 stayed above ~0.9 GB free
  throughout testing, no swap growth, no OOM.

## Limits and next steps

- Only concurrency 1 benefits. With two or more requests decoding, every request runs at
  K=7 (the sparse-attention backend only supports uniform CUDA-graph batches).
- The calibration was fitted on greedy prose/chat/code traces; pi agent traces were not in
  the fit. Refitting with agent traces (and all 190 held-out sequences) is cheap.
- About a third of cycles still re-prepare (K alternates between 1 and 3 on prose); a better
  predictor would recover a little more of the remaining overhead.
- The drafter's full-vocabulary token probability was slightly more informative than the
  selector confidence in simulation but needs the full softmax in the runtime.
