# Step 1: batch verify cap, runtime control, prefix-hit fix, prefill cadence, argmax clamp (2026-09-30)

Overview: `docs/DECODE-SPEEDUPS.md`, step 1.
Image: `spark-vllm:0.29.0-nvme4-stridefix-kvtier-verifycap6-batch-20260930`, which is the production base plus
`runtime/vllm029/verify_cap_overlay/`.
Everything was measured in one boot. Arms were switched through the runtime control file on rank 0, with no restarts.
Measured on an abliterated graft of GLM-5.3 with the same quantization envelope as GLM-5.3 Int4-Int8Mix (Int4
group-128 routed experts, Int8 group-128 elsewhere, identical shapes).

## Design

- **Batch verify cap** (`verify_cap.py`):
  - **When it applies:** the step is 1 to `VLLM_VERIFY_CAP_BATCH_MAX` (default 4) greedy decode requests with 7 drafts
    each, and nothing else. Otherwise K=7, as before.
  - **What it does:** every request in the step verifies the same K ∈ {1,3,5,7}, so the verify batch stays uniform.
    The sparse MLA backend needs that for FULL CUDA graphs.
  - **Graphs:** (n, n·(K+1)) FULL graphs are captured for n = 1..4 and K ∈ {1,3,5}, 12 in total. Captured FULL graphs
    went from 15 to 24; graph capture takes 1.79 GiB in total.
  - **Decision rule:** maximize Σ_r E_r(K) − λ_n·T(n,K) (policy `lambda`), or Σ_r E_r(K)/T(n,K) (policy `ratio`,
    the v3 rule). Per-request online bias learning now also runs inside batches.
- **Runtime control:**
  - `~/verify-cap-live/control.json` on the head. It is re-read on change, about once a second, with no restart.
  - Fields: mode `auto`/`fixed`/`off`, `fixed_k`, `batch`, `policy`, `alpha`, `costs` (a cost-table file that
    replaces the one in the image), and `prefill_cadence` (read by the scheduler).
  - Rank 0 writes live cycle periods to `periods.json` per (n, K, context band). A period is kept only for
    consecutive steps.
  - Whether a step takes part in the K broadcast still depends only on the scheduler output, so ranks cannot
    disagree.
- **Prefix-hit fix** (`VLLM_PREFIX_HIT_FIX=1`):
  - The EAGLE last-block drop is applied only to the DFlash draft group (`kv_cache_coordinator.py`); the log reads
    `Prefix-hit fix: EAGLE last-block drop only for draft groups [1]`.
  - The sliding-window finder now drops before aligning in its no-full-match path
    (`single_type_kv_cache_manager.py`).
- **Prefill cadence** (`VLLM_PREFILL_CADENCE` or control `prefill_cadence`): when no voice request is active,
  prefill chunks run only on every Nth step while others decode.
- **Argmax clamp** (vllm#50843): `tl.minimum(..., vocab_size - 1)` at the three tile-argmax sites. It is bit-exact
  for every valid id.
- **Guard:** `launch.sh` refuses `MAXLEN` > 400000 (the SM121 `persistent_topk` limit).
- **Tests:** `test_verify_cap.py` (batch path, control reload, periods) and `test_voice_first.py` (cadence) in
  `runtime/vllm029/verify_cap_overlay/` pass.

## Prefix-cache hit (`prefix-hit-fix-on.json`, `prefix_hit_probe.py`)

The same salted prompt was sent twice; the table shows the second send. "Before" is derived from the unpatched finder
logic, which reproduces the 256-token hit measured for the 638-token voice prompt in
`docs/VOICE-FIRST-SCHEDULING-RESULTS.md`.

| prompt tokens | GPU hit before (derived) | GPU hit now | NVMe hit |
|---:|---:|---:|---:|
| 918 | 640 | **768** | 0 |
| 1,529 | 1,152 | **1,408** | 0 |
| 3,717 | — | 0 | 3,584 |
| 7,476 | — | 0 | 7,296 |

Short prompts gain 128–256 cached tokens. Long prompts are served from the NVMe tier; the GPU hit of 0 on an
immediate repeat is pre-existing behaviour and is noted for later.

## Batch cost grid (`costgrid/periods.json`)

Fixed K through control. Median live cycle period in ms, short context, prose and code:

| K | n=1 | n=2 | n=3 | n=4 |
|---|---:|---:|---:|---:|
| 1 | 94.3 | 120.1 | 139.3 | 158.0 |
| 3 | 114.2 | 149.0 | 188.8 | 209.4 |
| 5 | 127.4 | 182.3 | 217.6 | 251.5 |
| 7 | 142.3 | 201.6 | 246.5 | 284.8 |

The n=1 row matches the historical screen (K=1 93.9, K=7 141.6). That screen's K=3 and K=5 values at 4K (106.7 and
132.3) were off, as expected: they were measured before speculative preparation.

## C2–C4 aggregate decode (`ab/`, `runtime/vllm029/conc_bench.py`)

- **Protocol:** 3 rounds per cell. Each round is n streams started together, 512 max tokens, greedy, thinking off.
  "mix" rotates prose, code and a replayed pi agent turn. 81 rounds, 0 contaminated.
- **Metric:** mean aggregate tok/s over the window in which all n streams decode. The number in brackets is tokens
  per cycle.
- **Arms:** `base` = batch off, which is production behaviour. `ratio` and `lambda` = batch on, with each policy.

| n | set | base | ratio | lambda |
|---|---|---:|---:|---:|
| 2 | prose | 24.1 (2.29) | 30.0 **+24.5%** | 31.1 **+29.3%** |
| 2 | code | 46.5 (4.72) | 47.8 +2.8% | 46.3 −0.3% |
| 2 | mix | 31.7 (2.83) | 36.7 **+16.0%** | 36.1 **+14.1%** |
| 3 | prose | 28.8 (2.30) | 37.3 **+29.3%** | 37.9 **+31.6%** |
| 3 | code | 56.2 (4.88) | 56.2 +0.1% | 56.8 +1.1% |
| 3 | mix | 36.6 (2.86) | 42.3 **+15.6%** | 42.0 **+14.7%** |
| 4 | prose | 33.8 (2.40) | 44.6 **+31.9%** | 45.3 **+34.0%** |
| 4 | code | 65.1 (4.95) | 64.9 −0.4% | 67.2 +3.2% |
| 4 | mix | 43.3 (2.95) | 47.9 **+10.6%** | 48.4 **+11.7%** |

**Reading:**
- Prose and mixed traffic gain 11–34%. Code is flat: its drafts are long, so K=5–7 stays optimal.
- Between the two policies, λ is +0.3 to +3.6 points better on prose and code, and 0.3 to 1.9 points worse on mix.
  The difference is within noise.

**Open items from the logs:**
- Batch steps re-prepare on about 50% of decisions: the per-batch-size prediction is a mode of the last 8 decisions,
  and batch decisions vary more. This is the step-4 target (prepare-8-then-fixup).
- The decision wait is 0.00 ms at batch sizes, so the host, not the GPU, is on the critical path at C≥2.

## Memory

- Rank-0 MemAvailable was 1,051 MB idle after boot. The minimum under the mix runs was **451 MB** (from `memlog.txt`
  on the head).
- This is below the plan's 0.8 GB gate. The earlier image was never measured under this same concurrent long-context
  load, so how much of the drop is caused by the extra graphs is not known.
- Step 2 (drafter diet) frees about 0.7 GB/rank net. Re-check this gate after it.

## C1 policy check (`c1-ratio/`, `c1-lambda/`; mean decode tok/s, same 40 prompts as verify cap v3)

| set | verify cap v3 (09-30) | step 1, ratio | step 1, lambda |
|---|---:|---:|---:|
| prose (15) | 20.78 | 20.58 | 20.75 |
| code (15) | 34.77 | 34.33 | 34.13 |
| prose+think (5) | 20.02 | 19.62 | 19.50 |
| code+think (5) | 32.56 | 31.87 | 31.68 |

C1 is unchanged within run-to-run noise. Per-prompt spread is about ±15%, so a 15-prompt mean moves about ±3%
between boots. The two policies are equal; **ratio stays the default** because it is the validated v3 rule.

## Prefill cadence (`cadence.json`, `cadence_probe.py`)

Setup: two greedy count-to-600 decoders run while a fresh, uncached ~67K-token prompt is prefilled. The cadence was
switched through the control file.

| cadence N | long prompt TTFT | decoders during the prefill (each) | decoders before |
|---:|---:|---:|---:|
| 1 (today) | 117.6 s | 2.4 tok/s | 41–43 |
| 2 | 133.2 s (+13%) | 4.1 tok/s (1.7×) | 44 |
| 4 | 140.0 s (+19%) | 7.6 tok/s (3.2×) | 42–43 |

It works as designed. The default stays **N=1 (off)** until the operator chooses, because it trades the long
prompt's TTFT for bystander decode speed.

## Decision

**Keep:**
- the batch verify cap: default `VERIFY_CAP_BATCH_MAX=4`, ratio policy;
- the runtime control;
- the prefix-hit fix: default `PREFIX_HIT_FIX=1`;
- the argmax clamp;
- the max-len guard.

The prefill cadence ships off. The memory gate is resolved by step 2.
