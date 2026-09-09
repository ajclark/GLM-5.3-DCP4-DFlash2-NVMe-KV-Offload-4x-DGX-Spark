# Adaptive verification experiment

Experimental phase complete, 2026-09-09. Production policy remains **off**. The experiment
keeps GLM-5.3, TP4/DCP2, the 180224 context limit, 12 sequences, and the 6 GB/rank
KV allocation. DFlash still proposes seven tokens with its trained block of
eight. Only target verification is shortened, to K=1,3,5,7.

The exact original containers, images, configurations, mounts and source files
have been restored on all four nodes and generation verified. All experiment
controllers and power samplers are stopped. The
[completion decision](completion-decision.json) retains the unmeasured wall-energy
gate and unresolved strict tracing-overhead bound; broad promotion is withheld.

## Completed R6 held-out Spark evaluation

The frozen version 4 controller completed 30 synthetic prompts × 3 balanced
paired repeats, 256 output tokens per request, thinking off:180 requests.
These are actual four-Spark C1 measurements, not VM speed estimates.

| Workload | Fixed7 aggregate tok/s | Adaptive aggregate tok/s | Paired geometric gain (95% prompt interval) |
|---|---:|---:|---:|
| Coding |25.03|25.41|+0.99% (-2.11% to+4.44%)|
| Prose |15.36|17.80|+15.80% (+13.24% to+18.73%)|

Rates divide total decoded tokens by total decode time. Paired gains give
each prompt equal weight. The original [gate report](heldout-adaptive-r6/gate-report.json)
passes prose, TTFT and p95 emission-gap gates. Worst matched-case median
TTFT ratio is 1.0127 and p95-gap ratio 1.0027. Coding narrowly misses the
predeclared lower 95% ratio ≥0.98; no policy is promoted.

A single fixed follow-up doubles coding repetitions from 3 to 6, retaining
the same 15 coding prompts, policy, calibration and sampling. Its declaration
is [coding-extension-lock.json](adaptive-v4-20260909-r6/coding-extension-lock.json).
The original result remains unchanged. The combined follow-up interval is
exploratory after viewing the initial uncertainty, not independent confirmation.
The completed [extension report](heldout-coding-extension-r6/extension-report.json)
contains 90 coding pairs: aggregate25.07→25.47tok/s, paired+1.14%,
95% interval-1.30–3.64%, inside the2% regression margin. No source or
calibration changed between the initial and extension runs.

The [device energy report](heldout-adaptive-r6/device-energy-report.json)
has full four-node coverage for all 180 requests. NVIDIA-device energy per
decoded token falls 17.37% for prose (95% interval 15.37–19.60%) and 2.50%
for coding (interval -0.93–5.96% savings). Whole-request energy per output
token, including prefill, falls 17.08% and 2.41% respectively. These sensor
measurements do not include a verified whole-cluster wall-power source.

All four ranks dispatch real FULL M2/4/6/8 target graphs while the drafter
keeps capacity 7. Capture logs report 1.62–1.63 GiB. Held-out minimum available
memory was 3411 MiB on rank 0 and 4788/5118/5152 MiB on the other ranks, with
zero new swap-out, OOM events, or pressure trips. Runtime versions, scheduled
caps and telemetry integrity pass the locked audit.

The masked first-repeat prose review preferred adaptive 3 times, fixed 2,
and tied 10. It reviews 256-token openings only. R6 complete-function and
transition/fallback/cancellation checks pass. The repeated C2 counting control
measures about81 combined tok/s and an adaptive/fixed7 ratio of1.00018
(+0.02%), with all outputs correct. Reasoning-mode controls, final adaptive
long-context checks, instrumentation comparison and restart checks are complete.
All 370 local tests pass. The expanded coding follow-up measures 2.85% lower device decode
J/token (exploratory 95% savings 0.40–5.39%); see its
[energy report](heldout-coding-extension-r6/combined-device-energy-report.json).

## R6 repository-context screen

One 256-token request per policy and task; these are single-repeat screens.
The [raw report](adaptive-v4-20260909-r6/repository-context-report.json)
retains cap counts, device energy and timing.

| Context | Coding fixed7 → adaptive tok/s | Gain | Prose fixed7 → adaptive tok/s | Gain |
|---|---:|---:|---:|---:|
| 4k |19.91 → 19.87|-0.2%|18.77 → 16.98|-9.5%|
| 32k |17.75 → 19.17|+8.0%|18.14 → 16.71|-7.9%|
| 100k |6.66 → 8.96|+34.6%|6.52 → 8.94|+37.2%|
| 170k |6.25 → 8.86|+41.7%|6.25 → 8.89|+42.1%|

The initial 170k coding request had a 402-second cold prefill. Subsequent
requests reused prefixes, so TTFT and whole-request energy do not establish
policy gains here. Device decode J/token fell33–39% at100k/170k but rose
4–5% for the slower4k/32k prose cases. No wall-energy claim is made.

Three additional paired repeats are declared for each 4k/32k task, using
unchanged source and calibration in R7. The completed
[32k follow-up](repo32k-followup-r7/followup-report.json) measures +7.2% coding
and +24.1% prose. The completed
[4k follow-up](repo4k-followup-r7/followup-report.json) measures +4.9% coding
and +15.3% prose. The initial prose regressions did not repeat. These remain
exploratory, with one prompt per category.
The first4k/32k prose outputs diverged during full-seven warm-up before the
first short verification; output variation is a possible contributor, not
a reason to discard the measured regressions. Broad promotion remains closed.

The R6→R7 restart test passed after a fresh 102360-token prompt was stored
once. R7 recovered 102016 prompt tokens from disk with zero local hits and
11.06 GB of load activity, then reproduced all127 continuation token IDs.
Generated KV was not stored: the deployed prompt-only offload policy remained
unchanged. See the [restart result](durable-reload-r7/summary.json).
Random 32k/100k marker-and-count checks passed under fixed7/fixed3/adaptive.
At32k all66token IDs match. At100k adaptive uses one newline after the marker
where fixed uses two; the marker and complete count are correct under all three.

Across all R6 serving tests, available memory reached a minimum of 1641 MiB
on rank0 and 3098/3280/3322 MiB on the other ranks. The head node swapped
out 2.60 MiB across the longer tests; full-memory PSI avg10 remained zero,
with no OOMs or guard trips. The
[final runtime audit](adaptive-v4-20260909-r6/final-runtime-audit.json)
checks 30,154 verification events and reports no integrity errors.

The final [tracing comparison](adaptive-v4-20260909-r7/instrumentation-overhead-report.json)
used nine identical-output pairs with matching graphics clocks. Tracing increased
decode time by 0.56% centrally (95% prompt interval -0.11% to +1.81%). The
upper bound exceeds the initial 1% budget; separate-boot effects and only three
prompts limit the conclusion. Tracing remains off by default.

R7 serving headroom reached 1793 MiB on the head and at least3168 MiB on other
ranks, with8 KiB new swap-out on the head, fullPSIavg10=0, and no OOM or guard
trip. Its source/graph/patch checks are in the
[final R7 audit](adaptive-v4-20260909-r7/final-runtime-audit.json).

## Measured fixed-cap results

R3 uses actual FULL target graphs at M=2,4,6,8. The first implementation used
PIECEWISE for the short caps; its R2 measurements must be kept separate.

| Short screen, two repeats | K=7 tok/s | K=3 tok/s | Change |
|---|---:|---:|---:|
| Prose | 16.65 | 19.52 | +17.2% |
| Code | 26.27 | 25.90 | -1.4% |
| Async repository task | 19.91 | 21.15 | +6.2% |

The separate 12-prompt development screen (one repeat) has paired geometric
mean gains of 16.4% for six prose prompts and 4.4% for six coding prompts.
Individual coding ratios range from 0.880 to 1.176. These are development
results, not promotion evidence for the adaptive controller.

NVIDIA-reported device energy per decoded token fell by about 19.1% for the
development prose prompts and 9.5% for coding at K=3. Integration uses the
first and last token timestamps and requires all four nodes with no sample
gap above five seconds. Sampling is every two seconds. **This is not wall
power or whole-cluster energy.** No idle power reduction is claimed.

Full graph capture takes 1.62–1.63 GiB/rank, about 0.15–0.21 GiB more than R2's
partially graphed prototype. Production rank-0 logs report 1.42–1.98 GiB across three boots; allocator
accounting varies, so capture log deltas do not establish an exact incremental
allocation. The runtime headroom checks are retained alongside those logs.

## Correctness and pressure checks

- 370 local tests pass, including actual Triton CPU interpreter kernels,
  async scheduling transitions, graph dispatch, censored feedback, and rollback.
- K=1,3,5,7 produce identical 60-token counting outputs in R3.
- The live C1→C2→C1 trace changes K=1→7→1. Both counting requests complete
  correctly. Nonzero temperature and repetition penalties verify all seven.
- Closing a stream cancels its request; the next K=1 request completes correctly.
- No OOM, new swap-out, or pressure trip in the R3 short workload checks.
  Post-development available memory was about 3.9/5.3/5.3/5.4 GiB across ranks.
- Cold/warm checks passed at 32823 and 100055 prompt tokens for K=7 and K=3.
  Minimum available memory was 2674 MiB on rank 0, with no full PSI stalls or
  new swap-out on any node. Four generated Python functions passed executable
  checks under both fixed settings. Final adaptive checks are recorded above.

The original greedy baseline is not bitwise repeatable on ordinary prose/code.
Output hashes therefore cannot establish sampler correctness alone. Fixed
logit/kernel fixtures and high-margin counting checks supplement live outputs.

## Long-context observation

At 100055 prompt tokens the drafter accepted zero proposals during the
counting test at all four caps. All answers were correct. K=1 reached
9.18 tok/s versus 7.04 at K=7 (+30%). This is a small diagnostic workload;
the unchanged-runtime control reproduces the behavior: 455 draft tokens,
zero accepted, and 7.05 tok/s. It predates these patches. The checkpoint advertises 1048576
positions and a 2048-token sliding window, so its stated context limit
alone does not explain the result.

A separate R4 experiment uses actual repository excerpts with coding and
prose tasks, 256 output tokens per request, and all four fixed caps:

| Prompt context | Coding K=7 / best cap tok/s | Prose K=7 / best cap tok/s |
|---|---:|---:|
| 4k | 16.93 / 19.26 (K=3) | 17.26 / 19.94 (K=5) |
| 32k | 20.42 / 20.42 (K=7) | 16.12 / 19.92 (K=3) |
| 100k | 6.44 / 9.32 (K=1) | 6.53 / 9.21 (K=1) |
| 170k | 6.49 / 9.36 (K=1) | 6.31 / 9.37 (K=1) |

The 100k gains are 44.7% for coding and 40.9% for prose. Fixed-seven
calibration at 100k accepted the first draft token in only 4 of 486 eligible
cycles, and never the second. These are one-repeat development results.
The 4k and 32k prefixes were already cached by the earlier 100k request;
their TTFT values must not be compared as cold-prefill measurements.

Cost tables are fitted separately within each context band. Version 3 can
interpolate measured anchors within an explicit operating range; it does
not reuse short-context costs unchanged or claim all intermediate contexts
were measured. The five-point calibration is frozen in
`adaptive-v2-20260908-r4/costs-v3-curve.json`, with a snapshot of its source
trace and hashes linking each fitted point. At 170k, minimum headroom was
2166 MiB on rank 0 and at least 3689 MiB elsewhere, with no new swap-out,
full PSI stalls, or OOM events.

## Controller and reproduction

Readable reports and frozen calibration files are committed individually. Request
records, power/memory samples, traces and historical run scripts are bundled in
`measurements.tar.gz` (about 21 MB compressed). From the repository root, restore
their original relative paths for analysis with:

```sh
tar -xzf results/adaptive-spec/measurements.tar.gz -C results/adaptive-spec
```

[ARTIFACT-PROVENANCE.json](ARTIFACT-PROVENANCE.json) records the archive hash and
each record's original and published hashes. Publication follows this repository's
existing identifier scrub: personal homes become `~`, SSH users become `user`,
and private addresses become `10.99.A.B`. Embedded historical hashes still refer
to the original bytes; use the provenance mapping for scrubbed records. An exact
original capture backup is retained locally. Runtime source, corpora, the frozen
cost curve and the R6 evaluation lock are unchanged. Archived per-run scripts
contain historical path placeholders; supported reusable drivers are in `bench/`.
The R7 cache trace duplicates the R6 trace through cache reuse and is not a new
trace-off measurement.

Policy version 2 estimates conditional acceptance at each verified position,
with decayed risk sets, eight completed observations before adapting, 3%
hysteresis, and a full-length probe every 16 eligible scheduled steps.
Unknown tails use an optimistic bound. Unsupported requests and contexts
outside the cost table use K=7. R4 tested this controller: paired development
gains were 3.49% for coding (95% interval 0.31–6.23%) and 3.76% for prose
(0.51–6.86%). The prose result fails the predeclared 8% target. Optimistic
tail estimates caused excessive full-length rechecking.

Version 3 was tested on R5. It adds
a weak conditional prior fitted only from fixed-seven calibration, with
two observations' worth of weight. Runtime feedback still determines each
request's estimates. It also supports context cost curves and lets the
telemetry writer block indefinitely when idle. The 30 held-out prompts were kept separate during controller development. Using the supported flat calibration fields, V3 gained
12.21% on development prose (95% interval 7.29–17.56%). Coding gained 0.54%
(interval -5.46–5.14%); one coding prompt regressed 13.1%. Coding
noninferiority is therefore unresolved.

The first nested-table request was rejected by the API before generating
output. Version 4 fixes only this encoding: transmit the context table as
a JSON string, then parse it once per request, with a 16 KiB bound. The
same controller, calibration, and held-out corpus are retained. The exact
deployed API field annotation and Pydantic version are recorded and tested
locally. R6 passed the live API and fixed-cap counting checks. Its frozen
30-prompt, three-repeat held-out evaluation is complete; see the results
above. The declared coding follow-up keeps the policy and calibration unchanged.

R3 is valid only for fixed-cap measurements; its older
adaptive estimator had a censoring bias that was corrected before R4.

Run local checks with:

```sh
PYTHONPATH=tests .venv/bin/python -m pytest tests/ -q
```

The guarded experiment runner supports `prepare`, `run --hold`, `collect`,
`restore`, and `verify-restore`. Each label must be new. It preserves the
original image, container ID, mounts, and KV directory. A `finish` file in
the local experiment result directory ends the hold and restores the originals.
There is also a bounded hold deadline; node watchdogs stop only experiment
containers if memory pressure rises or controller heartbeats disappear.

`bench/adaptive_spec.py` compares interleaved variants and refuses an occupied
endpoint. `--integration` runs bounded transition/cancellation probes;
`--context-smoke` checks roughly 32k and 100k prompts. `bench/fit_spec_costs.py`
requires actual FULL graph dispatch and enough measured fixed-cap cycles.
`bench/analyze_adaptive_spec.py` separates policies and bootstraps by prompt.
`bench/evaluate_spec_gate.py` checks complete locked pairs, runtime version,
trace integrity, throughput, and matched-case latency. It keeps wall energy
and deployment approval separate from the performance decision.

`bench/copy_opportunity.py --lookback 180224 --cap 7` scores committed-history
proposals with an incremental exact index on the sandbox VM. On fixed-seven
coding trajectories it found 19 matching future tokens across 435 short
development boundaries, zero across 77 repository boundaries at 32k, and
15 across 246 at 100k. Prose at 100k had 48 matching tokens across 248
boundaries. This limited screen does not establish a broad coding benefit
or predict live throughput. Copy retrieval has not been added to the GPU path.

The gate decisions and execution state are in
[STATUS.md](STATUS.md); the design and predeclared promotion thresholds are in
[the plan](../../docs/ADAPTIVE-SPECULATION-PLAN.md).
