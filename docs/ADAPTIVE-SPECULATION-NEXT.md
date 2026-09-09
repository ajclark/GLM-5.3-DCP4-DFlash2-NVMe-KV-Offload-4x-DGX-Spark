# Adaptive speculation follow-up: measured findings and decisions

The goal is better single-stream coding/prose throughput and lower active and
loaded-idle power on the four-Spark GLM-5.3 deployment. The user authorized
autonomous research, implementation and bounded experiments, using the sandbox
VM first and checking Spark memory pressure before and during inference.
Pi is invoked through herdr, with a separate 60-second state/pane watcher.

The [research survey](ADAPTIVE-SPECULATION-RESEARCH.md) and its pinned evidence
inventory led to the experiments below. Historical V4 held-out results remain
frozen under [results/adaptive-spec](../results/adaptive-spec/README.md). New
results do not overwrite that evaluation. In particular, its original
100k/170k acceptance collapse and apparent 35–42% adaptive gains are confounded
by the subsequently reproduced draft-cache table defect.

## Current decisions

| Avenue | What was implemented or measured | Decision |
|---|---|---|
| Long-context cache geometry | Per-group V2 block-table sizing and bounds tests; same-capacity boundary/100k/170k Spark validation | Repair is supported; retain a separate target-quality gate |
| Pi workload and phase hints | Opt-in Pi extension, server-owned priors limited to eight completed observations, actual Pi requests and 16 paired-control requests | Continue opt-in evaluation; no broad coding/prose promotion |
| Candidate confidence | Bounded asynchronous score transport, live identity checks, on/off controls and 1050 learnable records from 12 prompts | Reject the tested lag-two predictor for serving; current-proposal use requires a separate scheduling design |
| Active energy | Four-prompt interleaved frequency/governor screen; separate frozen-adaptive 1600 MHz bracket | 1600 MHz is a useful energy/throughput candidate; no permanent clock change |
| Loaded idle | Explicit GPU clock locks with independent rollback watchdogs and device power sampling | 600 MHz saves about 9.8 W across four devices; automatic clock mode was worse |
| True K0 | Actual sampler/input tests, park/arm/resume state-machine prototype and context-state memory analysis | Defer integration until measured target-only/resume costs justify it after the cache repair |
| Training, trees, copying, draft memory | Exact CPU tree selector; source, parameter-state, copy-opportunity and trained-block-boundary screens | Keep their explicit evidence gates; no additional model allocation or training beside the resident target |
| Target repeatability | Same-prompt functional/logprob diagnostics and source/kernel audit | Isolated atomic1/atomic0 numerical control in progress; quality promotion remains withheld |

The [K0 feasibility report](research/K0-FEASIBILITY.md) and
[architecture screens](research/ARCHITECTURE-SCREENS.md) contain the implementations,
CPU results and reasons for deferring larger serving changes. A negative result
or a concrete missing-evidence gate is an outcome; synthetic results are not
reported as Spark performance.

## Cache repair and new long-context measurements

Commit `8f684a4` fixes the V2 worker's replicated draft group. The target group
is CP2 with 1408 logical columns; the draft group is CP1 and needs 2816.
All four nodes log the corrected geometry. Additional table metadata is
132 KiB/rank; the model, TP4/DCP2, 180224 context limit, twelve sequences,
6 GB/rank KV pool and reported 198551-token capacity are unchanged.

Fixed-K7 probes at 89055 and 92056 prompt tokens emit identical 60-token
outputs, retain the marker and count correctly. Both accept 56 of 70 drafted
tokens across ten cycles, at 42.77 and 43.00 tok/s. These are diagnostic
outputs, not a broad benchmark. The
[long-context report](research/LONG-CONTEXT-DIAGNOSTIC.md) connects the CPU
reproduction to the actual Spark validation.

New fixed-cap calibration measures complete cycles at short, 100k and 170k
contexts. Two-repeat paired screens use one coding and one prose prompt per
context and 256 output tokens per request:

| Context | Coding fixed7 → adaptive tok/s | Paired gain | Prose fixed7 → adaptive tok/s | Paired gain |
|---|---:|---:|---:|---:|
| 100k | 16.76 → 17.53 | +4.59% | 16.92 → 18.62 | +10.05% |
| 170k | 16.90 → 17.56 | +4.13% | 16.26 → 17.94 | +10.16% |

Device decode J/token ratios are 0.892/0.852 for code/prose at 100k and
0.911/0.861 at 170k. One prompt per domain/context cannot support an independent
prompt-level interval. The first 100k baseline also has a longer TTFT than its
warm controls; whole-request energy is separate from decode energy.

Complete-function checks found an interval-semantics failure on a request
labelled adaptive that actually stayed at K7. Further controls reproduced a
failure under fixed K7. Shared-prefix target logprob differences change by up
to 2.375 at the semantic branch in the original six-request diagnostic. This
is not established as a tiny final-token tie or an adaptive cap-change bug.
The [repeatability report](research/REPEATABILITY-DIAGNOSTIC.md) preserves the
failures and narrows the first numerical control to dense Marlin atomics.

## Actual Pi integration and paired controls

Actual Pi/herdr requests completed four Python functions and a prose scene
against the repaired V5 server. All functions passed their checks. Their
traces show valid code/prose hints during warm-up and 125 valid confidence
packets. These establish integration, not a paired speedup.

The first explicit HTTP controls exposed an API-boundary bug: `vllm_xargs`
normalizes JSON booleans into integers, defeating an `is True` check. Those
comparisons are marked invalid and preserved. Commit `f71eac0` fixes the
switches, tests the captured OpenAPI boundary, and requires activation proof
in paired reports. A live two-request gate proved both on/off states before
the corrected comparisons. Trace snapshots allow the background writer's
one-second idle flush; an initial incomplete snapshot is retained separately.

The corrected hint screen contains two actual Pi prompt payloads × two repeats
× four variants, with confidence disabled. Relative to inference-only adaptive
verification, correct hints measure 0.9720 coding and 1.0844 prose tok/s ratios;
device J/token ratios are 1.0448 and 0.9637. Wrong hints measure 0.9241 and
0.9790 tok/s ratios. Both correct coding-hint requests remain K7 throughout,
so their slower observed rate does not establish a cap-change regression.
Only one of four correct-hint pairs has identical output IDs. All twelve
complete coding outputs across the independent confidence and hint controls
pass functional checks. These small screens do not justify global enablement.

An additional actual Pi session read and edited a scratch Python fixture using
its tools. The edit passed functional checks, and server traces applied the
code prior on the initial user turn while excluding both real tool-followup
requests. This is integration evidence, not a tool-workflow speedup.

The independent eight-request fixed-K7 confidence control proves 229 valid
on packets and zero off packets. Its two coding pairs have identical outputs
and a throughput ratio of 0.99935; prose outputs vary, preventing a precise
collection-overhead conclusion. A strict one-percent upper overhead bound is
not established. See [Pi hints](research/PI-HINTS.md) for protocols and reports.

## What the confidence data says about architecture

The twelve-prompt fixed-seven development trace yields 1050 valid learnable
records. Of these, 1026 have usable confidence by the decision deadline,
and every available feature is two proposal steps old. The first two records
per request abstain. Whole-prompt leave-one-out prefix Brier loss is:

| Predictor | Coding | Prose |
|---|---:|---:|
| Available acceptance history | 0.19396 | 0.11118 |
| History plus available lag-two scores | 0.19933 | 0.11144 |
| Same-proposal scores, intentionally noncausal diagnostic | 0.12354 | 0.07990 |

Lower is better. The tested lagged predictor adds no demonstrated value.
Current-proposal scores are informative, but their diagnostic use ignores the
existing scheduling deadline. Moving that decision requires measured changes
to graph dispatch, TP agreement and immutable asynchronous bookkeeping.
The [confidence report](research/CONFIDENCE-FEASIBILITY.md) records the full
causal checks and costs. Its same-boundary utility numbers are not rollout
or measured throughput; they reuse the frozen repaired atomic1 cost curve,
without assuming that the shadow collector has zero overhead.

## Active and idle power

The fixed-K7 clock/governor screen uses four development prompts, 256 tokens,
and same-prompt baselines before and after each profile:

| Profile | Tok/s ratio | Device decode J/token ratio |
|---|---:|---:|
| GPU 2200 MHz | 0.9893 | 1.1919 |
| GPU 1800 MHz | 0.9483 | 0.9080 |
| GPU 1600 MHz | 0.9537 | 0.8200 |
| CPU schedutil, GPU 2000 MHz | 0.9386 | 1.0804 |

A separate baseline/1600/baseline adaptive screen gives 0.9290 throughput and
0.8355 device J/token ratios, retaining the same frozen 2000 MHz cost curve
at both frequencies. It does not claim calibration at 1600 MHz.

Loaded-idle baseline power is approximately 32 W across four GPUs. Automatic
GPU clocks plus schedutil increased it to 47.08 W; GPUs rose above 2400 MHz.
Explicit 600 MHz measured 22.14 W, and 300 MHz measured 21.56 W. The extra
saving below 600 MHz is only about 0.5 W after adjacent-baseline correction.
A correct generation after restoration proves that resident model state
survived; it ran after a settled baseline, so immediate wake latency is unmeasured.

No permanent clock/governor policy was installed. GPU device sensors exclude
CPU/package and wall energy. Existing ConnectX-7 savings are a separate result;
network power settings were unchanged in these tests. See
[power experiments](research/POWER-EXPERIMENT.md) and [idle integration](IDLE-POWER.md).
An automatic idle policy needs authoritative shared-server quiescence,
restoration before work and a watchdog. One idle Pi session cannot establish
that the whole server is idle.

## Validation and remaining work

The full local suite at `f71eac0` passes **787 tests**, including API-boundary,
request ownership, delayed transport, cap parity and cache-layout checks.
R1's eighteen completed request/profile guards record no OOM or guard trip,
minimum head availability 1758 MiB, full-PSI average zero and at most 8 KiB
swap-out per phase. Loading and restoration have separate pressure records.
These observations are not universal admission thresholds.

The corrected hint/confidence phase is complete and its controller is restoring
the original containers. The atomic-enabled six-request numerical baseline
reproduced one interval-semantics failure; warm controls share 99968 local
prefix-cache hits and zero preemptions. The remaining declared work is the
separate atomic-disabled control, final restoration and evidence publication.
