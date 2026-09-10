# Bounded-lossy verification for prose: implementation plan (lever A)

**2026-09-09. Plan only; nothing built, nothing run on the Sparks.** Lever A of
[SPEED-ARCHITECTURE-OPTIONS.md](SPEED-ARCHITECTURE-OPTIONS.md): relax greedy
verification where the target's decision margin is small, per request, opt-in,
prose only. Every path below is the file that actually runs on the Sparks
(`~/lmcache-mg/spark-src/vllm` is the byte-exact dist-packages copy; `overlay/`
is what the launcher bind-mounts over it).

## 0. What the source says (read before designing)

- **Acceptance is decided in one Triton kernel, not in the DFlash2 code.**
  `v1/worker/gpu/spec_decode/rejection_sampler_utils.py`: `_compute_block_stats_kernel`
  (:47-153) reads every target logit row once in 8192-wide blocks; for greedy rows
  (`temp == 0.0`, :96-114) it stores only the per-block max and argmax; for sampled
  rows (:115-153) it stores per-block max and sum-exp (and the draft's). `_rejection_kernel`
  (:156-303) then walks positions 0..K-1 per request: greedy accepts iff
  `target_argmax == draft_sampled` (:250) and stores the draft or the argmax (:251-254);
  sampled accepts iff `log p(d) > log u + log q(d)` (:298). `_resample_kernel` (:306-431)
  and `_insert_resampled_kernel` (:434-490) skip greedy non-bonus rows entirely
  (:353-356, :468-471). The host wrapper `rejection_sample` (:493-670) allocates
  scratch each call (:535-548, :577-582, :622-629) and launches eagerly: **the
  rejection path is not inside a CUDA graph**, so a per-request scalar costs nothing
  in capture terms.
- **All quantities a margin rule needs are one gather away.** The draft's own logit
  is a single load (already done for sampled rows at :256-258); the global argmax and
  max come from the block stats; a runner-up (second max) and a log-sum-exp need one
  extra reduction each per block in a kernel that already has the logits in registers.
  No vocabulary-wide extra pass.
- **Every TP rank runs the identical sampler on identical full logits.**
  `model_executor/layers/logits_processor.py:114-122` all-gathers logits on CUDA;
  `overlay/.../model_runner.py:1362-1410` (`sample_tokens`) samples on the last PP rank,
  PP=1 here, so all four ranks execute `_rejection_kernel` on the same tensor with
  the same per-request parameters. Rank agreement is the existing invariant (it is
  what keeps `num_sampled` consistent today) and the new rule keeps it as long as it
  uses fixed-order reductions and no atomics.
- **Per-request scalars already flow `SamplingParams -> UVA buffers -> kernel`.**
  `v1/worker/gpu/sample/states.py`: `SamplingStates.__init__` (:17-40) owns
  `temperature/top_k/top_p/min_p/seeds` as `UvaBackedTensor(max_num_reqs)`,
  `add_request` (:42-63) fills them from `SamplingParams`, `apply_staged_writes`
  (:65-70) copies; the runner calls `sampler.add_request` at `model_runner.py:831`.
  `SamplingParams.extra_args` (`sampling_params.py:324`) is the `vllm_xargs` dict and
  is present on the worker copy. `RejectionSampler.__call__` passes
  `sampling_states.temperature.gpu`/`seeds.gpu` (rejection_sampler.py:127-128).
  This is the exact pattern to extend.
- **The wire normalises JSON booleans to ints** (fixed in `f71eac0`; test
  `tests/test_spec_xargs_boundary.py:22-35`). New fields must be numeric with range
  checks, never `is True`.
- **Draft q at T>0 exists** (DFlash2 caches a sparse fp32/-inf selector distribution:
  `dflash2/speculator.py:139-165, :193`), so a sampled-mode rule is implementable, but
  see §1b for why it is not worth it.
- **The trace/audit path exists.** `overlay/vllm/v1/spec_decode/adaptive.py`
  `VerificationPolicy.complete()` (:366-430) already joins per-step runner output to a
  JSONL verify row per request; `GLM_SPEC_POLICY=shadow` logs without changing caps.
  Codex's confidence packet shows how to carry a small per-request tensor from the
  worker to that row: `async_utils.py:21-78` (copied with the async output) and
  `outputs.py:283-285` (`ModelRunnerOutput.spec_confidence`).

## 1. The rule

Notation per verified position: target logits `z` (fp32, after
`apply_sampling_params`), draft token `d`, argmax `a`, `z1 = z[a]`,
`z2 = max_{v != a} z[v]`, `L = logsumexp(z)`.

### 1a. Greedy requests (pi/opencode at T=0; the daily case)

```
lossless:  accept iff d == a
relaxed:   accept iff d == a
                 or ( z[d] >= z2                      # d is the runner-up (rank 2)
                      and z1 - z[d] <= M              # margin in nats
                      and z[d] - L  >= log(p_min)     # optional absolute floor
                      and d not in STOP and a not in STOP )
emit d on accept; on reject emit a and stop (unchanged); bonus token unchanged.
```

This is MARS (arXiv 2601.15498, Algorithm 1: "if drafted token equals top-2 and
`r = z_(2)/z_(1) > theta`, accept") written with a scale-free margin instead of their
raw-logit ratio; at their default `theta = 0.9` and typical `z1 ~ 20-30` the ratio
test is a margin of 2-3 nats, i.e. a probability ratio near 0.05-0.1. MARS is
training-free, applies to greedy and sampling, and reports at K=7 with EAGLE-3:
accepted length 5.66 -> 6.39 (LLaMA-3.1-70B, +13%), 4.14 -> 5.14 (Qwen3-235B, +24%),
5.64 -> 7.20 (Vicuna-13B, +28%), with ROUGE-L within 0.0008, BLEU 29.67 vs 29.71,
MT-Bench within +-0.1. Our prose sits at 2.7 of 8 with more low-margin positions
than their benchmarks, so the upside is plausibly larger; that is the thing to measure.

Why the extra conditions: `rank 2` bounds the damage to the runner-up (MARS's rule);
`p_min` is a guard against relaxing on flat distributions where even the runner-up
is improbable (default off; swept); `STOP` (the three EOS ids 154820/154827/154829
from `config.json`, plus the `</think>` id to be read from the tokenizer on rank 0)
prevents a relaxed accept from ending or un-ending a generation or a thinking block.
Correctness of the continuation is unchanged: the target's logits at position i+1
were always computed conditioned on `d`, and the KV cache already holds `d`; the
only thing that changes is the emitted token at i.

### 1b. Sampled requests (T>0): implement the flag as ignored in v1

The July-2026 survey (arXiv 2607.26627, §collaborative verification) gives the
lenience rule `h(x) = min(1, p(x) / (l q(x)))` with output distribution
`{q for p < q <= p/l; p/l for q >= p/l; l-interpolated below}` and residual
`max(p - l q, 0)`. Its Table 1 (MBPP+) is decisive: the quality-preserving
variant ("ceiling the overshoot region", `{lambda p + (1-lambda) q if q <= p/lambda;
p/lambda otherwise}`) gives block efficiency 5.54-5.59 against ~5.5 lossless with
Pass@1 75.1-75.9 unchanged, while every variant that actually raises BE (5.95-9.15)
collapses Pass@1 to 50-66. The safe sampled rule buys nothing; the fast one is not
bounded. Since pi and opencode run T=0, v1 ignores `spec_lossy_*` when
`temperature != 0` and logs `ineligible=temperature`. If a sampled rule is ever
wanted, it is a two-line change (`+ log(l)` at :298, `exp(log q + log l - log p)`
at :388) plus a residual test, and it needs its own quality study.

## 2. Kernel and plumbing (overlay; ~170 lines over 6 files)

| file (overlay path = fork path) | change | ~lines |
|---|---|---:|
| `v1/worker/gpu/spec_decode/rejection_sampler_utils.py` | block stats: for greedy rows with `margin >= 0` also store per-block second max and sum-exp (new `target_local_second` buffer; sum-exp into the existing `target_local_sumexp`). Rejection kernel: greedy branch computes `z2 = max(second-best block max, winner block's second)`, gathers `z[d]`, computes `L` only when `min_logp > -inf`, applies §1a, counts relaxed accepts into a new `relaxed_steps[num_reqs]` output. Stop ids as a small int64 tensor + `NUM_STOP_IDS: tl.constexpr`. Host wrapper allocates/passes them. Runtime branch on the loaded scalar, same style as `temp == 0.0`; no new constexpr variants. | 90 |
| `v1/worker/gpu/spec_decode/rejection_sampler.py` | pass `sampling_states.lossy_margin.gpu`, `lossy_min_logp.gpu`, `stop_ids`; put `num_relaxed` on `SamplerOutput` (optional field) | 15 |
| `v1/worker/gpu/sample/states.py` | two `UvaBackedTensor`s: `lossy_margin` (fp32, init -1 = off), `lossy_min_logp` (fp32, init -inf); `add_request` parses `extra_args` only when the module-level `GLM_SPEC_LOSSY == "1"` and `temperature == 0`; `apply_staged_writes` copies | 30 |
| `v1/worker/gpu/async_utils.py`, `v1/outputs.py` | copy `num_relaxed` with `num_sampled` in `AsyncOutput`; `ModelRunnerOutput.spec_relaxed: dict[str,int] | None` (confidence-packet pattern) | 25 |
| `v1/spec_decode/adaptive.py` | `select()` records `lossy` params from xargs on the chosen row; `complete()` adds `relaxed` from `runner_output.spec_relaxed` to the verify row | 15 |
| `launch-glm53big-dcp.sh`, `stage/`, `patches/` | three new mounts (`v2_rejection_sampler_utils.py`, `v2_rejection_sampler.py`, `v2_sample_states.py` in the `DCP_FILES` list at :124-131 and their `-v` lines), `-e GLM_SPEC_LOSSY` beside :290-296, three patches, `SHA256SUMS`, `patches/apply.sh` | 20 |

Cost: two extra block reductions for greedy-lossy rows in a kernel that already
holds the logits; 19 blocks x 8 rows; estimate <0.1 ms per cycle (to be read from
the trace's kernel timings). Memory: `3 x [max_num_reqs] fp32/int32` persistent,
`[num_logits, 19] fp32` scratch per step. No graph change, no new model buffers.
Prometheus counter (`spec_decode_num_relaxed_tokens_total`) is deferred: it needs
the scheduler/metrics files, which are not in the overlay set; the JSONL rows and
`/metrics` accepted-token deltas are enough to audit v1.

Rank agreement: identical inputs on every rank (§0); reductions are `tl.max`/`tl.sum`
over fixed block order; no atomics; the decision is per request in one program.
For the trial boot only, add `GLM_SPEC_LOSSY_CHECK=1`: every 64th step all-reduce
`max - min` of `num_sampled` across TP and raise on nonzero (one 48-byte collective
per 64 cycles; off in any promoted configuration).

## 3. Request-level opt-in

`vllm_xargs` fields (flat numerics; the wire schema is `dict[str, str|int|float|list]`):

| field | type | default | meaning |
|---|---|---|---|
| `spec_lossy_margin` | float in (0, 5] | absent = off | `M` in nats |
| `spec_lossy_rank` | int in {2} (v1) | 2 | runner-up only; other values -> off |
| `spec_lossy_min_p` | float in [0, 0.5) | 0 | `p_min`; 0 disables the floor |
| `spec_label` | str | existing | audit key already used by the trace |

Server side: honoured only when the boot has `GLM_SPEC_LOSSY=1` (default 0, so the
production launcher is unchanged even if a client sends the fields), the request is
greedy, not structured-output, no logit bias/penalties/bad words (reuse the checks of
`adaptive.py:eligible`, :118-133, minus the C1 condition: the kernel is per request,
so mixed batches are fine and no scheduler shape changes). Invalid values fail
closed to off and are logged. Every verify row carries the effective `M/p_min`
and `relaxed`, so any response can be audited for which rule fired and how often.

Client side: extend `extensions/pi-speculation-core.mjs:preparePayload` with a
`lossy: {margin, minP}` option and `pi-speculation.ts` with `/spec-lossy on|off`
plus `PI_SPEC_LOSSY_MARGIN` (default unset). The pi prose persona sets it; the
coding persona never does. opencode: its provider `options` extra-body path needs
checking before promising the same switch there.

## 4. Tests (CPU, Triton interpreter; `tests/test_spec_lossy_verify.py`, ~300 lines)

Follow `test_spec_k0_screen.py:101-103`: `harness.extract` the overlay's
`_compute_block_stats_kernel`, `_rejection_kernel`, `_resample_kernel`,
`_insert_resampled_kernel`, `rejection_sample` with the gumbel namespace, and the
pinned fork originals from `tests/fixtures/spec_runtime/rejection_sampler_utils.py`
(hash in its manifest) as the lossless oracle. Seeded random logits, V=154880
truncated to 2 blocks + a partial block so the padded-block path is exercised.

1. **Identity when off.** `margin = -1` for every request: overlay `rejection_sample`
   equals the fixture bit-for-bit (`sampled`, `num_sampled`) over greedy and sampled
   rows, K in {1,3,5,7} and the K0 mixed batch from the K0 tests.
2. **Fires exactly when expected.** Hand-built rows where `d` is the runner-up at
   margins 0.3 / 1.2 / 3.0 nats: `M=1.0` accepts only the first, `M=2.0` the first
   two; emitted ids are the draft ids; `num_sampled = accepted + 1`; the first
   rejected position emits the argmax.
3. **Guards hold.** rank-3 draft within margin -> rejected; runner-up within margin
   with `p(d) < p_min` -> rejected; argmax or draft in STOP -> rejected; `p_min = 0`
   never computes/uses `L` (assert via a NaN-poisoned sum-exp buffer).
4. **Bookkeeping unchanged.** seven relaxed accepts -> bonus is the argmax of row 8,
   `num_sampled = 8`; reject at i -> `sampled[i] = argmax`, `num_sampled = i + 1`;
   resample/insert kernels produce fixture-identical output for greedy rows.
5. **Mixed batch.** three requests in one launch (greedy-lossy, greedy-exact,
   sampled T=1): the exact and sampled rows equal the fixture, the lossy row equals
   case 2; `relaxed_steps` is zero for the two controls.
6. **Determinism / permutation.** three runs with shuffled `idx_mapping` and request
   order give per-request identical outputs; the extracted kernel source contains no
   `tl.atomic`; `z2` equals a torch `topk(2)` reference for every row.
7. **Wire parsing** (in `test_spec_xargs_boundary.py` style): `1`/`1.0`/`"1"` and
   out-of-range values, `temperature=1` ignores the fields, `GLM_SPEC_LOSSY` unset
   ignores them, adapter schema unchanged.

## 5. Quality gates

Corpora: `bench/spec-development.json` (6 `prose_*` of 12) for the screen,
`bench/spec-heldout.json` (15 `prose_*` of 30) for the locked evaluation, plus a new
20-prompt `bench/spec-prose-checks.json` with deterministic constraints (word-count
window, required named items, forbidden items, ordered sections) so prose has an
executable check analogous to `spec_code_check.py`.

| gate | measure | threshold (pre-declared) |
|---|---|---|
| G1 code unaffected | code persona sends no fields; trace shows `relaxed = 0` and `M` absent on every code request; `spec_code_check.py` four-function checks pass on both variants | zero relaxed rows; 4/4 |
| G2 blind openings | `bench/spec_blind_quality.py` (generalise its `fixed/adaptive` mode names) on the 15 held-out prose prompts, 256 tokens, human A/B/tie | lossy wins or ties >= 60% of pairs; no pair judged incoherent |
| G3 pairwise judge | new `bench/spec_prose_judge.py`: an external judge model, position-swapped, both repeats of the 15 held-out prose prompts at 1024 tokens (60 judgements) | lossy win rate >= 45%; at n=60 the 95% half-width is ~12 points, so this detects only gross loss — say so in the report |
| G4 deterministic prose checks | 20 constrained prompts x 2 repeats | pass rate not below lossless on the same prompts |
| G5 divergence profile | first-divergence position and top-1 agreement over the first 64 tokens vs lossless | reported, not gated (greedy already varies run to run, `REPEATABILITY-DIAGNOSTIC.md`) |
| G6 long generation | 4 prose prompts x 2048 tokens: 4-gram repetition rate, EOS behaviour, judge on the final 512 tokens | repetition rate within 1.5x of lossless; no truncated/unfinished ending attributable to a relaxed STOP (should be impossible by construction, verify) |

Sweep grid on the development prose subset, 2 repeats, 256 tokens, interleaved:
`M in {0.5, 1.0, 1.5, 2.5}` nats (ratios 0.61/0.37/0.22/0.08; 2.5 ~ MARS theta=0.9),
`p_min in {0, 0.1}`; `rank = 2` throughout. 8 variants x 6 prompts x 2 repeats
= 96 requests plus 12 lossless anchors, ~40 min at C1. Freeze one `M*` (and at most
one runner-up) before touching the held-out set; no tuning on held-out results.

## 6. Measurement protocol

`bench/adaptive_spec.py` gains variants `lossy-m<M>[-p<p_min>]` (the variant table
at :156-200 maps names to `vllm_xargs`; today `spec_policy`/`spec_verify_cap`);
lossless control is `fixed7` with the policy in `shadow` so both variants trace.
Paired AB/BA per prompt and repeat (existing counterbalancing), 256-token outputs,
thinking off for the screen and on for one held-out repeat (pi runs effort high).
Contexts: short, `--repo-context 32000` and `100000` prose tasks (cold then warm),
on the repaired-cache runtime. Report per prompt and pooled: accepted/cycle,
decode tok/s, TTFT, p95 inter-token gap, relaxed accepts per cycle and per
position, and the per-position acceptance curve of the *next* draft after a
relaxed accept (the selector-interaction check, §8). `bench/spec_request_report.py
--trace` (activation proof, :128, :47-49) must see `M` echoed and `relaxed > 0`
for every lossy request and `relaxed = 0` for every control. Device power via
`bench/spec_power.py` as before; J/token is reported, not gated.

Promotion arithmetic to pre-declare: prose 18.1 tok/s at DCP2 fixed K7; the memo's
claim is 27-30 at 4-4.5 accepted. The gate is the paired lower bound, not the mean.

## 7. Rollout

Guarded experiment only: `bench/spec_experiment.py prepare <label> --lossy` (new
flag that exports `GLM_SPEC_LOSSY=1`, `GLM_SPEC_POLICY=shadow`, `GLM_SPEC_TRACE`, and
for the first hold `GLM_SPEC_LOSSY_CHECK=1`), then `run <label> --hold --hold-minutes
120`, inference only after `HOLD`, `touch results/<label>/finish` to restore, then
`verify-restore`. The flow stages the overlay into an isolated per-node directory,
verifies source hashes on all four ranks, keeps the originals as
`vllm_glm53big_backup_<label>`, and restores them on any failure (spec_experiment.py
:104-136, :257-284, :308-353). Admission is the existing `MemoryGuard`
(`bench/spec_memory.py`: 512 MiB hard floor :28/:50/:66, 768 MiB preflight :112,
swap-out and full-PSI rules :42-43/:60-61/:128) plus the remote watchdogs; this
change adds no model memory, so the admission question is only "is the cluster
quiet". Never run the launcher by hand; never leave the stack half-launched.
Rollback is the exact-container restore; the production launcher never sees the
new files until promotion adds the mounts and `GLM_SPEC_LOSSY=1` deliberately.

## 8. Risks and the check that resolves each

| risk | check |
|---|---|
| Greedy baseline is not run-to-run deterministic, so byte identity cannot gate anything | gate on rule-level proof: test 1 (kernel identity when off) plus trace `relaxed = 0` on controls; quality gates are paired judgements, not hashes |
| Selector interaction: a relaxed prefix conditions the next DFlash2 draft on non-argmax tokens and may lower its acceptance, eating the gain | per-position acceptance of the draft following a relaxed cycle vs following an exact cycle (§6); if the next cycle's first-position acceptance drops by more than the gain, lower `M` |
| Error compounding over long generations (the survey's "difficulty exposes losses") | G6 and G3 at 1024-2048 tokens, not only 256-token openings |
| Runner-up is a format token (quote, bracket, newline) in prose with embedded code or JSON | G4's constrained prompts include a list-and-table prompt; if it fails, add a `rank-2 and same character class` restriction or keep the persona split strict |
| STOP/think boundaries | constructed by exclusion (§1a); test 3 plus G6 EOS accounting |
| Rank disagreement (hang) | identical inputs by construction; `GLM_SPEC_LOSSY_CHECK=1` in the trial boot; watchdog heartbeat already aborts a hung engine |
| Interaction with the adaptive cap policy | trial runs fixed K7 in shadow mode; a combined adaptive+lossy run is a later, separate comparison |
| Thinking traffic: relaxation inside reasoning changes the reasoning path | report relaxed counts inside vs outside `<think>` (the trace has positions); G4/G3 with thinking on for one repeat |
| The `</think>` id and any harness stop strings are not known to the kernel | read the tokenizer on rank 0 (read-only) for the STOP set; harness `stop` strings are handled by the detokeniser after the fact and are unaffected |
| Logprobs consumers see a non-argmax sampled token with a lower logprob | expected and honest; note in the API doc |

## 9. Work breakdown and order

| step | size | days |
|---|---:|---:|
| 1. Kernel + `states.py` + rejection sampler plumbing, unit tests 1-7 | ~170 + ~300 lines | 2 |
| 2. Output threading (`async_utils`, `outputs`, `adaptive.py` rows), launcher mounts, stage, patches, `SHA256SUMS`, full local suite green | ~60 lines | 0.5 |
| 3. Bench: `adaptive_spec.py` variants, `spec_blind_quality.py` modes, `spec_prose_judge.py`, `spec-prose-checks.json` + checker, `spec_experiment.py --lossy` | ~350 lines | 1.5 |
| 4. pi extension option + `/spec-lossy`, `test_pi_speculation.mjs` cases | ~40 lines | 0.5 |
| 5. Spark hold 1: boot, smokes, rank-agreement check on, development sweep (§5), pick `M*`; restore | cluster ~2 h | 0.5 |
| 6. Spark hold 2: held-out 15 prose x 3 repeats AB/BA at 256 and one repeat at 1024 with thinking on, 32k/100k pairs, code control, G6 long runs; restore | cluster ~4 h | 1 |
| 7. Report in the codex style (`results/lossy-verify-<label>/README.md`, provenance, evidence index), decision | | 0.5 |

Total about six working days; steps 1-4 need no cluster time.

**Decision criterion for promoting `spec_lossy_margin = M*` to the pi prose
persona:** on the held-out prose set, paired decode tok/s gain with a 95% lower
bound >= +15% over fixed K7 (the same shape of gate codex used for adaptive), G1-G4
and G6 met, TTFT and p95 gap within noise, the rank-agreement check never fired, and
the code persona proven untouched (G1). If the gain lands but G3/G4 fail at `M*`,
try the next smaller `M` once; if that also fails, record the negative result and
stop. Promotion means: add the three mounts and `GLM_SPEC_LOSSY=1` to
`launch-glm53big-dcp.sh` via `rollout_dcp.sh`, and set the persona field in pi;
nothing else in production changes.

## Sources

MARS: https://arxiv.org/html/2601.15498 (Eq. 4/6, Algorithm 1, Tables 1/3/4/7).
Revisiting Lossy Verification: https://arxiv.org/html/2607.26627v2 (lenience rule,
Table 1 ablation, Lemma 1). Fork source as cited by path:line above; overlay files
under `overlay/vllm/`; codex's protocol in `ADAPTIVE-SPECULATION-PLAN.md` §9-10 and
`results/adaptive-spec/README.md`.
