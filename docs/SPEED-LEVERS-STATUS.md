# Autonomous speed-lever execution: status log

**Mandate (user, 2026-09-10):** work through every lever in
`SPEED-ARCHITECTURE-OPTIONS.md` §3 autonomously, in the order that best eke
out measured gains, no user input required; consult codex when useful.
Standing rules: `project-dcp-constraints` (never OOM/hang a node; Spark
changes only through the guarded flows; `rollout_dcp.sh` is classifier-blocked
for Claude, so production promotion is left as a `! ./rollout_dcp.sh <label>`
line for the user; experiment boots go through `bench/spec_experiment.py`).

## Order (rationale: measured gain per unit of risk and cluster time)

| # | lever | why here | status |
|---|---|---|---|
| 1 | A bounded-lossy verification (prose) | plan ready, no memory cost, opt-in, ~1 week, +15-30% prose | **speed gate passed: held-out +19.7% [17.0, 22.2]**; quality gates pending; promotion is a user rollout |
| 2 | H small cycle items (LSE fold, indexer-merge fusion, glue fusion, fp8 AR payload) | pure overlay + CPU tests, one boot, -6-10 ms for everything | LSE fold measured neutral (146.0 vs 144.9 ms count100; pure-torch pack/unpack ate the saving); off by default |
| 3 | B MTP K=2 prose lane | **measured (hold 3)**: prose 20.0 vs DFlash-K7 15.0 (+33%), repo 22.2 vs 19.4, code 24.8 vs 26.3 (-6%), count smoke 22.9 vs 52 (K=2 caps a cycle at 3 tokens) | prose-session boot option; not a default (structured/tool-call output loses) |
| 4 | D-lite (drafter fp8, indexer int8) | codex inventory: indexer W8A16 0.73 ms, drafter fp8 (QKV bf16) ~2.7 ms; ~2.3% total | tooling done (`research/DENSE-BYTES-PLAN.md`, `bench/repack_dense_int8.py`); cluster trial deferred behind H/B |
| 5 | G prefill profile + prefill chunk | profile in hold 2; then hold 4 = `--maxbatched 4096/8192` at DCP2 with cold 40k/100k prefills under the watchdog (the launcher note deferred this to "a measured profile run") | hold 4 scripted |
| 6 | E ring one-shot collectives: microbench gate + design | needs the stack down or a co-resident verbs probe; 2-3 weeks to build | queued |
| 7 | F trained draft | — | **declined by the user (2026-09-10 02:10 UTC)**; no corpus, no training, no spend |
| 8 | C/D-full requant (3-bit experts, 4-bit dense) | external GPU or maintenance windows; tooling + gates | queued |

## Log

- 2026-09-09 23:50 UTC: started. Cluster serving `dcp2-cachefix-prod`. Codex
  tasked with A's bench/quality tooling (plan §5-6); Claude on kernel/plumbing/
  tests/launcher lane (plan §2-4, §7).
- 2026-09-10 ~00:15 UTC: A implemented locally. Overlays:
  `v1/worker/gpu/spec_decode/rejection_sampler_utils.py` (runner-up + sum-exp
  per block for lossy greedy rows; MARS margin rule with rank-2, optional
  p_min floor, stop-id exclusion; `num_relaxed` output), `rejection_sampler.py`
  (stop ids from `GLM_SPEC_LOSSY_STOP_IDS`, `GLM_SPEC_LOSSY_CHECK` TP
  all-gather every 64 calls), `sample/states.py` (`spec_lossy_*` parsing,
  fail-closed, gated on `GLM_SPEC_LOSSY=1` and T=0), `async_utils.py` /
  `outputs.py` / `model_runner.py` (`spec_relaxed` per request),
  `adaptive.py` (rows echo `lossy_margin`/`lossy_min_p`/`lossy_enabled`,
  `relaxed`). Launcher: three mounts + three envs. Packager: `--lossy`,
  `--lossy-check`. pi: `/spec-lossy on [M [p]]|off`, `PI_SPEC_LOSSY_MARGIN`.
  Tests: `tests/test_spec_lossy_verify.py` (10; identity vs pinned fork kernel
  incl. sampled rows, exact-fire, guards, bookkeeping, mixed batch, Python
  reference over planted random rows with permuted request order, padding,
  wire parsing). Full suite 810 passed. Patches/stage/SHA256SUMS refreshed
  (39 entries). Not committed, not deployed.
- 2026-09-10 ~00:30 UTC: experiment harness now accepts vetted overlays that
  production mounts (`bench/spec_node.py` checks the mounted source hash
  against `vetted-overlays.json` = staged ∪ last committed `stage/glm-dcp`;
  test added). `prepare lossy-20260910-r1 --lossy --lossy-check` verified all
  four ranks (six production overlays recognised). `run ... --hold
  --hold-minutes 120` launched: boot, count smokes, 24 fixed-cap controls,
  then HOLD for the development sweep with codex's tooling. Restore is the
  harness's exact-container restore on `finish` or any trip.
- 2026-09-10 ~00:45 UTC: while the A boot loads, lever H item 1 drafted:
  `dcp_fold_lse_rs` in `overlay/.../mla_attention.py` (payload
  `[out*w | w | 0..]` bf16, w = exp(clamp(lse, 80)), one reduce-scatter,
  divide on receipt; -inf/NaN rows weight zero; all-empty rows give zeros),
  gated by `GLM_DCP_LSE_FOLD=1` (launcher `DCP_LSE_FOLD`, harness
  `--dcp-lse-fold`). Tests `tests/test_dcp_lse_fold.py` (3) against
  `ref_lse_merge` at DCP2/DCP4 incl. empty/NaN rows and the overflow clamp.
  Expected saving at DCP2: 78 x (~36 us + launch) ~ 3-4 ms/cycle; to be
  measured in a later boot (count100 acceptance/cycle, not hash identity).
  Also added a `--profiler` harness lane (PROFILER_DIR) for lever G.
- 2026-09-10 00:48 UTC: lossy-20260910-r1 reached HOLD (health 00:40:45,
  smokes K1/3/5/7 = 20.1/33.7/45.4/52.4 tok/s, 24 fixed controls match the
  earlier screens). Dev sweep started 00:49 (108 requests, results/lossy-dev).
  Codex handed lever D-lite (tensor inventory, indexer int8 / drafter fp8
  loader analysis, offline `bench/repack_dense_int8.py`).
- 2026-09-10 ~01:00 UTC: lever B lane built while the sweep runs: launcher
  `mtp` branch (mounts `$DCP_DIR/mtp_speculator.py`, `MTP_K`), node launch
  honours `lane.json` (SPEC_MODE/MTP_K/MAXLEN/KVBYTES/GLM_SPEC_POLICY),
  packager `--mtp K --kvbytes B --maxlen L` with validation (MTP needs the
  pool cut; MAXLEN <= pool/30200 tokens at DCP2). Planned first MTP boot:
  `--mtp 2 --kvbytes 3200000000 --maxlen 90112` (MTP layer ~2.8 GB/rank).
- 2026-09-10 01:24 UTC: A development sweep done (108 requests, 6 prose
  prompts x 2 repeats, 256 tokens; `results/lossy-dev/`). Paired geomean
  tok/s vs fixed7 (prompt-bootstrap 95%): m0.5 1.021 [0.996,1.045];
  m1.0 1.056 [1.035,1.078]; m1.0-p0.1 1.088; m1.5 1.088 [1.072,1.103];
  m1.5-p0.1 1.061; **m2.5 1.126 [1.069,1.183]**; m2.5-p0.1 1.112. Relaxed
  accepts/cycle 0.05 (m0.5) .. 0.21 (m2.5). Following-cycle first-position
  acceptance after a relaxed accept 0.743 vs 0.713 after exact (m2.5): no
  selector-interaction penalty. The p_min floor never helped. Frozen M* =
  `lossy-m2.5` (`results/lossy-dev/FROZEN-VARIANT`); held-out started 01:25.
  First-divergence positions are early (8-19 tokens) and first-64 agreement
  low, as expected once a relaxed token changes the trajectory; quality is
  therefore gated by G2-G4/G6, not by agreement.
- 2026-09-10 01:30 UTC: codex delivered D-lite (939 tests): real-header
  inventory shows the bf16 indexer is 21 layers (1.64 ms), lm_head bf16
  (1.98 ms), int8 dense 19.7 ms; indexer W8A16 saves 0.73 ms, drafter fp8
  (MLP/output/fc, QKV bf16) ~2.7 ms. Memo §D corrected. Cluster trial of
  D-lite deferred behind H (LSE fold) and B (MTP lane). Codex moved to the
  cheaper model on its rate-limit prompt (budget).
- 2026-09-10 01:35 UTC: `rollout_dcp.sh` now forwards `GLM_SPEC_LOSSY` and
  `DCP_LSE_FOLD` to the launcher (promotion = `GLM_SPEC_LOSSY=1
  ./rollout_dcp.sh <label>`; opt-in per request, so production behaviour is
  unchanged for requests that do not send the fields). Lever F's corpus prep
  (private pi transcripts) was refused by the permission classifier; not
  worked around — the user decides whether to build the corpus.
- 2026-09-10 01:54 UTC: **A held-out done** (`results/lossy-heldout/`,
  15 prose prompts x 3 repeats x fixed7/lossy-m2.5, AB/BA, repeat 2 with
  thinking): paired geomean tok/s ratio **1.197, prompt-bootstrap 95%
  [1.170, 1.222]**; every prompt positive (1.07-1.28); pooled decode 15.93 ->
  19.13 tok/s; accepted/cycle 1.28 -> 1.73; relaxed 0.217/cycle; TTFT 0.44 s
  and p95 emission gap 0.151 s unchanged; following-cycle first-position
  acceptance 0.723 (after relaxed) vs 0.728 (after exact). Activation proof:
  every lossy row echoes the margin with relaxed > 0; every control row
  relaxed = 0. The pre-declared +15% lower-bound speed gate is met. Quality
  gates G2-G4/G6 not yet run (need ~2 h of generation; plan: run against
  production after the opt-in switch is promoted, since it changes nothing for
  requests that do not send `spec_lossy_margin`).
- 2026-09-10 02:00 UTC: blind quality screen of the held-out 256-token
  pairs (Claude as judge, position-swapped, manifest unread): 54 ties, 2
  lossy wins, 4 lossless wins over 60 judgements (tie-adjusted 0.48). No
  gross loss at opening length; G3/G4/G6 still needed for promotion to the
  pi prose persona. Hold 1 released 01:54 (exact restore in progress); hold
  2 (LSE fold + prefill profile) boots automatically afterwards.
- 2026-09-10 02:10 UTC: user declined lever F (workload-trained draft). No
  corpus or training work will be done; the code multiplier stays at the
  shipped DFlash2 draft's acceptance.
- 2026-09-10 02:12 UTC: the user ran `./rollout_dcp.sh lossy-prod` (without
  `GLM_SPEC_LOSSY=1`) while hold 2 (`lsefold-20260910-r1`) was mid-boot. The
  rollout's preflight saw the experiment stack's /health, tore it down and
  launched the staged tree as production on all four nodes; the hold-2
  backups were displaced by the launcher's `docker rm -f`. Claude stopped the
  hold-2/3 chain, the experiment controller and the node watchdogs (which
  only ever kill labelled experiment containers) so nothing could act
  against the rollout. Hold 2 (LSE fold + prefill profile) and hold 3 (MTP
  K=2) must be rerun later. Lesson: one orchestrator at a time; a production
  rollout while a guarded experiment is up is a race between two restores.
- 2026-09-10 02:22 UTC: rollout `lossy-prod` healthy (520 s, OK generation,
  27 mounts, `GLM_SPEC_LOSSY=0`, `GLM_DCP_LSE_FOLD=0`, capture 1.50 GiB, head
  headroom 3.6 GB). Production behaviour unchanged; the lossy switch still
  needs a `GLM_SPEC_LOSSY=1` rollout. Post-boot checks running.
- 2026-09-10 02:30 UTC: post-boot checks on `lossy-prod` vs `dcp2-cachefix-prod`
  (`results/lossy-prod/`): count100 identical text, 54.1 tok/s @ 144.9 ms
  (was 54.0 @ 145.3); prose 18.1 vs 17.9; code 40.3 vs 39.1; TTFT equal; no
  new JIT warnings; head headroom 3.4-3.6 GB. The lossy overlays in exact
  mode are free. Switch still off (`GLM_SPEC_LOSSY=0`).
- 2026-09-10 02:40 UTC: user asked about restoring the 8192 prefill chunk
  (launcher note: 2048 validated under DCP4, raise only with a profile). At
  DCP2 the gathered-head transients are half of DCP4's. Added `MAXBATCHED`
  to the experiment lane overrides (`--maxbatched 2048|4096|8192`) and a
  hold-4 script (4096 then 8192; cold 40k/100k prefills via
  `bench/spec_boundary.py`, watchdog on). Runs after hold 3.
- 2026-09-10 02:47 UTC: hold 2 (`lsefold-20260910-r2`). LSE fold: count100
  53.7 tok/s @ 146.0 ms, text identical, vs production 54.1 @ 144.9: neutral,
  the ~6 torch ops per layer cost what the 78 all-gathers saved; needs a
  fused pack kernel to matter. **Prefill profile** (4158 uncached tokens,
  two 2048 chunks, 8.5 s = 490 tok/s; rank 1 GPU 8.3 s busy of 8.5 wall):
  MoE Marlin 1.53 s (18%), copies/indexing glue 1.40 s (17%, 8806 kernels),
  TP all-reduce 1.18 s (14%, 510 x 2.3 ms = ~11 GB/s on a 200G link),
  attention/indexer 1.11 s (13%), DCP all-gather 0.89 s (10%, ~13 GB/s),
  dense Marlin 0.83 s (10%), DCP reduce-scatter 0.67 s (8%). Communication
  is 33% of prefill and runs at about half the link rate (one QP per
  connection); DCP2's own gathers/scatters are 18%; the glue is 17%. The
  MoE runs at ~31 TFLOPS/rank (~25% of bf16 peak). Levers, in order: link
  utilisation (`NCCL_IB_QPS_PER_CONNECTION`, both HCA functions), prefill
  glue fusion, larger chunk (hold 4), then MoE prefill kernels.
- 2026-09-10 02:55 UTC: prefill glue attribution (rank 1 trace): CPU-side
  `aten::copy_` 2.5 s nested (`to`/`_to_copy` 1.5 s, `clone` 0.9 s,
  `contiguous` 0.8 s), `mul` 0.65 s, `topk` 0.5 s, `clamp_` 0.43 s (234),
  `aten::all` 0.32 s (234: one host sync per layer-chunk), `masked_fill`
  0.4 s (702), `where` 531, `full_like` 531. Concrete targets for a prefill
  glue-fusion overlay after holds 4/5. Made `NCCL_IB_QPS_PER_CONNECTION` a
  lane override (launcher default stays 1) and chained hold 5
  (`ibqps2-20260910-r1`: QPs=2, cold 40k/100k prefills + decode bench) after
  hold 4. Note: the NVMe tier's 16 worker threads show in the trace only as
  waits, not GPU time.
- 2026-09-10 03:00 UTC: prefill glue call sites (rank-1 trace, Python
  stacks): (1) `GroupCoordinator.reduce_scatter(out, dim=1)` copies the whole
  attention output to move the scatter dim first: 468 x 1.6 ms = 0.74 s per
  4k prefill (the stock path pays it too); fix = have the LSE-correction
  kernel write [H, B, D] so the scatter is dim 0 with no copy. (2) the DCP
  compaction/filter path in `flashmla_sparse._forward_fp8_kv_mixed_batch`
  / `sparse_utils.compact_dcp_candidate`: one `aten::all` host sync (0.32 s
  CPU), `clamp_` (0.43 s), `masked_fill` x2, `where`, `full_like`, `fill`
  per layer-chunk; fix = one Triton kernel, no sync. (3) my LSE fold adds
  ~0.5 s at prefill (fp32 mul + casts + zeros): another reason it stays off.
  Estimated prefill saving for (1)+(2): 1.2-1.5 s of 8.5 s (~15%) at DCP2;
  decode ~1-2 ms/cycle. Work item: "prefill glue overlay" after holds 4/5
  show what chunk size and QPs give.
- 2026-09-10 03:15 UTC: **hold 3 (`mtp2-20260910-r1`, MTP K=2, pool 3.2e9,
  window 90112, policy off) booted and held**: codex's tuple-contract
  overlay works under DCP2. Fixed screen (256 tokens, 8 repeats per prompt,
  caps inert): prose 20.0 tok/s (DFlash K7 15.0, K3 19.0), repo 22.2 (19.4 /
  21.2), code 24.8 (26.3 / 25.7); count smoke K7 22.9 vs DFlash 52.4. MTP
  wins prose by a third and loses structured output badly (3-token cycle
  ceiling ~31 tok/s). Verdict: a boot-per-workload prose lane. Queued: hold
  6 = MTP K=2 + lossy (they stack in the rejection kernel), hold 7 = DFlash
  lossy at m3.5/m5.0 for A's ceiling.
- 2026-09-10 03:20 UTC: hold 3 bench (`results/mtp2-20260910-r1/bench.json`,
  dcp_probe prompts, 2 reps): MTP K=2 cycle 104.5-107.5 ms; count100 27.7
  tok/s (2.99 of 3, text identical to DFlash), prose 21.7 (2.27 of 3) vs
  DFlash-K7 18.1, code 27.5 (2.91 of 3) vs 40.3. Prose +20% on this prompt,
  +33% on the dev corpus; structured output -30..-50%. Per-workload boot
  option only.
- 2026-09-10 03:45 UTC: hold 4 (`chunk4096-20260910-r1`, max-num-batched-
  tokens 4096 at DCP2): boot and cold prefills fine under the watchdog;
  40,055 tokens in 80.1 s (500 tok/s), 100,055 in 217.9 s (459 tok/s) vs
  ~473/460 tok/s at 2048. No gain: prefill cost scales with tokens, not
  chunks, at this size. The 8192 leg was cut (no upside, boot-3 memory
  risk). The launcher's 2048 stays.
- 2026-09-10 04:16 UTC: hold 5 (`ibqps2-20260910-r1`, NCCL_IB_QPS_PER_CONNECTION
  =2): 40k prefill 79.8 s (502 tok/s), 100k 215.1 s (465 tok/s) vs 473/460
  at QPs=1; decode unchanged (count100 54.3 @ 144.4 ms, identical text). No
  gain: the ~11 GB/s collective rate is the one-channel/one-CTA ring limit,
  and the ring cannot take more channels. Prefill knobs closed: chunk size
  (hold 4) and QPs (hold 5). Remaining prefill levers: glue overlay (~6-7%),
  MoE prefill kernel efficiency (25% MFU), or DCP1 for prefill-heavy use.
- 2026-09-10 04:45 UTC: hold 6 (`mtp2lossy-20260910-r1`, MTP K=2 + lossy):
  dev prose, 12 pairs per arm vs the MTP baseline (20.5 tok/s, 1.13 of 2
  accepted): lossy-m1.5 0.990, m2.5 1.004, m3.5 1.031 (accepted 1.11 / 1.14
  / 1.20). **No stacking**: the rule fires rarely on MTP drafts. Lossy is
  a DFlash-specific gain; MTP alone is the prose-lane alternative. (The
  trace-based report cannot run on the MTP lane because its policy is off;
  ratios computed from the result files.)
- 2026-09-10 05:22 UTC: hold 7 (`lossywide-20260910-r1`, DFlash K7 + lossy
  wide margins, dev prose, 12 pairs per arm; fixed7 16.5 tok/s, 1.37 accepted
  /cycle): m2.5 1.109 [1.060, 1.157] (relaxed 0.18/cycle), m3.5 1.155
  [1.096, 1.213] (0.27), **m5.0 1.219 [1.180, 1.266]** (0.31, accepted 1.89);
  next-cycle first-position acceptance unchanged (0.785 vs 0.77). Still
  monotonic in the margin; A's DFlash ceiling is ~+30% held-out at m5.0
  (runner-up down to ~0.7% of the top probability), where the survey's
  overshoot failures become plausible. Recommendation: promote m2.5 (held-out
  validated); evaluate m3.5/m5.0 with G3/G4/G6 against production later.
- 2026-09-10 05:25 UTC: **all holds complete; production `lossy-prod`
  verified; safe for the user's `GLM_SPEC_LOSSY=1 ./rollout_dcp.sh
  lossy-prod2`.** Then: quality gates G3/G4/G6 against production, pi prose
  persona on, prefill glue overlay, and the long items (C, E).

## Night 2 (2026-09-10 05:45 UTC onward): harness-study avenues

User direction: work the avenues from the harness-study revival autonomously
with codex; start with think-span-only lossy verification (#3); glm_best-style
pi prompting is already in use; re-instrumentation (#1) goes last.

- 05:55 UTC: **think-scoped lossy verification built**: `spec_lossy_scope =
  "think"` relaxes only while the committed stream is inside a `<think>`
  span. Kernel keeps a per-request think state (initialised from the prompt
  tail at admission: thinking-on prompts end `<think>`, thinking-off
  `<think></think>`; updated over committed tokens including the emitted
  argmax on rejection and the bonus token in the insert kernel); scope and
  state are UVA/GPU buffers in `SamplingStates`; rows echo `lossy_scope`.
  Launcher env `GLM_SPEC_LOSSY_THINK_IDS`. Four new CPU tests (span
  tracking, initial state and persistence incl. rejection-closed and
  bonus-opened spans, mixed batch/permutation, parsing). Suite 943 green.
  Codex: bench variants `lossy-think-m<M>`, report invariants (zero
  relaxation when the span never opens), `spec_think_check.py`, hold recipe.
  Codex's `research/HARNESS-STUDY-REVIVAL.md` ranks this avenue #6
  (+6-7% decode on the 09-01 token mix); the user chose it first.
- 06:15 UTC: harness capture for avenues #4/#5 started against production
  (no hold): `capture_proxy.py --per-call --record-text` on :8001 in front of
  the cluster, `bench/pi_humaneval.py --levels 1 --limit 12` routed through it
  (`PI_HE_BASE_URL`), two passes; per-call records carry texts, TTFT,
  prefix-cache hit rate, accepted length by position. Private data
  (`results/harness-capture-20260910/`, to be gitignored). Chain: capture ->
  hold 8 (`lossythink-20260910-r1`: thinking-on dev corpus, arms fixed7 /
  lossy-m2.5 / lossy-think-m2.5 / lossy-think-m5.0 at 1536 tokens, plus a
  thinking-off control proving zero relaxation under scope think). Codex
  queued: per-call attribution and the tool-argument copy screen on the
  captured streams (offline; /tokenize only).
