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
- 07:55 UTC: **hold 8 (`lossythink-20260910-r2`)** thinking-on dev corpus
  (12 cases x 1 repeat, 1536-token budget): paired tok/s vs fixed7 — code
  lossy-m2.5 1.195, lossy-think-m2.5 1.220, lossy-think-m5.0 1.276; prose
  1.169 / 1.172 / 1.266; pooled 20.3 -> 24.0 / 24.4 / 25.8 tok/s; relaxed
  0.25 / 0.24 / 0.35 per cycle. Thinking-off control under scope think:
  **relaxed 0.0** (invariant holds), ratio 0.98 (noise). Reasoning token
  count ratio 0.78 (m2.5) / 0.69 (m5.0) pooled, visible answers 1.12x /
  1.19x longer within the fixed budget. Code checks uninformative: the dev
  prompts do not match `spec_code_check`'s contract and most hit the budget
  on both arms. Next: task-level quality (HumanEval-through-pi, official
  tests) on a lossy boot with the proxy injecting the think-scoped fields.
- 07:55 UTC: codex's `research/HARNESS-CAPTURE-20260910.md` (67 real pi calls):
  not-length-limited mix is ~73% tool arguments (pure tool calls 6.79
  emitted/cycle), ~26% reasoning, ~1% content; one runaway reasoning call
  (32,768 tokens, 720 s) was 71% of all wall time — a harness-level issue.
  Copy screen skipped pending a tokenize window; attribution tables written.

### 2026-09-10 08:05 UTC — tool-argument copy screen: negative; per-call TTFT floor found

- **Tokenized prompt-lookup screen (codex `toolarg_copy_screen.py --tokenize`, production up, no hold, 67 captured pi calls):** every copy policy loses to the DFlash baseline on the real tool-argument streams — tool-cycle gain n4 −8.0%, n8 −3.4%, n16 −1.5%, longest-anchor −5.1%; all-token cycle gain −1.3% … −0.2%; the >3% gate fails for all four. Anchor hits are common (40% at n4) but correct ≥8-token copies are rare (16–19%) because DFlash already emits ~6.9 tokens/cycle on tool arguments (0.85 acceptance). **Harness avenue #2/#4 (copy/prompt-lookup drafting for tool args) is closed** without a cluster prototype. Output: `results/harness-capture-20260910/copy-screen.{md,json}` (no request text).
- **Per-call TTFT (harness avenue #5, from the same capture, no text read):** wall 1010 s = server prefill 65 s (6.4%) + server decode 940 s (93.1%); but the three runaway reasoning calls are 80.7% of wall. For the typical call (median 1265 prompt tokens, prefix-cache hit 0.73 median, median 88 generated tokens) TTFT is **~1.0 s flat regardless of uncached prompt size** (0–500 uncached tokens: median 0.97 s, max 1.18 s; 500–2000: median 1.00 s) and is 33% (median) / 46% (mean) of the call's wall; 28/67 calls generate ≤64 tokens. Server-side `prefill_time` ≈ 0.97 s/call matches, so the floor is inside the engine, not pi/proxy. Earlier bench runs put TTFT at 0.40–0.45 s for 65–140-token uncached prompts (`results/lossy-heldout`, `lossy-think-*`), and the 4096-token prefill profile (hold 5) ran at ~480 tok/s; ~350 uncached tokens at ~500 tok/s ≈ 0.6 s + the ~0.4 s small-prompt floor reproduces the 1.0 s. So there is **no hidden fixed cost**: pi-call TTFT is prefill throughput (~500 tok/s at DCP2) plus a ~0.4 s floor (one forward + first decode cycle + drafter prompt pass). The lever is therefore the prefill-throughput overlay already identified in hold 5 (copies/indexing 16%, TP all-reduce 14%, all-gather 10.5%, reduce-scatter 8% of GPU time): a 20% prefill gain would take ~0.12 s off a typical 3 s pi call (~4%), and more on long-context reads. Priority stays below hold 9; the copies/indexing bucket (reduce-scatter movedim copy, compaction, per-layer `.all()` syncs) is the only part that does not need NCCL changes.

### 2026-09-10 08:35 UTC — prefill glue overlay written: head-major DCP merge (`GLM_DCP_RS_HEADMAJOR`), awaiting hold 10

- Implements the 03:00 work item (1)+(2) as one opt-in boot switch, default off, exclusive with the LSE fold: `dcp_lse_ag_out_rs_headmajor` in `overlay/.../mla_attention.py` runs a copy of the stock LSE-correction Triton kernel that writes its rescaled partial **[H, T, D]** instead of in place, so `reduce_scatter(dim=0)` needs neither of the communicator's two `movedim().contiguous()` relayouts (468 × 1.6 ms of the 8.5 s 4k prefill), and `_v_up_proj(head_major=True)` consumes the head-major result directly (its bmm wanted [H, T, L] anyway). In `flashmla_sparse.py` the compaction kernel now also returns the empty-row mask (`compact_dcp_candidates(..., return_empty=True)`), which replaces the `(topk == -1).all(dim=-1)` rescan of the [T, topk] table, and the `[T, H, D]` `masked_fill` of the attention output is skipped under the flag (the kernel stores exact zeros wherever the rescale factor is zero; the `-inf` LSE mask stays). Same arithmetic as the stock path, so count100 text should be byte-identical.
- Plumbing: launcher `-e GLM_DCP_RS_HEADMAJOR=${DCP_RS_HEADMAJOR:-0}`, rollout forwards `DCP_RS_HEADMAJOR`, `spec_experiment.py prepare --dcp-rs-headmajor` → marker `dcp-rs-headmajor-enabled` → `spec_node.py` env; stage copies + `stage/SHA256SUMS` refreshed. Tests: `tests/test_dcp_rs_headmajor.py` (kernel vs transcribed stock merge at DCP2/DCP4 under the Triton interpreter, NaN/empty rows → exact zeros, head-major layout, compaction empty mask, plumbing), merge/compaction tests updated; suite green locally.
- Expected: ~0.9 s of 8.5 s off the 4k prefill (~10%: two relayout copies 0.74 s + output masked_fill ~0.14 s + rescan), ~1 ms/cycle at decode. **Hold 10** (after hold 9): `prepare --dcp-rs-headmajor` boot, count100 smoke (text must match production), `bench/prefill_profile.py --tokens 4096` vs the hold-5 profile, prose/code decode screens.

### 2026-09-10 08:30 UTC — hold 9 (`lossythink-tasks-20260910-r1`, lossy boot, pi HumanEval through recording proxies): pass 1 only, treatment arms inert; rerun as hold 9b

- Three proxies (control :8001 unchanged requests; :8002/:8003 meant to inject `vllm_xargs` think-scoped m2.5/m5.0 + `spec_label`), `bench/pi_humaneval.py --levels 1 --limit 12` (12 tasks at concurrency 1, thinking high, `PI_HE_MAX_TOKENS=8192`). Pass 1: control 12/12 in 186 s (36 calls, 4892 output tokens), "m2.5" 12/12 in 148 s, "m5.0" 12/12 in 137 s — but the spec trace shows every call unlabelled with `lossy_margin null`: the proxy's `--force-xargs` only injected on the Anthropic `/v1/messages` branch, and pi uses `/v1/chat/completions`. The treatment arms were therefore extra controls (and their 15–25% faster walls are cross-arm cache/drift noise on 2–3-minute runs, not lossy). Fixed in `capture_proxy.py` (chat-completions branch now injects and records `forced_xargs`).
- Also: **pi's HumanEval calls do not think** — `reasoning_tokens` 0 in every task of every arm (a few calls had 100–2400 reasoning chars at most), so think-scoped lossy has nothing to relax on this workload as posed. Hold 9b adds a task-prompt suffix asking for step-by-step reasoning (`PI_HE_INSTRUCTION_SUFFIX`, checked on production first), a trace-label activation probe that aborts the hold if the label is missing, and one controller only.
- Ops lesson: the hold was released at 08:19:59 by my own earlier "stop after pass 1" watcher (started when I misjudged the pass length) while a second controller was starting pass 2 — its `/metrics` fetch hit a server already in teardown. **One controller per hold; kill helpers before starting a replacement, and verify they are gone.**
- 08:40 UTC: codex reviewed dce63ce (head-major merge) before boot: kernel arithmetic identical to the stock `_correct_attn_cp_out_kernel` incl. NaN/inf guards and bf16 store; dim-0 reduce-scatter makes both `movedim().contiguous()` no-ops and rank r receives heads [rH/dcp,(r+1)H/dcp); `_v_up_proj` consumes the TP-local 16 heads correctly (DCP2 gathers 32, scatters back 16); skipping the output masked_fill is safe (no other consumer; the -inf LSE forces exact zeros); the compaction empty mask equals the old all(-1) test; no host sync or data-dependent allocation for graph capture. One open item: byte identity is unproven (bmm input strides change → cuBLAS kernel selection may differ), so hold 10 compares the count smoke token ids with the stock-path hold 9 smoke and the fixed screens' texts before any promotion. Hold 10 is chained after hold 9b (production prefill timing 4k×3/40k×3 first, then the flagged boot, screens, timing, profiled 4k prefill).
- 08:31 UTC: hold 9b production trial — the reasoning suffix does **not** make GLM-5.3 think on these pi coding calls either (36 and 70 reasoning chars on two of eight calls, 0 on the rest), so hold 9b's think-scoped arms will be inert by construction; it still yields the activation probe (labelled trace rows, relaxed count on a thinking probe) and a task-level invariance result (treatment ≡ control when nothing is relaxed). The informative task-level test is **unscoped** lossy (`spec_lossy_scope: all`, m2.5/m5.0) on real pi coding calls with official HumanEval tests as the quality gate; I did not switch the running hold's proxies to it (the mid-hold restart was blocked by the permission classifier) — proposed as hold 11, labels `all-tasks-m2.5`/`all-tasks-m5.0`, same driver and proxies, ~25 min.

### 2026-09-10 09:20 UTC — hold 9b (`lossythink-tasks-20260910-r2`): think-scoped lossy at task level — **negative: reasoning runaways fail the one reasoning task**

- Setup: lossy boot, three recording proxies (control unchanged; :8002/:8003 inject `spec_lossy_scope=think` m2.5/m5.0 + `spec_label`), activation proven by the probe (26/16 labelled rows, 6/3 relaxed) and by the trace (control 963 rows, 0 relaxed; m2.5 1017 relaxed; m5.0 1548 relaxed). 12 HumanEval tasks × 2 passes × 3 arms through real pi coding calls (thinking high, `PI_HE_MAX_TOKENS=8192`, reasoning suffix — which did not make the model think more).
- **Quality:** control 24/24; think-m2.5 22/24; think-m5.0 22/24. Every treatment failure is HumanEval/132, the only task where GLM-5.3 reasons at all (control: one ~3k-char reasoning call, passes both times). Under relaxed acceptance inside `<think>` that reasoning ran away to the 8192-token cap in 3 of 4 treatment runs (25–30k reasoning chars, `finish=length`, 250 s) and the task failed in all 4 (the fourth had shorter reasoning and still failed). Deterministic across passes and margins. This is the tail hold 8's averages hid ("reasoning 0.78×/0.69× shorter" on prose prompts): **think-scoped lossy can derail a reasoning chain into a loop**; with pi's 32k budget that is a 12-minute stall.
- **Speed:** per-task wall geomean vs control 1.21× (m2.5) / 1.25× (m5.0) slower overall, and still 1.17×/1.12× slower excluding task 132 with 25–31% more output tokens — but the inert arms of hold 9 pass 1 already differed by ±20–25% in tokens and calls between identical requests (tool results and prefix-cache state make pi runs non-deterministic), so 12-task speed ratios are noise; the pass-agreement matrix is the robust part. Pooled decode tok/s (+15–20%) and accepted/cycle (6.3/6.6 vs 5.4) are inflated by the runaway calls and mean nothing here.
- **Scope check (1:1 call↔request pairing by order):** on calls with no reasoning, relaxed tokens are 58/5170 (1.1%, m2.5) and 74/4235 (1.7%, m5.0) — small but not zero, i.e. think-scope is relaxing a little outside `reasoning_content` (empty `<think>` spans the template opens, or the kernel's think-state bookkeeping after a rejection; under review). Inside reasoning: ~28–34 relaxed per 1k reasoning chars.
- Verdict for the pi-alias question: **do not ship think-scoped lossy as a default for coding sessions**; if it is ever used, it needs a reasoning-length guard (e.g. fall back to exact verification after N relaxed tokens per span, or a hard per-call cap) and the small out-of-span leak fixed. The unscoped m2.5 task test (hold 11, official tests as the gate) runs after hold 10. Report: `results/lossy-think-tasks-20260910-r2/report.{md,json}` (activation windows are being fixed in the tool; the label/relaxed counts above are from the trace directly).

## Morning recap (written 11:35 UTC, holds 10 and 11 still queued)

| # | Item | Result | Status |
|---|------|--------|--------|
| 3 | Think-scoped lossy, request level (hold 8) | +22/+17% code/prose decode at m2.5, +28/+27% at m5.0 with thinking on; inert with thinking off | measured |
| 3 | Think-scoped lossy, **task level via pi** (holds 9/9b) | control 24/24 tasks; think-m2.5 22/24; think-m5.0 22/24 — the only reasoning task runs away to the token cap under relaxed acceptance (3 of 4 treatment runs) and fails every time | **negative; do not ship as a default** |
| 3 | Think-scope leak | 1–2% of tokens relaxed on tool-call-only outputs: first token after prefill bypassed the state machine (`</think>` missed) | fixed (28e9fc5), unmeasured |
| 2/4 | Tool-argument copy / prompt-lookup drafting | every policy loses to DFlash on 67 real pi calls (−0.2…−1.3% all-token) | closed, negative |
| 5 | Per-call TTFT | ~500 tok/s prefill + ~0.4 s floor; no hidden fixed cost; lever = prefill throughput | closed |
| G | Prefill glue overlay (`GLM_DCP_RS_HEADMAJOR`) | head-major DCP LSE merge: lossless (count identical), decode neutral, prefill +1.3% (4k) / +1.7% (40k) paired vs production — the copies were overlapped with NCCL, so 0.9 s of kernel time bought ~0.1 s of wall | measured; opt-in kept |
| A | Unscoped lossy m2.5/m5.0, task level via pi, official tests (hold 11) | quality unchanged (23/24 every arm), decode tok/s and accepted/cycle unchanged (peaked code distributions), tasks 1.3× slower from longer trajectories (more reasoning, extra tool calls) | **negative for coding**; prose-only lever |
| 1 | Re-instrumentation | deferred last per instruction | not started |

Ops notes for the morning: production is the user's `lossy-prod` (lossy overlays mounted, `GLM_SPEC_LOSSY=0`); every hold restored and was verified. `~/spark-cluster-experiments/capture_proxy.py` (separate dir, uncommitted) now injects `--force-xargs` on `/v1/chat/completions`. Lessons: one controller per hold; verify kills with `pgrep` (a `kill` on the `$!` of `nohup bash … &` hit a wrapper, not the script); never let a guard `pgrep -f` pattern be matchable by my own tool shell.

### 2026-09-10 12:20 UTC — hold 10 (`hm-20260910-r1`, `GLM_DCP_RS_HEADMAJOR=1` + profiler): head-major DCP merge is lossless and worth ~1.5% of prefill, not 10%

- Boot clean; count smokes byte-identical to all five stock-path holds today (the 256-token screens differ between *every* pair of boots, stock included — common prefixes 3–144 tokens from Marlin's nondeterministic MoE reductions — so they are not an identity gate). Decode screens within ±6% of the fold-lane reference with mixed signs: neutral.
- **Prefill timing, same command on production immediately before the boot (uncached random prompts, thinking off):**
  - 4096: production 539 tok/s (wall 8.21–8.38 s, n=3) → head-major 546 tok/s (wall 8.13–8.20 s, n=3): +1.3% throughput
  - 40000: production 492 tok/s (wall 87.54–88.02 s, n=3) → head-major 500 tok/s (wall 86.22–87.13 s, n=3): +1.7% throughput
- **Profiled 4k prefill** (rank 1): GPU busy 7.42 s vs 8.30 s in the hold-5 profile, copies/indexing bucket 0.42 s vs 1.40 s (3892 vs 8806 kernels): the relayout copies and the output masked_fill are gone as designed — but the hold-5 reference ran the LSE-fold lane, whose own casts/copies (~0.5 s, noted at 03:00) inflated that bucket, and the remaining copy time overlapped NCCL on another stream, so the wall gain is the ~1.5% above, not the 0.9 s of kernel time. Per-token: 545–547 tok/s head-major vs 538 production (4k), 500 vs 492 (40k).
- Verdict: keep as an opt-in (`DCP_RS_HEADMAJOR=1` on the rollout) — lossless, codex-reviewed, ~1.5% prefill, decode neutral. Enabling it by default is a one-line launcher change once it has served a longer session; it is not a speed lever on its own. The 17% "torch glue" in the profile is largely overlapped; the true prefill critical path is NCCL (ring ceiling) + MoE, as the 03:00 entry already ordered them.
- 12:22 UTC addendum to hold 9b: in hold 11's control pass 1 HumanEval/132 **failed under the control too** (16 s, 454 tokens, no long reasoning) — the control passes it when the model happens to reason (~3k chars) and fails it when it does not, so 132 is borderline at temperature 0 (3 passes / 1 fail over four control runs today). The hold-9b signal that stands is the behaviour, not the pass bit: under think-scoped relaxation the reasoning ran to the 8192-token cap in 3 of 4 treatment runs (never in any control run today), which is what would stall a pi session.

### 2026-09-10 12:50 UTC — hold 11 (`lossyall-tasks-20260910-r1`): **unscoped** bounded-lossy on real pi coding calls — no per-token gain, longer trajectories, quality unchanged

- Same driver as hold 9b (12 HumanEval tasks × 2 passes × 3 arms, official tests), proxies :8004/:8005 injecting `spec_lossy_scope=all` m2.5/m5.0 with labels `all-tasks-*`; activation proven by the probe and by the report (`--scope all`): control 0 relaxed / 1325 verify rows, m2.5 436 relaxed / 1992 rows, m5.0 489 / 1893. Report `results/lossy-all-tasks-20260910/report.{md,json}` — status complete.
- **Quality:** 23/24 in every arm; the only failures are HumanEval/132 (borderline under the control too, see 12:22). Pass agreement m2.5: 23 both-pass, 1 both-fail; m5.0: 22 both, 1 control-only, 1 treatment-only.
- **Speed:** pooled decode tok/s 37.5 (control) / 37.5 (m2.5) / 39.0 (m5.0) and accepted/cycle 5.93 / 5.98 / 6.19 — on tool-call and code output the target is peaked enough that a 2.5–5 nat margin almost never fires (436 relaxed of ~11.9k generated tokens at m2.5 = 3.7%), so there is **no per-token speedup**. Per-task wall geomean **1.30× / 1.31× slower** (bootstrap 95% [1.07, 1.60]) with 26–34% more output tokens: the relaxed arms reasoned far more (12.7k / 10.9k reasoning chars vs 1.3k in pass 1) and took extra tool iterations on several tasks (2 → 4 calls), i.e. the lossy trajectories drift longer, not wrong.
- **Verdict on lever A for coding sessions: negative.** Bounded-lossy is a prose/reasoning-token lever only (+19.7% held-out prose at m2.5, hold 8's think-span gains), and inside reasoning it carries the runaway risk found in hold 9b. For pi: no `GLM_SPEC_LOSSY` default; if a prose alias is wanted, a proxy that injects `spec_lossy_margin=2.5, spec_lossy_scope=all` for that alias's port is the mechanism that works today (pi's provider config has no extra-body hook), and it should not be used for coding.
