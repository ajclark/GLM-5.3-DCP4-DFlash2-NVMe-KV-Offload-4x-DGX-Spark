# Adaptive speculation execution status

User authorized autonomous implementation/testing, local development first and
careful memory observation before every Spark experiment.

## Current state:2026-09-09 05:34 UTC — experimental phase complete

- All R6/R7 experiments, declared follow-ups, correctness, restart, overhead and
  final source audits are complete. No controller, benchmark or sampler remains.
- Exact original container IDs/images/config/hostconfig/mounts/mounted-source hashes
  restored on all4nodes; final generation passed. TP4/DCP2,maxlen180224,maxseq12,
  KV6GB/rank unchanged. Proof:adaptive-v4-20260909-r7/original-restoration-audit.json.
- Implemented version4 adaptive verification remains opt-in; defaultpolicyoff.
  Source/calibration/corpora stayed frozen throughout evaluation. All370 local tests
  passed9.49s, threepatches reproduce overlays exactly, stage checksums match.
- Initial180requests:prose+15.80%(95%13.24–18.73%);coding+0.99%(-2.11–4.44%).
  Initial coding gate narrowly missed. Declared90request coding extension yields
  90coding pairs,+1.14%(95%-1.30–3.64%),inside2% margin; follow-up evidence only.
  Aggregate code25.07→25.47tok/s,prose15.36→17.80. Initial locked report retained.
- Device decode energy:prose-17.37%;combined coding-2.85%(95%savings0.40–5.39%).
  Whole-cluster wall-energy gate unmeasured; no hardware idle-watt improvement
  claimed. CX-7 cycling remains paused. No broad promotion.
- R6 one-repeat code/prose repo gains:4k-0.2%/-9.5%;32k+8.0%/-7.9%;
  100k+34.6%/+37.2%;170k+41.7%/+42.1%. R7 declared3repeat follow-ups:
  4k+4.89%/+15.33%;32k+7.19%/+24.07%. Initial prose regressions did not repeat.
  One prompt/category; exploratory. First170k prefill402s cold;decode-only gains.
- Correctness:fourcompletefunctions identical245IDs;C1/C2/cancel/fallback passed;
  C2overlap~81tok/s ratio1.00018;six256-token reasoning controls;blind prose
  openings10ties/adaptive3/fixed2. Random32k/100k count30+marker all pass;32k
  all66IDs identical,100k differs only one newline token before adaptation.
- Once-store/restart passed:102360freshprompt,13.53GBstore,0hits; R7 loaded
  102016external cached tokens,0localhits,11.06GBload,exact127continuationIDs.
  Prompt-only disk policy preserved; supplied80generatedIDs recomputed as suffix.
- Tracing9identical-output pairs:centraltime+0.56%,95%-0.11–1.81%, clocks match.
  Strict upper1%budget not established; separateboots/threeprompts limitation.
- R6audit30154verifyevents,0integrityerrors,FULL M2/4/6/8 on allranks in R6/R7.
  No OOM/guardtrips. Serving min1641MiBhead(R6)/1793MiB(R7); other ranks>=3098.
  Head swap-out2.60MiB R6 and8KiB R7; fullPSIavg10=0. Do not claim zero swaps.
- Copyselector offline CPU/opportunity screen complete; insufficient evidence for
  GPU integration on edit/continuation. See copy-branch-decision.json.
- README.md,COMPLETION-AUDIT.md,and completion-decision.json are final reports.
  All experimental raw results and frozen locks remain available locally.

## Historical execution log

As of 2026-09-08:

- Implemented opt-in C1 greedy verification caps 1/3/5/7, keeping DFlash2's
  seven-token draft capacity and original async placeholder replenishment.
- Added target graph candidates M=2/4/6 after upstream graph-size rounding;
  drafter graphs retain their existing shape. Logs report actual dispatch.
- Added bounded async JSONL telemetry and a censored-prefix controller prototype.
  Adaptive mode has not been enabled or tuned; no throughput benefit established.
- 306 local tests pass, including deployed Triton kernels on CPU, all rejection
  boundaries, two in-flight async steps with cap changes, graph dispatch,
  terminal/abort feedback, pressure thresholds, and rollback safety.
- Fixed existing pytest collection issues (standalone slab tests exited during
  import; one fork source path did not expand '~').
- The stage copy of multinode.py is newer than overlay; preserved the deployed
  stage version. Do not regenerate the entire stage from overlay blindly.
- Original K=7 baseline: 256 tokens, 2 repeats, prose 14.92/15.50 tok/s,
  code 27.60/29.69, repo-review 18.87/20.31. Greedy baseline itself diverges at
  output tokens 11/20/30 respectively; whole-sequence hash equality is not an
  adequate standalone correctness gate.
- Baseline min available MiB: rank0 2789.8; rank1 4248.8; rank2 4370.3;
  rank3 4826.7. No new swap-out, 1.33 MiB swap-in on rank0, no pressure trips.
- Prepared isolated experiment `fixed-caps-20260908-r1` on all four hosts;
  all relevant running Python hashes matched the pinned rank0 inventory.
- Controller RUN STARTED: exec session 7347 running
  `.venv/bin/python bench/spec_experiment.py run fixed-caps-20260908-r1`.
  Poll it; do not start another experiment while it runs.
  It should automatically restore original containers after smoke and fixed
  cap comparisons or on any failure. Verify restoration and collected logs.

Remote experiment directory: ~/glm-spec/fixed-caps-20260908-r1 on each host.
Original containers are retained as vllm_glm53big_backup_<label> while testing.
Each experiment container carries label glm.spec.experiment=<label>.
Node watchdog aborts on <1536 MiB available, new swap-out >32 MiB/2s,
new swap-in >64 MiB/2s, full PSI >0.5% avg10 or >1% interval, OOM counter
increase, reboot, or >60s controller heartbeat gap. Preflight requires
2048 MiB available and a quiet observation interval. No NIC/power changes.
The controller automatically restores the exact original image/args/mounts;
original overlay and /var/tmp/kvcache are untouched. New cache uses experiment
subdirectory and the same 150 GB sparse slab cap; >200 GB free checked.

Next: inspect experiment results, actual graph padding, memory headroom and
telemetry. Fix issues locally before another rollout. Measure sufficient paired
runs before enabling a controller. Copy proposals only if justified by gates.
Whole-node power has not been measured; do not claim energy/idle-power gains.
The paused CX7 power work must remain paused.

Update after first trial / user correction:

- Trial r1 aborted during weight loading on a 38 ms full-PSI spike in a 2 s
  interval, with 15 GiB available. Original containers were restarted.
  The original stack also triggered the old guard during ordinary reclaim;
  the first controller exited while originals continued loading. No OOM.
- User explicitly said the guard was too conservative and that normal rollout
  leaves little free memory. Revised guard to be phase-aware: hard floor
  512 MiB; preflight 768 MiB and quiet serving; abort low-headroom sustained
  pressure (full PSI >10% serving / >20% loading below2 GiB), sustained
  >512 MiB swap-out/10s with stalls below2 GiB, or severe interval pressure
  below1 GiB. Normal loading reclaim with ample available RAM is permitted.
  OOM/reboot/disconnect/heartbeat protections retained. Tests cover observed
  false trigger as well as sustained exhaustion.
- EXEC SESSION 80698 is now monitoring/verifying the ALREADY RESTORING
  original stack (`spec_experiment.py verify-restore ...r1`). Poll it until
  health and generation pass. No new experiment running.
- Next prepare/run a fresh r2 label with corrected guard after original
  restoration verifies. Do not reuse r1 (its watchdog/backup state is retained).

Latest update:
- Original stack restored, health+generation verified. Settled available MiB:
  rank0 4568, rank1 5814, rank2 6245, rank3 6167. Verify-restore session80698
  completed successfully. No OOM.
- 316 full-suite tests passed, plus a new count-smoke validation test passed.
- Prepared `fixed-caps-20260908-r2`, all source hashes verified on all ranks.
- RUN r2 now starting under revised guard; poll the latest exec session from
  tool output. It automatically rolls back original containers afterward.
- r1 diagnostic logs/trace are under its result directory; local STATUS above
  preserves the guard correction/user steering. No speculative throughput
  improvement has been measured yet.

Current live process: exec session38805, guarded run r2. It is loading normally
with ~18 GiB available per node, and must be polled through automatic restore.

Offline baseline screening at emission boundaries (excluding endpoints):
- Prose mean emitted2.108/cycle; K3 retains96.5% (requires >3.5% cycle saving).
- Code mean emitted4.116/cycle; K3 retains70.9%; K5 retains88.4%.
- Repo-review mean emitted2.807/cycle; K3 retains86.0%; K5 retains95.9%.
- Exact suffix-copy opportunity on these SHORT outputs is sparse: code3/63
  and5/59 boundaries (mean accepted1.33 and3.4), prose0, repo1/92 and3/86
  (both zero accepted). This does not evaluate long repository-edit contexts.
  Sandbox-only `bench/copy_opportunity.py` selects using committed history;
  future baseline output is scoring only. No live copy integration yet.

R2 screen finished; original stack is reloading automatically (session38805
still running). All four counting outputs have identical60 token IDs. Caps
actually use M2/4/6/8, but small caps used PIECEWISE while K7 used FULL. The
indexer forces FULL_AND_PIECEWISE (UNIFORM_BATCH support), requiring explicit
uniform C1 descriptors. This is now fixed locally and tested, not yet deployed.
R2 two-run medians: prose K7=16.03,K3=17.05(+6.3%); repo K7=18.44,
K3=21.45(+16.3%); code K7=25.39,K3=23.34(-8.0%). Other caps slower.

Next r3 contains full C1 small graphs and request-scoped calibrated costs
(spec_cycle_ms, spec_cost_lane, spec_cost_context). It can run --hold for
30 minutes of additional guarded experiments after the screen; touch its
local results/<label>/finish to trigger exact-container restoration early.
No serving/inference changes have been promoted. 323 tests now pass.
`bench/spec-development.json` holds 12 development prompts; held-out and
long-context validation remain outstanding. Adaptive cost tables require a
context_range; out-of-range requests fall back to full verification.

R2 restored successfully (session38805 completed). Telemetry2607 steps,
all cap/actual lengths matched, zero dropped trace records. Overall median
cycles: K1=117.80ms, K3=123.54ms, K5=137.53ms, K7=143.01ms.

CURRENT LIVE SESSION: 29070 runs
`.venv/bin/python bench/spec_experiment.py run full-caps-20260908-r3 --hold`.
R3 prepared/verified on all ranks. It will boot, count-smoke four caps, run
same24 short fixed-cap requests, then HOLD up to30 minutes under memory and
heartbeat watchdogs for additional benchmark commands. Create
`results/adaptive-spec/full-caps-20260908-r3/finish` to end hold and restore.
Poll until it is healthy, verify SPEC_GRAPH says FULL for all four caps,
then collect/fit cost table and run adaptive/12-prompt comparisons duringhold.
Do not run another workload while its initialscreen is active (idlecheck).

New `bench/spec_power.py` can sample NVIDIA-reported power/clocks independently
without CUDA imports. It has NOT been started. This is not wall energy;
whole-node meter belongs to user Grafana, and no automated source was found.
R3 results record started_at UTC for aligning inference with power samples.

Power sampler IS RUNNING: session62467, file
full-caps-20260908-r3/nvml-power-r1.jsonl, duration3600s. Stop it when done
(use signal or session input; non-TTY may require kill of exact local PID).
It reads NVIDIA device power/clocks, not wall power.

While r3 holds, collect live logs with `spec_experiment.py collect <label>`,
then run `fit_spec_costs.py <results/label> --out <costs.json>`. Fit refuses
without FULL M2/4/6/8 evidence and >=50 calibration cycles/cap; scope0..512.
Use `adaptive_spec.py --out ... --corpus bench/spec-development.json
--variants fixed7,fixed3,adaptive --costs ... --tokens256 --repeats1` for
interleaved12-prompt screen. --cases defaults to all corpus cases. Results
now include policy and started_at. Hold deadline is30min after fixedscreen.
First command r3 live session29070 still booting; no other inference until
its initial24-request screen completes and printsHOLD.

IMPORTANT BEFORE ADAPTIVE TESTS:
A local statistical audit found a censoring bias in the initial PrefixStats:
counting known tail failures on short rejected blocks while omitting censored
successful tails biases unconditional tail estimates downward. Replaced with
conditional acceptance risk sets (verified position AND all predecessors
accepted); prefix survival is their product. Inadequate tail evidence uses an
optimistic bound, favoring full verification. Added deterministic regression:
A=0 or7 equally, 15/16 cycles capped3 => correct Eemit7=4.5 (old method would
underestimate tail as1/17 rather than1/2). Policy tests pass.

R3 RUNNING CODE STILL HAS THE OLD CONTROLLER; fixed-cap data is unaffected.
DO NOT ENABLE ADAPTIVE ON R3. Use its hold for broader FIXED-cap validation,
cost calibration, C2/fallback checks and power data. The corrected controller
requires a fresh deployment (r4) after local testing. Local overlay adaptive.py
changed, but stage copy/checksums have NOT been refreshed after this correction.
R3 full graphs all verifiedFULL atM2/4/6/8; countoutputsidentical60IDs; graph
capture memory1.63GiB vsr2's1.42GiB (+0.21GiB); noOOM or guardtrips.
First prose pair K7=16.68,K3=19.83 tok/s (~+19%). Power samplerworks.

R3 fixed screen complete: FULL graphs M2/4/6/8; median prose K3 +17.2%, code -1.4%, repo +6.2%. Costs calibrated from >450 cycles/cap: 95.29/115.25/129.73/144.28ms (K1/3/5/7), context0..512. Broader12-prompt FIXED7 vsFIXED3 active session6903. Local corrected conditional estimator is policy version2; stage andSHA256SUMS ARE now refreshed. 324 tests passed before analyzer additions;8 focused analysis/deployment tests passed afterward. Held-out30 synthetic prompts frozen in bench/spec-heldout.json before adaptive development measurements. No adaptive requests on r3.

R3 broader fixed screen done (session6903 ended):12 prompts, paired geometric mean K3/K7 prose1.1643 (6 prompts), coding1.0443 (6 prompts, individual0.880..1.176). Device-energy ratios0.8095 prose,0.9054 code, NOT WALL ENERGY. Integrated samples bracket first/last token time; all4nodes required, reject gaps>5s. 326 local tests pass.
Live integration session60577 completed: temperature and repetition penalty fallback actualK7; C1-to-C2-to-C1 actualtraceK1-to-K7-to-K1, counts correct; cancelled stream left endpoint idle; nextK1count correct. Trace checks saved in integration-r3/trace-validation.json.
CURRENT additional inference session35256: context-r3 --context-smoke. 32823-token coldK7 passed (TTFT64.63s), warmK3passed (TTFT0.76s), 100055-token coldK7currentlyrunning. Memory~2.7GiB rank0/4.2..4.5GiB others, PSI0/new swapout0/OOM0. R3hold session29070 deadline~23:37UTC; after context finishes create full-caps-20260908-r3/finish and await exactoriginalrestore. Then prepare fresh r4 with version2controller; use --hold --hold-minutes120 for development and heldout comparisons. Power sampler62467 stillruns until~23:51UTC.
Copy screen on12developmentprompts shows sparse matches, no broad live-copy case established. Raw outputs/traces retained. README.md now summarizes fixedresults; adaptive tests outstanding.

R3 inference COMPLETE. Localfinishfilecreated~23:30UTC. Controller29070 now restoring/reloading exact originals; IDs fb1a5f3be3dc / bdfb37379a64 /8c4e76275f05 /5625163086e0 confirmed restarted, health pending. Power62467 stillsampling.
Longcontext32k/100k bothK7andK3passed markerretention+count1..30, warmTTFT0.76/1.07s. ColdTTFT64.63s at32823tokens,214.8s at100055tokens. Memoryminimum2673.7MiB rank0,4184.8/4425.0/4474.6MiB others,0PSI/newswapout/OOM. Both fixed7/fixed3 generated four complete pure Python functions passing bounded local executable checks (code-check-r3).
Unexpected100kbehavior: all65verificationsteps accepted0drafttokens at everycap. K7=7.04tok/s,K3=7.98,K5=7.51,K1=9.18 (+30%K1vsK7), samecountcorrect. Draftconfigmax_position_embeddings1048576/sliding_window2048, so not an obvious advertisedcontextlimit. Need originalbaselinecontrol before attributing to overlay. Longcosttable fitted55cycles/cap:110.16/126.65/134.34/143.32ms, scope100000..101000, in full-caps-20260908-r3/costs-100k.json.
Preparing adaptive-v2-20260908-r4 now (safe stagingwhileoriginalreloads). Next await original health, consider baseline100k control, then run r4 --hold --hold-minutes120. Corrected version2 already staged/tested. No adaptive deploymentyet.

Original r3 restoration COMPLETE and generationverified (session29070 exited), healthyafter407s. Newsource for r4 successfullyprepared/verified (session44399 completed), but DO NOT runituntil currentoriginal100kcontrolfinishes. Starting baseline-100k-r3 control on originalviaadaptive_spec.py --deadline-seconds900. This checks the zero-draftacceptance finding against unchangedruntime. Preparedr4 includesversion2overlay andrecordedevaluation-lock.json (30syntheticprompts,3repeats,256tokens,thinkingoff). No policy promotion.
User sidequestion: would Cdecoder improve tok/sec? Answeredincommentary: heavykernels alreadycompiled; nativehostlooponlyrecoversCPU/launchcriticalpath; efficientblocking mattersforidlepower morethanlanguage. Existingprofile suggestsGPU/collectives dominate, but its idlewindow accounting was disputed in review so do not claimpreciseCPUoverhead ceiling. Continueoriginaladaptivegoal.

R4 prep finished. Before launching r4, wait for original baseline100k control session70197 (currentlyprefilling,guardstable). Localdriver now records approximate Prometheus spec counter deltas, supports per-request deadline up to900s for longcoldprefix, thinking-on control, and counterbalanced AB/BA bycaseindex+repeat. It also has --code-smoke for boundedlocalgeneratedfunctionchecks and --context-smoke optionalcosts forout-of-range/adaptivelongchecks. 326tests pass; changedpatchesreproduceoverlays exactly.

Baseline100k control session70197 COMPLETE on original:215.08sTTFT,7.050tok/s,455drafttokens/65draftsteps/0accepted,66single-tokenSSEblocks. Matches r3zeroacceptance; this predatesourpatch. Baselineoutputhashb412e8fa matchesr3K1. No memorypressure/oom.
CURRENTLIVESESSION41060: `.venv/bin/python bench/spec_experiment.py run adaptive-v2-20260908-r4 --hold --hold-minutes120`. Preparingr4alreadydone. Nowbooting; will runcountsmoke+24fixedscreen beforeHOLD. DO NOT send inferenceuntilHOLD. Then collect,fitfreshshortcosts, run12dev paired fixed7/adaptive andvalidateactualversion2/caps; ifpromising runlocked30heldout3repeats (~50min). Useadditional100k/think-on/codechecks afterward. R4finishfiletriggerrestoreatresults/adaptive-spec/adaptive-v2-20260908-r4/finish.
Power62467 stillrunninguntil~23:51UTC. Sandboxps cannotseeescalatedprocessPID; letitendatown1hdeadline. StartnewR4power sampler onceitends (7200s) beforedev/heldoutmeasurements. No needtokillforeignprocesses.

At23:46UTC,r4session41060stillloadingnormally (~18GiBavailable,0OOM/trips). R3power62467ends~23:51UTC. Do not send C1benchmarkrequests before r4printsHOLDafterits24requestfixedscreen. Sidequestionextraevidence: sandboxCPU-onlypolicyselect/schedule/feedback microbench~8.19us; notwholePythonhostloopandnotSparkCPUmeasurement. Savedpolicy-cpu-overhead.json.

R4power sampler session48524 started~23:51UTC for7200s, file adaptive-v2-20260908-r4/nvml-power-r1.jsonl. Oldr3sampler62467willendautomaticallysoon; brieflyoverlappingread-onlynvidia-smiprocessesarenotCUDAallocations. R4graphcapture1.62GiBheadcompleted23:51:16, healthpending/soon. Session41060stillownsmodel andwillperformfixedscreen beforeHOLD; no externalinferenceyet.

R4 is NOW HOLDING (session41060); deadline~02:00UTC09Sep. Allfixedcaps/countsmokescomplete; fullgraphsM2/4/6/8 verified; tracesversion2only,0drops. Freshcosts95.69/115.71/130.34/145.10ms,>480cycles/cap. CURRENTINFERENCESESSION41788: dev-adaptive-r4,12devprompts,fixed7vsadaptive,256tokens,1repeat,counterbalancedAB/BA. Aftercompletecollect/analyzetraces+power anddecideheldout30x3. Nootherinferenceuntil41788ends.
Oldpower62467endednormally. R4power48524runninguntil~01:51UTC. Runtimecontroller41060holdingunderwatchdogs,finishfiletriggersoriginalrestore.
Localdriveradded --repo-context32000|100000 tobuildactualrepoexcerptswithtwo code/prosetasks viaCPU/API token count whileendpointidle. Prefixbands target..target+384, maxoutput256 fitscostbandtarget..target+1000. Usefulaftershortdev/heldout tocheckwhetherlongrandomfillercollapsealsooccurswithrealrepocontext. Newbuilderhasnotbeenrunyet. --deadline-seconds900neededforlongcoldprefill. Per-requestcosttablesneededforlongadaptive; firstcalibratefourfixedcapsinthesameband withfit_spec_costs.py --context-range.

2026-09-09 00:40 UTC checkpoint (supersedes older live-session notes):
R4 controller41060 is HOLDING until about02:00UTC; power48524 samples to01:51UTC.
Current inference43603 runs guarded repo170k-fixed-r4 (two tasks, four fixed caps,256tokens).
No other inference may overlap. 170132/170125 prompt tokens; existing6GBKVpool,
180224window. Headroom~2.2GiBhead/3.7..4.0GiBothers;0PSI/newout/OOM.
V2 development FAILED prose gate: code1.03493 CI[1.00313,1.06232],
prose1.03758 CI[1.00507,1.06865]. Held-out30prompts remain untouched.
Local policy VERSION3 adds weak fixed-calibration conditional prior(strength2),
bounded context CostCurve interpolation, and idle-blocking telemetry writer.
339tests passed before the offline copy-index addition;5copytests now pass.
Stage/changedpatches are currentV3; R4 remains V2. Do not run V3adaptive onR4.
R4 fixed repository calibration complete at4k/32k/100k. At100k,K1vsK7:
code9.324vs6.444tok/s(+44.7%),prose9.206vs6.533(+40.9%).
At32kcodebestK7=20.419/prosebestK3=19.917;at4kcodebestK3=19.259,
prosebestK5=19.938. Current costs-point-*.json fit initialR4fixedscreen and
eachcontextlabel separately;finalrefitafter170kcollect beforecurvemerge.
100kK7calibration486cycles:4first-tokenacceptances,0second-tokenacceptances.
So nearlyzeroacceptance,notstrictlyzeroontheactualrepotasks. Originalrandom
100kcontrol hadexactlyzero and confirmedpreexistingruntimebehavior.
Full-windowofflinecopyindex nowtestslookupacross180224tokens;noGPUintegration.
Fixed7code:short13matches/435boundaries,19acceptedfuturetokens;32k3/77and0;
100k17/246and15. Prose100k29/248and48. These arebaseline-trajectory
opportunitycounts,notliveperformance. No broadcodingcaseforcopyestablished.
Next:finish170k,collect/fit/mergecalibration,freezeV3evaluationlock,finishR4
andawaitexactoriginalrestoration,prepare/runfreshR5V3underwatchdogs,
12devscreen thenlocked30prompt3repeatheldout plusV3code/long/thinkingchecks.
No policy promotion;wallenergy/idlewattsunmeasured,NICcyclingstillpaused.

2026-09-09 00:49 UTC checkpoint:
- All R4 inference is complete. 170k fixed results: coding K1 9.360 vs K7
  6.491 tok/s (+44.2%); prose K1 9.372 vs K7 6.315 (+48.4%). Minimum
  headroom 2166 MiB on head, >=3689 MiB elsewhere; no PSI/new swap-out/OOM.
- R4 finish file created. Controller session 41060 is restoring the exact
  original containers; all original IDs restarted, still loading normally.
  Wait for health AND successful generation before running R5.
- R5 adaptive-v3-20260909-r5 prepared successfully (41383 completed). Local
  stage is VERSION3 with weak prior, context curve, idle-blocking writer.
- R4 five-point cost curve and immutable calibration-trace.jsonl are frozen.
  R5 evaluation-lock.json freezes source hashes, cost table, 30 held-out
  prompts, 3 repeats, 256 tokens, AB/BA order. Held-out inference untouched.
- 343 full tests passed; subsequent analysis refinements have 6 focused
  tests passing (no new core changes). New gate evaluator audits complete
  pairs and trace version/cap integrity; reports latency, aggregate decode
  rate, and coding/prose subgroups. No interval claimed from one prompt.
- R4 power sampler PID 412290 has been sent SIGTERM after measurement ended.
  Start a fresh 7200s sampler for R5. No inference is currently active.
- Next: await 41060 complete; run R5 --hold --hold-minutes120, wait through
  boot/counting/24 fixed controls until HOLD. Check FULL M2/4/6/8 and short
  cycle-cost stability; run dev fixed7/adaptive with frozen V3 curve. Then
  locked held-out 30x3 if promising; no tuning on held-out results.

2026-09-09 01:01 UTC checkpoint:
- R4 restoration completed successfully, healthy and generating after 423s.
- LIVE controller session 97314 runs R5 adaptive-v3-20260909-r5 --hold
  --hold-minutes120. It is still booting, then runs counting smokes and
  24 fixed controls. DO NOT send other inference until it prints HOLD.
- LIVE power sampler session 32625 started about00:54UTC, duration7200s.
  Output adaptive-v3-20260909-r5/nvml-power-r1.jsonl. Old R4 sampler ended.
- V3 evaluation lock is frozen; all locked source hashes verified unchanged.
- Local-only durability helpers added while R5 loads: spec_experiment
  supports --screen-repeats0|1|2 (default2) and prepare --reuse-cache-from
  <stopped-label>. Node helper checks identical original image/config,
  all DCP and mounted-kernel hashes, launch script, runtime manifest, and
  EXACT deployed slab salt including unmounted kernel Python sources. It
  attaches only a stopped, disarmed, untripped experiment's isolated cache
  through an empty new experiment directory. No data copied/deleted, no
  production cache reused, no salt check weakened. 13 deployment tests pass.
  R5 was packaged before these helper additions; its current flow unchanged.
- NEW bench/spec_durable_check.py first|reload creates a unique ~100k prompt
  once with adaptive verification, validates marker+count100, records prompt
  and exact request, then checks byte-identical output and actual offload
  load-byte counters after an engine restart. No live use yet. Intended flow:
  run first as LAST R5 inference, finish/restore R5; prepare R6 with cache
  reuse from R5 (identical core/source), run R6 --screen-repeats0 --hold,
  then durable reload, finish/restore. This checks first-store without a
  warm revisit and durable reload without changing the deployed identity.
- R4 long-context NVIDIA energy/token K1/K7:100k code0.614/prose0.614;
  170k code0.596/prose0.577. Single repeats, device energy only, not wall.

2026-09-09 01:12 UTC checkpoint:
- R5 controller97314 is now HOLDING; deadline about03:10UTC. All four
  counting outputs are identical60tokenIDs. All ranks dispatch FULL graphs
  at M2/4/6/8, fixed draft capacity7; capture1.61..1.66GiB. No pressure/OOM.
- 24 fixed controls complete. R5/R4 cycle-time ratios are1.00345/1.00355/
  1.00173/1.00077 for K1/3/5/7. Frozen R4 cost curve retained unchanged.
- CURRENT inference session21934: dev-adaptive-r5,12 development prompts,
  fixed7 vs adaptive V3,256tokens,one repeat,paired AB/BA order. No other
  inference until it completes. Then collect/analyze traces and power; if
  promising run locked heldout30x3 with the same untouched source/costs.
- Power32625 samples until about02:54UTC. 352 local tests pass. No core
  or locked benchmark source changes since the V3 evaluation lock.
- R6 durability helper additions remain local only, not installed on R5.
  Use unique first-store prompt as the LAST R5 inference, then restore;
  prepare R6 --reuse-cache-from adaptive-v3-20260909-r5, run with
  --screen-repeats0 --hold, and do the durable reload probe. Source salt
  comparison must pass; do not weaken it. Both runs restore originals.

2026-09-09 02:52 UTC checkpoint (supersedes prior live notes):
- Interruption revalidated: all R5 processes ended; exact original containers
  running. R5 restored.json was written at02:39; verify-restore passed again.
- V3 API nested-table request failed HTTP400 before producing tokens. The
  actual chat vllm_xargs schema permits scalars/flat lists, not nested dicts.
  V3 flat-field development completed: prose ratio1.12214 CI[1.07292,1.17558];
  coding1.00541 CI[0.94538,1.05140], one code_cache case-13.1%. Promising
  prose, unresolved coding noninferiority. No held-out inference yet.
- LOCAL AND STAGED policy VERSION4: same V3 decision rules; context table
  encoded as JSON string and parsed once/request, bounded16KiB. Bad/deep/
  oversized JSON falls back. Pinned API annotation+source hash saved; exact
  Pydantic2.13.5 installed in VM; schema rejects old dict and accepts string.
  Full suite360passed. No further core changes after V4 evaluation lock.
- Fresh R6 adaptive-v4-20260909-r6 prepared on all ranks. Its evaluation lock
  retains original frozen curve and30held-out prompts3repeats256tokens.
  New run uses --hold --hold-minutes120 --screen-repeats1 (count4caps and
  12fixed calibration controls; previous deployment already had24controls).
- A new7200s NVIDIA power sampler starts with R6. Record actual sessions
  from latest tool outputs; do not send external inference until HOLD.
- Next: verify FULLM2/4/6/8 andV4 trace, collect/fit R6 cost stability; run
  two-case API smoke(code_queue,prose_incident) fixed7/adaptive with curve,
  then full frozen held-out30x3. Afterward thinking/code/integration/long
  checks and unique100k once-store as LAST inference. Restore R6, prepare
  R7 with --reuse-cache-from adaptive-v4-20260909-r6 and same untouched
  core/source salt, --screen-repeats0 --hold; durable reload, then restore.
- R5 cache CANNOT be reused for V4: source salt changed. Durability first
  must be generated on R6 and reloaded on identical-source R7.

2026-09-09 03:04 UTC LIVE checkpoint:
- Current session97314 is R6 (this numeric session ID was reused; it is NOT
  the old R5 controller). Label adaptive-v4-20260909-r6, --screen-repeats1,
  --hold --hold-minutes120. Healthy after519s, countsmokes identical60IDs,
  FULL M2/4/6/8 confirmed all ranks. Initial12fixed controls nearly complete.
  No external inference before HOLD. Current power session90673 started
  about02:51UTC for7200s, ends about04:51UTC. No other inference active.
- V4 frozen evaluator sources unchanged. Full360tests passed before two
  deployment trace-control tests and one C2 accounting test were added;
  all new focused tests pass. No core or locked benchmark change afterward.
- NEW local-only helper controls for R7: prepare --trace-off writes a flag
  consumed by node launch, leaving policy=shadow, same graph allocation and
  source salt. R7 can both reload R6's identical cache and measure telemetry
  overhead. Core/source unchanged, no cache identity bypass.
- bench/spec-overhead.json freezes3 predictable numeric prompts. Run onR6
  with --caps7 --tokens768 --repeats3, then identical requests on trace-off
  R7. Compare decode intervals, identical outputs/acceptance and clocks;
  report uncertainty, do not infer <1% overhead just from the CPU microbench.
- bench/spec_concurrency.py added: guarded2simultaneous streams, fixed vs
  adaptive,3repeats, measures only shared decode interval (excludes prefill
  and C1tails), validates complete180-number outputs. One local accounting
  test passes. Intended R6 check after heldout, alongside existing integration.
- Completion audit created in COMPLETION-AUDIT.md with unresolved gates.
  No wall-power exporter in running Spark containers; no source found in
  related experiment repos either. Wall energy remains unmeasured.
- Next: HOLD; collect/fit R6 cost stability; API smoke code_queue and
  prose_incident, fixed7/adaptive with frozen curve; verify V4 costs/prior
  and actual adaptive caps; then full locked heldout30x3. Later: reasoning,
  complete-code, integration, C2, adaptive repo contexts, overhead3x3, then
  unique100k once-store as LAST R6 inference. Restore R6; fresh R7 prepare
  --reuse-cache-from adaptive-v4-20260909-r6 --trace-off, run --hold
  --screen-repeats0; durable reload and overhead controls; final restore.

2026-09-09 03:07 UTC:
R6 controller97314 HOLD deadline about05:05UTC; power90673 ends about04:51.
API smoke COMPLETE: encoded curve admitted, version4 on all events, costs
present, adaptive K3 on82/97 code_queue steps and95/110 prose_incident;
remainingstepsK7 warmup/probes,0drops/mismatches. Fixed controls costratios
within0.22% of frozenR4. No calibration/controller/corpus change.
Starting heldout-adaptive-r6:30frozen prompts,3repeats,256tokens,paired
fixed7/adaptive. No other inference until that new session completes.
Then collect and run evaluate_spec_gate.py with R6 evaluation-lock.json,
tracesR6 and powerR6. Do not tune on these held-out results.

Held-out live inference session is7807, started03:07UTC. Controller97314
(R6) holds until~05:05UTC; power90673 until~04:51UTC. No other inference.
A local prose-blinding helper is ready: after first repeat's15prose pairs
exist, run spec_blind_quality.py heldout-adaptive-r6 --out <newqualitydir>.
Review blind-pairs.json and write preferences before opening policy-map.json.
Review is limited to coherence/instruction adherence of256token openings;
complete-code checks and correctness fixtures are separate evidence.
