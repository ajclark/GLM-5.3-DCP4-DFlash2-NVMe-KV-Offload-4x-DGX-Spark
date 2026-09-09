# Adaptive speculation completion audit

2026-09-09, experimental phase complete. Broad promotion is withheld. Requirements
come from `docs/ADAPTIVE-SPECULATION-PLAN.md` and the active objective.

| Requirement | Current evidence | Result / limitations |
|---|---|---|
| Inventory exact running runner, scheduler, sampler and source | `inventory/`, pinned runtime fixtures; actual V2 DFlash2 identified | Proven for experimental lane |
| Preserve GLM target, TP4/DCP2, 180224 window, maxseq12 and KV6GB/rank | Launch arguments, node preparation assertions, running logs | Verified in final original-restoration-audit.json: exact original containers/config/mounts/source and generation |
| Keep trained draft block8/capacity7 while verifying1/3/5/7 | Runtime input/rejection fixtures and live graph logs | Proven through R7 on all four ranks |
| Real small FULL graphs with no padding to8 | Four-rank M2/4/6/8 dispatch logs, graph-manager tests | Proven through R7 on all four ranks; graph-shape-audit.json |
| Extra graph memory and safe normal rollout pressure | Capture1.61–1.66GiB, per-rank /proc guards, no experimental OOM | R6 min1641MiB head/3098+others;2.60MiB head swap-out,fullPSIavg10=0,0OOM/trips; R7 min1793MiBhead/3168+others,8KiB head swap-out,0OOM/trips |
| All rejection boundaries and fixed draft stride | `test_spec_runtime.py`, actual Triton CPU kernels | Proven in tested fixtures |
| Two queued steps, rollback, positions and request incarnation | Runtime transition tests and stale-feedback tests | Proven in tested fixtures |
| DCP causal/local bounds at widths2/4/6/8 | `test_dcp_spec_decode_verify.py` | Proven in tested fixtures |
| C1→C2→C1, unsupported sampling and cancellation | R3 integration trace1→7→1 and count checks | R6 transition/fallback/cancel checks pass; 3-repeat C2 overlap ratio1.00018, all counts correct |
| Censoring, probes, hysteresis and missing-data fallback | Conditional risk-set, prior, curve and stale-feedback tests | Proven locally; all30154 R6 verification events pass runtime integrity audit |
| Bounded telemetry, no hot-path I/O/synchronization | Queue tests, trace drops0, writer blocks when idle | Measured nine exact pairs:central+0.56% time,95%-0.11–1.81%; upper1%budget not established; trace defaults off |
| Fixed-cap measurement before adaptation | R3/R4 tables and frozen five-point calibration | Complete |
| 12 development prompts separated from30 held-out | Frozen corpora and evaluation locks | Complete separation; locked R6 evaluation complete |
| ≥3 balanced paired repeats and prompt-cluster bootstrap | Driver and gate evaluator tests | Complete:90 matched pairs,3 balanced repeats,30 prompts |
| Prose≥8%, positive95% interval | R6 held-out+15.80%, interval13.24–18.73% | Passed |
| Coding lower95% ratio≥0.98 | R6 held-out+0.99%, interval-2.11–4.44% | Initial gate narrowly missed; declared6-repeat combined follow-up+1.14%,95%-1.30–3.64%, meets margin but exploratory |
| TTFT and p95 emission gaps≤5% regression | R6 worst matched-case median ratios:TTFT1.0127,p95gap1.0027 | Passed |
| Fresh/edit/continuation/tool subsets and aggregate tok/sec | R6 gate-report.json includes all four coding subsets and aggregate rates | Complete for initial locked run |
| Thinking-off and actual coding reasoning configuration | Thinking flag retains configured high effort | R6 six reasoning-mode controls completed;256-token openings only |
| Executable code and prose quality | R3 four complete functions passed under fixed caps | R6 complete-function checks pass for fixed7/adaptive; blind first-repeat prose:10 ties,adaptive3,fixed2 |
| Short/32k/100k contexts, cold versus warm labeled | R3 random context checks; R4 repo4k/32k/100k/170k | R6 screens complete; R7 three-repeat32k follow-up code+7.2%/prose+24.1%; R7 three-repeat4k follow-up code+4.9%/prose+15.3%; initial prose regressions did not repeat; R7 random32k/100k fixed7/fixed3/adaptive complete-marker/count checks passed |
| Once-store, eviction, engine restart, durable reload | Local slab crash/powerloss tests; R4 offload load counters | Passed R6→R7:102016external cached tokens,0local hits,11.06GB loaded,all127continuation IDs identical; prompt-only policy preserved; actual runtime explicitly skips generated-KV disk stores (`adaptive-v4-20260909-r6/cache-policy.json`) |
| Copy proposals only when evidence supports integration | Full-window exact CPU index; 170k VM build~0.266s, update+lookup~3.2us; held-out first-pass continuation/tool matches exceed fresh-code matches | All-repeat screen complete; no GPU integration justified for this selector (copy-branch-decision.json) |
| Active whole-cluster wall J/token | NVIDIA-reported device energy available; no wall source found in repo, related experiment repos or running node containers | Unmeasured; device energy cannot satisfy this gate |
| Idle power considered separately | Telemetry idle wakeup removed; NIC cycling remains paused | No measured hardware idle-watt improvement claimed |
| Safe rollback, isolated caches, preserved cache identity | Exact original IDs restored repeatedly; reuse guard checks full source salt | R6/R7 exact originals restored and generation verified; all controllers/samplers stopped |
| Final report, reproducible commands, conservative default | README/plan/bench tools present; policy defaults off | Complete:README,completion-decision.json,reproducible drivers,defaultoff |

The R6 policy changes only the API encoding from R5: a bounded JSON string
replaces an API-rejected nested object. Acceptance rules and calibration are
unchanged. R6 passed the pinned API smoke and fixed-cap counting controls before held-out inference. Its frozen 30-prompt, three-repeat evaluation is complete; the declared coding extension is complete (90 total coding pairs). The adaptive code and calibration have not changed during evaluation.

Implementation, available measurement, copy-branch decision and final rollback are complete.
Wall-energy promotion remains unmeasured and the strict tracing-overhead upper bound
is unresolved; neither is represented as a passing gate. No experiment is left running.
