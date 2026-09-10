# Harness study revival: optimize checked task completion, explain the token mix

2026-09-10. Consultation only: local sources read; no cluster access, model calls, configuration changes,
training, or commits. This document is the only new artifact. `E` below means
`/home/napta2k/spark-cluster-experiments`; `F` means `/home/napta2k/lmcache-mg/spark-src/vllm`; unprefixed
paths are this repository. Read: E/{HARNESS-COMPARISON-20260901,PLAN-HARNESS-PROMPT,PLAN-OPENCODE,
PLAN-OPENCODE-50,MVP-HARNESS-20260902,PI-PARITY-20260902}.md and their harness/proxy/attribution code.

Code shorthand: `adaptive.py` = `overlay/vllm/v1/spec_decode/adaptive.py`;
`sample/` and `spec_decode/` = subdirectories of `overlay/vllm/v1/worker/gpu/`;
`serving.py` = `F/entrypoints/openai/chat_completion/serving.py`;
`pi-speculation-core.mjs` = `extensions/pi-speculation-core.mjs`.

## 1. Starting point: findings through September 10

| Evidence | Result | What it establishes |
|---|---|---|
| Sep 1, four tasks, matched T=1/max | pi 3.97 emitted/cycle, 27.2 tok/s, 2,130 tokens, 6 calls, 95 s/task; opencode 5.10, 31.3, 702 tokens, 4.5 calls, 23 s | A real harness/loop difference; one repeat, tiny tasks |
| Sep 1, optimized opencode, eight tasks ×3, high/T=0 | 6.13 emitted/cycle, 41.6 tok/s, 422 tokens, 4 calls, 27 s; 21/21 checks | Combined prompt/tool/sampling treatment, not a controlled comparison with pi at max |
| Sep 2, tuned pi, native ×3 | 5.21, 35.5 tok/s, 274 tokens, 3.9 calls, 15 s; 21/21 checks | Wall-time parity already reached versus native opencode's 16 s |
| Sep 2, MVP relative → absolute paths | 37.9 →42.6 tok/s; 268 →489 tokens; 10 →16 s/task | Predictable path padding improves tok/s while making work slower |
| Sep 2, opencode MVP-style tools | 17 →12 s/task, 471 →283 tokens; 7/7 suite and 10/10 HE checks | Useful schema/result-format gains can lower reported tok/s |

Sources: E/HARNESS-COMPARISON-20260901.md “Method/Results”; E/PLAN-OPENCODE-50.md §7; E/PI-PARITY-20260902.md
“Flagless validation”; E/MVP-HARNESS-20260902.md “Absolute-path variant/Backport”. The eight-task suite had
seven scored tasks: never relabel 21/21 check executions as 24 independently checked tasks. The proxy/native
comparison also reported ~10 s/task overhead; cache state and transport were not fully isolated, so reproduce
that difference before attributing it.

Latest evidence takes precedence over older projections: [speed status](../SPEED-LEVERS-STATUS.md) and
[architecture
§3b](../SPEED-ARCHITECTURE-OPTIONS.md#3b-measured-on-2026-09-10-guarded-holds-production-restored-after-each).
DFlash lossy m2.5: held-out prose +19.7% [17.0,22.2], pooled 15.93 →19.13 tok/s; opening-length blind screen
only, full quality gates pending. MTP K2: prose +20–33%, structured output −30–50%; lossy did not materially
stack with MTP. Chunk4096 and NCCL QPs=2 did not solve prefill; current 2048 stays. LSE fold was
neutral/harmful. Production `lossy-prod` mounts the overlays with **GLM_SPEC_LOSSY=0** (status log 02:22,
02:30, 05:25); this is recorded state, not a fresh live inspection. The launcher already sets effort **high**
(`launch-glm53big-dcp.sh:359`): “defaults to max” describes the old template/harness failure mode, not today's
guaranteed effective setting. Capture pi's explicit overrides before changing anything.

Define `a` = accepted **draft** tokens (0..7), `e=a+1` = emitted/cycle on complete nonterminal K7 cycles. Old
reports called `e` “accepted/cycle” (`E/call_attrib.py:74–79`); new trace `accepted` means `a`
(`overlay/vllm/v1/spec_decode/adaptive.py:475–513`). Always report both. Primary objective: checked task wall
time and success rate. Diagnostics: `pooled_decode = Σcompletion_tokens / Σdecode_seconds`, `e =
1+Σa/Σcycles`; do not average call tok/s or silently count a bonus at truncated boundaries. Use `Ttask =
Tcritical_decode + Tcritical_prefill/queue + Ttools/client`, with overlap accounted for; reducing tokens or
calls can beat increasing decode rate.

## 2. First avenue: per-call pi attribution without perturbing production

**Mechanism / effort:** 1–2 days to adapt the existing capture/report path, then a small labeled pilot. Keep
current endpoint/model/policy fixed. Native pi observation already captures request settings, delta
kinds/times and usage (`bench/pi_throughput/observe.ts:11–32`); the speculation extension creates a
`spec_label` and local request record (`extensions/pi-speculation.ts:43–79`, `pi-speculation-core.mjs:44–54`).
Generalize identifiers to run/task/session/call/retry with collision-free labels <=96 characters; capture
side calls separately. The extension's single `current` slot is insufficient for overlapping side requests.

Revive `E/capture_proxy.py` against the configured production URL as a **pass-through** client proxy, with no
forced kwargs in the observational arm. Its actual code logs request summaries and response character
composition, not full token spans (`:137–174,213–254`); `--dump-dir` was a plan, not an implemented option.
Replace synchronous per-call metrics scrapes (`:177–210`) with labeled response usage/timestamps for routine
collection; retain a quiet-C1 metrics validation pass. Timestamp upstream first token, downstream first token,
first visible token, last token, HTTP completion, tool start/end, agent-settled and optional exit work.
Reasoning-first TTFT and visible-answer latency are different measures. Calibrate native vs lightweight proxy
on the same requests/cache stratum: require <1% wall overhead or report native wall and use separate proxy
attribution runs.

Join `spec_label` to existing verify traces, retaining request hash, step, cap, scheduled_k, accepted,
sampled, terminal, learned, cycle_ms, latency_ms, lossy_margin/min_p/enabled, relaxed, writer_error/dropped
(`adaptive.py:431–440,496–517`). Fail joins on missing/duplicate steps; never join only by a broad wall-clock
window or multiply scheduler records across TP ranks. `cycle_ms` is scheduler receipt spacing, not isolated
kernel duration. Inspect actual mode/cap so fixed7 and adaptive calls cannot be accidentally pooled. Trace
availability is conditional: the sink exists only with `GLM_SPEC_TRACE` (`adaptive.py:354–362`; launcher
`:308`). Mounting code does not enable a sink. Use an existing operator-exported trace if available; if
absent, the production pilot can report call metrics now, but exact cycle attribution needs telemetry enabled
at a separately coordinated boot. Do not claim an HTTP call enables it.

**Category accounting:** reasoning / visible prose / tool-call arguments, plus delimiters/unknown; split tools
into read paths, bash commands, edit-old text, edit-new text, write bodies and JSON scaffolding. Use raw
emitted IDs with a local pinned tokenizer/parser, not `characters / 4`. The API supports `return_token_ids`
(`F/entrypoints/openai/chat_completion/protocol.py:368–375`) and emits IDs even when text parsing buffers a
delta (`serving.py:639–686`). Validate stream-ID completeness, parser delimiters and usage reconciliation on
the pilot; if IDs are absent, preserve character shares as character shares. SSE frames may contain multiple
tokens/cycles: arrival time cannot identify which accepted tokens were reasoning. Align raw token offsets to
ordered emitted-cycle lengths, handling prefill's first token and censoring gaps/terminal cycles. A fully
aligned pure-category cycle yields exact category `a/e`; mark transition cycles mixed and disclose coverage
rather than assigning their whole cost to one span. More exact transition accounting needs output
offsets/category/relaxed-position masks in server telemetry, a separate overlay experiment.

`E/call_attrib.py:81–115` uses character-share **unweighted linear regression** of emitted/cycle; that is an
indicative mixture fit, not measured per-category token acceptance (and rates do not mix linearly in token
shares). Keep it only as a historical comparison; prefer pure cycles and, secondarily, fit seconds/token
against token shares with task-clustered uncertainty. Current scalar `relaxed` cannot locate relaxed positions
(`bench/spec_request_report.py:142–172`). Report relaxed/cycle, category exposure, and following-relaxed
versus following-exact acceptance without pretending the scalar supplies a span mask. **Decision/value:** a
trustworthy time budget, not an immediate tok/s gain; require >=95% labeled call coverage and publish
unresolved token/cycle coverage separately. Record explicit margin+enabled even if a small treatment call has
zero relaxed accepts; that call shows no activation, not a quality failure. Demand nonzero relaxed on the
treatment corpus and zero on every exact control.

## 3. Second avenue: pi effort, prompt, side work and tool design

**Effort 1–3 days; production-request-only A/B.** Freeze the real daily pi config as A, including plugin
capabilities; compare one factor at a time on development tasks, then combine winners. Native pi explicitly
sends effort; historical parity already changed it to high. Log top-level reasoning_effort,
chat_template_kwargs, pi thinking level, temperature and output cap after all extensions/proxy edits.

| Factor / mechanism | Measurement that decides it | Plausible value / historical bound |
|---|---|---|
| Explicit high vs accidental max; max as hard-task control, low/off as diagnostic | Official tests, hard-task success, reasoning tokens/time, truncation and wall p90 | Old max spirals were large; no assumed gain if pi is already high. Low/off did not beat high on tuned coding |
| Terse action-first prompt; short useful final message | Tokens and wall at equal required explanations/checks | Historical pi 59 →15 s combined configuration; not a new 4× promise. Target >=10% wall gain on a modern corpus |
| Remove optional title/exit-summary calls in the benchmark arm | Count/tag every side call; interactive-settled and background-complete wall separately | opencode title removal 4.9 →3.8 calls, +2.8 tok/s; pi exit summary was 45% of tokens on tiny tasks |
| Batch independent reads; concise complete tool results; no redundant rereads | Calls, new prompt tokens, missed dependencies, tool latency | One avoided call saves its measured TTFT+generation+tool overhead; 2 calls ×1–2 s TTFT =2–4 s before token savings |
| Path first, old text before new; raw read text; edit/write “ok” | Useful edit token acceptance and official diff checks, total emitted bytes | MVP edit 6.83 vs pi 5.70; prior ~6% decode hypothesis, stronger wall evidence from compact tools |
| Targeted edits vs whole-file writes; minimal sufficient tool set | New code vs copied old text, calls, output tokens, wall, check pass | Whole writes were neutral on small tasks; large rewrites can cost more despite higher acceptance |

Sources: E/PLAN-HARNESS-PROMPT.md §1–5; E/PLAN-OPENCODE.md §1 and results; E/PLAN-OPENCODE-50.md §7;
E/PI-PARITY-20260902.md stages A–C; E/MVP-HARNESS-20260902.md. Do not revive failed “think in two sentences”
or work-procedure prose as defaults: both increased reasoning. Do not pad absolute paths to win tok/s. Compare
relative paths and schema order separately. Keep the daily memory/tools arm: Sep 2's four-tool pin/summary
suppression became **PI_BENCH-only**, after hiding useful interactive tools. Removing required memory or
verification is a capability change, not a free optimization. Evaluate memory/session tasks before adopting
it.

## 4. Third avenue: harness-aware verification and drafter routing

**A. Prose-labeled request default (1–2 days client/routing work after quality gate).** Use
`spec_workload=prose` as a candidate routing feature; send explicit `spec_lossy_margin=2.5, spec_lossy_rank=2,
spec_lossy_min_p=0` only for an explicitly opted-in prose persona/call. Require greedy, no
tools/tool_followup/structured output; mixed, unknown, code and explicit opt-out stay exact. This conservative
allowlist is proposed behavior, not today's extension: the classifier is a weak last-user-intent heuristic
(`pi-speculation-core.mjs:9–24`), and current lossy controls are independent of that label (`:49–53`). A
“prose” request can still emit code or tools. The server's structured_outputs guard does not prove the absence
of ordinary parser-based tool calls (`sample/states.py:37–62`). A client allowlist needs no new kernel once
the boot switch is on; a server default keyed on the label needs explicit opt-out precedence, trusted caller
scope, audit fields, classifier false-positive tests and a reviewed server change. **Decider/value:** G3/G4/G6
prose quality plus official task tests, no relaxed code/tool rows, matched call speed. The measured +19.7%
applies to eligible prose calls, not entire pi tasks: `speedup=1/(1-f+f/1.197)` for affected decode-time share
f. f=10% gives +1.7% pooled decode; on tuned ~2–3% prose streams, likely <1% task gain. More useful for
genuinely prose-heavy sessions.

**B. Think-span-only relaxation (about 1 week + guarded kernel/state tests).** Reasoning is a plausible
low-acceptance target: the old opencode fit was 25% of response characters, ~3.03 emitted/cycle versus tools
~6.11, not a current pi measurement. Treat the often-quoted “25% of tokens” as a hypothesis to verify.
Implement an explicit request opt-in and token-level state machine for the actual template's reasoning
delimiters, including a prefilled open span. Existing margin is per request
(`sample/states.py:80–85,113–116`), read once by the verifier
(`spec_decode/rejection_sampler_utils.py:260–261,300–346`); proxy text inspection cannot change verification
inside an in-flight call. Use a per-position eligibility mask/state update over the **accepted prefix**;
delimiter/tool/answer tokens verify exactly. A rejected proposed closing delimiter must not mutate the
committed state, and a cycle crossing `</think>` must not relax subsequent answer positions. Preserve request
incarnation, async scheduling, cancellation, mixed batches and all-rank decisions; test malformed/nested
markers, stops, tokenized partial delimiters and rejected boundary candidates. Trace entry/ exit state plus
per-position relaxed masks to prove zero relaxation outside think. **Decider/value:** official hard-task
correctness, tool validity, runaway/repetition rate and wall; hidden reasoning influences visible answers, so
“not user-visible” is not a quality exemption or visible-output identity guarantee. If reasoning were 25% of
tokens at half tool speed it would consume ~35–40% of decode time: applying a hypothetical 1.197× reasoning
gain gives only +6–7% pooled decode (~33.4 →35.6 tok/s). On tuned pi's much smaller reasoning share expect
less. Transfer of the prose gain to reasoning is unmeasured; reject if retries or reasoning length grow.

**C. MTP routing (existing boot lane; 1 day evaluation, multi-drafter routing larger).** Keep MTP K2 for whole
prose sessions: measured ~105 ms cycles, max 3 emitted; code 27.5 vs DFlash 40.3 tok/s. A label cannot switch
draft model inside the current DFlash server. Resident MTP+DFlash routing needs new scheduler/cache plumbing
and reduced KV admission; session-level boot comparison requires a hold. Do not stack advertised prose gains:
hold 6 measured MTP+lossy ~flat. Gate on task-mix wall and memory/context limits, not prose speed alone
(status 03:15–04:45).

## 5. Fourth avenue: a bounded exact-copy branch for real tool arguments

**Screen only, 2–3 days CPU; integration 1–2 weeks only if it passes.** Structured spans merit a narrower
hypothesis, not reopening generic n-gram speculation: old tool shares ranged 60–88% (90–94% after tuning), but
copied paths/JSON keys often already draft well. Novel code and shell commands may not repeat at all. [Earlier
decision](../../results/adaptive-spec/copy-branch-decision.json): code 145/3019 matching boundaries, tool
subset 69/576 (~12%), 287 agreed copied tokens; edited code only 29/1151. Repaired long-context screen had
4/86 and 1/108 code matches ([architecture
screens](ARCHITECTURE-SCREENS.md#exact-copy-reuse-current-opportunity-is-sparse)). These are
baseline-trajectory opportunities, not incremental accepted tokens.

On newly recorded **study-task** pi traffic, replay actual rendered prompt and committed output IDs at each K7
anchor. Separate scaffold/path, edit-old, edit-new, write and bash. Compare most-recent suffix 16/8/4
(original control) against a bounded prompt-lookup index favoring repeated tool spans/unique matches; test
JSON escaping with actual emitted IDs, not decoded substring matches. Fix source selection before inspecting
future output; no oracle best-match selection, no cross-request leakage. Use held-out tasks/repos, no training
or hidden-state corpus. Report hit coverage, agreed-prefix length, false hits, **incremental** accepted prefix
over DFlash at the same anchor, lookup p95/99 and index memory at 180k. Trace accepted length plus exact
output alignment supplies the DFlash baseline; without that join, only an opportunity screen is possible. Keep
changed-trajectory claims for live validation. Existing plan explains proposal ownership and re-entry
(`docs/ADAPTIVE-SPECULATION-PLAN.md:269–309`; [K0 feasibility](K0-FEASIBILITY.md)).

Predeclare a screen gate: estimated held-out end-to-end gain >=3% after charged lookup cost and task-bootstrap
lower bound >0, including misses; otherwise stop. At e=6.1, +0.2 genuinely additional tokens/cycle is only
+3.3% before overhead; at 12% hit rate even +1 additional token/hit yields ~2%. Charge lookup on every cycle,
not only hits. Report deployment cost separately from this estimate. First live design chooses one full K7
proposal source while still running DFlash to maintain context/cache state; no claim of saving its ~10.9 ms.
Reject stale anchors; exact target verification remains mandatory. Only a later tested bypass can save draft
work, with catch-up/re-entry and async ownership resolved. Such a branch needs a guarded hold. Do not increase
copied padding to manufacture hits.

## 6. Fifth avenue: TTFT, stable prefixes and prefill glue

**Client work 1–3 days; glue overlay several days plus hold.** Measure each call's prompt P, cached C,
uncached U=P−C, TTFT, cache tier when available, queue time, tool-result length and rendered-token longest
common prefix with the preceding request. Keep system/tool schemas stable and messages append-only where
semantics permit; stabilize metadata ordering, avoid reinserting mutable memory near the front, and cap
irrelevant tool output without truncating required evidence. Append-only JSON alone does not prove prefix
reuse: reasoning removal, tool serialization, chat-template rewriting, compaction and eviction can break it.

Use `stream_options.include_usage=true` and `usage.prompt_tokens_details.cached_tokens` if the server exposes
it (`F/.../chat_completion/serving.py:92–105,754–766`). That detail requires a server option; absence is
**unknown**, not zero. Report per-call C/P, pooled ΣC/ΣP and fraction of calls with C>0 separately. The old
proxy's prefix_hits/queries is a global-counter ratio (`E/pi_coding_bench.py:98–105`), usable per call only in
uncontended intervals. A token-LCP estimate bounds reuse opportunity, not actual cache hits; disk hits can
still incur TTFT. Enabling absent server details/trace must wait for a coordinated boot, not be assumed
available.

At the measured **460–500 uncached tok/s**, every additional 1,000 uncached tokens costs ~2.0–2.2 s; six calls
adding 1k each imply ~12–13 s prefill/task. Re-prefilling 10k avoidably costs ~20–22 s; a cold 100k costs
~200–217 s. These omit queue/cache load and context-dependent attention. Reducing calls/prefix invalidation is
more valuable than polishing a subsecond client runtime. Decide using U and TTFT per call, not system-prompt
character count alone. Latest glue estimate is **6–7% prefill**, not the earlier optimistic 15% from nested
trace attribution: ~0.5–0.6 s per observed 8.5 s/4158-token prefill. If prefill is 15–30% of task wall, that
is ~1–2% task saving, plus an unmeasured ~1–2 ms/cycle decode hypothesis. Targets: reduce-scatter layout copy
and DCP compaction/host-sync fusion (status 03:00,04:16; architecture §3b). Keep this separate from client
A/B; chunk4096, QPs=2 and unfused LSE fold already failed.

## 7. Evaluation protocol and production/hold boundary

Build 24 real repository tasks: 16 development, 8 unseen held-out; bug repair, multi-file refactor, API/CLI
change, test-driven implementation, structured edits, shell/build repair, prose/documentation with executable
examples, and memory/ session continuity. At least eight hard tasks, stratified across both splits. Pin repo
revisions, input state, dependency versions and original official test commands; retain graders outside the
editable workspace. Add fixed HumanEval official tests as a continuity panel, not a substitute for agentic
work. The existing evaluator executes `problem['test']` and entry-point checks separately
(`bench/pi_humaneval.py:56–72,186–198`); legacy runner checks are merely configured shell commands
(`E/harness_bench.py:79–83`), so audit their provenance/coverage.

Screen one factor on development once; freeze two finalist configurations before 8 held-out tasks ×3 repeats
×2 arms (48 runs), AB/BA alternated within task/repeat. Fresh identical workspaces and pinned local harness
versions; same required tools, sampling, budgets and test access except the named factor. Record failures,
timeouts, checks and retries; never silently drop a failed run from the speed panel. Compare daily pi, tuned
pi and opencode/MVP-style reference as separate contrasts; do not compare today's stack to old absolute rates.
Reuse the runner's fixture reset/task labels (`E/harness_bench.py:54–76`), but randomize arm order and fix the
workspace path length. Match cold-first-call and warm-follow-up strata; no global cache reset on production.
Use balanced order and measured cache hits rather than warming only the second arm. Runtime-generated labels
stay outside prompt text.

Report pooled decode, p50/p90 task wall, calls/task incl. side work, generated/new code/copied tokens, TTFT
and C/P per call, category a/e and exposure, relaxed/cycle, correctness overall/hard strata and capability
regressions. Count reasoning once inside completion usage. Report prompt-time and tool/client critical-path
shares. Bootstrap paired task log ratios, clustered by task (10k resamples); promotion target >=10% median
wall reduction, 95% lower bound >0, no observed official-check or hard-task regression. Eight held-out tasks
detect only large regressions; expand before defaulting lossy reasoning or dropping daily capabilities.

**Production, no hold:** C1 ordinary requests, native/proxy observation, prompt/ effort/tool-schema ablations,
local official tests and offline copy screens. Wait when the endpoint is busy and stop issuing study work when
user traffic arrives; never cancel user requests. Global `/metrics` deltas are contaminated by overlapping
traffic or side calls: discard those attribution intervals, retaining labeled wall/ usage.
`E/harness_bench_par.py:76–97` uses stage-wide deltas and a hard-coded 145 ms C1 estimate; do not reuse that
estimate or its “batch-independent” assumption. The pilot is not executed by this consultation. Current
switch-off production can only supply exact controls: lossy requests remain inert until a coordinated
`GLM_SPEC_LOSSY=1` production rollout. After that rollout, explicit per-request lossy quality pairs need no
hold. **Needs a guarded hold/boot:** new think-span kernel/state/telemetry, server label-default policy, copy
proposal branch, MTP lane/multi-drafter routing, prefill glue, or currently absent server telemetry. One
orchestrator owns changes; do not overlap a rollout with a hold (status 02:12).

## 8. Ranked continuation

1. **Instrument current pi and establish the native wall/correctness baseline** (1–2 days): resolve category, cache and side-call attribution before claiming a new lever.
2. **Recover useful pi prompt/effort/side-call/schema gains** (1–3 days): historically 59 →15 s; modern target >=10% wall improvement, preserving daily capabilities and avoiding filler.
3. **Repair client prefix stability and excessive tool-result/call volume** (1–3 days): ~2 s per 1k avoided uncached tokens; likely larger task benefit than another few percent decode.
4. **Complete prose-lossy quality gates, then opt-in prose routing** (1–2 days plus generation): measured +19.7% eligible-call speed; little whole-task upside when prose is already sparse.
5. **Screen exact copying on real study tool streams, offline only** (2–3 days): cheap falsification; integrate only above the incremental 3% gate, not because tool share is high.
6. **Think-span-only lossy experiment if reasoning still dominates** (~1 week + hold): conditional +6–7% decode on the old mix; downstream correctness decides it.
7. **Prefill glue overlay and dedicated MTP prose sessions** (separate holds): ~6–7% prefill estimate / +20–33% prose measured; neither is a general pi code-speed replacement. Training remains declined.
