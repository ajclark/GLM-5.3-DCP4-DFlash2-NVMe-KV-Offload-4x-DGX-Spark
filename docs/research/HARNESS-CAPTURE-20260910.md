# Pi capture, 2026-09-10: attribution and prompt-lookup screen

Status: attribution measured; exact-token copy screen **pending a confirmed no-hold window**.
A CPU-only `/health` probe returned HTTP 200; health does not establish absence of a guarded hold.
No `/tokenize` or generation call was made for this report. Do not read an unmeasured screen as a negative result.

## Snapshot and provenance

The completed local snapshot contains **67 calls, 12 task identities ×2 passes, 24 task sessions**;
run1 has 36 calls, run2 has 31. This differs from the approximate 80-call input estimate.
Input: private `results/harness-capture-20260910/proxy_calls.jsonl`, 596399 bytes,
SHA256 `992806d832e6eff61cc71793352ff402379cfa64161663ebd0d28109796b8693`.
The source grew during implementation; all numbers below use this single 67-record snapshot.
Runner artifacts supply task timing and official checks: **23/24 passed**. No official tests were rerun here.

Aggregate artifacts: [attribution JSON](../../results/harness-capture-20260910/call-attribution.json),
[per-task/per-call tables](../../results/harness-capture-20260910/call-attribution.md),
[copy-screen status](../../results/harness-capture-20260910/copy-screen.json).
Only numeric aggregates and pseudonymous task/session IDs leave the private inputs. The requested
`proxy_*.jsonl` and `run*/` paths are gitignored. No raw texts or token IDs are written by either analyzer.
Sources: `bench/harness_capture.py:24,92`; runner timestamps/checks `bench/pi_humaneval.py:189–198`.
Here `E/` means `~/spark-cluster-experiments/`.

## Attribution: report the length-limited call separately

| Cohort | Calls | Generation tokens | Emitted/cycle | Pooled decode tok/s | Call wall s | Prefill s | Decode s |
|---|---:|---:|---:|---:|---:|---:|---:|
| All | 67 | 40501 | 7.142 | 43.08 | 1010.266 | 64.578 | 940.219 |
| Not length-limited | 66 | 7733 | 5.518 | 35.02 | 289.923 | 63.667 | 220.812 |
| Length-limited reasoning call | 1 | 32768 | 7.675 | 45.55 | 720.343 | 0.911 | 719.407 |
| Run1 | 36 | 5072 | 5.155 | 32.67 | 194.729 | 36.319 | 155.239 |
| Run2 | 31 | 35429 | 7.559 | 45.13 | 815.537 | 28.259 | 784.980 |

The long reasoning call contributes 80.9% of generation tokens and 71.3% of call wall.
Its high acceptance does not establish useful task progress. Removing it is a sensitivity analysis,
not the headline throughput result or an exclusion from correctness scoring. This capture cannot be
summarized as “reasoning always accepts at half the tool rate.” The old 09-01 mix was a different sample
(`docs/research/HARNESS-STUDY-REVIVAL.md:168–197`).

Tokenization is pending: the following **estimated token shares** allocate each call's usage tokens
in proportion to its parsed category character counts, following the original attribution convention.
They are not GLM tokenizer measurements. The completed `/tokenize` command below replaces character
weights with independently tokenized category lengths and normalizes them to the same usage total.

| Category | Estimated tokens, all | Estimated share, all | Estimated share, not length-limited | Call-mixture emitted/cycle proxy | Pure-category calls / emitted per cycle |
|---|---:|---:|---:|---:|---:|
| Reasoning | 34759.95 | 85.82% | 25.76% | 7.286 | 1 / 7.675 |
| Content | 92.00 | 0.23% | 1.19% | 2.029 | 23 / 2.029 |
| Tool arguments | 5649.05 | 13.95% | 73.05% | 6.548 | 40 / 6.789 |

“Emitted/cycle” means `1 + Σaccepted_draft_tokens / Σdrafts`, including the bonus token;
pooled decode is `Σgeneration_tokens / Σserver_decode_s`. Within-call acceptance by span is absent.
The mixture proxy allocates a call's cycles/accepts by its category fractions; it cannot identify the
reasoning portion of a mixed reasoning/tool call. Pure-category rows are directly pooled counters.
The weighted category cycle-cost regression is **unidentified**: insufficient independent mixtures
with at least five draft cycles, so it returns no invented category coefficients.
Short content-only calls retain the earlier `side/title` **heuristic** bucket; they are not proven title requests.
Sources: `E/call_attrib.py:30–41,81–115`; `bench/harness_call_attrib.py:15,49,93`.

## TTFT, cache reuse and task cost: avenue #5

| Call position | Calls | Mean TTFT s | Prompt-weighted cache hit ratio | Estimated uncached prompt tokens |
|---|---:|---:|---:|---:|
| First in task session | 24 | 1.004 | 55.30% | 11897.96 |
| Follow-up | 43 | 1.061 | 75.60% | 15364.42 |
| All | 67 | — | 69.57% | 27262.39 |

Total prompt tokens sent: **89598**; estimated cached: **62335.61**. Across 24 sessions:
2.79 calls, 3733.25 prompt tokens sent, 1135.93 estimated uncached tokens, and 2.69 s measured prefill
per task. Follow-ups alone account for 640.18 estimated uncached tokens/session. Each call's TTFT,
cache ratio, prompt/uncached estimates and order appears in the aggregate Markdown artifact.

All **43/43 within-session transitions preserve the previous message list as an exact prefix**;
43/43 also preserve the tools schema. Mean serialized-character common-prefix fraction is 99.979%
(the previous JSON array terminator prevents literal 100% on append). Thus the client's *captured*
prefix is stable; this is not a rendered-token LCP measurement or proof of full KV reuse.
Tool-result additions still need prefill; template serialization, removal of reasoning, block rounding,
cache eviction and cache loading can all change effective reuse. There is no evidence here for a client
that rewrites the front of every request. Sources: `bench/harness_call_attrib.py:28`;
`docs/research/HARNESS-STUDY-REVIVAL.md:201–215`.

Cache counts are estimates: the proxy saved global `prefix_cache_hits_total / prefix_cache_queries_total`
deltas, not `usage.prompt_tokens_details.cached_tokens`. Compute `C≈P*h`, `U≈P−C`; absence is unknown,
never assumed cold. All captured `api_calls` deltas equal one, and no call's server prefill+decode
materially exceeds its forward wall, but neither proves absence of concurrent counter contamination.
Sources: `E/pi_coding_bench.py:69–106`; `E/capture_proxy.py:177–215`.

Measured call wall comprises **6.39% prefill, 93.07% decode, 0.54% other**. Without the length-limited
call it is **21.96% prefill, 76.16% decode, 1.88% other**. TTFT overlaps prefill and is not added again.
Runner task wall totals **1013.181 s**, versus 1010.266 s proxy forward wall; the 2.915 s difference
includes tools/client/proxy scrape work and timing boundaries. It is not a pure client-runtime measurement.

At the measured 460–500 uncached tok/s ceiling, the 27262 estimated uncached tokens imply
54.5–59.3 s, or 2.27–2.47 s/task, before queue/cache-load effects. Measured prefill is 64.58 s:
these counter-ratio estimates and a long-prefill ceiling are not exact predictions for short calls.
A 6–7% reduction in measured prefill would save **3.87–4.52 s total, 0.16–0.19 s/task**,
about **0.38–0.45% of all task wall**. On the non-length-limited call cohort the corresponding
call-wall reduction is **1.32–1.54%**. This is a prefill-only estimate; do not add the unmeasured
1–2 ms/cycle decode-glue hypothesis. Sources: `docs/SPEED-LEVERS-STATUS.md:168–179,198–207`;
`docs/research/HARNESS-STUDY-REVIVAL.md:216–224`.

**Verdict #5:** preserve the already stable client prefix; measure/cache the unavoidable new suffix and
avoid unnecessary follow-up calls when correctness permits. Prefill glue offers a modest task saving here.
The length-limited reasoning failure is a much larger wall-time issue. Pair any call-batching or reasoning
budget change with official checks; cached-prefix estimates alone do not authorize dropping evidence.

## Exact-copy screen: avenue #4

**No new exact-token copy percentages or gains are available yet.** `/health` succeeded, but a no-hold
window was not confirmed, so `/tokenize` was not used. The screen emits `status=skipped` and no synthetic
“token” results. The historical “sparse” finding is neither confirmed nor falsified on this snapshot.
The routine token mix (estimated 73.05% tool arguments) makes the screen relevant; high pure-tool
DFlash acceptance (6.789 emitted/cycle) leaves limited incremental upside.

Implemented protocol (`bench/toolarg_copy_screen.py:34,40,91,134`):

1. Flatten tools schema then chronological messages, preserving literal tool-argument strings.
   This approximates chat-template order; role delimiters/escaping and hidden reasoning are not reconstructed.
   Use CPU `/tokenize`, no special tokens, at most 2048 characters per POST, sequentially; health before
   every POST, memory-only cache. Independent chunk boundaries introduce tokenization differences.
2. At every argument-output position with a complete K7+bonus future, search the preceding 4/8/16 tokens
   in the *static request context*. Choose the longest anchor, then its most recent complete occurrence.
   Also report each width separately. Never select a source by inspecting future output.
3. Compare that proposed continuation with the recorded future. Copy emission is `1+min(agreement,7)`;
   fallback on no source uses `e_D=1+sum(accept_rate_by_pos)`. False hits are charged, never oracle-filtered.
   Report anchor coverage and positions where the selected copy correctly continues for at least 8 tokens.
   Short tails retain DFlash. Output stream boundaries and actual cycle boundaries are not captured.
4. For call i, estimate tool-token count T_i from normalized tokenizer category shares. Pool tool emission
   as `ΣT_i / Σ(T_i/e_i)`. Pool all-token emission as
   `ΣG_i / Σ[(G_i−T_i)/e_Di + T_i/e_i]`. Compare against the same calls' DFlash baseline;
   this is an expected-cycle screen on the fixed trajectory, not a live speedup measurement.
5. Charge worst observed offline scan seconds/position at every modeled cycle. This includes future-agreement
   scoring, but omits index construction and server integration costs; report both raw and charged gains.
   **Do not integrate below +3% pooled gain**. Above it, require a held-out/task-bootstrap confirmation and
   a guarded prototype with official tests. Keep DFlash running initially to maintain draft state; no
   10.9 ms draft-pass saving is claimed. Current verdict: gate unestablished; defer integration.

Reference source selection: `bench/copy_opportunity.py:15–34,38–78`; old sparse outcomes
`docs/research/ARCHITECTURE-SCREENS.md:111–137` and
`docs/research/HARNESS-STUDY-REVIVAL.md:168–197`. Captured tool arguments are concatenated across
streams (`E/capture_proxy.py:235–263`); matches across those unavailable boundaries are a screen limitation.
A call's acceptance curve also cannot establish the acceptance specifically inside its argument span.

## Commands

Entirely offline, character-share attribution and explicit skipped copy status:

```bash
.venv/bin/python bench/toolarg_copy_screen.py \
  results/harness-capture-20260910/proxy_calls.jsonl \
  --out results/harness-capture-20260910/copy-screen.json \
  --attribution-out results/harness-capture-20260910/call-attribution.json
```

After the operator confirms production is outside a guarded hold, run both in one process to reuse
private tokenizer results in memory. `--no-hold` is an operator attestation, not a hold detector;
stop before any scheduled hold. Failed health/tokenization produces a skipped copy report and
falls back to explicitly labelled character-share attribution.

```bash
.venv/bin/python bench/toolarg_copy_screen.py \
  results/harness-capture-20260910/proxy_calls.jsonl \
  --tokenize --no-hold --model glm-5.3 \
  --out results/harness-capture-20260910/copy-screen.json \
  --attribution-out results/harness-capture-20260910/call-attribution.json
```

Standalone attribution:

```bash
.venv/bin/python bench/harness_call_attrib.py \
  results/harness-capture-20260910/proxy_calls.jsonl \
  --out results/harness-capture-20260910/call-attribution.json
.venv/bin/pytest -q tests/
python3 -m py_compile bench/harness_capture.py bench/harness_call_attrib.py \
  bench/toolarg_copy_screen.py tests/test_harness_capture_tools.py
```

Tests cover harmonic pooling, false-copy penalties, non-oracle source selection, width priority,
K7 tails, invalid curves, cache unknowns, stable/changing prefixes, pseudonymous task ordering,
private-output exclusion, and health/hold/network failure behavior using synthetic inputs only.
