# Bounded-lossy prose: bench and quality tooling

**2026-09-10. Local tooling; no Spark measurements or quality claim.** Implements
[plan §5–6](../LOSSY-VERIFICATION-PLAN.md), including the following-draft check in
§8. The [architecture memo, A](../SPEED-ARCHITECTURE-OPTIONS.md#a-bounded-lossy-verification-for-prose-opt-in)
motivates the experiment; it does not establish this model's speed/quality result.
Run generation commands only during Claude's guarded `--lossy` experiment after
`HOLD` (boot: `GLM_SPEC_LOSSY=1`, `GLM_SPEC_POLICY=shadow`, trace enabled).

[adaptive_spec.py](../../bench/adaptive_spec.py) keeps existing request bodies
unchanged. `lossy-m<M>[-p<pmin>]` adds float margin, rank 2, and float probability
floor to `fixed7`; omitted floor means `0.0`. Bounds: `0 < M <= 5`,
`0 <= pmin < 0.5`. Both arms request fixed K7; boot shadow enables tracing.
`--prose-only` selects `prose_*`, including after `--repo-context` builds its
reference. Corpus entries may be strings or `{prompt, constraints}` objects.

**Development: 108 requests**, 8 treatments × 6 prompts × 2 repeats + 12 anchors.
Nine-arm blocks retain the existing seeded shuffle, with one shared anchor per
prompt/repeat; two-arm comparisons retain `(case_index + repeat) % 2` AB/BA.

```bash
.venv/bin/python bench/adaptive_spec.py \
  --corpus bench/spec-development.json --prose-only \
  --variants fixed7,lossy-m0.5,lossy-m0.5-p0.1,lossy-m1.0,lossy-m1.0-p0.1,lossy-m1.5,lossy-m1.5-p0.1,lossy-m2.5,lossy-m2.5-p0.1 \
  --tokens 256 --repeats 2 --out results/lossy-dev
```

Freeze `LOSSY_VARIANT` to the selected development name before opening held-out
results. **Held-out: 90 requests**, 15 prompts × 3 repeats × 2 arms, AB/BA.
`--thinking-repeat 2` enables thinking for both arms in the third repeat; the
first two stay off. Both arms retain the same total completion-token budget.

```bash
.venv/bin/python bench/adaptive_spec.py \
  --corpus bench/spec-heldout.json --prose-only \
  --variants "fixed7,${LOSSY_VARIANT:?Set the frozen development variant}" \
  --tokens 256 --repeats 3 --thinking-repeat 2 --out results/lossy-heldout
.venv/bin/python bench/spec_request_report.py results/lossy-heldout \
  --trace "${LOSSY_TRACE:?Set the collected local spec-trace.jsonl path}"
.venv/bin/python bench/spec_blind_quality.py results/lossy-heldout \
  --variants "fixed7,$LOSSY_VARIANT" --repeat 0 --out results/lossy-openings
```

Use the same report command for `results/lossy-dev`. Supply one authoritative
trace stream, not concatenated rank replicas. The report rejects missing trace,
lost C1 eligibility, dropped rows, wrong policy/cap, mismatched margin/floor,
disabled lossy rows, any lossy request without a positive `relaxed` count, and
any control row with nonzero `relaxed`. Old rows without lossy fields count as
zero relaxed. Existing hint/confidence controls also reject relaxation.

[spec_request_report.py](../../bench/spec_request_report.py) writes request,
per-prompt and pooled accepted/cycle, relaxed/cycle, relaxed/scheduled-position,
decode tok/s, TTFT, p95 SSE emission gap, plus prompt-clustered paired speed
intervals from [analyze_adaptive_spec.py](../../bench/analyze_adaptive_spec.py).
Cycle statistics exclude terminal/nonlearnable rows; activation checks include
them. `following_relaxed` / `following_exact` count accepted prefixes at positions
1–7 in the **next** cycle, joining adjacent rows within each request in scheduling
order. Excluded rows break joins. Denominators are scheduled positions, so these
are survival curves, not acceptance conditional on reaching a position.

The scalar trace `relaxed` cannot locate individual relaxed tokens. Exact pooled
`relaxed_per_scheduled_position = sum(relaxed) / sum(scheduled_k)` is available;
each position also has count bounds and a null exact count when ambiguous. A
position mask would be needed to narrow those bounds. First divergence (zero
based), first-64 paired token agreement, four-gram repetition and finish reasons
are reported; agreement does not reconstruct logits on a diverged prefix, and
SSE gaps measure token blocks. `--power <local-power.jsonl>` adds device J/decode
token using the existing four-node [power sampler](../../bench/spec_power.py);
these are device measurements, not wall energy.

**G3: full prose judging**, 15 prompts × 2 repeats × 2 variants = 60 generations;
position swapping produces 60 judgements over 30 completion pairs. The second
repeat below enables thinking. The protocol exports only final completion text;
keep blank/truncated completions as evidence when reasoning consumes the budget.

```bash
.venv/bin/python bench/adaptive_spec.py \
  --corpus bench/spec-heldout.json --prose-only --variants "fixed7,$LOSSY_VARIANT" \
  --tokens 1024 --repeats 2 --thinking-repeat 1 --out results/lossy-judge-runs
.venv/bin/python bench/spec_prose_judge.py export \
  results/lossy-judge-runs results/lossy-judge-runs \
  --treatment "$LOSSY_VARIANT" --repeats 0,1 --out results/lossy-judge-pairs
.venv/bin/python bench/spec_prose_judge.py score \
  results/lossy-judge-pairs/manifest.json judgements.json \
  --out results/lossy-judge-score.json
```

[spec_prose_judge.py](../../bench/spec_prose_judge.py) also accepts two separate
result directories. Give only `pair-*.json` to a human or external judge; retain
`manifest.json` privately (position map, source hashes, pair hashes). No endpoint
is contacted by this tool. Ingest an exact JSON mapping `pair-id -> A|B|tie`;
missing, extra and duplicate votes fail. `win_rate` counts strict lossy wins over
all judgements; its 95% Wilson interval treats ties as non-wins. The report also
gives half-credit `tie_adjusted_score`, decisive win rate/interval, and swap
consistency. Declare the tie convention before judging; the plan's ≥45% win-rate
gate uses the reported strict rate unless explicitly amended before evaluation.
**n=60 detects only gross loss**: nominal independent decisive judgements near
50% have a ~12-point half-width; swaps/repeats are dependent and cannot establish
a promotion confidence bound. G2 separately records A/B/tie and incoherence.

**G4: executable prose**, 20 prompts × 2 repeats × 2 arms = 80 generations.
The [corpus](../../bench/spec-prose-checks.json) declares word windows, required
named items, forbidden items, ordered headings, and a list/table/embedded-JSON
case. The [checker](../../bench/spec_prose_check.py) scores each constraint and
each completion, preserves failures, and compares matched pass rates to fixed7.
Word counts include headings/table/JSON; JSON checks reject duplicate keys,
non-JSON numbers and wrong typed values. These checks establish declared form
and items; literary/factual quality remains with the blind reviews.

```bash
.venv/bin/python bench/adaptive_spec.py \
  --corpus bench/spec-prose-checks.json --prose-only --variants "fixed7,$LOSSY_VARIANT" \
  --tokens 1024 --repeats 2 --out results/lossy-checks
.venv/bin/python bench/spec_prose_check.py results/lossy-checks \
  --variants "fixed7,$LOSSY_VARIANT" --repeats 0,1
```

For context pairs, use `--repo-context 32000` or `100000`, `--prose-only`, the
same frozen two-arm variants, `--repeats 2`, and `--deadline-seconds 900` in fresh
directories. Cold/warm runs share a prefix; record request order since only the
first request starts with a cold cache. G6's four preselected prose cases can use
`--cases <four names> --tokens 2048`; finish reasons and token repetition do not
replace review of endings or the final-512-token judge. G1's existing
[four-function checker](../../bench/spec_code_check.py) and code-persona wire
proof remain separate. The [plan](../LOSSY-VERIFICATION-PLAN.md#5-quality-gates)
requires all quality gates and the held-out speed lower bound before promotion.

Local verification: [bench tests](../../tests/test_lossy_bench.py) cover legacy
wire bytes, the 108/90 request designs and trace proof/causal counts;
[quality tests](../../tests/test_spec_prose_quality.py) cover all 20 constraint
cases, incomplete pairs, swapped exports, vote validation and Wilson arithmetic.
Run `.venv/bin/pytest -q tests/`; compile all changed Python files with
`python3 -m py_compile`. No cluster test is part of this local validation.
