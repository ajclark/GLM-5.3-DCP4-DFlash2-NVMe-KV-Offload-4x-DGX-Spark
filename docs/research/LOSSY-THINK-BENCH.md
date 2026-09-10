# Think-scoped lossy verification: bench and invariant proof

2026-09-10. Offline implementation; no Spark/GLM requests or deployment performed.
Server implementation belongs to Claude. Run these commands only inside his
guarded, think-scope-capable DFlash K7 hold with `GLM_SPEC_LOSSY=1`.

`bench/adaptive_spec.py` accepts `lossy-think-m<M>[-p<pmin>]`: identical to the
existing fixed7-based lossy request plus `vllm_xargs.spec_lossy_scope="think"`.
Existing fixed/adaptive/shadow/lossy variants retain their request serialization;
old lossy variants omit scope and therefore mean `all`. Margins remain (0,5],
rank=2, min_p=[0,0.5). The server tracks committed `<think>`/`</think>` tokens
154841/154842, initialized from the prompt tail; a client flag alone is not
proof that relaxation occurred in the intended scope.

## Hold commands

Development = the unchanged `bench/spec-development.json`: six `code_*` and six
`prose_*` cases. Every arm receives the identical 1024-token **total completion**
budget, including reasoning. `--thinking` applies to every arm/every repeat;
`--thinking-repeat N` instead applies to every arm of zero-based repeat N.
Two-arm blocks alternate AB/BA across cases/repeats; larger blocks retain the
existing seeded per-case/repeat shuffle, with one fixed7 anchor per block.

```bash
# 12 cases × 2 repeats × 4 arms = 96 requests, thinking ON.
.venv/bin/python bench/adaptive_spec.py \
  --out results/think-dev --corpus bench/spec-development.json \
  --variants fixed7,lossy-m2.5,lossy-think-m2.5,lossy-think-m5.0 \
  --thinking --tokens 1024 --repeats 2 --deadline-seconds 300

# 12 cases × 1 repeat × 2 arms = 24 requests, thinking OFF.
.venv/bin/python bench/adaptive_spec.py \
  --out results/think-off --corpus bench/spec-development.json \
  --variants fixed7,lossy-think-m5.0 \
  --tokens 1024 --repeats 1 --deadline-seconds 300
```

Outputs must be new directories. Keep their basenames short: trace labels have a
96-character bound. Each request checks endpoint idleness and the existing
four-node memory guard; these commands do not boot/restore the service. Stop on
an error and let the hold owner handle restoration. Both blocks fit within one
hold; budget the hold for up to 120 × 300 s plus preflights in the worst case,
not merely the expected decode time. Do not silently shorten one arm's budget.

## Trace report and the thinking-off invariant

After the hold owner exports the scheduler trace locally, set `THINK_TRACE` to
that file. The same trace can cover both result directories; labels isolate runs.

```bash
export THINK_TRACE=/path/to/exported/spec-trace.jsonl
.venv/bin/python bench/spec_request_report.py results/think-dev --trace "$THINK_TRACE"
.venv/bin/python bench/spec_request_report.py results/think-off --trace "$THINK_TRACE"

.venv/bin/python bench/spec_think_check.py results/think-dev results/think-dev \
  --treatment lossy-think-m2.5 --out results/think-dev/quality-m2.5.json
.venv/bin/python bench/spec_think_check.py results/think-dev results/think-dev \
  --treatment lossy-think-m5.0 --out results/think-dev/quality-m5.0.json
```

Activation proof checks every verify row: fixed cap seven, C1 eligibility, complete
telemetry, requested margin/min_p and enabled status. Think treatments must echo
`lossy_scope="think"`; missing, `all` or `invalid` fails. With thinking on, **each
think treatment request must have at least one `relaxed > 0` row**. With thinking
off, **every row must have `relaxed == 0`**, even though margin/scope/enabled remain
armed. Every fixed7 control must also have zero relaxed tokens. A missing relaxed
field counts as zero; unknown thinking metadata cannot masquerade as thinking off.
An off request that unexpectedly opens another think span and relaxes fails this
negative-control block; inspect its completion, never waive the invariant.

`request-control-report.json` adds `thinking_splits[variant].on/off/unknown`.
Its `pooled_decode_tps` is Σtokens-after-first-emission-block / Σpost-first-block
seconds; `decode_tokens`, `decode_seconds` and excluded request count expose the
denominator. The older `decode_tps` field remains the mean request rate for
compatibility. Paired thinking settings and completion budgets must match. Prompt
token/message hashes, censoring and following-relaxed/exact curves remain checked.
These scalar traces prove the thinking-off invariant and scope dispatch; they
cannot independently locate individual relaxed tokens inside mixed-span cycles.

## Reasoning and visible-answer accounting

Each runner result records `thinking`, `max_tokens`, `reasoning_tokens`,
`reasoning_tokens_source`, `reasoning_tokens_estimated`, `reasoning_words`, and
visible-answer word/estimated-token counts. Exact provider
`usage.completion_tokens_details.reasoning_tokens` (or `usage.reasoning_tokens`)
wins when available. Otherwise use streamed `reasoning`/`reasoning_content`, then
raw think spans: **round(1.04 × Unicode word count)**, explicitly a tokenizer-free
estimate. There are no `/tokenize` calls. Do not treat these estimates as exact
acceptance attribution, particularly for code or non-English reasoning.

Visible answer = text after the final `</think>`, or parsed content if reasoning
arrives in a separate field. An unclosed raw span has no visible answer; a raw
thinking-on stream can start inside the prompt-prefilled span without repeating
`<think>`. The runner records whether a separate reasoning field was present so
that this case can be distinguished from already-parsed visible content.
Provider reasoning counts and text-derived estimates may differ; source counts
are included in the quality report rather than hidden inside an unlabeled ratio.

`spec_think_check.py` accepts separate result directories or the same multi-arm
directory; use `--treatment` to select one think arm and repeatable `--repeat N`
to select thinking-on repeats from a mixed run. It checks identical paired
prompts/settings and rejects thinking-off inputs. Ratios are treatment/control
of pooled reasoning-token counts and **visible-answer word counts**; zero control
denominators yield null. It retains empty answers, length stops and code failures.
At 1024 tokens, a reasoning spiral may leave no answer: that is an outcome, not
a missing sample. Visible verification being exact does not imply identical or
correct answers after a changed reasoning prefix.

## Code checker applicability

Every `code*` case in both arms is passed to `bench/spec_code_check.py:check` on
its visible answer only. That existing bounded local checker tests four pure
Python functions (`lower_bound`, `merge_intervals`, `run_length`, `stable_unique`).
The six unchanged development prompts instead request modules, TypeScript, SQL
and patches. Their checker failures are recorded but **are not functional scores
for those tasks**. Reports separate all pass counts from prompts matching the
checker's contract. Do not promote from the 96-request speed screen alone.

For a meaningful additional functional probe, create a matching corpus locally
and run it during the same authorized hold (six extra requests, outside 96+24):

```bash
.venv/bin/python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'bench')
from spec_code_check import PROMPT
Path('/tmp/think-functions.json').write_text(json.dumps({'code_functions': PROMPT}))
PY
.venv/bin/python bench/adaptive_spec.py \
  --out results/think-func --corpus /tmp/think-functions.json \
  --variants fixed7,lossy-think-m2.5,lossy-think-m5.0 \
  --thinking --tokens 1024 --repeats 2
.venv/bin/python bench/spec_think_check.py results/think-func results/think-func \
  --treatment lossy-think-m2.5 --out results/think-func/quality-m2.5.json
.venv/bin/python bench/spec_think_check.py results/think-func results/think-func \
  --treatment lossy-think-m5.0 --out results/think-func/quality-m5.0.json
```

Offline tests cover request-byte compatibility, 96/24 schedules, all-arm thinking
and budgets, usage/field/span counting, prefilled/truncated spans, echoed scope,
armed-but-zero thinking-off traces, pooled denominators, paired settings, and
actual bounded function checks on both visible answers. Run `.venv/bin/pytest -q tests/`
and compile the four changed/added bench Python modules plus the new test file.

## Task-level report

Hold 9 evaluates the same 12 official HumanEval tasks in two passes, ordered
control → m2.5 → m5.0 per pass, C1, pi thinking high. **The completion limit is
8192 (`PI_HE_MAX_TOKENS`), reduced from production's 32768**; this is a protocol
deviation, not a production-equivalent reasoning-budget experiment.

After all six lane summaries exist, report entirely locally:

```bash
.venv/bin/python bench/spec_think_tasks_report.py \
  results/lossy-think-tasks-20260910/ \
  --trace /path/to/spec-trace.jsonl \
  --out results/lossy-think-tasks-20260910/task-report
```

Add **`--allow-partial` before reading an experiment still being written**.
It reports missing passes/tasks and an unfinished JSONL tail, uses available
paired tasks, and never marks a partial snapshot promotion-ready. Without it,
completion summaries are checked before private task/call files are opened.
If no trace is available, explicitly substitute `--no-trace`; activation is
then **unproven**. Output is `<prefix>.json` plus `<prefix>.md`. Exit 1 means a
reportable validation failure; exit 2 means unusable input/CLI configuration.

The tables include per-arm/pass and pooled official `check.passed` counts,
tasks with errors, elapsed lane wall, task median, output/reasoning tokens,
API and length-stop counts, proxy-pooled decode (`Σtokens/Σdecode_seconds`),
and emitted/cycle (`1+Σaccepted/Σdrafts`, **including the bonus token**).
Reasoning character share is a separate character-based measure. Task wall
comes from `seconds`; pooled arm wall sums lane-summary elapsed times.
Sources: `bench/pi_humaneval.py:evaluate,summarize` and
`bench/spec_think_tasks_report.py:aggregate,compare`.

Pair by `(task_id, pass)`, treatment/control. Report wall/reasoning/output
ratios, pair medians, and the both-pass/control-only/treatment-only/both-fail
matrix. Geometric means weight tasks equally; seeded percentile bootstrap
resamples whole task IDs, retaining both paired passes together. Defaults:
`--seed 20260910 --bootstrap-samples 10000`. Zero denominators are undefined
and counted; zero numerators remain zero. Length-bound tasks have any proxy
call ending `finish_reason=length`; differing or unknown paired completion
budgets are flagged. Small task counts and budget-censored outcomes limit
interpretation; faster completion alone is not a quality win.

Activation requires matching treatment labels/margins, scope `think`, enabled
lossy verification and positive relaxation; control rows require zero relaxed
accepts and explicit null margin. Verify times use proxy **call-start** `t`
and inclusive `[t,t+wall_s]` windows. A label in the wrong arm's window fails;
rows outside every proxy window count as other traffic, even for a recognized
label. Missing labels, overlapping arm windows, dropped/ineligible telemetry
and writer errors fail proof. No trace is joined through private request text.

Only whitelisted counts, public HumanEval IDs and fixed diagnostics are
serialized. Private texts, tools/messages, check output, error payloads and
trace request IDs never enter JSON, Markdown or stdout. Tests use synthetic
fixture trees exclusively:

```bash
.venv/bin/pytest -q tests/test_spec_think_tasks_report.py
python3 -m py_compile bench/spec_think_tasks_report.py tests/test_spec_think_tasks_report.py
```
