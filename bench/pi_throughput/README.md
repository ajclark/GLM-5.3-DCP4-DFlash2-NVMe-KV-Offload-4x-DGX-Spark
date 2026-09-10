# HumanEval via pi and herdr

Run the deployed TP4/DCP2 GLM-5.3 lane against the frozen repaired V5 adaptive
speculation build, using real pi RPC sessions. All model calls go through pi's
provider adapter and tool loop. The controller runs in a herdr pane; HTTP reads
outside pi are health and metrics only.

From an available herdr shell pane in this repository:

```bash
PI_HE_TASK_TIMEOUT=1800 python3 -u bench/pi_humaneval_compare.py \
  --out results/pi-humaneval-compare-YYYYMMDD \
  --label pi-he12-YYYYMMDD-r1
```

The 1800-second per-task watchdog accommodates long high-thinking responses;
it does not alter the provider's output-token limit. The September 9 run began
with a 600-second watchdog, preserved an incomplete C3 attempt, and retried that
whole cell with this longer bound and unchanged pi request settings.

The controller prepares an isolated experimental directory on each Spark,
verifies staged sources against the saved `hints-conf-20260909-r2` sources,
measures the currently deployed stack at every C1–C12, temporarily switches to
the adaptive stack, repeats the sweep, restores the exact original containers,
and measures restored C1/C12 anchors. Original image IDs, container IDs, commands,
selected environment, and Python mount hashes must match after restoration.
Four-node memory monitoring and the existing remote experiment watchdogs remain
active. The experiment changes no GPU clocks or production source files.

A single-lane pilot without a deployment switch:

```bash
python3 -u bench/pi_humaneval.py \
  --out results/pi-humaneval-pilot-YYYYMMDD --levels 1,2 --limit 2
```

The main run uses all twelve tasks at every concurrency, for 288 scored solves
across the two main lanes. Each cell has a fresh pi session per task and up to C
active sessions, with a barrier before the first wave. The same predetermined
task order and workspace paths are used on the two stacks. This finite-batch
workload includes tool pauses and the draining tail; it does not maintain a
constant number of GPU decode requests. Extra warmups and restored baseline
anchors are reported separately.

The corpus preserves the ten tasks selected in the previous HE10 benchmark:
49, 20, 132, 154, 55, 100, 69, 48, 75, 122. Tasks 0 and 22 fill C12 without
duplicating prompts. `humaneval12.json` pins the upstream commit, source hash,
prompts, tests, and selection. Canonical solutions are deliberately omitted.
The HumanEval code/data license is in `HumanEval-LICENSE`.

Each prompt asks pi to create `solution.py`. Hidden official tests are evaluated
after completion without feedback, inside a networkless bubblewrap namespace
with only system libraries visible and CPU/memory/wall limits. Agent tool use is
included in timing. The results are a small agentic HumanEval subset, not the
standard full-dataset completion-only HumanEval score.

Pi uses a temporary snapshot of the user's model entry, settings, and SYSTEM.md:
`glm53/glm-5.3`, thinking high, T=0, top_p=1, and max output 32768. The native
read/bash/edit/write tools stay enabled. Optional packages, skills, context
files, and other extensions are disabled equally in both lanes. No benchmark
speculation hints, seeds, output caps, or sampling overrides are injected.
`observe.ts` records provider request times, streaming activity and final usage
without changing payloads. Pi processes do not inherit the controller's herdr
identity, so they cannot overwrite its pane status.

The primary rates are aggregate output tokens over batch wall time, and output
tokens over summed provider-request time. Native usage includes tool arguments
and any reasoning tokens; reasoning counts are never added a second time.
`post_first_tok_s` uses time after the first streamed delta and is approximate
because a first speculative chunk can contain several tokens. Prefer the
separately reported server decode rate for an engine decode estimate. It too is
a ratio across request durations, not aggregate cluster throughput. Server
counters cross-check client request and output totals; mismatches stop the run.

The adaptive stack includes the repaired draft-cache table and other V5 runtime
changes. Its measured cost curve is frozen, confidence collection and client
hints are off, and policy tracing is on. Adaptation is C1-only and returns to
K7 when multiple decode requests are active. This compares the two stacks as
requested, not the controller in isolation. Main lane order is deployed then
adaptive; caches warm naturally, and the restored anchors help assess drift.
Twelve solves per cell do not establish narrow confidence intervals.

Raw events, responses, tests, metrics, memory samples, runtime inventories and
rollback evidence remain under the result directory. The run writes `COMPLETE`
only after the main sweeps and successful restoration checks finish.

Rebuild the report, CSV and audit from saved evidence with
`python3 bench/pi_humaneval_report.py results/pi-humaneval-compare-YYYYMMDD`.
Add `--plot` in an environment with matplotlib to produce PNG/SVG charts.
