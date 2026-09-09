# DFlash2 confidence: useful current scores, unsuccessful lagged predictor

**2026-09-09. Decision:** bounded shadow collection is implemented and validated
on the Sparks, disabled by default. The 12-prompt live dataset shows that the
tested lag-two predictor does not improve on available acceptance history.
Do not integrate it into serving. Same-proposal scores are informative, but
using them requires a separately measured worker/scheduler change: the present
transport delivers them after their scheduling decision. Detailed live results
and their limits follow below.

The implementation adds an opt-in collector and a formal optional runner-output
field; graphs, model weights, sampler and adaptive cap selection are unchanged.
It includes [an offline screen](../../bench/spec_confidence_screen.py),
[a verify-trace converter](../../bench/spec_confidence_convert.py),
[23 existing semantic tests](../../tests/test_spec_confidence_screen.py), and
[56 collector/transport tests](../../tests/test_spec_confidence_trace.py).
The initial source-audit task produced no Spark requests; subsequent parent
experiments validated the transport and collected the real dataset below.
Historical traces without scores cannot establish confidence quality retroactively.

## What the score actually means

The active checkpoint's selector top-k is 16, with seven proposed tokens and
an eight-token query block. In captured
[qwen3_dflash2.py](../../tests/fixtures/spec_confidence/qwen3_dflash2.py),
`compute_candidates` obtains 16 candidate IDs and unary logits for each draft
position. `_score_edges` constructs a tensor of shape
`[batch, 7, predecessor_candidate, child_candidate]`:

```text
edge_score(j, parent, child) = unary_logit(j, child)
    + dot(predecessor_codebook[parent_token] * projected_hidden[j],
          successor_codebook[child_token])
```

At the first position, the predecessor token is the bonus/anchor token,
repeated across all predecessor indices. At later positions, it is the token
at the preceding position's candidate index. The score is a learned lexical
interaction plus a unary candidate logit. It is not a target logit, a target
probability, or a directly calibrated acceptance prediction.

The captured [selector walk](../../tests/fixtures/spec_confidence/dflash2_speculator.py)
starts with predecessor index 0. At every position it loads the row for the
**previously chosen candidate index**, selects a child, writes that entire
16-element row to `_selector_scores`, and uses the chosen child index as the
next predecessor. Thus `_selector_scores[batch_row, j, :]` already contains
the correct row along the realized path. Taking the maximum over every parent
or using unary logits alone would describe a different decision.

For greedy requests, the chosen child maximizes that realized row. Candidate
IDs need not be sorted by token ID or score. The CPU tests independently check
the edge formula and replay the actual Triton walk with nonidentity token IDs,
request-state permutations, and absolute positions 32 and 100055. Neither
absolute position nor random seed changes this fixed-score greedy walk.
Sampling uses a different noise-dependent path and remains outside this first
experiment's eligibility contract.

Three useful reductions of each realized row are:

| Feature | Definition | Interpretation |
|---|---|---|
| Margin | Selected score minus best other score | Separation within these 16 candidates |
| Candidate pmax | `1 / sum(exp(score - selected_score))` | Selected probability after normalizing only this candidate set |
| Normalized entropy | Entropy of that candidate-set softmax divided by `log(16)` | Uncertainty among the proposed alternatives |

Compute reductions in FP32 from the existing FP32 realized buffer. This does
not recover precision lost in earlier model operations. Nonfinite rows
abstain. Constant logit offsets should not change any feature; score scaling
does change confidence and needs model-specific calibration. A missing target
token can coexist with a very confident selector, so high pmax is not evidence
that target verification may be omitted. The full target verifier stays
authoritative.

The existing score buffer is only **448 bytes per C1 proposal**
(`7 × 16 × 4`). Reducing to three FP32 features per position would produce
84 bytes. No vocabulary-sized tensor or new model is needed. First collecting
the raw 448-byte rows makes alternate feature calculations possible offline;
fusion into the existing walk should be considered only after the predictor
is useful and extraction overhead is measured.

## Availability is the binding constraint

The relevant ordering in the active
[V2 runner](../../overlay/vllm/v1/worker/gpu/model_runner.py) is:

| Worker order | Data available |
|---|---|
| Target verifies proposal P | Target output and P's accepted/rejected counts |
| Construct `AsyncOutput` | Starts copying target results |
| Postprocess sampled tokens | Commits request-state updates needed for drafting |
| `speculator.propose` | Generates proposal P+1 and its selector scores |
| Return async output | CPU may process results while later GPU work continues |

Captured [async_utils.py](../../tests/fixtures/spec_confidence/async_utils.py)
makes this explicit: `AsyncOutput.__init__` waits the copy stream on the main
stream, copies the sampled output, and records its completion event.
`get_output` later synchronizes **that event**. Adding P+1's scores to the
already-launched copy cannot make those not-yet-produced values valid. Moving
the copy after drafting would extend its dependency and change the overlap
being optimized.

There is also no ordinary greedy draft-token copy to extend for free.
Captured [DraftTokensHandler](../../results/adaptive-spec/inventory/runtime/v1/worker/gpu/spec_decode/utils.py)
copies IDs for structured-output validation; without that need it keeps no
CPU draft array. Its exceptional copy path synchronizes on retrieval, which
is not a suitable template for adding an unconditional new wait.

CPU scheduling can already have queued the next verification shape. Reducing
a GPU score to a cap does not itself change that CPU-selected shape. Masking
tail tokens inside an already-selected M8 graph is not proof that M2 target
work executes. The measured finite action set remains K=1/3/5/7 and genuine
target M2/4/6/8 graphs.

[EVICT](https://arxiv.org/html/2605.00342v1) combines candidate-benefit estimates
with profiled verification costs and graph-compatible selection. Its useful
transfer here is the cost-aware objective. Its integration does not establish
that this fork can consume same-block GPU signals without a new dependency.

For a chain, the candidate objective remains
`(1 + sum(prefix_survival[0:K])) / measured_cycle_cost[K]`. DFlash2 has already
computed the complete parallel draft when its selector scores exist, so this
can reduce target verification work; it cannot retroactively save that draft
pass. A GPU-controlled graph dispatcher or a deliberate host wait is a larger
separate experiment, with its own measured break-even condition.

## Minimal viable collection and ownership contract

Prefer collecting P's immutable score packet with the ordinary target output
that verifies P, then use it only at a scheduling boundary where the packet
has actually arrived. At that point it is naturally paired with P's label and
is a **lagged** predictor for a later proposal. Record its age; do not promise
a fixed one-step lag. Existing asynchronous queuing can make the usable age
two or more proposals.

The implementation follows this contract:

1. Assign every proposal a monotonic sequence within a request incarnation.
   Retain runtime/checkpoint identity, run identity, request identity and
   epoch, proposal sequence, actual anchor position, actual live batch row,
   and any row/slot generation. The verification result must refer to that
   exact proposal and anchor. GPU request-state index alone is not identity.
   The live packet supplies worker epoch/proposal metadata; runtime/run IDs
   and explicit prompt-family provenance are added by the offline converter.
2. Snapshot the selected-path scores into a private GPU packet before
   `_selector_scores` can be overwritten by the next draft. A retained tensor
   reference or `record_stream` protects allocation lifetime, not mutation of
   a persistent buffer. Use explicit stream/event ordering and slot ownership.
3. Copy only the actual live C1 row into explicitly pinned host storage. This
   implementation ties private packet lifetime to the existing bounded
   in-flight AsyncOutput lifetime and caps collection at 128 packets per
   request incarnation. Source and host tensors are released after the
   existing event completes and CPU lists are formed. A reusable ring/pool
   could reduce allocation overhead later, but would need its own lease tests.
4. Attach only ready host packets to an ordinary internal runner-output
   message. The scheduler stamps receipt with its own monotonic clock. Do not
   compare a remote worker's wall clock or CUDA event time to a scheduler
   deadline. If using a separate post-proposal copy instead, test readiness
   without waiting and deliver it at a later normal output boundary.
5. In the offline screen, use the latest compatible packet whose receipt
   precedes the decision and whose proposal age is within the calibrated
   range. Unknown epoch, cancellation, reset/preemption, malformed values,
   unsupported sampling, stale age or missing packet means abstain. The
   scheduler still owns the selected cap and sends one consistent decision
   to all ranks. The live collector never reads packets to choose a cap.

**Padded graph rows require special care.** DFlash input preparation pads
`sample_idx_mapping` with zero. The selector's `req_state >= 0` check therefore
does not distinguish padded graph rows from real request state 0. The actual
kernel tests demonstrate both masked negative rows and finite outputs from
zero-mapped padded rows. Export must use the real `input_batch.num_reqs`, not
the graph's padded count. `_selector_scores` is indexed by batch row, while
the sampling mapping indexes request state; they cannot be interchanged.

The collector runs on target TP rank 0 only, with PP1/DP1 and asynchronous
V2 DFlash K7. A maximum-size packet holds 448 bytes of FP32 scores, seven
int64 sample positions, seven int64 draft IDs, and at most eight target
positions/IDs: **656 bytes on GPU plus 656 bytes in pinned host storage**,
before allocator overhead and JSON expansion. Only one actual request row is
copied, even if the graph pads to twelve. Verify TP rank 0 packet/target
agreement in the live identity trial rather than averaging different ranks.
Additional kernels, allocation, serialization and copy ordering still need
an overhead control.

No new GPU synchronization should occur in the serving path. Extending the
existing output copy with an **already-produced, stable** packet adds a small
copy and may extend that copy's completion time; it is not zero-cost. The
control must measure this effect. Never attach current P+1 scores to P's
verification result merely because they share the same worker call.

### Implemented integration and first-boot contract

The [collector helper](../../overlay/vllm/v1/spec_decode/confidence_trace.py)
registers an incarnation on every `add_requests`, including replacement after
streaming updates, and clears ownership on removal/preemption. It requires
one registered request and one actual scheduled request. C2 admission,
prefill, unsupported sampling, a missing predecessor, or malformed source
shapes clears ownership. The first decode establishes ownership for its next
proposal and is intentionally not captured. Collection stops after 128
packets, including invalid diagnostic packets, per incarnation.

Before constructing [AsyncOutput](../../overlay/vllm/v1/worker/gpu/async_utils.py),
the runner clones the **previous** proposal's scores, sample positions and
draft IDs together with the actual current target positions/input IDs. The
copy stream waits on the main stream at its existing boundary, copies into
explicitly pinned destinations, and records its existing event. The next
proposal can then overwrite its persistent buffers while the private clones
remain valid. No extra CUDA event, `synchronize`, `.item()`, or GPU-to-host
scalar decision is introduced. The event's copy duration can increase.

After that event completes, host validation requires
`sample_positions[0] - 1 == target_positions[0]`, contiguous original absolute
positions, and `draft_tokens[:K] == target_input_ids[1:K+1]`. Tests execute the
actual DFlash preparation and target input kernels to prove this mapping.
Nonfinite scores and mismatches become invalid diagnostics, with scores
removed. Target tokens are never modified. The optional
[`ModelRunnerOutput.spec_confidence`](../../overlay/vllm/v1/outputs.py) field
carries CPU values keyed by internal request ID.

The scheduler's existing weak-reference/incarnation, invalid-KV, terminal,
eligibility, cap and discarded-async-output gates remain authoritative. A
packet's accepted prefix must also match the actual sampled prefix. Valid
and invalid diagnostics join the existing `verify` row; its request identifier
remains hashed. `decision_ns` is recorded on entry to selection, and
`receipt_ns` on entry to completion, using the same scheduler's
`monotonic_ns`. The collector's proposal age of one worker step is **not** its
usable age at a later scheduling decision.

For the root agent's later guarded boot, set `GLM_SPEC_CONFIDENCE_TRACE=1`
together with a non-off speculation policy and `GLM_SPEC_TRACE`. The shadow
collector works with fixed/adaptive experiment policies but never changes
their choices. The first dataset should use fixed cap 7. A request with
`vllm_xargs.spec_confidence_trace=false` disables its packets in the same boot;
omit the key or use boolean `true` to collect. Invalid types abstain. The
environment flag alone does not create a scheduler trace sink or enable an
otherwise off verification policy.

Start with a short known counting request and inspect packet ownership,
anchor agreement, nonfinite/mismatch counts, sample limit and trace drops on
the output rank. Inspect per-rank memory pressure before boot/capture and
during the request. Then randomize identical fixed-seven requests with
capture enabled/disabled in the same boot, preserving all other controls.
Measure the first 128 eligible cycles separately: whole-request averages
would dilute overhead after collection stops. Inspect tok/sec, device and
wall joules/token, copy/CPU time, trace-writer backlog, pinned-host allocation
and peak GPU memory. An initial acceptance gate is no identity failure, no
new quality failure, no pressure excursion, no trace loss, and overhead below
the measured noise floor or a justified small bound before collecting broadly.

The existing [100k complete-function quality failure](../../results/adaptive-next/cache-width-r1/complete-100k/complete-100k-adaptive-k7.json)
remains an independent no-promotion gate. Its adaptive request scheduled cap 7
throughout, and a later repeat passed. This does not establish a cap-transition
bug or a target-logit tie. Shadow collection must not weaken that investigation's
prompt-level correctness checks.

## Calibration that respects what was observed

For an actual cap K and accepted prefix A:

- Conditional position j is observed only when `j < K` and `j <= A`.
  Positions before A succeed, the first rejected position fails, and later
  conditional outcomes are unknown.
- Prefix-survival labels have a different mask. After a rejection, every
  longer prefix is false. If a short cap is fully accepted, its unverified
  longer prefixes remain unknown.
- A lagged feature from P paired with the outcome of P+2 is a predictor for
  P+2. It must be trained with that relationship and age. Training it against
  P's label and then using it two steps later silently changes the problem.

The new screen implements these masks and checks exact proposal/anchor joins.
It extracts margin, pmax and entropy; its initial calibrator uses pmax bins
conditioned on context band, feature age and block position, with bounded
shrinkage toward causally available acceptance history. It compares a
history-only screen, available confidence and a separately named same-block
noncausal diagnostic. The last comparison estimates how much signal is lost
to timing; it is not a deployable policy or a mathematical upper bound.

All repeats of a prompt/case are held out together. No request incarnation can
cross split groups. The screen refuses mixed runtime identities: geometry
repair changed the long-context data-generating process, so original and
repaired traces must not be pooled. Other request incarnations cannot provide
history merely because a GPU slot or request name was reused.

Only fixed-seven records enter the all-cap utility comparison. The metric
evaluates different prefix choices at the same recorded boundaries, omitting
changes to subsequent proposals. It also omits the serving controller's warmup,
periodic probes and hysteresis, so its history screen is not a controller
reproduction. Any serving proposal still needs a real inference-only control.
Version 5 with hints disabled is locally parity-tested against the repaired V4
decisions. Calibration metrics alone are not
tok/s or energy measurements.

Before a live policy experiment, collect development-only fixed-seven shadow
records after the geometry repair. Include coding, prose, actual thinking
settings, tool transitions and distinct context bands. Fit bins on development
prompts and lock a separate evaluation set. Compare acceptance-history only,
harness hints only, confidence only and hybrid predictions. Add a shuffled
confidence control that preserves age/context and calibration splits; keep
it clearly separate from a deployable causal predictor. Do not add a
classifier-model request or train alongside the loaded service.

The original 96-record demo remains explicitly synthetic: current scores arrive after their decisions,
so only age-two packets are available for 88 records. The tests establish
that this distinction is honored. Demo utility values are properties of
invented data and must never enter Spark benchmark summaries.

## Implement/defer gates

| Next action | Decision and evidence needed |
|---|---|
| Bounded immutable shadow packets | **Implemented and validated on the Sparks, off by default.** Identity checks pass; the small on/off screen does not establish a strict one-percent overhead bound. |
| Lagged confidence affecting caps | **Reject the tested predictor for serving.** At the actual two-step delay it is slightly worse than history alone on the 12 development prompts. A different predictor needs new held-out evidence. |
| Current-block host cap selection | **Defer.** It requires changing an existing dependency or graph-dispatch contract. Prototype/measure that cost only if same-block signal materially exceeds the usable lagged signal. |
| Learned head or selector fine-tuning | **Defer.** First separate candidate recall failures from selector/path errors; raw margin calibration may already suffice. |

For a bounded live trial, retain C1 greedy eligibility, finite graph shapes,
full target verification and fallback on unavailable features. Compare against
the hint-disabled V5 runtime, then add Pi hints as an ablation. Keep the current
warmup/probe/hysteresis behavior initially rather than confounding this test
with another cold-start change. Require no new correctness/pressure failures,
a positive paired confidence interval on the targeted workload, no material
regression in the other workload, and no regression in measured J/token.
Loaded idle watts are a separate policy problem; a workload confidence score
does not lower them.

## Running the offline prototype

```sh
.venv/bin/pytest -q tests/test_spec_confidence_screen.py
.venv/bin/pytest -q tests/test_spec_confidence_trace.py
.venv/bin/python bench/spec_confidence_screen.py
.venv/bin/python bench/spec_confidence_screen.py --demo
.venv/bin/python bench/spec_confidence_convert.py --trace verify.jsonl --out joined-confidence.jsonl --runtime-id REPAIRED_RUNTIME --run-id BOOT_ID --case-map prompt-cases.json
.venv/bin/python bench/spec_confidence_screen.py --trace joined-confidence.jsonl --max-age 2
```

The command without data reports that no feature dataset was supplied. The
prototype never contacts a model, reads KV contents, or launches CUDA. Its
tests run the captured Triton walk under the existing CPU interpreter. The
56 additional collector tests run actual modified async/runner methods with
deferred fake streams/copies, lifecycle changes, and source-buffer overwrite;
they also exercise real preparation kernels, trace receipt joins and invalid
conversion counterexamples. These establish CPU contracts, not CUDA timing.
All tests work from a fresh checkout. Original selector, model and async
sources are pinned in [tests/fixtures/spec_confidence](../../tests/fixtures/spec_confidence),
with source paths and SHA-256 hashes in its
[manifest](../../tests/fixtures/spec_confidence/manifest.json). The source-replay
tests verify those hashes and have no ignored-inventory dependency.

Each joined JSONL record requires:

| Fields | Meaning |
|---|---|
| `runtime_id`, `run_id`, `request`, `epoch` | Stable owner identity; runtime ID includes repaired checkpoint/runtime provenance |
| `case` | Prompt-family split key, identical for every repeat |
| `proposal_id`, `anchor` | The proposal actually verified; IDs increase within an incarnation |
| `feature_proposal_id`, `feature_anchor` | Origin of this record's raw score packet; must equal proposal/anchor |
| `verified_proposal_id`, `verified_anchor` | Verifier's recorded origin; must match exactly |
| `decision_ns` | Scheduler-local time when this proposal's verification cap was chosen |
| `feature_available_ns` | Scheduler-local packet receipt time, or null if unavailable |
| `feedback_available_ns` | Scheduler-local availability of this record's verification result |
| `scheduled_k`, `accepted`, `eligible`, `terminal` | Actual feedback and censoring/eligibility fields |
| `realized_scores` | Seven rows of 16 finite scores; absent data or invalid rows abstain |
| `candidate_ids`, `selected_tokens` | Optional seven candidate rows and selected IDs, supplied together for mapping validation |
| `costs_ms` | Positive context-matched complete-cycle costs keyed by `"1"`, `"3"`, `"5"`, `"7"` |

The screen selects older available packets itself. Do not replace a record's
score packet with an old row while leaving its proposal identity unchanged.
`prompt-cases.json` explicitly maps experiment labels to prompt/case IDs;
all repeats share the same case, and missing mappings fail. The converter
counts absent/invalid/unlearnable packets and missing measured cost tables,
refuses missing timestamps or incorrect joins, and never substitutes worker
wall time. It emits the compact offline format from the implemented internal
runner/verify-trace diagnostic; this is not a public serving response field.

Captured source hashes (SHA-256): selector speculator
`4dac8303f68baafc0a2e675c053df8a15df2f1345b4032ef9348122d1f71cef9`;
DFlash2 model
`c141daa4b2059c0098224ac36471c2197b7052c100bef0a4dbc2ca79b627053f`;
async utilities
`a6256b706253868e340641a7a8fd6d327eedf7c513e492c54df7cf34d486af73`.

## Live repaired-runtime confidence screen

Experiment `hints-conf-20260909-r2`, runtime `f71eac0`, collected fixed-seven
shadow data from twelve frozen development prompts: six coding and six prose,
256 output tokens each, one repeat. No confidence-based cap decision was made.
The collector produced **1050 valid learnable records**; twelve initial records
had no packet and twelve terminal/censored records were excluded. There were
no invalid packets in this development trace.

Of the 1050 evaluated records, **1026** had a usable packet by their scheduling
deadline, and every such feature was **two proposal steps old**. The first two
records per request therefore abstain. These are actual scheduler-local
receipt/decision times with request epoch and proposal identity checks, not
assumed GPU completion times or backdated features.

The screen fits conditional-risk bins with one entire prompt held out at a
time, including all records belonging to that prompt. Mean prefix Brier loss
is averaged equally across the six prompts in each domain; lower is better:

| Predictor | Coding Brier loss | Prose Brier loss |
|---|---:|---:|
| Available acceptance history | 0.19396 | 0.11118 |
| History plus available lag-two confidence | 0.19933 | 0.11144 |
| Same-proposal confidence, intentionally noncausal diagnostic | 0.12354 | 0.07990 |

The lagged variant also loses in the offline same-boundary utility screen:
ratios 0.9831 for coding and 0.9952 for prose against history alone. The
same-proposal diagnostic ratios are 1.0807 and 1.0423. **None are measured
tok/s gains or a closed-loop rollout.** Warm-up, periodic probes, hysteresis
and the changed future trajectories are absent from that screen. Its costs
come from the frozen measured `8f684a4` repaired-runtime atomic1 curve, supplied
to the `f71eac0` fixed-seven shadow requests; collector overhead was not
recalibrated independently at every cap. Brier losses do not depend on those
costs. The separate Pi on/off control validates switch activation and packet
transport, but does not supply a strict overhead bound.

The [screen report](../../results/adaptive-next/hints-conf-r2/confidence-screen-report.json)
links source hashes for the joined data and full per-prompt analysis. This is
development evidence, not a locked independent evaluation or evidence about
uncollected long-context/tool/reasoning features. It supports a concrete
negative decision: the tested lagged bins should not be added to the serving
controller merely because current-proposal scores correlate with acceptance.

A current-proposal design would need to resolve worker graph selection,
TP-consistent cap choice, immutable asynchronous step counts, input positions,
accept/reject bookkeeping and ownership of queued proposals. Simply copying
a score to the host earlier, or shortening a tensor after FULL M8 dispatch,
does not establish saved target compute. Prototype the changed dependency and
measure its synchronization cost separately before claiming that the apparent
same-boundary opportunity can survive serving integration. The unresolved
[target repeatability investigation](REPEATABILITY-DIAGNOSTIC.md) is another
reason to keep this architectural change separate from the validated transport.
