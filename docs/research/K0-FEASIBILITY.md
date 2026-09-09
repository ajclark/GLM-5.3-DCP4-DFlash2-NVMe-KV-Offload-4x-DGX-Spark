# True K=0 fallback in the asynchronous DFlash2 runtime

**2026-09-09. Decision: defer serving implementation until repaired-runtime
calibration identifies a sustained low-acceptance workload.** A zero-draft
target step is supported by the input and sampling machinery. Skipping the
drafter safely is a separate change: it needs genuine M=1 target dispatch,
explicit worker ownership, and preservation of the draft context cache.
If measurement justifies that work, implement **eager context-KV maintenance
with the draft query pass parked** first. Defer retained-state catch-up until
its additional cache-publication contract is implemented and tested.

This investigation produced an [offline screen](../../bench/spec_k0_screen.py)
and [23 passing CPU tests](../../tests/test_spec_k0_screen.py). It changed no
runtime or deployment and launched no Spark requests. There is no measured
true K=0 timing, transition cost, or energy saving in this report.

## Why priority changed

The repaired runtime removes a major reason previously advanced for K=0:
the apparent collapse of draft acceptance at long context. The separate
[long-context diagnostic](LONG-CONTEXT-DIAGNOSTIC.md) identified an incorrectly
divided replicated draft block table. The root agent has now validated the
repair on the Sparks. Its [boundary report](../../results/adaptive-next/cache-width-r1/boundary-report.json)
records identical output token IDs at 89,055 and 92,056 prompt tokens,
56 accepted draft tokens in 10 cycles for both, and 42.77/43.00 decode tok/sec.
These are short counting diagnostics, not a representative coding benchmark.

New repository-context requests give the following observations at cap 7:

| Request | Prompt tokens | First-position acceptance | Accepted tokens / draft cycles | Decode tok/sec |
|---|---:|---:|---:|---:|
| Code, first run | 100,157 | 33/51 | 76/51 | 15.77 |
| Code, warm repeat | 100,157 | 35/47 | 83/47 | 16.63 |
| Prose, first run | 100,150 | 34/47 | 81/47 | 16.71 |
| Prose, warm repeat | 100,150 | 35/51 | 76/51 | 15.38 |

Sources: [code first](../../results/adaptive-next/cache-width-r1/repo100k/repo100k-code_repo_context-r0-fixed-k7.json),
[code repeat](../../results/adaptive-next/cache-width-r1/repo100k/repo100k-code_repo_context-r1-fixed-k7.json),
[prose first](../../results/adaptive-next/cache-width-r1/repo100k/repo100k-prose_repo_context-r0-fixed-k7.json),
[prose repeat](../../results/adaptive-next/cache-width-r1/repo100k/repo100k-prose_repo_context-r1-fixed-k7.json).
Each completion is 128 tokens. Aggregate draft counters can include terminal
work, so they should not be treated as exact counts of user-visible output.

The old approximately 6.4 tok/sec and near-zero first-position acceptance
used a different long prefix and 256-token outputs. This is evidence that
acceptance recovered, **not a clean paired speedup measurement**. Neither
the original low-acceptance prior nor its apparent K=0 opportunity should
be used to choose a policy for the repaired runtime. Fixed-cap recalibration
and confidence observation take priority. The local result paths above are
experiment artifacts; the CPU screen and tests do not depend on them.

## What already works, and what does not

The tests execute extracted methods and actual Triton kernels in the CPU
interpreter, using the existing pinned runtime fixtures plus a new
[hash manifest](../../tests/fixtures/spec_k0/manifest.json). They do not load
model weights or prove CUDA graph execution.

| Boundary | Evidence | Remaining implementation requirement |
|---|---|---|
| Empty speculative schedule | Empty dictionary, or an entry with an empty list, gives zero proposed tokens and one target logit for C1 | Represent the parked state explicitly; an empty entry is not persistent state |
| Target inputs | Actual input kernels ignore stale draft IDs when the scheduled count is zero | Preserve absolute positions and actual per-step counts |
| Sampling | `GPUModelRunner.sample` selects the ordinary sampler for an all-zero-draft batch | Keep target sampling authoritative |
| Mixed zero/positive rejection | Full rejection path, including the captured Gumbel resampling kernel, passes greedy fixtures containing caps 0 and 7 | Mixed-batch production support still needs ownership and graph validation |
| Commit/rollback | Zero draft commits one sampled token and rejects zero; queued 7→0 bookkeeping passes | Retain the immutable count associated with each in-flight step |
| Target graph | Current policy configuration dispatches C1 zero draft to PIECEWISE M=2; a test-only extra M=1 candidate dispatches FULL M=1 | Add and capture a real M=1 target graph, then inspect actual dispatch on the GPU |
| Drafter work | The runner still invokes `speculator.propose` whenever its speculator exists | Skip the query pass deliberately; empty scheduling alone saves no draft invocation |

Source anchors: [V2 model runner](../../overlay/vllm/v1/worker/gpu/model_runner.py),
[adaptive graph sizes](../../overlay/vllm/v1/spec_decode/adaptive.py),
[async scheduler](../../tests/fixtures/spec_runtime/async_scheduler.py),
[base DFlash speculator](../../tests/fixtures/spec_runtime/dflash_speculator.py),
and [captured resampling kernel](../../tests/fixtures/spec_k0/gumbel.py).
The actual-input tests cover positions 61, 2,047, 100,055, and 170,125 and
check untouched buffer tails as well as resulting positions and token IDs.

Do not set the configured draft length to zero. The trained DFlash2 block
remains eight tokens, with seven proposals and its existing selector/conv
geometry. K=0 is a per-step verification and worker-action choice. Current
positive caps remain 1/3/5/7. The asynchronous scheduler repopulates speculative
placeholders from `num_spec_tokens_to_schedule`; leaving that value at seven
restores seven placeholders after an otherwise empty step.

## A safe park/arm/resume protocol

The CPU prototype freezes each step as `(request ID, request epoch, sequence,
intent generation, verified cap, worker action)`. Later host decisions must
not rewrite a step already queued to the worker. There are four effective
states:

| State | Newly scheduled verification | Worker after target sampling |
|---|---|---|
| Active | Positive cap | Update context and generate the next proposal |
| Parked | K=0 | Maintain context eagerly, or retain committed combined states |
| Arming | K=0 until compatible acknowledgement | Restore missing context if needed, then generate a proposal |
| Armed but queued K=0 remains | K=0 for already scheduled work | Continue generating a fresh proposal after every target step |

The final row is essential. An arm acknowledgement can arrive after another
K=0 step is queued. That step advances the target position and invalidates
the earlier proposal's anchor. Generating a proposal once during arming and
then skipping subsequent queued K=0 steps would resume verification with
stale IDs. The prototype rejects precisely this broken one-shot design.
Keeping drafting active after arming makes the latest proposal match the
next positive verification, even while old zero-cap work drains.

Park requests similarly leave already queued positive-cap steps intact.
Those steps still need proposals and context maintenance. Acknowledgements
must be associated with the request epoch and current intent generation;
a stale arm acknowledgement cannot override a later park or a reused request
slot. Cancellation cannot release retained storage until queued work and
the relevant GPU events/copies have drained. The CPU tests exercise two
queued steps, stale acknowledgements, request replacement, and cancellation.

The worker owns the actual proposal anchor and draft-cache frontier. The
scheduler owns future scheduling intent. An acknowledgement's validity must
follow an explicit stream/event contract, not merely the return of a Python
function. [AsyncOutput](../../tests/fixtures/spec_confidence/async_utils.py)
begins its result copy before the subsequent proposal in the
[current runner](../../overlay/vllm/v1/worker/gpu/model_runner.py). An arm-ready
flag cannot be attached to that earlier event while referring to unfinished
proposal work. A future implementation must publish readiness at a later
valid boundary without adding an unconditional GPU synchronization.

Initial eligibility remains C1 greedy. If a second client or an unsupported
request arrives, a parked request cannot instantly fall back to cap 7 with
no proposal. It must pass through the same arm protocol or a separately
validated admission transition.

## What must be retained to resume

The captured [DFlash model](../../tests/fixtures/spec_k0/qwen3_dflash.py) and
[checkpoint configuration](../../tests/fixtures/spec_k0/draft-config.json)
establish the dimensions. The target feature layers are
`[5, 19, 33, 47, 61, 75]`. The constructor builds a linear combination from
`6 × 6144 = 36864` features to 6,144 features. CPU tests execute that actual
constructor with allocation-free layer stand-ins and check the dimensions.
The six-layer draft uses a 2,048-token sliding window.

`precompute_and_store_context_kv` directly projects these combined target
states into each draft layer's K/V, applies normalization and absolute-position
RoPE, and writes the slots. It does not depend on replaying historical draft
query layers. This makes bounded catch-up conceptually possible: retain
committed combined states for the live window and reconstruct missing context
K/V before generating a new query block.

At a completed target step, only its accepted prefix plus the ordinary target
input row belong to the committed context. Rejected speculative input tails
must be excluded. The newly sampled bonus token has no target hidden state
yet; its position is the next query anchor. Store original absolute positions,
not rebased window coordinates, and copy retained states out of reused target
buffers. Recompute physical slots from the current request's block table on
re-entry. Retaining old physical slot IDs across block reuse is unsafe.

### Memory is manageable only with a bounded catch-up implementation

All figures below are per rank and assume BF16 retained features. The
screen reserves 2,048 window rows plus eight rows of slack for one C1 lease.

| Storage/workspace | Source-derived size |
|---|---:|
| One combined target row | 12,288 bytes, or 12 KiB |
| Six raw auxiliary rows for one token | 73,728 bytes, or 72 KiB |
| Combined-state ring, 2,056 tokens | 24.09 MiB |
| Raw-auxiliary ring, 2,056 tokens | 144.56 MiB |
| Combined ring plus estimated 128-token replay workspace, 2 KV heads/rank | 28.97 MiB |
| Combined ring plus estimated full-window workspace, 2 KV heads/rank | 102.19 MiB |

The workspace estimate includes normalized context, an explicit copy for a
wrapped ring, flat and rearranged K/V projections, normalized K, and repeated
positions. These are tensor-size estimates, **not measured peak allocations**;
existing weights, KV pools, graph memory and allocator overhead are excluded.
Actual KV-head sharding must be inspected rather than inferred from a draft
TP configuration label. A 128-token replay chunk also stays below the initial
64 MiB screen at eight KV heads/rank. Full-window replay does not pass that
screen even at two heads/rank.

Allocate one C1 retention lease, not one ring for each of twelve possible
request slots. Retaining combined states still runs the combining linear
layer on each parked target step. Retaining raw auxiliary states would avoid
that work but exceeds the proposed memory budget. Recomputing an entire long
prefix to resume is not an acceptable bounded alternative.

### Deferred context writes change the GPU prefix-cache contract

This is the strongest reason to prefer eager maintenance initially. The
[async scheduler](../../tests/fixtures/spec_runtime/async_scheduler.py) calls
`kv_cache_manager.cache_blocks(request, computed - output_placeholders)` after
committed results. That shared frontier assumes each cache group has valid
materialized content. If the target advances while draft K/V writes are
deferred, the target frontier alone cannot establish that a draft page is
valid for publication and reuse.

A deferred implementation therefore needs a per-group materialized frontier
or equivalent suppression of draft-prefix publication, preserved through
preemption, offload/load, cache reset, and request reuse. Prompt-only disk-cache
policy does not establish this property for the GPU prefix cache. The CPU
prototype explicitly rejects publication beyond its draft materialized frontier.
If the presumed base cache becomes invalid, it also refuses re-entry until
the full needed window is available. Having retained a short recent gap is
insufficient when the earlier supposedly valid base has been lost.

Eager maintenance retains the existing materialization contract: run feature
combination and context-KV projection/storage on every committed step, but
skip the eight-token draft query pass while parked. It needs little additional
persistent storage and avoids deferred-prefix invalidation. Its saving could
be smaller; that is a measurement question, not a reason to omit the contract.

## Cost and energy gates

For conditional acceptance probabilities `p[j]`, the screen computes
`E[K] = 1 + sum(i=1..K, product(j=1..i, p[j]))` and chooses the lowest measured
positive-cap cycle cost divided by `E[K]`. Let that effective cost be `T`.
A true K=0 trial must first satisfy `t0 < T`, where `t0` includes all retained
maintenance, target M=1 execution, and scheduling overhead. Over a parked
interval of `N` tokens, time saving additionally requires:

```text
N * (T - t0) > resume_and_transition_cost
```

The transition term includes draining/arming work and every queued K=0 step
that generates proposals while armed, not just the cheapest context replay
kernel. The implementation prints unknown timings as `unmeasured`; optional
timing inputs are explicitly labeled scenario assumptions.

The default historical v3 cost curve yields modeled thresholds of 47.14,
52.97, and 50.75 ms per emitted token around 220, 4,229, and 32,267 context
tokens. Its approximately 109 ms thresholds at 100k/170k use pre-repair data
and are unsuitable for promotion decisions. These calculations demonstrate
the screen; they do not estimate true K=0 performance. Replace both cost and
acceptance inputs with repaired, workload-relevant calibration.

Proceed to a bounded eager-maintenance trial only if repaired traces show
sustained low acceptance and credible dwell time. Require actual FULL M=1
dispatch, no parked query pass, fixed-logit greedy/kernel invariants, correct
cache reuse after park/resume, and bounded memory before broader testing.
On the Sparks, observe memory pressure before capture or allocation, use one
short controlled request first, then paired coding/prose runs. A candidate
should beat the best repaired positive-cap policy on held-out latency or
energy without a material regression in the other objective. Inspect per-rank
peak memory and transition tails as well as average tok/sec.

Measure device joules/token and integrated wall energy per completed task
separately. Fewer draft invocations could save active energy, but longer target
execution can offset that saving. Parking a resident drafter does not unload
weights or establish a reduction in loaded idle watts. An idle power policy
remains a separate experiment.

## Reproduce the local evidence

```bash
.venv/bin/pytest -q tests/test_spec_k0_screen.py
.venv/bin/python bench/spec_k0_screen.py
.venv/bin/python bench/spec_k0_screen.py --costs /path/to/repaired-costs.json
```

The tests require no ignored runtime inventory or result artifacts. Newly
pinned sources and their original SHA256 hashes are in
[tests/fixtures/spec_k0](../../tests/fixtures/spec_k0/manifest.json); existing
sampler, scheduler and input fixtures are reused. The CLI's default historical
cost file is an experiment input and is explicitly replaceable. CPU tests
validate protocol counterexamples and exact kernel behavior under the
interpreter; GPU ordering, capture memory, actual cache lifecycle integration,
and real K=0 timings remain to be established.
