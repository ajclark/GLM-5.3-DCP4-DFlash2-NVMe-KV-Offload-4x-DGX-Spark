# Remaining architecture screens after the long-context repair

**2026-09-09.** This is a local implementation and feasibility screen, not a
new Spark benchmark. It covers selector training, exact-copy reuse, tree
verification, draft-memory compression and longer draft blocks. The
[offline script](../../bench/spec_architecture_screen.py) has
[38 passing CPU tests](../../tests/test_spec_architecture_screen.py), including
an exact ancestor-closed tree dynamic program checked against brute force.
No deployment, network call, training run or checkpoint change was made.

The repaired runtime changes the priority of these ideas. The new 170k
calibration observed the following decode rates:

| Workload | K1 | K3 | K5 | K7 |
|---|---:|---:|---:|---:|
| Repository code | 15.88 | 17.10 | 16.12 | 14.11 |
| Repository prose | 15.18 | 17.41 | 19.11 | 17.87 |

These are individual calibration runs, not paired confidence intervals.
Sources: the root agent's [170k calibration artifacts](../../results/adaptive-next/cache-width-r1/cal-170k)
and [cost table](../../results/adaptive-next/cache-width-r1/costs-170k.json).
Measured complete-cycle costs for K1/3/5/7 were approximately
110.42/133.24/151.13/168.25 ms. Eligible full-seven cycles after warmup had
109 first-position acceptances out of 174 trials. The old apparent
long-context acceptance collapse is no longer a sound premise for an
architectural replacement. Healthy acceptance and differing code/prose
preferences support measuring the smaller interventions first.

## Decisions and the evidence that would change them

| Direction | Current decision | Next evidence gate |
|---|---|---|
| Selective selector training | Prepare an offline data/ablation design; defer training | Show that the correct target token is often in the candidate set but the selected path is wrong |
| Exact-copy reuse | Keep as an opportunity screen; defer live bypass | Demonstrate substantially more useful matches on held-out edit/copy tasks, after accounting for the existing drafter |
| Branch/tree verification | CPU selection implemented; defer target integration | Obtain valid branch probabilities and independently measured tree-mask costs |
| Draft-only compression | Defer broad changes; measure physical draft allocation first | Establish releasable bytes or bandwidth savings large enough to exceed conversion cost and acceptance loss |
| Longer blocks | Defer changing the loaded checkpoint's geometry | Find frequent full-seven acceptance and validate a suitably trained checkpoint before new graphs/rollback paths |

The [confidence shadow collector](CONFIDENCE-FEASIBILITY.md) is the nearest
measurement step for distinguishing selector uncertainty from ordinary
acceptance history. Its current packets intentionally omit candidate IDs and
hidden features, so they cannot answer every gate in this table.

## Selective selector training: the head is not tiny

The pinned [checkpoint configuration](../../tests/fixtures/spec_k0/draft-config.json)
sets hidden width 6,144, vocabulary 154,880, selector rank 256 and top-k 16.
The actual [CandidateSelector constructor](../../tests/fixtures/spec_confidence/qwen3_dflash2.py)
allocates two full vocabulary-by-rank codebooks and a bias-free replicated
hidden projection. The CPU test executes that constructor using meta tensors
and checks actual parameter shapes without allocating model weights.

| Component | Parameters | BF16 weights |
|---|---:|---:|
| Predecessor and successor codebooks | 79,298,560 | 151.25 MiB |
| Hidden projection, 6,144 → 256 | 1,572,864 | 3.00 MiB |
| Complete selector | 80,871,424 | 154.25 MiB |

Codebooks are ordinary full-sized parameters; the projection is explicitly
replicated. These are per-process weight sizes, not a count divided by target
TP. The inference implementation marks the codebooks `requires_grad=False`.
A training experiment must deliberately change the training model's parameter
selection; merely importing the serving module does not train them.

An explicit mixed-precision Adam scenario uses 16 bytes per trainable
parameter: two bytes each for BF16 weights and gradients, four for an FP32
master weight, and eight for two FP32 moments. Training the entire selector
therefore requires **1,234 MiB of parameter state before activations and
workspace**. This is not a measured peak, and other optimizer/dtype choices
change it. Frozen draft weights, stored features, temporary gradients,
communication and allocator overhead must also be included. The screen leaves
`full_training_peak_bytes` unknown. Do not train on a Spark alongside the
resident target model based on the 154 MiB inference-weight figure.

More bounded candidates are:

- Freeze both codebooks and train the projection: 24 MiB of trainable
  parameter/optimizer state plus 151.25 MiB of frozen codebooks, before other
  model and activation costs.
- Add rank-16 LoRA to that projection: 102,400 trainable parameters and
  1.56 MiB of the same parameter-state scenario. The original frozen selector
  weights and activations still exist. This is a proposed adaptation, not
  support already present in the serving `ReplicatedLinear` path.
- Update a restricted vocabulary subset only with an explicitly sparse
  parameter/optimizer design. Ordinary indexing into a full parameter does
  not guarantee sparse gradients or sparse optimizer state. This is a more
  complicated first experiment than projection-only training.

The useful failure decomposition is **candidate coverage versus path choice**.
At each reached verification position, did the target's greedy token appear
among the 16 candidates? If absent, training the selector alone cannot choose
it. If present but not selected, ask whether projection/codebook interactions
can rank it correctly using the available features. A globally positive
temperature rescaling preserves greedy argmax; calibration alone can help a
cap policy but does not automatically repair the proposed token.

A bounded future dataset would capture candidate IDs, unary scores, the
anchor token, reached target labels and frozen draft hidden states. Seven
BF16 hidden rows alone cost 86,016 bytes per proposal, versus the current
656-byte confidence packet. Limit examples explicitly and store them offline;
do not widen every production trace. Hold out complete prompts and all their
repeats, separate repaired from original-runtime data, and respect censoring
after rejection. Train a small projection adaptation only if coverage and
selector-error counts justify it. Compare against the original selector and
acceptance-history/confidence policies with the same target verifier.

No trained artifact is produced here. A candidate must improve held-out
accepted tokens per measured complete cycle, preserve task-level code/prose
quality, and pass memory/energy controls before promotion.

## Exact-copy reuse: current opportunity is sparse

The root agent's repaired-context copy screen uses suffix widths 16, 8 and 4,
a bounded 180,224-token lookback, and a seven-token candidate continuation.
The following are **matches on the recorded baseline trajectory**:

| Context / workload | Matching boundaries | Mean agreed copied prefix when matched | Agreed copied tokens per baseline boundary |
|---|---:|---:|---:|
| 100k code | 4 / 86 | 0.75 | 0.0349 |
| 100k prose | 4 / 90 | 3.00 | 0.1333 |
| 170k code | 1 / 108 | 3.00 | 0.0278 |
| 170k prose | 5 / 83 | 1.60 | 0.0964 |

Sources: [100k copy report](../../results/adaptive-next/cache-width-r1/copy-100k.json)
and [170k copy report](../../results/adaptive-next/cache-width-r1/copy-170k.json).
There was one full-seven continuation match in the 100k prose screen and
none in its code counterpart. These counts do not measure a copy drafter's
closed-loop acceptance, its marginal benefit over DFlash2, tok/sec or energy.
Changing proposals changes subsequent target boundaries and lookup inputs.

The current samples do not justify a serving bypass. More promising tasks to
screen are exact patch application, repeated API scaffolding, structured
editing and requested quotation of supplied source. Use the actual tokenized
prompt/output rather than character matches, preserve request isolation, and
compare the copied continuation with DFlash2 at the same opportunities. A
copy candidate that merely duplicates a well-accepted DFlash proposal may
save draft-query work but adds little target-verification benefit.

An index over 180,224 int32 tokens has about 704 KiB of raw token payload per
request; index and hash storage are additional. A modest CPU index is feasible
in the VM. The asynchronous serving dependency is the harder issue: the CPU
may not yet know all tokens preceding a queued target step. A prefix match
must be checked against the exact current anchor and request incarnation,
not stale received text. GPU lookup or a later proposal boundary may be
needed to avoid a host wait. Any skipped query pass must preserve draft
context-KV maintenance and re-entry, as detailed in the [K=0 screen](K0-FEASIBILITY.md).
The full target verifier remains mandatory.

## Trees: an exact CPU selection screen, with costs kept separate

The new `tree_frontier` dynamic program selects an ancestor-closed subset of
a rooted candidate tree for every node count up to a supplied budget. The
ordinary target anchor is a virtual root. Every candidate has a parent, a
token and an estimated conditional target-agreement probability. Siblings
must represent **different, mutually exclusive tokens**, and their
probabilities must sum to at most one. Missing probability mass represents
an outcome outside the candidate set.

For node `v`, its reach probability is the product of conditionals along its
ancestor path. For an ancestor-closed selected set `S`:

```text
expected emitted tokens = 1 + sum(reach_probability(v) for v in S)
```

Only one branch can be accepted, which is why sibling exclusivity matters.
The program uses exact rational arithmetic for the supplied decimal values,
merges child-subtree budget tables, and breaks equal-value ties
deterministically. Twenty randomized small-tree cases are compared with
exhaustive subset enumeration, including shuffled input order. Additional
tests cover chains, complete exclusive branches, impossible probability sums,
duplicate sibling tokens, cycles, missing ancestors and unknown parents.

This is exact for **expected emissions at a node budget under the supplied
probability model**. It is not a global latency optimum. One topology with
slightly fewer expected tokens could execute faster than the selected
frontier topology. The cost screen therefore ranks only the emitted frontier
and states that limitation.

Costs are keyed by the full selected parent-index topology, including anchor
index zero. A two-node chain `[0,1]` cannot borrow a two-child branch cost
`[0,0]`. The input requires `kind=tree_shape`, explicit measured/synthetic
provenance, runtime identity, context and independently supplied complete-cycle
costs. Missing shapes abstain. Chain cap costs, including the repaired 170k
curve above, are deliberately unusable as tree costs. Even with measured
costs, the returned ratio is a modeled expectation, not measured tok/sec.

The `--demo` scenario supplies invented probabilities and invented shape
costs, prominently labels both, and serves only as a reproducible algorithm
exercise. It supplies no evidence that trees are faster on the Sparks.

### The current score trace cannot construct an alternate-parent tree

The actual DFlash2 model can compute edge scores with shape
`[batch, 7, predecessor candidate, child candidate]`; each edge combines a
unary logit and a codebook interaction. The live confidence collector stores
only the seven **realized predecessor rows**, each of width 16. Those rows
cannot reconstruct alternative-parent scores. Nor does a softmax over the
truncated candidate set provide calibrated target probabilities or account
for an absent target token. Inventing branches from those rows would overstate
what was observed.

A tree data screen needs candidate IDs, alternative-parent edge information,
and properly calibrated probability mass, including the outside-candidate
outcome. Reached linear verification labels do not label every branch. The
pairwise selector's scores also do not establish that arbitrary deeper paths
have the same target-conditioned probability model as the selected path.
Collect limited additional evidence before choosing a tree family.

Target integration would then need ancestor-only attention masks, distinct
KV slots for same-position siblings, exact chosen-path commit/rollback,
branch-to-output mapping, and graph/DCP support for the resulting layout.
The current chain rejection kernel cannot simply consume a tree list. Begin
with a tiny CPU attention reference that compares each branch to its
independent causal prefix, then a fixed bounded topology and independently
profiled target graph. Retain target greedy verification and existing
request/epoch ownership. Only proceed beyond CPU screening if calibrated
branch benefit survives that measured cost and memory budget.

## Draft-only compression: measure physical savings before changing formats

The checkpoint has six sliding-attention layers, head dimension 128 and a
2,048-token window. The attention constructor divides its eight global KV
heads by the active TP group, but the declared draft TP setting alone does
not establish the runtime group. The screen therefore reports **2/4/8 KV
heads per rank as scenarios**, rather than claiming one as measured.

For BF16 K and V, one live window's payload is 12/24/48 MiB per rank across
those scenarios. Adding an eight-token query block and rounding to 64-token
allocation blocks gives 2,112 tokens and 12.375/24.75/49.5 MiB. Ideal int8
payloads are half those figures, excluding scales, metadata and quantization
workspace. These calculations are source-derived tensor sizes, not a report
of actual allocated cache pages or peak memory.

The existing launch skips sliding-window layers when selecting FP8 KV; the
draft's intended context cache remains the full-precision path. A compression
trial would require compatible store/load/attention kernels and calibration
for per-layer/head scales. Quantizing only the draft may preserve the final
target-verification rule while degrading acceptance or adding enough
conversion work to hurt tok/sec and active joules/token.

More importantly, reducing payload bytes does not reduce process RSS if the
KV pool stays at its configured fixed size. Measure allocated pages per group,
head count, dtype and occupancy, then determine whether a revised pool budget
can release physical memory without reducing required serving capacity.
The target's large weights remain resident. No idle-watt saving follows from
the ideal 6–25 MiB payload reduction alone.

Quantizing the 154 MiB selector weights or a retained combined-state ring is
a separate possibility with separate accuracy/conversion costs. The latter
only matters if deferred K=0 maintenance is implemented; its cache-publication
contract is already a prerequisite. Neither variant is implemented here.

## Longer blocks: changing a dimension also changes computation

The loaded checkpoint is trained/configured for block size eight: one anchor
and seven mask/proposal tokens. Its grouped convolution uses the active block
boundary. A CPU test runs the actual `_grouped_conv` and demonstrates that
changing an eight-token block to sixteen changes computation at position
eight, even with identical inputs and weights. A successful reshape or kernel
launch is therefore not evidence of equivalent trained behavior.

Blocks 12 and 16 would request 11 and 15 proposals. The selector's realized
FP32 rows would grow from 448 bytes to 704/960 bytes, while its all-parent
edge tensor grows from 7,168 to 11,264/15,360 bytes per request. Those small
tensors are not the important complete-memory bound: draft activations,
target rows, any vocabulary-sized rejection buffers, cache allocation and
graph workspace also scale and must be measured.

First establish that fully accepted seven-token prefixes are frequent enough
on the desired workload to make extra positions useful. First-position
acceptance alone does not show this. Require a checkpoint validated for the
longer block or an explicit offline retraining/extension study, then update
graph capture, buffers, scheduler placeholders, actual counts, rejection and
rollback together. Current adaptive caps 1/3/5/7 and measured target shapes
M2/4/6/8 do not provide timings or correctness guarantees for M12/M16.
Given the repaired code/prose calibration often favors K3 or K5, longer blocks
are presently behind selective prediction/selection improvements.

## Reproduce and extend the local screen

```bash
.venv/bin/pytest -q tests/test_spec_architecture_screen.py
.venv/bin/python bench/spec_architecture_screen.py
.venv/bin/python bench/spec_architecture_screen.py --demo
.venv/bin/python bench/spec_architecture_screen.py --copy-report results/adaptive-next/cache-width-r1/copy-170k.json
.venv/bin/python bench/spec_architecture_screen.py --tree candidate-tree.json --budget 7 --costs measured-tree-shapes.json
```

The tests depend only on tracked source/config fixtures and ordinary CPU
PyTorch; model parameter checks allocate on the meta device. Spark result
files are optional CLI inputs and evidence links, not test prerequisites.
The program never trains, contacts a model or launches CUDA. Every future
serving experiment still needs the existing memory-pressure guard, independent
quality checks, measured per-task device/wall energy, and explicit rollback.
Loaded idle power remains a separate policy objective.
