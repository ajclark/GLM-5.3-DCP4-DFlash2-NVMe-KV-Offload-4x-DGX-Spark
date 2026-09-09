# Target repeatability diagnostic

**2026-09-09.** The saved runs show a material target-distribution change,
including a functional coding failure, with the same prompt and matching
generated prefix. This is not adequately explained by tiny final-token ties.
The first focused control should disable **dense Marlin atomic reduction**:
the live environment enables it, and the actual model contains a frequently
executed projection eligible for that path. This is a falsifiable hypothesis,
not an identified root cause. Sparse-attention causality, cache identity and
asynchronous input ownership remain correctness questions until measured.

This investigation made no serving, deployment or inference changes. Its
[13 CPU tests](../../tests/test_repeatability_diagnostic.py) exercise actual
tracked sparse kernels. They establish causal contracts and counterexamples
for deliberately wrong metadata; they do not reproduce the live variation.

## What the six controlled runs establish

The [saved experiment](../../results/adaptive-next/cache-width-r1/quality-logprobs)
used the same 100,189-token prompt, temperature zero, seed 42, disabled thinking,
top-two target logprobs and a 768-token output limit. All six requests actually
scheduled **K7 throughout**, including requests labelled adaptive. They ran on
the repaired V4 runtime, before deployment of the confidence shadow collector.
Thus these failures do not demonstrate an adaptive cap transition problem.

| Run | Output tokens | Functional interval test | Output identity |
|---|---:|---|---|
| r0 fixed | 285 | Pass | Canonical |
| r0 adaptive | 285 | Pass | Canonical |
| r1 adaptive | 281 | Pass | Different |
| r1 fixed | 285 | Pass | Canonical |
| r2 fixed | 284 | **Fail: merges adjacent integer intervals** | Different |
| r2 adaptive | 285 | Pass | Canonical |

The [divergence report](../../results/adaptive-next/cache-width-r1/quality-logprobs/logprob-divergence-report.json)
contains source hashes and pairwise shared-prefix checks. Output positions below
are zero-based. At position 106, r0 fixed prefers `list` over `tuple` by
approximately 0.75 logprob units; r1 adaptive prefers `tuple` over `list` by
0.50, after 106 identical preceding output IDs. The change in the same
two-token logit difference is therefore approximately **1.25**. Even runs
with identical output IDs have different gaps at this position: 0.75, 0.625
and 0.125 occur among canonical outputs.

The stronger semantic example is r1 adaptive versus r2 fixed. They share
**140 output IDs**, then choose between `:\n` and ` +`:

| Run | log p(`:\n`) | log p(` +`) | Difference, colon minus plus |
|---|---:|---:|---:|
| r1 adaptive | −0.201954886 | −1.701954842 | +1.499999955 |
| r2 fixed | −1.223758698 | −0.348758698 | −0.875000000 |

That pairwise difference moves by approximately **2.375**, at the branch
which introduces the incorrect adjacency behavior. For two tokens in the
same distribution, subtracting log probabilities cancels the log-softmax
normalizer: `log p(a) − log p(b) = logit(a) − logit(b)`. This is evidence of
changed target preferences, assuming correct row association in the reporting
path; it cannot be attributed merely to a noisy normalization constant.

The captured [RejectionSampler](../../results/adaptive-next/runtime-inventory/v1/worker/gpu/spec_decode/rejection_sampler.py)
passes **target** logits to its logprob helper. The captured
[logprob kernel](../../results/adaptive-next/runtime-inventory/v1/worker/gpu/sample/logprob.py)
computes normalization in FP32 and retrieves the corresponding target logits.
These are not the drafter's confidence scores. The discrete-looking gaps do
not establish their cause or make a 2.375 preference change acceptable.

Matching prompt/output IDs still does not prove matching GPU verification
rows, draft suffixes, KV bytes, prefix-cache frontier, graph execution or
expert packing. Those are the next missing observations. Six runs are a
diagnostic sample, not a reliable estimate of the failure probability.

## The installed quantization path narrows the first control

The captured [target configuration](../../results/adaptive-next/runtime-inventory/target-config.json)
and [model class](../../results/adaptive-next/runtime-inventory/model_executor/models/deepseek_v2.py)
resolve the model as `GlmMoeDsaForCausalLM`, inheriting `DeepseekV2ForCausalLM`.
It has hidden width 6,144, 78 layers, three initial dense MLPs, 256 routed
experts with eight selected per token, and one shared expert of intermediate
width 2,048. Attention has 64 heads, Q rank 2,048, KV rank 512, NoPE dimension
192, RoPE dimension 64 and value dimension 256.

The checkpoint uses **W4A16 routed experts and W8A16 dense linears**, with
group size 128 for target layers. It does not declare dynamic INT8 activation
quantization. The first layer, routers and sparse indexers are excluded by
the quantization rules. Boot evidence selects `MarlinLinearKernel` and
`CompressedTensorsWNA16MarlinMoEMethod`. Calling the dense path W8A8, or
attributing it to per-batch activation scaling, would be incorrect.
The later [atomic-control build capture](../../results/adaptive-next/runtime-inventory/atomic-control-build.json)
also confirms `VLLM_MARLIN_INPUT_DTYPE` is unset, excluding that environment
override as an explanation for activation quantization here.

The [live environment capture](../../results/adaptive-next/runtime-inventory/repeatability-env.json)
reports `VLLM_MARLIN_USE_ATOMIC_ADD=1`; batch invariance and the captured
cuBLAS/TF32 overrides are unset. In the installed
[Marlin utility](../../results/adaptive-next/runtime-inventory/model_executor/layers/quantization/utils/marlin_utils.py),
`should_use_atomic_add_reduce` accepts CUDA projections with padded local
`N < 2048` and `K >= 2048` when that flag is enabled. The BF16 restriction for
pre-SM90 devices does not exclude SM121. `apply_gptq_marlin_linear` passes
this decision to the custom GEMM. FP32 reduction is separately enabled by
default; that argument does not cancel `use_atomic_add=True`.

The following dimensions follow from the actual classes and TP4. They assume
ordinary TP for shared experts; the boot's disabled sequence-parallel fusion
is consistent with that, but the exact `use_sequence_parallel_moe` value and
loaded per-module shapes should be recorded once at boot.

| Projection | Local input K | Local output N | Python atomic gate |
|---|---:|---:|---|
| Fused attention Q/KV A, replicated | 6,144 | 2,624 | No: N too large |
| Attention Q B | 2,048 | 4,096 | No |
| Attention KV B | 512 | 7,168 | No |
| Attention output | 4,096 | 6,144 | No |
| Dense MLP merged gate/up | 6,144 | 6,144 | No |
| Dense MLP down | 3,072 | 6,144 | No |
| **Shared-expert merged gate/up** | **6,144** | **1,024** | **Yes** |
| Shared-expert down | 512 | 6,144 | No |

The eligible shared projection occurs in layers 3–77: **75 instances**. It
is TP-sharded, while the fused attention A projection is replicated. A
separate 576-wide KV A projection would be eligible, but this target enables
Q/KV A fusion and computes the combined 2,624-wide output. Confusing the
unfused checkpoint names with the instantiated module gives the wrong answer.
Marlin padding does not cross the gate boundary for the highlighted shape.

This establishes Python dispatch eligibility, not that the compiled CUDA
kernel uses multiple reduction slices on every eligible call. The exact
compiled Marlin source/build identity, selected tile configuration and
`slice_count` remain useful evidence. The separate `upstream-vllm` checkout
has a different revision and must not be presented as the running binary.
The read-only build search found no `*marlin*.cu` files under `/workspace`,
`/opt` or the installed package root. That bounded negative result does not
establish that no build source exists elsewhere.

The captured [routed-expert Marlin implementation](../../results/adaptive-next/runtime-inventory/model_executor/layers/fused_moe/experts/marlin_moe.py)
explicitly passes **`use_atomic_add=False, use_fp32_reduce=True`** to both
expert GEMMs. A blanket claim that routed Marlin MoE atomics explain these
outputs is contradicted by the inspected path. Disabling the dense flag
therefore isolates a much narrower mechanism than replacing the MoE backend.

## Ranked hypotheses and what would distinguish them

This ordering prioritizes evidence and cheap isolation; it is not a probability
estimate. Several mechanisms may coexist.

1. **Dense atomic reduction changes intermediate states.** This has a live
   enabling flag and eligible repeated layers. Atomic reduction can have
   order-sensitive floating-point results; their magnitude and propagation
   on these real weights have not been measured. Hold the prompt, cache
   provenance, K7 graphs and other flags fixed, change only the atomic option,
   and compare repeated target preferences and functional outputs. If the
   variation persists with confirmed non-atomic execution, this is not a
   sufficient explanation.
2. **Verification-row or cache identity differs.** Asynchronous proposal
   ownership, accepted-prefix advance, rollback, cache reuse and DCP slot
   mapping must agree at the exact row returning the divergent logprob.
   Existing scheduler traces record K7 but not all actual GPU input IDs and
   positions. The upcoming shadow collector can validate proposal anchors
   and verified suffixes. Add the target row/output-index association when
   localizing a divergence; a shared emitted prefix alone cannot validate it.
3. **Forward arithmetic and discrete routing amplify a smaller difference.**
   The same K7 does not force identical expert-packed batches: future draft
   rows, acceptance boundaries and the row representing a given output token
   can differ. Routed token packing, reductions and sparse top-k selection
   can magnify earlier hidden-state changes. Record the first layer where
   matching input states yield different output states, then compare router
   scores, selected expert IDs and sparse candidate sets there. No such
   layer-level measurement exists yet.
4. **Sparse causal bounds, candidate ownership or reused cache state is
   incorrect.** The sparse consumer does not apply an independent absolute
   causal mask. Incorrect per-row bounds or stale candidate indices can
   expose speculative future tokens even when the tensor shape is correct.
   Assert global candidate positions never exceed the query position, local
   candidates belong to this DCP rank, physical slots belong to this request,
   and cache generation/frontier matches. This remains a correctness audit,
   not a demonstrated live bug.
5. **Preemption, cache restoration or other execution changes.** Compare
   request preemption counters, prefix-hit lengths, disk/GPU cache provenance,
   chunked-prefill boundaries and graph metadata. Do not assume preemption
   occurred merely because the model is large. Identical restored KV bytes
   with identical row state would weaken this hypothesis. Hardware faults
   are lower priority without device errors or a failing controlled tensor
   replay, but should not be dismissed if those appear.

One source/config mismatch deserves its own observation. The target config
contains `moe_router_dtype="float32"`, but its actual inherited constructor
does not consume that field or pass `params_dtype`, `out_dtype` or
`force_fp32_compute` to [GateLinear](../../results/adaptive-next/runtime-inventory/model_executor/layers/fused_moe/router/gate_linear.py).
The inspected specialized kernels target SM90/SM100 and other supported
shapes, not this SM121 6,144-by-256 case. Confirm the loaded router weight,
input and output dtypes and any later setter before alleging an FP32-policy
violation. Upgrading all 75 replicated router weights from BF16 to FP32
alone would add approximately **225 MiB per rank**, before workspaces; it
is not an unbudgeted follow-up to the atomic control.

## Sparse-attention audit and concrete CPU evidence

The tracked [indexer metadata](../../overlay/vllm/v1/attention/backends/mla/indexer.py)
expands per-token causal lengths before DCP localization; existing tests
cover why reversing that order is wrong. The
[sparse backend](../../overlay/vllm/v1/attention/backends/mla/flashmla_sparse.py)
explicitly relies on indexer candidates for causality, filters candidates by
DCP ownership, and normalizes empty-owner output/LSE before the collective
merge. The global-ID packing through FP32 is exact at these 100k/170k
positions, below the 24-bit integer precision boundary. This inspection
does not support a generic positional integer-overflow explanation.

The new tests execute `_fp8_paged_mqa_logits_rowwise_kernel` directly from
[sm12x_mqa.py](../../stage/glm-triton/sm12x_mqa.py). The model's 32-head,
128-dimensional indexer meets the rowwise wrapper's shape predicates. Tests
use eight verification rows, permuted physical pages, and logical chunk
starts 0, 100,096 and 170,016. They show:

- Poisoning keys and scales beyond every row's allowed prefix with NaN or
  infinity leaves valid logits unchanged; entirely future tiles become −∞.
- A changed key affects only query rows whose supplied bounds allow it.
- Changing one query row does not change other rows.
- Deliberately broadcasting the final row's bound exposes future keys to
  earlier rows. The kernel cannot repair incorrect metadata on its own.

Tests also execute `_fused_gather_dequant_attn_kernel` from
[sm12x_sparse_mla_attn.py](../../stage/glm-triton/sm12x_sparse_mla_attn.py)
using the real 656-byte FP8/scale/BF16-RoPE layout. Sentinel, out-of-bounds
and beyond-length candidates are masked. A physically valid slot designated
as future contributes whenever included in the candidate list: there is no
absolute query-position argument at this layer.

These are CPU address, masking and data-dependency tests. The indexer test
uses FP32 key storage to isolate those properties. The fused-attention test
adapts the installed interpreter's BF16 dot handling, which otherwise treats
BF16 storage bits as unsigned integers, to a NumPy FP32 dot. Neither test
simulates tensor-core rounding, atomics, CUDA graph timing or NCCL. Passing
them rules out the tested standalone kernel errors under correct inputs; it
does not prove that live metadata is correct.

## What official batch invariance can and cannot promise here

Current vLLM documentation describes `VLLM_BATCH_INVARIANT=1` as a **beta**
feature for reproducible outputs across batching changes. Its tested models
include several DeepSeek and Qwen variants; it does not establish coverage
for this GLM5.3 mixed W4A16/W8A16, SM121 sparse-MLA, DFlash2, TP4/DCP2 fork.
Performance can change when optimizations are disabled.
[Official batch-invariance documentation](https://docs.vllm.ai/en/latest/features/batch_invariance/)

Official reproducibility guidance does not promise reproducibility by
default; a fixed seed alone is insufficient. It documents batch invariance
for online serving. Disabling V1 multiprocessing is an offline alternative,
not a validated remedy for this online distributed deployment.
[Official reproducibility guidance](https://docs.vllm.ai/en/stable/usage/reproducibility/)

The [upstream tracker](https://github.com/vllm-project/vllm/issues/27433)
still lists work around prefix caching and speculative decoding. The open
[speculative-invariance PR #52522](https://github.com/vllm-project/vllm/pull/52522),
opened August 16, 2026, focuses on stochastic speculation and reconstruction
of proposal state after preemption. Its current description removes RNG
domain separation into independent work. It reports EAGLE3/DFlash/DSpark
tests, but does not validate this deployment. Its prefix-cache replay boundary
is relevant only if actual preemption/reconstruction is implicated here.

A separate open [forward-pass RFC #54506](https://github.com/vllm-project/vllm/issues/54506),
opened August 31, 2026, reports M-dependent kernels, compilation/fusion
differences and execution-form asymmetry on another FP16 hybrid MoE/MTP
stack. It distinguishes forward-pass invariance from sampling fixes and
warns that instrumentation can alter compilation. This is useful diagnostic
methodology, not evidence that its measured causes explain our larger logit
swings. The same distinction between individually deterministic operations
and batch-invariant inference is developed in the primary
[Thinking Machines investigation](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).

Our installed [batch-invariance module](../../results/adaptive-next/runtime-inventory/model_executor/layers/batch_invariant.py)
is particularly consequential: it changes cuBLAS workspace settings, overrides
several ATen operations, disables reduced-precision reductions, and forces
NCCL settings including one channel and Simple/Tree collectives. Its SM80
path replaces generic matrix multiplication; other architectures follow a
different branch. It does **not** explicitly disable the dense Marlin atomic
flag. The custom rowwise indexer explicitly requests `input_precision="tf32"`
inside Triton, which is not rewritten merely by a PyTorch TF32 preference.
Consequently, enabling the global flag would both confound the first control
and leave important custom paths requiring verification.

## Minimal next controls, with explicit stopping conditions

**First: one atomic-setting control.** Keep fixed K7, request payloads, model,
graphs, TP/DCP and logging constant. After the existing experiment queue is
empty, record host/GPU memory pressure on all four nodes, then make a guarded
boot with `VLLM_MARLIN_USE_ATOMIC_ADD=0`. Confirm the effective flag and one
eligible module's dispatch without copying weights. Run a bounded initial
set of eight same-prompt repeats; retain every functional result and target
top-two gap. Compare within-setting variation as well as between-setting
means. Measure decode TPS and device/whole-node energy separately if available.
No fixed future headroom allowance or new model allocation is needed by this
design, but graph/workspace changes still require measured memory checks.

Cache provenance must be explicit. Reusing an **identical documented warm
prompt-cache snapshot** across settings is a useful decode-only control.
It does not make the full run atomic-free because its prompt KV may have
been produced with atomics enabled. A separate fresh cache salt/rebuild
tests the full path. Never silently mix those interpretations. A clean
atomic-off sample supports further testing; eight successful outputs do not
prove determinism or a generally improved model. The prepared isolated launch
control at commit `d43f62f` refuses persisted-cache reuse, implementing the
fresh-cache form of this experiment; no atomic-off measurement was available
when this report was finalized. Persistent variation rules
out this flag as a complete fix and triggers localization.

**Second: establish actual row identity.** Use bounded shadow packets to
record the previous proposal's immutable IDs/positions and actual verified
suffix, request incarnation, target row and corresponding output index.
Reject mismatched anchors as invalid diagnostics without changing output.
Record per-row sequence bounds and compact hashes/min/max/counts of candidate
IDs for the divergent query; inspect exact candidate lists only at the first
suspect layer. A single 2,048-entry int32 list is 8 KiB, but copying it across
every layer/step would be a materially different instrumentation budget.

**Third, only if needed: localize the first changed state.** Prepare a replay
of the canonical accepted prefix on the sandbox, then compare a selected
target query with identical cache/position state and either identical or
changed future verification suffix. Separate fixed-shape repeated execution
from M1-versus-M8 tests. Capture one layer boundary at a time, moving from
attention/indexer to router/shared expert/routed expert once the first change
is located. Keep instrumentation placement and compiled graph structure
symmetric across controls. Record hashes plus selected-row values/norms;
do not retain all 78 layers of hidden states or full-vocabulary logits.

If the first discrepancy is an invalid slot, future candidate or mismatched
request/row, stop performance promotion and fix that invariant. If correct
identical inputs first diverge in a dense projection, run a small real-weight
kernel replay under atomic on/off before considering broader replacements.
If attention or routing first diverges, test its corresponding bounded path
or dtype control. Global batch invariance and whole-engine eager mode belong
after this localization, with an explicit compatibility and memory budget.

The performance work should continue to retain functional code checks and
prompt-level prose quality gates. Variation in fixed K7 is relevant baseline
evidence; it does not excuse an observed regression or establish that a faster
policy is ready for promotion.

## Evidence inventory and reproducibility

The live wheel identifies itself as `v0.23.1rc1.dev190+gab6660699.d20260830`;
local serving repairs and overlay changes are additional to that base.
Installed source hashes are recorded in
[repeatability-source-sha256.json](../../results/adaptive-next/runtime-inventory/repeatability-source-sha256.json)
and the paths in
[repeatability-paths.json](../../results/adaptive-next/runtime-inventory/repeatability-paths.json).
All 34 entries present at this audit were hash-verified, including the target
model/config. These ignored captures are experiment evidence, not portable
test prerequisites. The new tests read only tracked stage files and the
existing test harness.

Validation: `.venv/bin/pytest -q tests/test_repeatability_diagnostic.py` —
**13 passed**. This report's public-source statuses were checked September 9,
2026; current web documentation and open-PR bodies can change independently
of the pinned installed runtime.
