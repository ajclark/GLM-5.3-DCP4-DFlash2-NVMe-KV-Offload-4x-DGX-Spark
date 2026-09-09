# Adaptive speculative decoding: research and implementation opportunities

**Follow-up evidence:** the investigation below led to a reproduced and repaired
draft-cache table defect. The old near-zero long-context acceptance and its
apparent 35–42% adaptive gains are confounded by that defect. Subsequent Spark
power, Pi and confidence experiments are recorded in the
[follow-up report](ADAPTIVE-SPECULATION-NEXT.md); the literature assessment and
its original starting observations remain distinguishable below.

The most promising next step is to improve the information used by the existing verification controller, while investigating why the deployed DFlash2 drafter loses almost all acceptance at long context. A second large draft model is a poor starting point for this cluster. A genuine target-only fallback and a separately measured energy controller deserve bounded prototypes after those investigations.

The literature supports this direction, but does not establish a transferable speedup. Recent systems increasingly combine candidate-level acceptance estimates, hardware-specific verification costs, and explicit graph execution. Several attractive papers already have substantial implementations. The clearest gaps found are EVICT's public serving implementation, the full author implementation of Cascade, and recent research on compressed draft memory and energy-constrained scheduling. These are opportunities to build and evaluate an adaptation, not evidence that no implementation exists anywhere.

## 1. Recommendation and scope

This assessment covers publicly accessible work available on **September 9, 2026**, emphasizing 2024–2026 papers, author repositories, and official serving runtimes. Paper submission dates and implementation availability are distinct: for example, LongSpec appeared on arXiv in February 2025 and in ACL 2026; DSpark's vLLM adaptive verification implementation merged on August 12, 2026. Repository claims below refer to inspected snapshots, with identifiers and search coverage in the [evidence inventory](research/adaptive-speculation-evidence.json). No external project was executed or benchmarked on the Sparks for this report.

| Priority | Proposed investigation | Why it fits this deployment | Initial deliverable |
|---|---|---|---|
| 1 | Calibrated selected-path confidence, constrained by async scheduling | Adds information to the existing cap controller without new target weights or large draft weights | A feature-availability audit, calibration dataset, and shadow comparison against version 4 |
| 2 | Long-context drafter diagnosis and draft-only context repair | Near-zero long-context acceptance is a large measured loss of useful work | Position/window/cache differential tests and a measured breakdown of where acceptance collapses |
| 3 | Real K=0 fallback with bounded draft-state catch-up | Current K=1 still generates seven proposals and verifies two target rows | A CPU state-machine prototype and a break-even model including re-entry cost |
| 4 | Energy-aware cap and operating-point selection | Directly targets J/token while retaining a declared C1 latency floor | Offline Pareto analysis, then a slow controller over measured settings |
| 5 | Targeted improvement of the existing DFlash2 selector or drafter | Coding gains from cap selection alone remain small | An offline failure taxonomy and a small, separately trained adapter only if that taxonomy supports it |

The first three can share existing traces, CPU fixtures, and bounded instrumentation. They should remain independent experiments: a controller change should not be combined with new draft weights before its contribution is understood.

## 2. Starting point and the limits of the current result

The repository serves GLM-5.3 743B with mixed INT4/INT8 weights on four DGX Sparks, TP4/DCP2, a 180224-token context limit, twelve configured sequences, and 6 GB of KV allocation per rank. Its experimental controller changes the target verification cap among 1, 3, 5, and 7. The DFlash2 query block remains eight and draft capacity remains seven; target FULL graphs actually execute M=2,4,6,8. The policy is restricted to eligible C1 greedy decoding and defaults to off. The [design](ADAPTIVE-SPECULATION-PLAN.md) and [completed experiment report](../results/adaptive-spec/README.md) describe the implementation and evidence.

The initial locked evaluation measured +15.80% paired prose throughput, with a 95% prompt interval of +13.24% to +18.73%. Coding measured +0.99%, with an interval of −2.11% to +4.44%. An unchanged-policy coding follow-up produced +1.14%, with an exploratory interval of −1.30% to +3.64%. Long-context repository screens showed approximately 35–42% gains at 100k/170k, where acceptance was extremely poor. Those screens are not a broad long-context benchmark.

The present policy already handles censored acceptance, context-dependent cycle costs, a weak acceptance prior, delayed feedback, warm-up, periodic exploration, hysteresis, and graph dispatch. Reimplementing a generic adaptive-length heuristic would add little. Its principal limitations are the information available before scheduling, missing K=0 execution, and the quality of the proposals themselves.

Measured serving memory headroom reached about 1.6 GiB on the head and 3.0–3.4 GiB on other ranks. This is an observed operating condition, not a new universal free-memory threshold. New buffers, graph captures, training state, and cache growth must fit the actual per-rank pressure envelope. Merely describing a draft model as “small” or an adapter as “parameter efficient” is insufficient.

Device energy per decoded token improved in the completed experiment. Whole-cluster wall energy and a reduction in loaded idle watts remain unestablished. Those distinctions continue throughout this report.

## 3. What has changed in the research frontier

### Acceptance control is becoming a systems problem

Foundational speculative decoding preserves the target distribution by correcting the drafter's proposals with the target. Adaptation can change which proposals receive verification while retaining that correction. For greedy decoding, the proposed prefix must be checked against the unchanged target rule; for stochastic decoding, proposal probabilities and the rejection correction must remain consistent. Mathematical losslessness does not by itself establish bitwise reproducibility across different quantized GPU execution shapes. [1](https://proceedings.mlr.press/v202/leviathan23a.html)

Earlier adaptive methods focused on stopping an autoregressive drafter. Hugging Face's dynamic speculation stops drafting when assistant confidence falls below a threshold; this became an assisted-generation default in Transformers 4.45.0. AdaEDL uses draft entropy as an acceptance proxy. Both are useful baselines for deciding whether additional draft work is worthwhile. Neither means that truncating a block after a DFlash2 forward saves that forward's work. [2](https://huggingface.co/blog/dynamic_speculation_lookahead), [3](https://proceedings.mlr.press/v262/agrawal24a.html)

SpecDec++ formalized candidate-length selection as a decision process and introduced an acceptance-prediction head. Its useful contribution here is the separation between a token being probable under the drafter and being accepted by the target. Its reported additional gains were approximately 7–11% over its speculative baseline on a Llama-2 7B/70B pair; those are not expectations for this MoE cluster. [4](https://arxiv.org/abs/2405.19715)

For this repository, the basic objective remains

```text
expected committed tokens(k) = 1 + sum_j<=k P(accepted prefix reaches j)
predicted throughput(k)      = expected committed tokens(k) / complete cycle time(k)
```

The next improvement must sharpen those probabilities, reduce the cycle cost, or change the available proposals. Maximizing accepted length alone is inadequate. MoE experts, context length, graph padding, CPU scheduling, communication, and the fixed draft pass all affect the denominator.

### DSpark and current vLLM: borrow the scheduling contract

DSpark combines a parallel backbone with a lightweight sequential correction and a confidence head. Its scheduler uses estimated prefix survival and engine-specific cost profiles. The paper reports 60–85% faster per-user generation against its production MTP-1 baseline at matched throughput in DeepSeek-V4 serving. This is a different target, baseline, hardware, and load regime. The public DeepSpec repository includes draft training and evaluation, including confidence calibration metrics; it is not the entire proprietary production serving stack. [5](https://arxiv.org/html/2607.05147v1), [6](https://github.com/deepseek-ai/DeepSpec)

vLLM PR #47808 implements confidence-based budget selection with stale, double-buffered CPU confidence and live GPU allocation. It merged on August 12. Its documented FULL-graph path depends on supported attention backends on SM100; non-SM100 falls back to PIECEWISE. Its reported adaptive/fixed differences are within roughly ±3% through concurrency 64, with the major benefit appearing at high concurrency. This is valuable source material for buffer ownership, cost profiles, and GPU prefix allocation, but neither a ready SM121/DCP2 patch nor evidence of a new C1 gain. [7](https://github.com/vllm-project/vllm/pull/47808)

Inspecting `deepspec/eval/dspark/confidence_head.py` confirms per-position cumulative survival evaluation with calibration error, Brier score, and AUROC. These are useful checks to adopt when assessing a predictor. The current repository does not need a DSpark model swap to adopt calibrated prediction as an experiment. [8](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/confidence_head.py)

Upstream integration is still evolving. An August 27 vLLM issue reports adaptive profiling of decode shapes exceeding the scheduler's possible `max_num_seqs × (1 + K)` rows. The issue remained open at inspection. This is a reported upstream defect, not a reproduced local failure; it supports profiling only shapes and contexts this deployment can actually execute. [45](https://github.com/vllm-project/vllm/issues/54046)

### EVICT: the closest paper with an implementation gap

EVICT, “Making Every Verified Token Count,” is a May 2026 MoE verification paper. It estimates candidate benefit from drafter signals, selects a cost-effective ancestor-closed tree prefix using profiled cycle costs, captures multiple verification graphs, and fuses selection logic into the draft graph. It reports an average 1.21× improvement over EAGLE-3 across its evaluated settings. The paper describes an SGLang implementation, but no public author implementation was found in its paper links, exact-title searches, or relevant GitHub repository results. [9](https://arxiv.org/html/2605.00342v1)

The existing controller already implements the chain-shaped counterpart of much of this objective and graph strategy. The new contribution worth implementing is candidate-dependent benefit estimation that can operate within this fork's async contract. A whole tree verifier is unnecessary for the first experiment. EVICT's assumption that draft probabilities are sufficiently calibrated must be tested for the deployed quantized GLM/DFlash2 pair rather than inherited.

### AdaFlash: real code, with an important C1 qualification

AdaFlash combines on-policy diffusion-drafter distillation and an adaptive length head. Its training uses target feedback to address domain mismatch; its adaptive head addresses variation within draft blocks. The released tree contains training workers, offline head initialization, model heads, and serving scripts. It is substantive research code, not a paper-only proposal. [10](https://arxiv.org/html/2607.19223v1), [11](https://github.com/ZinYY/AdaFlash)

The inspected serving script explicitly says the adaptive head is used only when dynamic verification is enabled **and configured `max_running_requests` exceeds 48**; otherwise it uses fixed-window DFlash. That is a configuration threshold, not proof of behavior at a particular observed concurrency. The repository's Transformers model code also includes scalar extraction in its reference generation path. Consequently, the supplied pipeline is not a ready adaptation for this C1 async runner. Its training components are more useful as references for an offline experiment with frozen serving weights. [12](https://github.com/ZinYY/AdaFlash/blob/6445920e369ed66dab466c4ef204e4ca040797f5/scripts/serve/serve_thresh_head.sh), [13](https://github.com/ZinYY/AdaFlash/blob/6445920e369ed66dab466c4ef204e4ca040797f5/SpecForge/modeling/draft/dflash.py)

### Adaptive trees and reinforcement learning

BASTION is unusually relevant because its public scope includes C1 experiments with DFlash and a hardware-calibrated verification model. Its official release is a Transformers reproduction package, supporting selected Qwen3/Llama targets; it explicitly excludes serving backends such as vLLM and SGLang. The code builds a best-first tree from draft probabilities, stopping according to marginal benefit and estimated cost. Inspection found CPU transfers, Python heap processing, and scalar extraction in tree construction. Porting that loop unchanged would undermine this repository's async graph work. [14](https://github.com/kaist-ai-osi-lab/BASTION), [15](https://github.com/kaist-ai-osi-lab/BASTION/blob/1577b5d9ee28783e15358f1dd791844e9f966172/bastion/tree_draft.py)

Learning to Draft trains separate policies for draft depth and verification size using cycle throughput as reward, then co-adapts them. Its paper reports gains over EAGLE-3 ranging from approximately 4% to 36% across model settings. The official repository includes PPO training environments and evaluation code. This is implemented research, not missing code. Its autoregressive tree-depth action is largely absent from a one-pass DFlash2 block. A small supervised cost-sensitive policy is a better first local comparison than a live PPO training loop. [16](https://arxiv.org/html/2603.01639v1), [17](https://github.com/zhzihao/Learning-to-Draft)

JetSpec trains causal parallel trees and publishes a graph-based engine plus a vLLM integration. It is a credible larger architectural alternative, particularly for coding, but requires a compatible trained head and branch-aware verification. Published engine numbers concern much smaller Qwen targets on B200. A tree expands target rows and expert diversity, so it must compete against the existing chain on measured MoE cost, not accepted length alone. Its best use now is an offline candidate-coverage screen before committing to attention and KV-layout changes. [18](https://github.com/hao-ai-lab/JetSpec)

D-cut and Tencent AngelSpec provide another recent implementation direction: confidence-based allocation across requests, backed by a runtime cost model. D-cut's main motivation is high concurrency; the vLLM PR inspected on September 9 remains open. Its cross-request allocation contributes little at C1, although cost-profile and compact-prefix mechanisms can inform later C2+ support. [19](https://arxiv.org/abs/2607.14647), [20](https://github.com/Tencent/AngelSpec), [21](https://github.com/vllm-project/vllm/pull/47131)

### Online learning should account for the feedback actually observed

BanditSpec provides training-free UCB and EXP3-style adaptation with theoretical stopping-time regret analysis and public inference code. It is a useful alternative to hand-set exploration, but the repository's seven-position feedback is more structured than an ordinary independent-arm bandit: a verified prefix provides information about shorter caps, while unverified tails remain censored. Discarding that structure would waste data. Any bandit comparison must also account for delayed results, finite requests, and context drift. [22](https://arxiv.org/abs/2505.15141), [23](https://github.com/sail-sg/BanditSpec)

Not-a-Bandit studies selection among specialized drafters using more informative feedback. MemSpec adds resident-model management because switching costs on memory-constrained devices can erase adaptive gains; its reported results use Jetson Orin Nano. No public MemSpec implementation was found in the checked paper links and searches. Both reinforce a local design constraint: first adapt among choices that are already resident. Multiple separate domain drafters are not justified under the present memory envelope. [24](https://arxiv.org/abs/2510.20064), [25](https://arxiv.org/html/2608.10362v1)

## 4. Improving draft quality and long-context behavior

### DFlash2 already solves part of the local-coherence problem

DFlash2 is not plain DFlash. Its published design scores adjacent pairs among top candidates using a learned low-rank, context-gated selector, then walks the resulting scores. That provides local coherence without another backbone or vocabulary-head pass. DSpark/DeLS-style local correction must therefore be compared with this existing selector, not with independent top-1 DFlash proposals. Adding another correction head may duplicate work or interfere with the learned score scale. [26](https://inco.ai/blog/dflash2/)

The local source inventory independently confirms `DFlash2Speculator`, `_selector_walk_kernel`, and an existing FP32 `_selector_scores` buffer shaped as maximum requests × seven positions × configured candidate count. The selector writes the score row actually encountered along its chosen path. This supplies a concrete bounded feature surface; the published top-16 example is not assumed to be the deployed checkpoint's configuration. The inventoried path and its hash are recorded in the evidence inventory; the committed `dflash_speculator.py` test fixture is the base parent class, not this selector implementation.

DeLS-Spec keeps a DFlash backbone fixed and adds a separately trained short-context head. Its public repository contains runtime and evaluation code and released local-head weights; the training code is explicitly described as forthcoming. That is a **partial implementation gap**. Recreating a text-only training recipe could be worthwhile for a compatible tokenizer, but the evidence is against DFlash-style baselines and does not establish superiority over DFlash2. [27](https://arxiv.org/html/2607.07409v1), [28](https://github.com/dt-3t/DeLS-Spec)

Before retraining, separate failures into three cases: the target token is absent from the candidate set; it is present but the selector chooses another path; or early acceptance collapses because the drafter's context representation is wrong or out of distribution. Only the second case directly supports selector fine-tuning. Candidate recall is an oracle diagnostic; it is not an achievable throughput estimate.

### LongSpec, Windowed-MTP, and compressed draft memory

LongSpec addresses draft KV growth, position-related train/inference mismatch, and inefficient long-prefix tree attention. Its author repository contains training and inference code. These are useful diagnostic concepts for the current collapse, especially because a large advertised positional limit does not prove trained acceptance at that position. A position change must remain draft-only, be consistent with its cache, and be evaluated against unchanged target verification. [29](https://aclanthology.org/2026.acl-long.83/), [30](https://github.com/sail-sg/LongSpec)

Windowed-MTP provides an actual SGLang patch with draft-only sliding attention, sink tokens, and a compact ring buffer, while preserving full target attention. Its reproduction package targets B200 and million-token built-in MTP/NEXTN workloads. The deployed DFlash2 already advertises a 2048-token window. Therefore, simply adding “windowed drafting” is not a new optimization here: the immediate question is whether the active kernel honors the intended window, absolute positions, valid slots, and rollback at long prefixes. [31](https://zenodo.org/records/21522902)

“Strong Drafts Need Compact Memories” introduces learned memory slots for an independent drafter while retaining the full target KV. Its Llama-family experiments extend to 32k, reporting over 70% draft-memory reduction. No public implementation was found in the paper links or title/GitHub searches. Its strong independent drafts are not a demonstrated fit for the available Spark memory. A smaller adaptation would add bounded draft-side memory to the current head, but that changes the proposed architecture and needs its own training and evaluation. [32](https://arxiv.org/html/2608.30252v1)

The distinction between **reproducing MASW** and **borrowing its memory-slot idea for DFlash2** matters. The latter might retain sparse information outside the 2048-token draft window, but cannot inherit the paper's results. It also needs an explicit rejection rule for memory slots created from speculative tokens: no rejected token may survive through a compressed representation.

### Extending the trained horizon is a conditional opportunity

DBloom's August 31 preprint recommends looking for a spike in fully accepted blocks before widening a block-diffusion drafter. It reports that naive runtime widening can damage even early-position acceptance, and studies longer-block post-training. No public implementation was found in the checked paper and repository searches. This is a useful diagnostic and a possible training experiment only for high-ceiling coding subsets. It has little relevance to the near-zero-acceptance 100k regime. [33](https://arxiv.org/html/2608.30427v1)

The negative PEFT-BD result is also instructive: few trainable parameters do not imply cheap drafting when the method executes the full backbone twice. Avoid a “small adapter” design whose runtime still traverses the 743B target to produce each draft block. [34](https://arxiv.org/abs/2607.12422)

LayerSkip demonstrates a different kind of early exit: specially trained early layers draft, and later layers verify and correct while reusing computation. It has public code and checkpoints, but its training recipe changes the model to make early exits useful. It is not evidence that dropping layers from the existing GLM or DFlash2 checkpoint will accelerate this service with adequate acceptance. Draft-layer exit remains a later training experiment, with full target verification retained. [46](https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding/)

## 5. Energy, power, and idle behavior

An energy benchmark published at EACL 2026 explicitly shows cases where speculation finishes faster but consumes more energy. The policy objective should therefore be measured J/token or a declared throughput/energy trade-off, rather than assumed proportional savings from tok/s. Report device and wall measurements separately, and retain prefill-inclusive request energy alongside decode-only energy. [35](https://aclanthology.org/2026.findings-eacl.249.pdf)

PELM publishes a controller combining frequency selection, speculation, and verification depth. Code inspection confirms that verification can stop at a configurable depth. That changes the target computation; “comparable task performance” is not the same requirement as preserving this deployment's target verifier. Its DVFS and slow-control concepts are relevant, but its verification-depth action should be excluded from the lossless experiment. The launch script also configures platform thermal/fan settings; it is not appropriate to run it on the Sparks. [36](https://github.com/imec-nu/PELM), [37](https://github.com/imec-nu/PELM/blob/1c23267ac23948abbd057f9b8116feafb3e7b266/generators/control_exit_ss_generator.py)

GELATO uses a virtual energy queue and entropy-driven drafting in a device/edge system. It is a useful paper-only control lead: no public implementation was found in the checked sources. However, its evaluation section is simulation, and its energy model explicitly excludes the grid-powered edge verifier. Its headline energy reduction therefore cannot be applied to total four-Spark consumption. A local adaptation would use measured cluster energy, coarse control windows, and the existing finite action set; it would not inherit the paper's guarantees. [38](https://arxiv.org/html/2605.10124v1)

MoE-Spec reduces verification expert work by omitting or substituting experts. Its paper explicitly permits an accuracy/latency trade-off. This is outside an unchanged-verifier experiment. The useful diagnostic is the observation that additional verified rows can expand the union of active experts; measuring that effect could improve a cost model without dropping any expert. [39](https://arxiv.org/abs/2602.16052)

For a fixed amount of generated work over a fixed observation horizon, a useful accounting identity is

```text
total energy = active power × active time + loaded-idle power × idle time
```

Faster completion can reduce the active-time contribution. It does **not** lower the hardware's loaded-idle wattage by itself. A slow controller can trade cap and supported operating point against latency; idle power needs a separately measured idle-state policy. Keep background training, periodic polling, and model swapping out of idle intervals. The existing blocking telemetry writer is already consistent with this requirement. This report does not recommend restarting the paused CX-7 power-cycling experiment.

## 6. Reuse and coding workloads

SuffixDecoding is a credible implemented alternative: ArcticInference provides model-free suffix proposals and a vLLM integration. An empirical software-engineering study found model-based methods better suited to fresh generation, and model-free methods better suited to repeated repository editing and repair in its tested workloads. Those findings support stratifying coding tasks instead of treating “code” as one distribution. [40](https://github.com/snowflakedb/ArcticInference), [41](https://arxiv.org/abs/2604.26469)

The repository's own [copy screen](../results/adaptive-spec/copy-branch-decision.json) found sparse useful opportunities for its most-recent exact-match selector and did not justify GPU integration. The external work does not overturn that result. A richer retrieval branch needs a new corpus of actual repeated edits/tool scaffolding, a selector that beats the existing offline screen, and a cheap proposal-source decision. Memory for a CPU index still competes with unified system memory on Spark.

The recent `mlx-dspark` project is useful engineering reference material for unified-memory deployments: it contains adaptive truncation, prefix-cache handling, draft-context windows, and small-verification kernel specialization. Its MLX/Metal paths and reported speedups are platform-specific. Borrow hypotheses and failure tests, rather than attempting to port Metal kernels into the CUDA/DCP stack. [42](https://github.com/ARahim3/mlx-dspark)

## 7. Concrete experiment designs

The following are proposed local implementations, not changes already made or results already measured.

### Experiment A: confidence-informed verification

**Integration.** Start at the active DFlash2 selector's already computed candidate/path scores, the V2 runner's completion feedback, and `VerificationPolicy` in `overlay/vllm/v1/spec_decode/adaptive.py`. Enumerate each feature's production step, readiness event, request incarnation, and earliest scheduling decision that may read it. Record selected-score margins and bounded candidate statistics only where their extraction does not introduce a vocabulary-wide operation or device synchronization.

The ordering is critical: drafting follows a target pass, while CPU scheduling for subsequent work may already be queued. A current-block GPU score is not automatically available to a host choosing that block's target graph. First evaluate **only signals known by the ordinary scheduling boundary**, explicitly labeling their age. Test whether they predict the next useful cap beyond the existing acceptance history. Do not silently pair a previous block's confidence with the current block's acceptance label.

The checked-in V2 `ModelRunner` starts `AsyncOutput` copying before it calls `speculator.propose`, so new selector statistics cannot simply be attached to that already launched copy. A later bounded copy needs its own readiness dependency and buffer lifetime. Normalizing a truncated selector score row gives candidate-set probabilities, not calibrated probability of greedy target agreement.

**Local work.** Fit a tiny calibrated predictor from development-only data: selected-path confidence/margin, position within the block, context band, recent acceptance, and feedback age. Start with lookup bins or logistic regression. Preserve censor masks; a rejection observes that failure and earlier successes, not the hypothetical validity of the remaining tail. Compare against version 4, raw confidence, and a cost-aware oracle screen on exactly the same records.

**Memory and execution budget.** Proposed initial limit: under 1 MiB of predictor parameters and under 16 MiB total new bounded buffers per rank, excluding separately measured capture changes. CPU fitting needs no target model in the VM. These are design budgets, not measured allocations. Idle state has no polling or training worker.

**Decision gate.** Proceed to an active Spark test only if out-of-prompt development replay predicts at least 3% aggregate throughput benefit over version 4 in a relevant stratum, with no clear coding/prose regression after expected overhead. This is a screening threshold, not a speed claim. Then require live positive paired evidence under the shared gates below.

Same-block GPU selection is a separate second-stage design. It is admissible only after proving that reserved versus executed lengths, output placeholders, KV allocation, rollback, graph choice, and all TP/DCP ranks remain consistent. Compacting rows in the worker after the scheduler has committed different lengths would violate the current authority model. A new synchronization or graph-dispatch cost must be measured and charged; a masked M8 graph is not a smaller target pass.

**Request-hint variant.** Pi's extension API exposes `before_provider_request`, which can replace the final provider payload. This offers a CPU-available signal before inference begins. It does not change metadata inside an already running stream. [47](https://pi.dev/docs/latest/extensions)

A proposed Pi extension could attach new flat `vllm_xargs` fields such as `spec_workload`, `spec_phase`, and `spec_hint_confidence`. These fields and their interpretation are **not implemented**. Use known harness state or explicit task metadata, with no classifier-LLM request. Interpret the hint as a weak, calibrated initial prior or a declared latency/energy preference; server eligibility, cap limits, and measured inference feedback retain authority. Abstain for mixed or unknown tasks. A coding request can emit reasoning, tool JSON, or explanatory prose, so its task label cannot determine every block's policy.

The server should map labels to domain/phase-conditioned acceptance priors while retaining its hardware/context cost table. Today, `PrefixStats.choose` forces K=7 for the first eight observations regardless of the prior, and `GLM_SPEC_POLICY=off` constructs no policy. Therefore, sending metadata alone cannot shorten warm-up or enable production adaptation. An earlier prior-based decision needs a separately validated policy change; otherwise this variant can only affect choices after warm-up.

Test version 4, hint-only, inference-signals-only, and a hybrid on the same locked mixed workload; add shuffled hints to detect incidental correlations and an omitted-hint control to verify fallback. The practical question is whether hints improve cold-start decisions or short responses beyond the existing policy. They remove the GPU-feature timing problem for that initial signal, but do not establish an additional gain beyond the already measured prose improvement.

A separate waiting-for-user or long-tool hint could inform a coarse idle policy, subject to the server confirming that all clients have no active or queued work and accounting for wake cost. That is a separate power-intent contract: a prose/code label does not lower loaded idle watts.

### Experiment B: long-context diagnosis before learned memory

**Integration.** Inspect the active model's draft context-KV precomputation, `prepare_dflash_inputs`, slot mapping, position handling, selector, and rollback. The existing CPU interpreter fixtures provide a starting point; add minimal long-position and ring-wrap cases without allocating a full model. Compare the same local suffix at different absolute positions, cold versus prefix-cache reuse, and first execution versus restart.

**Diagnostic separation.** Measure target-token candidate recall, selected-path acceptance, and draft/target cycle components across 4k,32k,100k,170k. Use sampled aggregates or bounded short captures. Determine whether the first-token collapse is positional, cache-related, candidate coverage, or data mismatch. A sliding-window checkpoint alone cannot decide among these explanations.

**Conditional implementation.** Fix a demonstrated cache/position bug first. If behavior is internally correct, evaluate draft-only sink/window/position variants on a development corpus with unchanged target attention. If information outside the window matters, prototype bounded compressed memory with explicit commit epochs and rollback of slots derived from rejected tokens. A MASW-style memory implementation requires offline training on a suitable GPU and is not a CPU-only VM training task.

**Gate and budget.** Diagnostic mode starts below 32 MiB extra bounded storage per rank. A trained-memory branch gets a separate allocation proof capped initially at 128 MiB/rank. It proceeds only if projected accepted-token benefit exceeds measured extra draft work by at least 5%, and long-context marker, repository, rollback, and restart controls all pass. Long-context gains must be measured against version 4 as well as fixed seven.

### Experiment C: true target-only fallback

Cascade uses short test phases and longer set phases to choose speculation length, including disabling speculation when its utility is below one. Its paper describes a vLLM implementation; no full author release was found. A third-party llama.cpp fork explicitly labels its implementation a simplified Cascade-style controller, so this is an author-artifact gap rather than an entirely unimplemented idea. [43](https://arxiv.org/html/2506.20675v1), [44](https://github.com/angerybob/S2-MoE/blob/orin/MOE_UTILITY_SPEC.md)

**Integration.** Add a separately modeled mode that performs one target row and actually skips draft execution. It must retain sufficient committed target features to resume DFlash2, or rebuild the relevant draft window at re-entry. Define states for speculative execution, target-only execution, catch-up, cancellation, preemption, and request reuse. The scheduler remains the authority for scheduled lengths.

**Local work.** Implement the state machine and use small deterministic models or kernel fixtures to verify sequences such as 7→0→0→3, cancellation during catch-up, mixed C1/C2 transitions, and disk-prefix reload. Account for features already resident rather than copying an entire long prefix. If bounded catch-up cannot be guaranteed, do not expose hot switching.

**Break-even gate.** If target-only saves Δt per token and re-entry costs Tcatchup, the expected target-only dwell must exceed `Tcatchup / Δt`, plus a margin for uncertainty. Compare total interval time, not just its cheapest cycles. A new M1 graph must be shown to execute less work. Budget retained features from actual tensor dimensions, with an initial ceiling of 64 MiB/rank; reject the design if it cannot fit that ceiling without reducing the daily context limit.

This path is most attractive when acceptance is persistently near zero. It is unlikely to help already high-acceptance coding, and repeated probe/re-entry cycles can waste both time and energy.

### Experiment D: energy-aware operating points

**Local work.** Extend the existing energy analysis to compute a Pareto set over cap and context, then over supported operating points if those are separately measured. Use full-request and decode-only energy. Fit no model to the locked held-out prompts. Missing wall measurements must yield an explicitly device-only result.

**Controller.** Initially choose one setting per request or coarse workload window, not per token. A proposed objective is minimum measured J/token subject to at least 98% of the chosen throughput baseline and existing TTFT/gap constraints. A slow energy-budget controller can later adjust that choice using a token-normalized energy deficit; the accounting unit must match the stated objective. This is an engineering adaptation, not a reproduction of GELATO's per-step wireless model.

**Spark gate.** First measure a few already supported clock/power settings independently of adaptation, with normal fan and thermal protection. Use repeated sufficiently long windows, matched output work, randomized order, four-node coverage, and a wall meter if available. Require at least 5% lower energy per token with the declared throughput floor and no increased loaded-idle power. If no operating point dominates or meets the trade-off, retain the measured frontier and stop.

### Experiment E: improve the existing drafter selectively

Use the failure taxonomy from Experiment B to choose between selector calibration/fine-tuning, domain distillation, or a longer trained block. Keep a frozen target and an unchanged base drafter control. The preferred initial training object is a small head on existing features; do not run continuous on-policy training alongside the loaded service.

A useful sandbox deliverable is the data contract, label generation on tiny models, optimizer/checkpoint loading, and a memory estimate. Real GLM-aligned head training needs sampled target features and external GPU capacity; the VM cannot establish its final training cost or throughput. Bound collected features and sample on development prompts, rather than dumping every full-vocabulary target logit.

The go/no-go criterion is a live C1 improvement over both unchanged DFlash2/version 4 and a simpler confidence controller. Predeclare a 5% throughput target for this more expensive branch, unchanged correctness controls, and no increase in measured J/token. Reject a model that improves acceptance while increasing full-cycle cost enough to erase the benefit.

## 8. Shared correctness, measurement, and deployment gates

The target weights, target attention semantics, sampler, context window, and KV/offload policy remain fixed throughout the lossless track. Graphs must execute the selected amount of target work. Only a verified contiguous chain prefix may be committed; trees require ancestor-closed masks and correct branch-specific KV. Draft confidence is never permission to bypass target verification.

Use tiny-model distribution checks for any new sampling rule, deterministic fixed-logit/kernel tests for greedy correctness, and executable code plus long-context retrieval controls for end-to-end behavior. Because the current GPU baseline is not generally bitwise repeatable, identical output hashes are useful controls but not the sole correctness argument. Unsupported request modes retain their existing fallback until separately audited.

Preserve the completed benchmark as historical evidence. New development data and a new locked evaluation corpus should include full coding tasks, repository edits, tool transitions, explanatory and creative prose, long outputs, and the actual thinking configuration. Use at least thirty prompts across the main coding/prose comparison with balanced paired repeats; treat long-context strata independently. Increase sample size before locking if development variance makes a practical confidence interval unlikely.

Every active comparison includes fixed seven and version 4 where relevant. Report paired per-prompt gains and intervals, aggregate tok/s, TTFT, p95 emission gap, complete-task correctness, and active energy. A proposed common promotion gate is a positive lower confidence bound on the targeted throughput improvement, a lower ratio bound of at least 0.98 in the other primary workload, and no new correctness or serving-pressure failures. Instrumentation overhead requires its own paired control; the previous experiment did not establish a strict upper bound below 1%.

Before each Spark experiment, observe all ranks' available memory, swap-out deltas, full-memory PSI, process allocations, and model health. Estimate incremental buffers and graph peaks before admission. Do not require an arbitrary percentage of unused RAM once the model is loaded; use the measured normal envelope and the existing guarded controller's phase-aware limits. One experiment runs at a time, with watchdog, bounded duration, early stop, and exact restoration. Sampling pressure must continue during the run.

## 9. Implementation availability and open questions

“No public implementation found” means that the paper's links, exact-title/author searches, and relevant GitHub search results were checked on September 9, 2026. It is not proof of global absence. General searches for EVICT returned unrelated cache-eviction projects; none inspected was the paper's method. New releases may change these classifications.

| Work | Availability established | Suitable missing work here |
|---|---|---|
| EVICT | Paper describes SGLang implementation; no public implementation found | Confidence-informed chain adaptation with explicit async/graph contract |
| Cascade | Author vLLM implementation described; full author release not found; simplified third-party implementation exists | Real K=0 and switching-cost/catch-up handling for DFlash2 |
| MASW / Strong Drafts Need Compact Memories | Recent paper; no public implementation found | Small draft-memory prototype, conditional on diagnosis and training resources |
| GELATO | Paper and simulation; no public implementation found | Measured whole-cluster energy controller with different accounting assumptions |
| MemSpec | Paper; no public implementation found | Resident-choice and switching-cost ideas; multi-model runtime deferred |
| DBloom | Recent paper; no public implementation found | Acceptance-ceiling audit; longer-block training only if justified |
| DeLS-Spec | Runtime/evaluation and weights available; training promised | Training recipe, with DFlash2 overlap explicitly evaluated |
| DSpark / DeepSpec | Training/evaluation code and checkpoints; separate merged vLLM implementation | Selective adaptation of confidence/ownership mechanisms to SM121/DCP2 |
| AdaFlash | Training/head code; serving scripts with backend assumptions | C1-compatible use of a head, offline training and calibration |
| BASTION / LTD | Public research implementations | Small bounded components, not wholesale serving-loop ports |
| JetSpec / LongSpec / Windowed-MTP | Public implementations with specific model/runtime scopes | Later architectural forks if the simpler experiments fail |

The highest-value unanswered question is whether candidate information available without new synchronization improves decisions beyond delayed acceptance history. The second is whether the long-context collapse is a repairable runtime problem or a trained-drafter limitation. Resolving those questions determines whether the next implementation should be a small policy change, a cache/position fix, or a new trained component.

## Sources

Sources are primary papers, author repositories, or official runtime publications. GitHub snapshot identifiers and inspected paths are recorded in the accompanying JSON inventory. Publication dates below refer to the original paper or cited release where established; undated repository entries were inspected September 9, 2026.

1. Leviathan, Kalman, and Matias. [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html). ICML 2023.
2. Mamou et al., Intel/Hugging Face. [Faster Assisted Generation with Dynamic Speculation](https://huggingface.co/blog/dynamic_speculation_lookahead). October 8, 2024.
3. Agrawal, Jeon, and Lee. [AdaEDL](https://proceedings.mlr.press/v262/agrawal24a.html). NeurIPS ENLSP workshop, December 2024.
4. Huang, Guo, and Wang. [SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths](https://arxiv.org/abs/2405.19715). May 30, 2024.
5. Cheng et al. [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/html/2607.05147v1). July 6, 2026.
6. DeepSeek-AI. [DeepSpec](https://github.com/deepseek-ai/DeepSpec). Training/evaluation repository.
7. Wilkinson et al., vLLM. [DSpark confidence-scheduled verification, PR #47808](https://github.com/vllm-project/vllm/pull/47808). Merged August 12, 2026.
8. DeepSeek-AI. [Confidence-head evaluation source](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/confidence_head.py). Inspected snapshot.
9. Pan et al. [Making Every Verified Token Count: Adaptive Verification for MoE Speculative Decoding](https://arxiv.org/html/2605.00342v1). May 1, 2026.
10. AdaFlash authors. [AdaFlash: Adaptive Speculative Decoding via On-Policy Distilled Diffusion Drafters](https://arxiv.org/html/2607.19223v1). July 21, 2026.
11. AdaFlash authors. [Official AdaFlash repository](https://github.com/ZinYY/AdaFlash).
12. AdaFlash authors. [Adaptive-head serving script](https://github.com/ZinYY/AdaFlash/blob/6445920e369ed66dab466c4ef204e4ca040797f5/scripts/serve/serve_thresh_head.sh).
13. AdaFlash authors. [DFlash model/head implementation](https://github.com/ZinYY/AdaFlash/blob/6445920e369ed66dab466c4ef204e4ca040797f5/SpecForge/modeling/draft/dflash.py).
14. Oh et al. [BASTION official repository](https://github.com/kaist-ai-osi-lab/BASTION). Paper arXiv:2605.29727, May 2026.
15. BASTION authors. [Adaptive tree construction source](https://github.com/kaist-ai-osi-lab/BASTION/blob/1577b5d9ee28783e15358f1dd791844e9f966172/bastion/tree_draft.py).
16. Zhang et al. [Learning to Draft: Adaptive Speculative Decoding with Reinforcement Learning](https://arxiv.org/html/2603.01639v1). March 2, 2026; ICLR 2026.
17. Zhang et al. [Learning-to-Draft official implementation](https://github.com/zhzihao/Learning-to-Draft).
18. Hao AI Lab. [JetSpec code, engine, and evaluation](https://github.com/hao-ai-lab/JetSpec). Paper arXiv:2606.18394, June 2026.
19. Liu et al. [D-cut: Adaptive Verification Depth Pruning for Batched Speculative Decoding](https://arxiv.org/abs/2607.14647). July 16, 2026.
20. Tencent. [AngelSpec](https://github.com/Tencent/AngelSpec). Paper arXiv:2607.25852, July 2026.
21. Liu et al., vLLM. [D-cut implementation PR #47131](https://github.com/vllm-project/vllm/pull/47131). Open at inspection.
22. Hou et al. [BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms](https://arxiv.org/abs/2505.15141). May 21, 2025.
23. BanditSpec authors. [Official BanditSpec repository](https://github.com/sail-sg/BanditSpec).
24. Liu, Huang, Jia, Park, and Wang. [Not-a-Bandit: Provably No-Regret Drafter Selection in Speculative Decoding for LLMs](https://arxiv.org/abs/2510.20064). October 22, 2025; revised April 22, 2026; ICLR 2026.
25. Kim, Jeon, and Han. [MemSpec](https://arxiv.org/html/2608.10362v1). August 11, 2026.
26. Inco AI. [DFlash 2: Keep Drafting Parallel](https://inco.ai/blog/dflash2/). August 2026.
27. DeLS-Spec authors. [DeLS-Spec: Decoupled Long-Short Contexts for Parallel Speculative Drafting](https://arxiv.org/html/2607.07409v1). July 8, 2026.
28. DeLS-Spec authors. [Official DeLS-Spec repository](https://github.com/dt-3t/DeLS-Spec).
29. Yang et al. [LongSpec: Long-Context Lossless Speculative Decoding with Efficient Drafting and Verification](https://aclanthology.org/2026.acl-long.83/). ACL, July 2026; first arXiv February 24, 2025.
30. LongSpec authors. [Official LongSpec repository](https://github.com/sail-sg/LongSpec).
31. Valliappan. [Windowed-MTP: B200 reproduction package](https://zenodo.org/records/21522902). Version 1.0.0, July 24, 2026.
32. Yuan, Liao, and Wen. [Strong Drafts Need Compact Memories: Long-Context Speculative Decoding with Compressed KV Cache](https://arxiv.org/html/2608.30252v1). August 31, 2026.
33. Wu. [Ceiling-Clipped Acceptance Histograms Indicate Stranded Speed-up in Block-Diffusion Speculative Decoding](https://arxiv.org/html/2608.30427v1). August 31, 2026.
34. Javat and Kazakov. [Accepted Prefixes Are Not All You Need: A Negative Result on PEFT-Based Block-Diffusion Drafting](https://arxiv.org/abs/2607.12422). July 14, 2026.
35. Dutta et al. [Benchmarking the Energy Savings with Speculative Decoding Strategies](https://aclanthology.org/2026.findings-eacl.249.pdf). Findings of EACL, March 2026.
36. Yang and Xia. [PELM code artifact](https://github.com/imec-nu/PELM). SenSys 2026, [paper DOI](https://doi.org/10.1145/3774906.3802783).
37. PELM authors. [Controlled-depth verification source](https://github.com/imec-nu/PELM/blob/1c23267ac23948abbd057f9b8116feafb3e7b266/generators/control_exit_ss_generator.py).
38. Tang et al. [GELATO: Generative Entropy- and Lyapunov-based Adaptive Token Offloading for Device-Edge Speculative LLM Inference](https://arxiv.org/html/2605.10124v1). May 11, 2026.
39. McDanel et al. [MoE-Spec: Expert Budgeting for Efficient Speculative Decoding](https://arxiv.org/abs/2602.16052). February 17, 2026.
40. Snowflake. [ArcticInference](https://github.com/snowflakedb/ArcticInference). SuffixDecoding implementation; NeurIPS 2025 paper.
41. Li, Chen, Hu, and Xia. [An Empirical Study of Speculative Decoding on Software Engineering Tasks](https://arxiv.org/abs/2604.26469). April 29, 2026.
42. ARahim3 and contributors. [mlx-dspark](https://github.com/ARahim3/mlx-dspark). Inspected September 9, 2026.
43. Saxena et al. [Utility-Driven Speculative Decoding for Mixture-of-Experts](https://arxiv.org/html/2506.20675v1). June 2025.
44. S2-MoE contributors. [Simplified Cascade-style controller documentation](https://github.com/angerybob/S2-MoE/blob/orin/MOE_UTILITY_SPEC.md). Third-party implementation, inspected September 9, 2026.
45. vLLM issue author. [Adaptive verification profiles a batch shape the scheduler cannot produce, #54046](https://github.com/vllm-project/vllm/issues/54046). August 27, 2026; open at inspection.
46. Elhoushi et al., Meta. [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding/). June 14, 2024.
47. Pi. [Extensions: model events and provider request payload](https://pi.dev/docs/latest/extensions). Current documentation inspected September 9, 2026.
