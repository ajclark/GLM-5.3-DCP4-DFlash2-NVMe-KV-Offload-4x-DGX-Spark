**Serving while loading unrelated models**

The selected first priority is now the [generic NVMe loading implementation plan](NVME-SATURATED-MODEL-LOADING.md), using the supplied 4–5 GB/s bandwidth without a preliminary measurement phase. The alternatives and trace-first experiments below are historical design comparisons, not prerequisites for that plan.

Switching between unrelated models can overlap loading with useful work, and a sparse MoE target can potentially generate correct tokens before all its experts are resident. Durable KV helps only when the incoming model already has a compatible cached prefix. It cannot transfer a conversation's computed state from one unrelated model to another.

The most dependable near-term path is prepared per-rank weight loading with overlapping restoration of the incoming model's own KV, followed by progressive activation of optional serving features. The most promising larger experiment on the existing hardware is exact on-demand expert loading for an incoming MoE model. Dense layer streaming is implementable in principle, but usually starts computation early without delivering a dramatically earlier first token.

This note ranks options for switching A → B, where A and B can have different architectures, tokenizers, quantization, attention and parallel configurations. It assumes powered-on hosts and locally staged model artifacts. It complements [the startup analysis](FAST-START-MODEL-SWITCHING.md). Rankings are engineering judgments, not measured success probabilities. No runtime changes or new performance benchmarks were performed.

**What durable KV contributes**

KV contains model-derived attention state for a particular token sequence. It is not a model-independent representation of the conversation. The transferable object between unrelated models is the conversation text and application state; B applies its own tokenizer and computes its own activations.

| State when switching to B | What can be reused | What B must compute |
| --- | --- | --- |
| Only A has processed this conversation | Text/application history | B's entire prompt, except any independently matching B cache entries |
| B previously processed an exact prefix | B's compatible persisted prefix | Any uncached tail and new user/tool text, then generation |
| B has cached the system prompt but not the conversation | B's system-prompt prefix | The remaining conversation and new input |
| A → B → A, with unchanged A revision and cache compatibility | A's persisted prefix on return | A's new suffix and generation |
| B has changed weights or an incompatible execution/cache configuration | Generally no old B KV | Recompute under the new configuration |

An identical string is not enough for a hit. Tokens, positions, model revision and the cache compatibility contract must match. In this deployment, target/draft configuration and runtime layout also participate in durable-cache identity.[1]

Even a complete B KV prefix does not contain B's weights or arbitrary future outputs. Each new token still needs B's embedding, attention/query/indexer work, feed-forward path, normalization and output projection across the model. Historical KV avoids rerunning those operations for cached historical tokens. It does not allow skipping the layers for a new token.

Ordinary KV also does not preserve the final logits or full residual stream needed to resume at an arbitrary point in computation. The local vLLM cache manager explicitly leaves work to obtain logits. An exact session checkpoint could preserve more state, but that would remain specific to B and to that session.[2]

**Ranked options**

The order favors a useful, correct deployment over an impressive first-token demonstration. Additional hardware is called out explicitly; the primary target remains the existing four Sparks. No incoming model list was specified, so architecture-dependent options are conditional.

| Rank | Path | Likelihood of a useful result | Genuine new B tokens while B's main weights are still loading? | Main qualification |
| --- | --- | --- | --- | --- |
| 1 | Prepared B weights + concurrent B KV recovery + staged optional features | High for the loading/recovery path; medium for live feature activation | Generally no; yes while optional draft weights or unrelated cache state load | Best practical baseline; a cold B prompt still needs prefill |
| 2 | Keep a complete B ready on spare capacity; route immediately | High if capacity exists | Yes while other replicas/state load, but the serving copy is already ready | Requires both models to fit or a separate serving group |
| 3 | B's shared weights + an initial expert set; exact MoE misses load on demand | Medium, with substantial engine work | Yes | B must be sparse MoE; cache misses and fresh prefills can erase the advantage |
| 4 | Stream B's layers and B's KV in execution order | Medium technically; modest expected first-token benefit for dense B | Usually first output arrives near completion of needed layer loading | Every layer on the token's path must execute |
| 5 | Replace A's layers with B's behind A's final forward pass | Low on this stack | Potentially, with a custom execution schedule | Hard across unrelated layouts, speculation and tensor-parallel ranks |
| 6 | Transparent lazy process/VM restore or arbitrary memory paging | Low as the first engineering investment | Only if B's actual execution touches a small enough working set | Device mappings, fault handling, graph behavior and I/O stalls remain |

Rank 3 is the strongest answer to the specific question “can the incoming large model start producing correct tokens before its entire weight set has loaded on the same hardware?” Rank 1 establishes the baseline it must beat. Rank 2 is the strongest answer to “can selection between unrelated models feel instant?” when additional residency is acceptable.

**Rank 1: Load B efficiently, recover its own KV, then increase serving capability**

Keep a controller and model-specific worker skeletons ready where their retained memory fits. Prepare B's checkpoint for B's TP layout and quantization before switching. On activation, stream rank-local tensors through bounded staging buffers, while locating and restoring only B's compatible KV needed by admitted requests. Upstream vLLM provides sharded loaders and concurrent streaming building blocks, but not a general guarantee of partial-model serving.[3]

Once the target model and required request state are ready, serve ordinary target-model decoding. Prepare optional draft weights, additional graph shapes, and caches for other sessions afterward. This can produce output while parts of the broader stack load, but should not be reported as avoiding B's required target weights.

Live transition into DFlash is a proposed engine capability, not an existing launcher toggle. It needs draft context initialization, target feature availability, compatible cache pools and an idle execution boundary. Serving eagerly while preparing more graphs also requires scheduling and memory controls. A conservative first version can defer optional features until the next safe activation rather than changing them mid-request.

The existing durable-cache result is useful evidence: a 100736-token request recovered 100608 tokens from the external cache after restart and had 2.70-second first-token latency. This measurement was made with the weights loaded. It neither demonstrates overlap with loading nor applies to a different model's cache.[4]

Weights and KV on the same SSD compete for bandwidth. Overlap can hide CPU/GPU work and pipeline gaps; it cannot make their combined bytes free. Use a shared I/O scheduler with request-critical reads ahead of speculative reads, and bound pinned buffers against Spark's shared physical memory.

**Rank 2: Make model selection a routing operation**

If A and B both fit, keep both ready. If they do not, a second serving group can hold B while A serves. Switch the route when B is healthy, leaving existing A streams on A until they finish. This supports unrelated architectures cleanly because each group owns its model plan and cache namespace.

The current GLM deployment reports approximately 96 GiB of model-loading allocations per rank, before KV and other serving resources. Spark has a shared CPU/GPU memory pool. A second similarly large model cannot simply be parked in CPU RAM on those nodes.[5]

Preloading a portion of B into spare memory can shorten cutover, but cannot hide the full transfer if insufficient capacity exists. Keeping every checkpoint on disk is preparation, not residency. Likewise, B being ready does not eliminate a fresh conversation's B prefill. Full user-perceived immediacy may require B's common prompts or session state to have been computed in advance too.

For unrelated-model switching, this option gives the clearest latency contract. Its cost is capacity rather than a complicated weight-faulting engine.

**Rank 3: Exact MoE expert loading, with eventual full residency**

Keep B's common path resident: embeddings/output head, norms, attention and indexer weights, routers, dense feed-forward layers, shared experts, quantization metadata and communication resources. Load an initial set of routed experts. At each MoE layer, obtain B's real router decisions, ensure the chosen experts' rank-local tensors are present, then execute that layer. Missing experts delay execution; they must not be replaced with zeros, skipped, or silently rerouted.

A background loader fills the remaining experts at lower priority. As residency grows, demand misses should fall, and eventually the system can use its normal fully loaded path. For a B that fits in memory, this should be a temporary loading phase: avoid evicting already loaded B experts and turning startup into perpetual storage-backed inference.

This is supported as a research direction by work on MoE offloading and predictive expert scheduling. Those papers establish relevant techniques, not performance for this NVMe/GB10 deployment; many evaluate host-RAM-to-GPU transfers on other hardware.[6][7]

The deployed GLM configuration, read on September 13, 2026, reports 78 layers, 256 routed experts, eight selected per token, one shared expert and three initial dense layers. These facts characterize GLM when it is the incoming model. They must not be assumed for an unrelated B.

For such an architecture, one token selects only 8/256 routed experts per MoE layer. That is 3.125% of the expert count, not 3.125% of total model bytes. All common-path tensors remain necessary, and mixed quantization can give experts different byte costs. If B's routing and layout permit it, many unselected experts need not be present to produce that token.

Durable B KV changes the economics because a long cache hit can reduce the work from thousands of prompt tokens to a short suffix. Fresh B prompts can touch many experts across their token batch and drive required residency toward the whole model before any output is available.

**The uncached tail is a critical measurement**

Our persisted-prefix test still recomputed 128 tokens: 100736 minus 100608. A new user message or tool result can add considerably more. “99.87% cached” is excellent for ordinary prefill latency, but can still be a poor starting point for a sparsely loaded MoE.[4]

For intuition, assume 256 experts and eight uniformly selected experts per token, independently across tokens. Then the expected number of distinct experts needed at one layer for N tokens is:

`U(N) = 256 × [1 − (1 − 8/256)^N]`

| Tokens evaluated at the layer | Expected distinct experts | Fraction of experts |
| --- | --- | --- |
| 1 | 8.0 | 3.1% |
| 8 | 57.4 | 22.4% |
| 32 | 163.3 | 63.8% |
| 96 | 243.9 | 95.3% |
| 128 | 251.6 | 98.3% |

This is a toy model, not a measured GLM routing distribution. Real routing is correlated, grouped and skewed. It demonstrates why expert traces are required and why multiplying total model size by 8/256 is an unsafe startup estimate.

The same issue applies to speculative verification and concurrency. Multiple target positions broaden the expert union. Start the experiment with one request and single-token target decoding. Add speculation or concurrency only if accepted output per transferred byte improves. Changing speculation can change the random-number trajectory; exactness means using the correct model computation/distribution, not promising identical sampled text under every scheduling change.

An exact B session checkpoint could reduce artificial tail replay: seal valid partial target KV, preserve token position and sampling/constraint state, and record a pending sampled token or the logits needed to select it. Preserve draft state separately or initialize it later. This helps resume an existing B computation, but does not erase newly appended input, supply missing B history, or reuse A KV. It is an enhancement to durability semantics, not something the current prefix-block cache already guarantees.

**What the MoE engine would need**

Prepare independent, rank-local expert segments with their scales, layout metadata and integrity checks. Add an expert residency table and execution barriers. The current fused router/expert path assumes usable weight tensors; it must expose routing before executing missing experts. Splitting that path, or introducing a supported miss protocol, is real engine work.

On TP4, selected expert IDs are not a license for one rank to continue alone: each required shard and collective dependency must be ready in a consistent order. Use generation-tagged completion events, bounded waits and explicit failed-load handling. Keep a separate control mechanism so a rank waiting for disk does not strand the others in a mismatched collective.

Whole-model CUDA graphs cannot execute kernels that dereference absent expert storage. Begin with eager execution or graph segments separated at residency checks. Loading new bytes into inactive expert regions must not race with kernels or invalidate layout assumptions. Promote to the normal graph path only when its required allocations are ready.

Record recent expert IDs as a separate per-model/per-session hint stream if useful. Ordinary KV is not an expert-route log and does not determine future routing by itself. Previous routes can guide prefetch; the current router remains authoritative.

**Rank 4: Pipeline B's layer execution with B's weight and KV loading**

Store weights in execution order. For layer L, wait until L's necessary weights and B's attention state are ready; execute L while loading later layers. Retain loaded weights so subsequent tokens improve. This can overlap useful computation with loading even for a dense target.

For a dense model, a newly generated token still traverses all layers. Its first output cannot appear until all weights on that path have become usable. If storage is slower than computation, the token follows the loader and first-token latency remains close to the full required-weight transfer. A long fresh B prefill offers more compute to overlap, but produces no output until its required prompt computation finishes.

A durable B prefix reduces prefill dramatically, which also reduces the compute available to hide loading. Durable KV improves absolute latency while making layer-only overlap less transformative. Sparse expert loading is different because unselected experts can remain absent even after correct output begins.

Layer-wise KV recovery also requires a connector change. The current durable store exposes group/block transfers, verifies the payload CRC over the complete stored block, and reports completion through worker jobs. It does not expose verified readiness for arbitrary individual layer slices. Serving a slice before validating its current whole-block checksum would weaken the existing integrity contract.[1]

Start by restoring the needed B KV through the existing path while weights load. If measurements justify finer granularity, design a versioned layer/chunk index, per-chunk checksums and rank-aware readiness events. Avoid fetching small layer fragments from thousands of token blocks if it converts sequential recovery into costly scattered reads.

GLM's sparse attention suggests a further experiment: retain the indexer state needed to select positions, then fetch only selected attention payloads. This is a second on-demand storage problem with its own kernel and disk-layout work. Future queries can select different historical positions. It is not valid to load only the latest KV blocks and ignore the rest of the context.

**Rank 5: Replace A behind its final forward pass**

At a carefully chosen boundary, stop admitting A work. After A's final pass consumes a layer's weights, release those allocations and load B into the freed capacity. B's first pass could follow behind as its dependencies become ready. This reduces the temporary need for two full models.

Across unrelated models there may be no useful correspondence between layers, tensor sizes, attention workspaces or TP groups. A may revisit layers for another decode or speculative correction. Existing graphs, pending KV transfers and collectives complicate final-use detection. The safe boundary is the last use by all relevant work, not the end of one observed kernel.

Durable A KV helps resume A later, but cannot keep A generating once its required weights are overwritten. This is a specialized cutover scheduler with storage-limited latency and expensive rollback, so it ranks below straightforward B loading and MoE demand loading.

**Rank 6: Transparent memory faults or lazy VM restoration**

A lazy mapping can make the runtime appear restored while much of B remains on disk. Its value depends on how much B actually touches. Dense decoding touches the required path through every layer; sparse MoE can touch less, but a generic page-fault mechanism has less semantic information than explicit expert loading.

GPU mappings, DMA/pinning and graph execution make transparent fault recovery a platform-dependent undertaking. A restored CPU process does not establish that CUDA or RDMA resources are valid. The previous [snapshot analysis](FAST-START-MODEL-SWITCHING.md) covers those feasibility gates. A microVM cannot make A KV useful to B or remove B's required weight reads.

CPU execution of cold experts is another possible fallback, but CPU and GPU memory are shared on Spark. It still needs the correct B expert bytes from disk and compatible CPU kernels, so it is not an independent RAM tier that solves cold loading. Benchmark it only if device-copy or GPU contention is the demonstrated bottleneck.

**DFlash and cached output have narrower roles**

DFlash is a drafter conditioned on target features, not a standalone substitute for an absent target model. Its proposals must be verified before being exposed as the target's committed output. B's draft and target features are specific to B. A's drafter cannot be assumed compatible with unrelated B.[8]

Research such as SP-MoE combines speculative decoding with expert prefetching using correspondence between draft and target structures. That is a potential later enhancement to rank 3, not a reason to assume our DFlash can directly predict every target expert. Predictors may improve scheduling while exact misses still fetch the true experts.[9]

Persisted logits can allow sampling one next token at an already-computed B boundary. A durable outbox can replay previously generated, undelivered B tokens during recovery. Neither provides an open-ended fresh response to new input. They can smooth a reconnect or planned pause; they are not general unrelated-model switching solutions.

**Scheduling eventual loading without freezing the first tokens**

For a partially resident B, give the highest priority to expert/weight and KV reads blocking the current admitted request. Follow with near-future layer reads, bounded predicted experts, then bulk completion. Admit few requests initially and expand only as residency and latency permit.

Reserve a measured share of bandwidth for completing B so demand traffic cannot starve eventual residency indefinitely. If this reservation produces unacceptable stalls, limit admission further or finish loading before admitting more work. Background loading competes for SSD bandwidth, shared DRAM traffic, copy resources and CPU checksum/decode time.

For one rank, a rough storage lower bound is:

`T_required ≥ (required common weights + unique required expert bytes + required KV bytes) / useful SSD bandwidth`

Sequential router-dependent misses and compute add critical-path latency. Four local drives help in parallel, but the slowest required rank determines progress. During overlap, total bytes read from one SSD must be accounted for together rather than assigning full SSD bandwidth separately to weights and KV.

The approximately 103 GB per-rank full-state estimate in the previous note is a full-restoration scenario. A sparse first-token working set can be smaller; a dense path or a long uncached MoE suffix may approach the full requirement. Quantify B's actual tensors and routed experts before attaching a time prediction.

**Experiments that decide what to build**

1. **Establish A → B baselines.** For at least one unrelated dense B and one unrelated MoE B that fit the selected hardware, measure prepared full loading and first output. Run separately with no B cache, a B system-prefix hit, and a long B prefix plus 1/8/32/128/512 uncached tokens. Keep prompt token counts model-specific.
2. **Trace MoE working sets before writing a loader.** Collect expert IDs and byte costs for each layer over suffix replay and the first 32–128 generated tokens. Include C=1 and the intended concurrency, with and without speculation. Calculate perfect-prediction and measured-history-prediction read bounds. If even the optimistic bound barely beats full loading, stop this path early.
3. **Prove an exact miss path on a small MoE.** Hold selected experts out of residency, force real misses, and compare against fully resident execution using the same kernels where possible. Check logits and greedy outputs; test corrupt reads and a failed rank. Zero-filled missing weights must never reach execution.
4. **Add durable B KV recovery.** Preserve the existing identity, CRC and rank-completion semantics. Measure tail amplification explicitly. Prototype richer B session checkpoints only if replay is responsible for losing the sparse advantage.
5. **Add background completion and measure the whole response.** Record time to first new verified token, p50/p95/p99 inter-token gap, longest stall, time to first 32 tokens and time to normal throughput. Track peak physical RAM, useful bytes read, read amplification and time to full B residency. A fast first token followed by a long freeze is a failed improvement.
6. **Add prediction and optional features last.** Evaluate expert-history hints, DFlash-assisted prefetch, larger batches and graph promotion against the exact demand-loading baseline. Retain a full-load fallback for cold prompts or model architectures whose working sets are too broad.

The recommended commitment is rank 1 as the engineering baseline and a bounded trace-first investigation of rank 3. If the objective is immediate selection of arbitrary unrelated models, rank 2 is the direct capacity solution. Durable KV remains valuable in all three, but always belongs to the model that created it.

**Sources and evidence**

1. [Current durable KV implementation](../runtime/vllm029/overlay/vllm/v1/kv_offload/tiering/multinode.py): content identity, `SlabIO.read`, `BounceController`, and worker completion; offload scheduler source (local reference: `upstream-vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`). Local sources inspected September 13, 2026; upstream checkout details require checking against the pinned image before implementation.
2. Local KV cache manager (local reference: `upstream-vllm/vllm/v1/core/kv_cache_manager.py`), cache-hit limit for obtaining logits; vLLM, [automatic prefix caching](https://github.com/vllm-project/vllm/blob/main/docs/features/automatic_prefix_caching.md), accessed September 13, 2026.
3. vLLM, [sharded Run:ai model streaming](https://github.com/vllm-project/vllm/blob/main/docs/models/extensions/runai_model_streamer.md), accessed September 13, 2026. These are loading primitives, not proof of serve-while-loading support.
4. [Recorded release reload result](../results/vllm029-upgrade/upgrade-029-r8/extended/reload.json) and [upgrade report](VLLM-029-UPGRADE.md), September 11 deployment validation. 100736 prompt tokens, 100608 external hits and 2.703-second first-token latency.
5. [Startup and memory analysis](FAST-START-MODEL-SWITCHING.md), with original log and hardware references. Deployed GLM configuration read over SSH from `/var/tmp/models/GLM-5.3-Int4-Int8Mix/config.json` on `spark-06c4.local`, September 13, 2026: `GlmMoeDsaForCausalLM`, 78 layers, hidden size 6144, 256 routed experts, eight selected per token, one shared expert, three initial dense layers, MoE intermediate size 2048. This is GLM-specific evidence.
6. Eliseev and Mazur, [Fast Inference of Mixture-of-Experts Language Models with Offloading](https://arxiv.org/abs/2312.17238), December 2023. Research precedent for sparse-model offloading, not a Spark benchmark.
7. Yu et al., [LayerScope: Predictive Cross-Layer Scheduling for Efficient Multi-Batch MoE Inference on Legacy Servers](https://arxiv.org/abs/2509.23638), revised April 16, 2026; earlier title/abstract uses PreScope. Research precedent for predictive scheduling and asynchronous I/O.
8. Chen, Liang and Liu, [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036), revised May 28, 2026; [local DFlash proposer](../runtime/vllm029/overlay/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py), including target hidden-state inputs.
9. Chen et al., [SP-MoE: Speculative Decoding and Prefetching for Accelerating MoE-based Model Inference](https://arxiv.org/abs/2510.10302), revised November 6, 2025. Its structural draft/target assumptions require validation before applying the technique here.
