**Fast startup and model switching on the Spark cluster**

Follow-up decisions: the scope is switching [unrelated models](SERVE-WHILE-LOADING-UNRELATED-MODELS.md), whose KV caches are separate. The first implementation priority is [saturating NVMe-to-memory loading](NVME-SATURATED-MODEL-LOADING.md); snapshot and JIT-loading ideas remain later alternatives.

The strongest engineering direction is a persistent worker runtime, with model-specific execution plans and rank-local weight images that can be streamed from NVMe. A 30–60 second transition from stopped inference to useful generation looks physically plausible. Subsecond switching between arbitrary, nonresident models of GLM-5.3's size requires substantially more resident memory, additional serving hardware, or advance loading. Virtualization and snapshots can eliminate repeated setup, but do not eliminate weight-transfer time.

This is an architecture assessment, not a measured fast-start implementation. It uses the recorded September 11 vLLM 0.29.0 deployment, repository source, and primary documentation checked September 13, 2026. Predictions below are explicitly conditional. No serving processes were changed and no new storage or startup benchmarks were run.

**Define the latency contract first**

| Starting condition | What has survived | Appropriate objective |
| --- | --- | --- |
| Host powered off | Local disk only | Measure firmware, OS, driver, network and inference startup together; 30–60 seconds is unproven. |
| Hosts on, inference processes absent | Drivers, local images, weights and compile caches | 30–60 seconds to a real generated token is a useful development target. |
| Runtime asleep, weights absent | Workers, compatible execution plans, possibly communicators and graph allocations | Approach weight streaming time plus activation checks. |
| Target model resident and ready | All target state needed for the request | Routing overhead can be subsecond; prompt processing still adds latency. |
| Target model resident, conversation absent | Model but no compatible KV | Switching is fast, but a long cold prefill may dominate. |

Track time to first generated token on a fixed short prompt, time to resume a persisted long conversation, and time to full configured throughput separately. An HTTP health endpoint or an open socket is insufficient evidence of readiness. Report queue/drain time separately from loading time; completing an arbitrarily long existing generation has no fixed 60-second bound.

**The current evidence changes the priority order**

The recorded TP4/DCP2 deployment uses DFlash2 K=7, a 180224-token limit, 12 sequences, 6 GB KV per rank and a 150 GB disk slab per rank. The rank-zero log reports a 377.41 GiB target checkpoint across 282 files, a 4.58 GiB draft checkpoint, and 96.12 GiB of model-loading allocations. Target weight loading took 350.90 seconds; total model loading took 385.62 seconds. Other ranks reported roughly 364–369 seconds for total model loading. Engine initialization then took 58.78 seconds, including about five seconds of graph capture. These timers overlap by containment; do not add the weight timer to the total model-loading timer.[1]

The checkpoint size is the source set, not a measurement of bytes physically read by each rank. The logs demonstrate traversal of the checkpoint, but cannot establish fourfold physical I/O amplification: slicing, memory mapping and page-cache behavior affect actual reads. Nevertheless, a rank-specific artifact removes unnecessary traversal and can remove unnecessary reads and tensor transformations. Measure both physical bytes and useful output bytes to establish the gain.

There is a second concrete opportunity in cache placement. The launcher persists Triton artifacts under the mounted model directory, while recorded FlashInfer tuning writes under `/root/.cache/vllm/flashinfer_autotune_cache/...`. That latter directory is not explicitly mounted by the launcher. It can survive restarting the same container, but not necessarily replacing it. The log contains a roughly 35-second interval between announcing sparse MLA autotuning and the autotuner starting; its cause needs tracing before attributing the whole interval to tuning.[1][2]

The launcher already fixes the KV memory budget, and the log explicitly reports skipping memory profiling. Therefore “skip profiling” is not an unexplored saving here. Builds and extensive deployment regression gates should run before planned downtime where feasible, but those rollout operations must also be distinguished from the engine's own startup path.

**The physical budget**

DGX Spark has 128 GB of shared LPDDR5x per node. CPU and GPU allocations consume the same physical capacity. Retaining approximately 96 GiB of weights in CPU allocations does not create room for a second approximately 96 GiB model. Peak copy buffers, graph pools, OS memory and KV must also fit.[3]

For planning, use 96.12 GiB ≈ 103.2 GB per rank as an approximate restore payload. This is an allocation figure, not a serialized-weight measurement. An actual image can be smaller or larger depending on packing and included state.

| Sustained useful payload rate per rank | Time for 103.2 GB, transfer only |
| --- | --- |
| 3 GB/s | 34.4 s |
| 5 GB/s | 20.6 s |
| 7 GB/s | 14.7 s |
| 10 GB/s | 10.3 s |
| 20 GB/s | 5.2 s |

These are arithmetic scenarios, not measured SSD rates. All four ranks read their own local drives concurrently, so cluster activation follows the slowest rank, rather than four times one rank's duration. Shared network storage introduces an additional aggregate bandwidth constraint.

A useful pipelined model is `T ≈ T_prepare + max(S_i / B_i) + T_activate`, where `B_i` is useful end-to-end throughput into usable tensor allocations, including the limiting stage of I/O, copying and decoding. If those stages execute serially, their times must be added instead. With 15 seconds of non-transfer work, a 60-second target requires about 2.3 GB/s per rank; a 30-second target requires about 6.9 GB/s. For a resident skeleton with lower preparation cost, the bandwidth requirement is easier.

**Path 1: Prepare rank-specific execution images**

Prepare each model once, before anyone requests it. Save the exact per-rank tensors needed after tensor-parallel partitioning and any expensive quantization-layout transformations. Store large aligned segments plus a small manifest describing tensor offsets, shapes, strides, quantization scales, aliases, buffers, and the execution configuration. Keep each rank's image on that rank's local NVMe. Handle the draft model explicitly: replicated draft weights must be available on every node that runs the drafter.

First test the existing vLLM sharded-state loader and `runai_streamer_sharded`, rather than immediately implementing a custom binary loader. Upstream documents per-rank files and bounded concurrent streaming. The local source also contains `ShardedStateLoader`. Availability in the pinned image and compatibility with GLM's mixed quantization and replicated DFlash path still need direct verification.[4]

The important distinction is between checkpoint tensors and runtime tensors. Some quantized loaders change parameter names, shapes or layouts after loading; dumping their final `state_dict()` does not automatically produce something the ordinary loader can restore. A prepared runtime image needs a matching restoration path that avoids applying those transforms twice. Verify numerical equivalence against the normal loader, including scales and nonparameter buffers.

Stream chunks through a bounded pool of pinned buffers and overlap reads with copies into destination allocations. Start with hundreds of MiB of buffering and tune from measurement. Avoid materializing a whole approximately 100 GB CPU copy alongside the GPU allocations. Benchmark direct I/O against buffered reads; `O_DIRECT` is an option with alignment requirements, not a guaranteed win. Compare fastsafetensors' non-GDS path as well.[5]

Version each artifact by model content, runtime image, GPU architecture, tensor layout and parallel configuration. Hash chunks while streaming instead of rereading the entire image afterward. Inference weights are immutable: when evicting an unchanged model, discard them and later reload the existing image. There is no reason to write approximately 100 GB back to disk at every switch.

This architecture has a research precedent in ServerlessLLM's loading-oriented checkpoint format and multi-tier loading system. Its published gains are not a forecast for this cluster; its useful contribution here is the design pattern.[6]

**Path 2: Keep a runtime skeleton resident**

Separate three lifetimes:

| Lifetime | Candidate state |
| --- | --- |
| Host/runtime lifetime | API routing, controller, worker processes, CUDA contexts, loaded kernel modules, network topology, reusable buffer pools |
| Execution-plan lifetime | Model structure, tensor shapes, quantization method, TP/DCP groups, attention and KV layout, draft configuration, graph variants |
| Model/session lifetime | Weight bytes, model buffers, tokenizer and templates, prefix-cache identity, session tokens and KV |

A compatible weight update can reuse much of the execution plan. Different hidden sizes, attention types, expert layouts or TP degrees generally need a different plan. Maintain a small collection of preinitialized model-specific skeletons, with one active large model, rather than assuming one graph can serve every architecture.

The least invasive experiment is vLLM level-2 sleep: discard weights and KV, preserve the server, reallocate weights, reload, then reallocate KV. Level 1 backs weights up in CPU RAM and is a poor fit for parking another GLM-sized model on Spark. The existing sleep backend abstraction is a useful integration point, but its presence does not imply that a disk-snapshot backend is implemented.[7][8]

Measure what actually remains allocated after sleeping with this exact model, MRv2, quantization, DFlash and offload connector. Graph pools and workspaces may retain significant memory. Sleep also needs to drain connector transfers; a late completion must not write into an allocation reassigned to another model.

Within one compatible plan, preserve virtual addresses and overwrite tensor contents only after every referencing GPU operation completes. CUDA graphs embed addresses and may include collective resources. Stable weight pointers are necessary but insufficient: shapes, strides, modules, KV addresses and communicator references must also remain valid. If preservation is troublesome, recapturing the measured five-second graph set may be a better tradeoff than implementing complete graph persistence.

A separate worker group per skeleton is easier to isolate but retains multiple contexts and communicator sets. A single process with reusable device resources and multiple model plans can reduce that overhead, at substantially higher implementation cost. Do not attempt to share raw NCCL handles between unrelated processes.

**Path 3: Snapshot the reusable parts of the process**

CUDA-aware process checkpointing is worth a bounded feasibility experiment. NVIDIA documents ARM support starting with driver 595, and more IPC support in 610. The utility releases CUDA resources after copying device state to host memory and can combine with CRIU. UVM allocations and IPC created through `cuMemExportToShareableHandle()` remain documented limitations.[9]

Physical shared memory on GB10 is not itself proof that the application uses unsupported CUDA UVM allocations. Inventory the actual allocation APIs and installed driver. Conversely, ARM support is not proof that this complete inference stack is restorable. RDMA objects, sockets, shared memory, multiple processes and graph-contained collective references require an application-level protocol.

The attractive snapshot is a small, prepared runtime with bulk weights and recoverable KV excluded. Restoring a small skeleton and streaming an immutable model image avoids duplicating the largest data. Whether CUDA objects remain restorable after excluding those allocations must be demonstrated; this is a proposed snapshot boundary, not an existing guarantee.

A full approximately 100 GB snapshot still has the same storage lower bound as weights, may include useless scratch space, and can create a large transient host copy during capture. Transparent lazy restoration can defer those bytes, but then page faults land in first-token latency. GPU and DMA memory may additionally require prefaulting or registration before use.

Current NCCL provides communicator suspend/resume for releasing dynamic GPU allocations. This is useful for resident idle runtimes; it does not document persistence across process death or arbitrary NIC removal. The repository's optional NIC hotplug plugin is another relevant experiment, but it is a different mechanism and is disabled by default in the inspected launcher.[10][2]

**Path 4: Hardware-backed microVMs**

MicroVMs are attractive for packaging an already-initialized OS and userspace. The difficult boundary is the accelerator and its communication devices. Saving guest RAM does not, by itself, serialize CUDA contexts, device mappings, queues or the other three ranks.

Firecracker should not be described as simply having no PCI support: its current changelog includes optional PCI transport for virtio devices. Its GPU initiative documents passthrough prototypes, but the cited prototype does not support device snapshot/resume. Those facts do not establish a production GPU snapshot path, and the early GPU project's scope explicitly excluded GPU snapshots.[11][12]

Cloud Hypervisor is a more direct VFIO candidate. Its current documentation supports snapshot/restore for devices exposing VFIO migration v2 through an appropriate variant driver; generic `vfio_pci` is not migratable. Therefore test actual GPU and NIC capabilities, not merely whether a VM can run CUDA. GB10 guest-driver support, assignment/reset behavior and IOMMU grouping remain unverified for this proposal.[13]

The more promising VM arrangement is to assign hardware once, keep that VM alive, and switch prepared model images inside it. That retains the same warm-runtime architecture with a VM isolation boundary. Reassigning the GPU to a new VM for every model switch adds device and driver work to the latency budget.

A second option is a Firecracker frontend that talks to a permanent host GPU service over a coarse inference API. It is a useful isolation arrangement, but the host service still owns the model-loading problem. Proxying individual CUDA operations would require careful latency analysis.

**Path 5: Stream state directly into shared memory**

The compelling ideal is one immutable, file-backed weight mapping that both CPU I/O and GPU kernels can use, with no extra full-size copy. On Spark, shared physical memory does not mean every CUDA allocation is a coherent CPU or device-I/O destination. NVIDIA specifically documents that ordinary device allocations cannot be coherently accessed by CPU/I/O peripherals and that GPUDirect RDMA is unsupported; it recommends host-allocated buffers for RDMA.[14]

Consequently, begin with bounded host buffers and a supported copy path. Test mapped host allocations or registered file mappings as a separate optimization, checking GPU access performance, pinning, cache coherence, alignment, quantization kernels and graph compatibility. Do not assume a successful `mmap` is a GPU-ready tensor.

For discrete GPUs on supported systems, GPUDirect Storage is a relevant way to avoid host bouncing. Its applicability and effective path on this GB10 stack must be measured; a GDS API falling back to CPU staging is not evidence of direct DMA. The first success criterion is sustained useful bandwidth, regardless of branding.[15]

Restoring the entire memory image is often wasteful. Restore immutable model tensors, small execution metadata and the active session's compatible KV. Allocate scratch space without reading its old contents. Keep the rest of the 150 GB-per-rank KV slabs on disk and load blocks on demand.

**Making switching feel instant**

There are several materially different possibilities:

| Concept | What it buys | Constraint |
| --- | --- | --- |
| Two ready resident models or two serving groups | A small routing cutover | Additional memory/hardware; two GLM-sized copies do not fit on the current nodes. |
| Preload the expected next model | Hides some or all I/O before selection | Prediction and enough staging capacity; limited spare RAM on current nodes. |
| Resident base model plus LoRA adapters | Small weight changes and potentially very fast activation | Applies to related adapter variants, not arbitrary models. |
| Remote DRAM model cache | Potentially faster source than one SSD | Network and source-server bandwidth must feed all ranks; Spark requires a supported host-buffer path. |
| Layer-ordered loading | Begins correct prefill before the entire model loads | Every needed weight must arrive before its layer executes; output is still gated by the final layer. |
| MoE expert demand loading | Initially transfers only the experts actually used | Unpredictable routing and expert misses create latency spikes; throughput needs a working-set policy. |

Layer-ordered loading is especially interesting here: store each rank's weights in execution order, start prefill after the first layers arrive, and load later layers concurrently. For short prompts this cannot hide much of the total read, because execution quickly catches up with storage. For long prefills it may overlap meaningful work. It is not permission to run with missing weights or silently skip experts.

A more ambitious same-layout switch could overwrite old layers after the old model's final forward pass has consumed them, allowing a new model's first pass to follow behind. That needs a custom layer scheduler, cross-rank barriers, separate session state and a precise final-use protocol; full CUDA graphs and speculative lookahead make it harder. It reduces some overlap requirements, but does not remove the incoming-byte lower bound or support continuing arbitrary old-model requests.

Lossless compression helps only if saved I/O exceeds decompression cost. Quantized checkpoints may offer limited further compression. Shared tensor chunks or compact deltas are useful where model variants actually share bytes; unrelated checkpoints should be assumed distinct. More aggressive quantization changes the model and should be evaluated as a separate quality/performance option.

For an unpredicted 103 GB-per-rank target absent from RAM, a one-second switch implies over 103 GB/s of useful incoming bandwidth per rank before activation overhead. No microVM or lazy mapping changes that arithmetic. Truly instant arbitrary switching is primarily a residency and capacity decision.

**Conversation continuity and transactional switching**

The existing durable cache already addresses an essential second half of seamless switching. The release validation records 180.83 seconds for a cold long prompt and 2.70 seconds for first-token latency after loading its persisted prefix. This is a same-model persisted-prefix result, not a model-switch benchmark.[16]

On A → B → A, retain A's disk namespace so the conversation can resume cheaply. On A → B, B generally must compute its own KV from the conversation text. Changing the tokenizer can change the token sequence as well. A fast weight switch does not make A's activations valid for B.

The current slab identity uses file metadata fingerprints and runtime/configuration identity. In-process replacement can bypass assumptions tied to startup or file paths. Extend it with an explicit immutable model revision and activation generation; compute true content identity when preparing the image. Select the correct tokenizer, templates, target/draft pair and KV namespace together.[17]

A proposed switch protocol is:

1. Prepare and validate B's artifacts before the request to switch, wherever possible.
2. Stop admitting A requests; drain to a defined boundary or explicitly pause supported sessions. Fence GPU work and KV transfers on all ranks.
3. Retain required session metadata and seal its durable KV. Discard A's immutable weight pages when capacity requires it.
4. Load B's rank image into its compatible skeleton or initialize its separate plan. Recreate any resources that were not retained.
5. All ranks attest the same revision, plan and generation. Run a short distributed generation check, then commit the route atomically.

If one rank fails, keep B unavailable everywhere. Reload A if its allocations have already been overwritten. Keeping A's disk image gives a recovery path, but not instant rollback. Preserving a fully live A while preparing B requires spare capacity.

**The next experiments, in order**

| Experiment | Decision it resolves |
| --- | --- |
| Trace one ordinary startup per rank, separating disk-cold and cache-warm cases | How much time is I/O, tensor processing, module loading, network setup, warmup and admission? |
| Read each rank's prospective artifact through the actual bounded destination pipeline | Is ≥2.3 GB/s feasible for a 60-second budget with 15 seconds overhead, or ≥6.9 GB/s for 30 seconds? |
| Prepare sharded GLM + draft artifacts and compare restored outputs | Can existing loaders deliver the required layout without extra full copies or repeat repacking? |
| Persist measured FlashInfer/Torch/Triton caches under versioned paths | How much of the approximately one-minute setup phase disappears across container replacement? |
| Run repeated same-model level-2 sleep/reload cycles | Are graph addresses, draft state, communicator operation and connector transfers correct; how much memory remains? |
| Switch two compatible revisions, then two genuinely different execution plans | Where is the reuse boundary, and what is the cost of multiple sleeping skeletons? |
| Resume A's persisted conversation after A → B → A | Does model identity prevent stale KV while retaining fast same-model recovery? |
| Small CUDA/CRIU and VFIO feasibility probes on spare capacity | Does snapshot engineering offer savings beyond a resident skeleton and five-second graph recapture? |

Measure peak physical RAM and first-token latency throughout. Exercise a failed rank during activation and a pending KV write during sleep. Do not flush global page caches or stress storage while serving merely to obtain a cold benchmark; arrange an isolated run. Separate reusable model preparation from the timed switching operation.

The recommended first milestone is rank-local prepared weights plus persistent compilation/tuning artifacts and real first-token measurement. The second is a level-2 sleeping worker group with a bounded streaming loader and transactional model identity. Full CUDA snapshots and GPU-backed microVM restoration become worthwhile only if the remaining setup cost justifies their platform and distributed-state complexity.

**Sources and local evidence**

1. Recorded vLLM 0.29.0 deployment, September 11, 2026: rank-zero startup log (local reference: `results/vllm029-upgrade/upgrade-029-r8/running-spark-06c4.local.log`), lines around 2578–2589 and 3156–3234; rank-one log (local reference: `results/vllm029-upgrade/upgrade-029-r8/running-spark-365c.local.log`); validation log (local reference: `results/vllm029-upgrade/upgrade-029-r8/validated-spark-06c4.local.log`). Local evidence, not a new benchmark.
2. [Current release launcher](../runtime/vllm029/launch.sh). Local configuration, inspected September 13, 2026.
3. NVIDIA, [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/dgx-spark.pdf), hardware specifications. Accessed September 13, 2026.
4. vLLM, [Run:ai Model Streamer integration and sharded loading](https://github.com/vllm-project/vllm/blob/main/docs/models/extensions/runai_model_streamer.md); local sharded-state loader (local reference: `upstream-vllm/vllm/model_executor/model_loader/sharded_state_loader.py`). Main-branch capabilities require verification against the deployed image.
5. Foundation Model Stack, [fastsafetensors overview](https://github.com/foundation-model-stack/fastsafetensors/blob/main/docs/overview.md). Accessed September 13, 2026.
6. Fu et al., [ServerlessLLM: Low-Latency Serverless Inference for Large Language Models](https://arxiv.org/abs/2401.14351), 2024; [project repository](https://github.com/ServerlessLLM/ServerlessLLM).
7. vLLM, [Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/), August 6, 2026 documentation. Accessed September 13, 2026.
8. Local sleep backend abstraction (local reference: `upstream-vllm/vllm/device_allocator/sleep_mode_backend.py`); vLLM, [CUDA checkpoint/restore RFC #34303](https://github.com/vllm-project/vllm/issues/34303). An abstraction/RFC is not a validated backend for this deployment.
9. NVIDIA, [cuda-checkpoint README](https://github.com/NVIDIA/cuda-checkpoint), driver-version features and functionality limitations. Accessed September 13, 2026.
10. NVIDIA, [NCCL 2.31.2 communicator management](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommsuspend). Accessed September 13, 2026.
11. Firecracker, [changelog](https://github.com/firecracker-microvm/firecracker/blob/main/CHANGELOG.md), optional PCI virtio transport. Accessed September 13, 2026.
12. Firecracker, [GPU and PCIe discussion #4845](https://github.com/firecracker-microvm/firecracker/discussions/4845), project scope and November 2024 prototype limitations. Historical prototype evidence, not current deployment certification.
13. Cloud Hypervisor, [VFIO documentation](https://github.com/cloud-hypervisor/cloud-hypervisor/blob/main/docs/vfio.md), device assignment and migration-v2 snapshot requirements. Accessed September 13, 2026.
14. NVIDIA, [DGX Spark CUDA porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html), May 24, 2026 update. Accessed September 13, 2026.
15. NVIDIA, [GPUDirect Storage design guide](https://docs.nvidia.com/gpudirect-storage/design-guide/). Accessed September 13, 2026; general design, not evidence of GB10 support.
16. [vLLM 0.29.0 upgrade and validation report](VLLM-029-UPGRADE.md), cold and persisted-prefix measurements.
17. [Current durable KV implementation](../runtime/vllm029/overlay/vllm/v1/kv_offload/tiering/multinode.py), `_dir_fingerprint` and `_content_identity`.
