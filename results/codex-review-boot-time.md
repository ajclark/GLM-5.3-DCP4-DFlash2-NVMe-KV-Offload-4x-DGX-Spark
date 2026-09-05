# Codex second opinion: the "binary object" boot (2026-09-05)

Prompt: the human's challenge that nothing changes between restarts, so the
derived state (post-processed weights, compiled graphs, autotune results,
CUDA graphs) should be a binary object each rank loads or memory that stays
resident. Codex (gpt-6-astra, xhigh) read `docs/BOOT-TIME.md` and the fork,
did not run anything, and judged three routes. Verbatim substance below;
fork citations were re-checked by the Claude session and all hold.

**Framing.** "The human's framing is right: repeated derivation is largely
avoidable. But there are two different solutions: serialize immutable
artifacts, or preserve the live executor. A weights-only daemon preserves
less than you want; a warmed engine daemon, with a replaceable API
container, preserves weights, graphs, kernel handles and NCCL connections."

## A. tmpfs-backed parameters adopted via cudaHostRegister: plausible, performance unproven

- Shared physical DRAM does not guarantee identical GPU caching. NVIDIA's
  Tegra app note documents registered host memory as GPU-cached only with
  full system coherency, otherwise uncached; not enough evidence to promise
  cudaMalloc-equivalent speed on GB10.
- Only saves memory if initialization adopts the pages instead of
  allocating another 95 GB; a CPU tensor followed by `.cuda()` defeats it.
- vLLM needs a custom init path: skip destructive repacking, reconstruct
  aliases/attributes, account for memory outside its allocator. The startup
  admission check (`v1/worker/utils.py:405 request_memory`) would reject
  already-resident weights at utilization 0.91.
- Decisive experiment: 256 MiB / 1 GiB / 4 GiB registered mappings vs
  cudaMalloc; measure registration separately from page population,
  streaming bandwidth, a real Marlin kernel, graph replay after a consumer
  restart. 23-25 million 4 KiB pages make extrapolating registration time
  unsafe. Hugetlbfs is a separate experiment, not an assumed cure.

## B. CUDA-IPC holder daemon: plausible; preferable to A if supported

- Do not reject it for GB10 being integrated: the Tegra guide says memory
  IPC became supported with CUDA 13 on newer open-driver platforms; the
  installed driver still needs a probe.
- Experiment: 64 MiB exporter + disposable consumers; import, checksum,
  Marlin/graph replay, restart consumers while watching physical memory.
  Explicit allocations, not caching-allocator suballocations.
- Daemon death is a fatal dependency failure requiring reload, not durable
  memory. Needs the same adoption/accounting path as A. If legacy IPC
  fails, probe CUDA VMM shareable allocations separately.

## C. Post-processed NVMe image: works architecturally; Codex's first implementation choice

- Treat it as a versioned execution checkpoint, not a state_dict: final
  tensor bytes, shapes/strides, aliases, scales, MLA plain attributes,
  required non-tensor metadata. Key by weights, fork/kernel build,
  quantization backend, architecture, parallel layout.
- **Avoid "PWAL on empty weights"**: post-processing performs real
  transformations and parameter replacement. Separate metadata
  construction from transformation, then load final storage directly. The
  default loader (`model_loader/base_loader.py:80`) otherwise re-runs
  post-processing after `load_weights`.
- Without GDS, O_DIRECT needs bounded registered host staging and a copy
  into device allocations; it cannot read straight into cudaMalloc memory.
  On UMA that adds traffic, not a second model-sized copy.
- Decisive experiment: round-trip one representative post-processed
  MoE+MLA layer with aligned direct reads and bounded buffers; compare bytes
  and kernel outputs. Then a maintenance-only full-rank load measures
  whether 15-25 s is realistic. 10 s is the storage-only lower bound.

## Compilation, tuning, process snapshots

- Persist AOT/Inductor artifacts; the fork loads them before retracing
  (`compilation/decorators.py:565`). Persist Triton binaries and, where
  supported, autotune results (Triton `cache_results`). Validate with two
  fresh processes and unchanged manifests.
- BOOT-TIME.md understates the FlashInfer risk: its persistent cache is
  disabled for known key collisions
  (`model_executor/warmup/kernel_warmup.py:115`), potentially selecting
  invalid tactics. Don't merely flip that constant.
- Process-bound regardless: CUDA graphs, loaded module/function handles,
  streams/events, library workspaces, NCCL/RDMA connections.
- CRIU + cuda-checkpoint: no on the installed 580-series ARM drivers
  (NVIDIA added ARM checkpoint support in 595). Even upgraded, work must
  drain, RDMA connections need coordinated handling, and checkpointing
  copies device memory into host allocations, dangerous at this UMA
  footprint. Start with a tiny disposable process, never this engine.

## Floor estimate

"Retaining today's ~50-second startup and ~20-second tail, expect roughly
90-120 seconds plus registration/import time for A/B, or 105-145 seconds
for C, with persistent caches. These are engineering targets, not measured
limits. The document's 'under a minute is impossible' is too absolute:
keep the warmed engine alive, and frontend-only recovery can plausibly be
seconds."

## Where the Claude session lands after this

- A and B are off the table by the human's decision (cold power-up must be
  fast too, so the artifact lives on NVMe); C is the design.
- Take Codex's "no PWAL on garbage" as a preference for the manifest route:
  the bake records every final parameter (name, shape, dtype, strides) and
  every PWAL-created attribute, and the boot-time loader constructs those
  tensors directly and fills them from the image, overriding `load_model`
  so `process_weights_after_loading` never runs on the loaded data. The
  garbage-PWAL trick stays as the quick way to get shapes for the first
  round-trip experiment only.
- Floor difference (Claude 70-90 s vs Codex 105-145 s) is the
  pre-worker startup and server tail, which neither of us has measured
  after the load path changes; the flags-only boot will show them.
