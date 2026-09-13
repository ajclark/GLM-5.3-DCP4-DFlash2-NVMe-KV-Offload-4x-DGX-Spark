# GLM-5.3 on four DGX Sparks

Run GLM-5.3 Int4-Int8Mix across four DGX Sparks with **concurrent NVMe model
loading, a sharded KV cache, and durable prefix reuse**. The stack runs
vLLM 0.29.0 with DFlash2 speculative decoding over a switchless RoCE ring.

## Highlights

### Faster startup from original model files

The coalesced loader reads checkpoint tensors concurrently and overlaps NVMe
reads with GPU uploads. It uses 128 MiB batches on the Sparks and requires no
advance conversion into model-specific loading artifacts.

| Startup milestone | Previous Run:ai loader | Coalesced loader |
|---|---:|---:|
| Target weights loaded, per rank | 59–61 s | **41.8–42.4 s** |
| API healthy, from launch | 131.3 s | **114.7 s** |
| First streamed output, from launch | Not recorded | **115.2 s** |

These measurements start with the serving workers stopped on already booted
hosts, with compiler and driver caches present. They exclude shutdown time;
**they are not cold-machine boot times**. Each configuration has one full
activation measurement.

Validation passed sandbox tests, CUDA transport checks, exact Llama/GPT-2
loading comparisons, full-model generation, and same-model durable-KV recovery.
See the [loader implementation and results](docs/COALESCED-LOADER-IMPLEMENTATION.md).

### More context from the same KV memory

Decode context parallelism (DCP) distributes the target model's KV cache across
GPUs. The serving profile uses **DCP2**, doubling target KV capacity relative to
DCP1; DCP4 provides four times the capacity. DFlash2 retains its replicated draft
cache and runs alongside either configuration.

The earlier custom engine served a **500k-token prompt** under DCP4. That was a
separate capacity test with almost no memory headroom; the current serving
window is 180,224 tokens. Detailed capacity and throughput comparisons are in
the [DCP implementation report](docs/DESIGN.md) and
[historical concurrency measurements](results/concurrency-sweeps.md).

### Reuse long prefixes after a restart

Each worker stores KV blocks in a fixed-size NVMe slab. Cache identity, epochs,
and checksums protect reuse; incompatible or damaged entries become cache misses.
**KV belongs to the model that created it and is not shared between unrelated models.**

In the vLLM 0.29.0 validation, a 100,736-token retrieval request reached its first
token in **180.83 seconds** without cache hits. After an engine restart, it
recovered **99.87%** of the prefix from NVMe and reached its first token in
**2.70 seconds**, with zero GPU prefix-cache hits. These are request latencies
once the engine is serving. Full-machine reboot recovery was also validated on
the earlier custom engine.

See the [release validation](docs/VLLM-029-UPGRADE.md) and
[durable KV implementation](docs/NVME-DESIGN.md).

## Current serving configuration

| Component | Configuration |
|---|---|
| Model | GLM-5.3 Int4-Int8Mix |
| Runtime | vLLM 0.29.0, CUDA 13 ARM64 base pinned by digest |
| Parallelism | TP4 / DCP2 across four Sparks |
| Speculative decoding | DFlash2, K=7 |
| Context window | 180,224 tokens |
| Maximum concurrent sequences | 12 |
| GPU KV allocation | 6 GB per rank |
| Durable KV storage | 30 GB per rank in the tested coalesced profile |
| Loader image | `spark-vllm:0.29.0-nvme4` |

The coalesced deployment builds on the validated `upgrade-029-r8` release.
The standard release rollout uses a 150 GB NVMe slab per rank; the loader
controller selects its own 30 GB store. Historical benchmark configurations
are identified in their reports.

## Build and run

On the configured four-Spark cluster, build the loader image and activate
original-checkpoint streaming:

```bash
.venv/bin/python runtime/nvme_loader/build.py
.venv/bin/python runtime/nvme_loader/sparkctl.py stream --observe-boot
```

The controller defaults to coalesced CUDA loading with 128 MiB batches. It
retains the previous containers, monitors memory, validates generation, and
restores the last tested service if activation fails. Hostnames, checkpoint
paths, and mounts are specific to this cluster.

For setup, loader options, tests, and rollback, use the
[loader guide](runtime/nvme_loader/README.md). For engine builds and upgrades,
use the [vLLM runtime guide](runtime/vllm029/README.md). The root rollout scripts
select the vLLM 0.29.0 release; `VLLM_RUNTIME=legacy` selects the historical engine.

## Repository guide

| Location | Contents |
|---|---|
| [runtime/nvme_loader/](runtime/nvme_loader/) | Model loaders, deployment controller, and boot instrumentation |
| [runtime/vllm029/](runtime/vllm029/) | Pinned release image, runtime port, and regression checks |
| [tests/](tests/) | Loader, kernel, cache, and runtime correctness tests |
| [docs/](docs/) | Implementation details, validation reports, and operating notes |
| [results/](results/) | Recorded measurements and validation evidence |
| [baseline/](baseline/), [overlay/](overlay/), [patches/](patches/) | Historical engine sources and patches |

## Credits

Built on [tonyd2wild's GLM-5.3 Int4-Int8Mix TP4 recipe for four DGX Sparks](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark),
including its quantization, sparse-MLA kernels, and DFlash2 port. This project
adds the DCP integration, durable multi-node KV tier, and concurrent model loader.
