# GLM-5.3 on four DGX Sparks

Run GLM-5.3 Int4-Int8Mix across four DGX Sparks with **concurrent NVMe model
loading, a sharded KV cache, durable prefix reuse, and RDMA collectives**. The stack
runs vLLM 0.29.0 with DFlash2 speculative decoding over a switchless RoCE ring.

## Performance

Four DGX Sparks, GPU clocks locked at about 2000 MHz, greedy decoding, single requests unless noted.
**These numbers were measured on an abliterated graft of GLM-5.3 that uses the identical quantization
envelope** (Int4 g128 routed experts, Int8 g128 elsewhere, same shapes and kernels) as GLM-5.3 Int4-Int8Mix.
Speed per token is the same; DFlash2 acceptance, and so decode tok/s, may differ slightly on the original
weights.

**Decode** (2026-10-01; `spec_accept_probe.py`: 15 prose and 15 code prompts, 63 replayed coding-agent turns
of up to ~90K context):

| | stock (09-29) | now | change |
|---|---:|---:|---:|
| Prose, tok/s | 17.9 | **26.7** | +49% |
| Code, tok/s | 34.2 | **42.8** | +25% |
| Agent turns (median), tok/s | 22.7 | **33.0** | +45% |
| Verify cycle, 1 request × 1 draft | 93.9 ms | **72.0 ms** | −23% |
| Verify cycle, 1 request × 7 drafts | 141.6 ms | **116.0 ms** | −18% |

| 2–4 concurrent requests, aggregate tok/s (`conc_bench.py`) | base (09-30) | now |
|---|---:|---:|
| Prose, n = 2 / 3 / 4 | 24.1 / 28.8 / 33.8 | **37.2 / 46.7 / 53.7** |
| Mixed prose / code / agent, n = 2 / 3 / 4 | 31.7 / 36.6 / 43.3 | **44.8 / 48.1 / 54.7** |
| Code, n = 2 / 3 / 4 | 46.5 / 56.2 / 65.1 | **53.0 / 62.7 / 73.3** |

**Prefill** (2026-10-06; `prefill_bench.py`: uncached prompts with a unique `cache_salt`, median of 3):

| Prompt | before (10-05) | now | change |
|---:|---:|---:|---:|
| 4K tokens | 587 tok/s | **759 tok/s** | +29.2% |
| 32K tokens | 575 tok/s | **747 tok/s** | +30.0% |
| 60K tokens | 567 tok/s (TTFT 108.4 s) | **740 tok/s (TTFT 83.0 s)** | +30.5% |

The prefill changes leave decode unchanged (verify cycle at one draft 70.8–74.0 ms). What changed, each with its
own launcher switch and report:

| Change | Effect | Report |
|---|---|---|
| Draft-aware verification cap (verify K ∈ {1,3,5,7} from the drafter's confidence), single requests and batches of 2–4 | prose +16% at C1, C2–C4 prose +24–34% | [verify cap](docs/DFLASH2-VERIFY-CAP-RESULTS.md), [batches](results/step1-batchcap-20260930/REPORT.md) |
| Int8 DFlash2 drafter, `fc` split over TP, FP8 draft head | cycles −5 to −10 ms | [step 2](results/step2-drafter-20260930/REPORT.md) |
| RoCEnante one-shot RDMA collectives for the DCP pairs | byte-identical; cycles −4 to −6 ms | [step 3](results/step3-roce-20261001/REPORT.md), [review](docs/ROCENANTE-REVIEW.md) |
| RDMA ring for TP all-reduces ≤ 1 MiB, edges on both PCIe links | 95 → 39 µs at 8 tokens | [step 7](results/step7-ring-allreduce-20261001/REPORT.md), [latency](docs/RDMA-COLLECTIVES-LATENCY.md) |
| Rank-local K decision, L2 prefetch in collective windows, vocab-parallel argmax | single-request cycles −7.5 to −8.4 ms | [steps 4–5](results/step456-integration-20261001/REPORT.md) |
| Prefill DCP collectives on RoCEnante striped over both PCIe twins | prefill +10–11%, byte-identical | [prefill](results/prefill-item2-20261006/REPORT.md) |
| DCP combine without layout copies (`DCP_GLUE`) | prefill +6%, byte-identical | [prefill](results/prefill-item2-20261006/REPORT.md) |
| Early local-shard copy in prefill RoCE gathers | prefill +1–2%, byte-identical | [prefill](results/prefill-item2-20261006/REPORT.md) |
| NCCL TP all-reduce over both PCIe links (2 channels, 4 NICs) | prefill +7–8%; 2 channels reorder NCCL's sums (inside the A/A envelope), 4 NICs bit-identical to that | [prefill](results/prefill-item2-20261006/REPORT.md), [network](docs/NETWORK.md) |
| Threaded O_DIRECT NVMe KV tier, 400 GB slab per rank | restores at 9.5–11 GB/s per node | [KV tier](docs/NVME-KV-TIER-OPTIMIZATION-RESULTS.md) |
| Voice-first scheduling (requests with priority < 0 skip queued prefills) | voice TTFT behind a 60K prefill 98.5 → 6.0 s | [scheduling](docs/VOICE-FIRST-SCHEDULING-RESULTS.md) |

The decode steps, their order and how each was measured are in [decode speedups](docs/DECODE-SPEEDUPS.md).

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
| Model | GLM-5.3 Int4-Int8Mix (`/var/tmp/models/GLM-5.3-Int4-Int8Mix`) |
| Runtime | vLLM 0.29.0, CUDA 13 ARM64 base pinned by digest |
| Serving image | `spark-vllm:0.29.0-nvme4-stridefix-kvtier-verifycap12-e2-20261006` (release image + coalesced loader + threaded KV tier + overlay layer) |
| Parallelism | TP4 / DCP2 across four Sparks |
| Speculative decoding | DFlash2, K=7, int8 drafter, draft-aware verification cap |
| Collectives | RoCEnante for the DCP pairs (decode and prefill), RDMA ring for TP all-reduces ≤ 1 MiB, NCCL on both PCIe links for prefill ([network](docs/NETWORK.md)) |
| Context window | 180,224 tokens |
| Maximum concurrent sequences | 12 |
| GPU KV allocation | 6 GB per rank |
| Durable KV storage | 400 GB NVMe slab per rank, threaded O_DIRECT I/O |
| Scheduling | Voice-first priority scheduler; prefix-hit fix |

All of this is the default of `runtime/vllm029/launch.sh`; every feature has its own switch there (for example
`VERIFY_CAP=0`, `ROCE_DCP=0`, `ROCE_TP=0`, `DCP_GLUE=0`, `NCCL_HCAS=…`).

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

To build and start the full serving stack (each step runs from the repository root; the image builds do not stop
serving):

```bash
# 1. release image (runtime/vllm029/build.sh), then the coalesced loader layer on all four Sparks
NVME_IMAGE=spark-vllm:0.29.0-nvme4-stridefix-20260920 .venv/bin/python runtime/nvme_loader/build.py
# 2. threaded NVMe KV tier on top of it, on every Spark
docker build -t spark-vllm:0.29.0-nvme4-stridefix-kvtier-20260928 -f runtime/vllm029/Dockerfile.kvtier runtime/vllm029
# 3. overlay layer (verification cap, scheduler, drafter, glm_fast, RDMA collectives), built on all four Sparks
TAG=spark-vllm:0.29.0-nvme4-stridefix-kvtier-verifycap12-e2-20261006 bash runtime/vllm029/verify_cap_overlay/build.sh
# 4. int8 DFlash2 drafter (compressed-tensors W8A16), on every Spark
python3 tools/drafter_int8.py /var/tmp/models/GLM-5.3-DFlash2-draft /var/tmp/models/GLM-5.3-DFlash2-draft-int8
# 5. static address on each node's port-0 PCIe twin (docs/NETWORK.md), once
bash node/roce-p0-twin.sh apply
# 6. coordinated start of all four ranks (validate first with --dry-run)
bash start-glm53.sh --dry-run && bash start-glm53.sh
```

`start-glm53.sh` checks every node's launcher and network before stopping anything, starts ranks 3/2/1/0 and
waits for `/health`. The containers use restart policy `no`; after a host reboot, run it again.

For setup, loader options, tests, and rollback, use the
[loader guide](runtime/nvme_loader/README.md). For engine builds and upgrades,
use the [vLLM runtime guide](runtime/vllm029/README.md). The root rollout scripts
select the vLLM 0.29.0 release; `VLLM_RUNTIME=legacy` selects the historical engine.

## Repository guide

| Location | Contents |
|---|---|
| [runtime/nvme_loader/](runtime/nvme_loader/) | Model loaders, deployment controller, and boot instrumentation |
| [runtime/vllm029/](runtime/vllm029/) | Pinned release image, runtime port, regression checks, benches (`prefill_bench.py`, `cycle_bench.py`, `conc_bench.py`, `spec_accept_probe.py`) |
| [runtime/vllm029/verify_cap_overlay/](runtime/vllm029/verify_cap_overlay/) | Overlay layer: verification cap, voice-first scheduler, drafter, `glm_fast` (L2 prefetch, argmax, DCP glue) |
| [runtime/vllm029/roce/](runtime/vllm029/roce/) | RoCEnante and ring RDMA collectives (vendored b12x), pair and ring tests, NCCL benchmark |
| [start-glm53.sh](start-glm53.sh), [node/](node/), [tools/](tools/) | Coordinated start, node network profile, int8 drafter converter |
| [tests/](tests/) | Loader, kernel, cache, and runtime correctness tests |
| [docs/](docs/) | Implementation details, validation reports, and operating notes |
| [results/](results/) | Recorded measurements and validation evidence |
| [baseline/](baseline/), [overlay/](overlay/), [patches/](patches/) | Historical engine sources and patches |

## Credits

Built on [tonyd2wild's GLM-5.3 Int4-Int8Mix TP4 recipe for four DGX Sparks](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark),
including its quantization, sparse-MLA kernels, and DFlash2 port. This project
adds the DCP integration, durable multi-node KV tier, concurrent model loader, verification cap and the
collective and prefill work above.

The RDMA collectives and several decode optimizations come from other people's work. This project's own part is
routing the DCP pairs (not only the TP group) through RoCEnante, the ring transport for TP on the switchless ring,
striping over both PCIe twins, and the prefill changes.

| What we use | Origin | How it is used here |
|---|---|---|
| RoCEnante one-shot RDMA collectives (`b12x.comm.roce`) | Luke Alonso ([@lukealonso](https://github.com/lukealonso)) and Jason Cook ([@original-el8](https://github.com/original-el8)), [Local Inference Lab b12x](https://github.com/local-inference-lab/b12x) (local-inference-lab/b12x#295), Apache-2.0 | Vendored subset at `b58f34ea` under `runtime/vllm029/roce/b12x/`, byte-identical except one reviewed proxy patch; also the base of the ring proxy and `_pipe_proxy.c` |
| vLLM communicator for RoCEnante (capability vote, eligibility limits, worker health check) | Jason Cook, local-inference-lab/vllm#597 | Ported in `roce/glm_roce/adapter.py` |
| Port of #597 to the DGX Spark vLLM tree (all-gather switch, async-output forwarding) | tonyd2wild, [GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark) (`speed-night-2026-09-18/roce`) | The path our adapter came by |
| RoCE routing shim (`glm_roce/`) and the vendored b12x subset we started from | [knapcio, GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4) @770d115, MIT | Adapted to route the DCP groups, then the TP ring (MIT notice in `roce/glm_roce/LICENSE`) |
| One-shot RDMA collectives on DeepSeek (SGLang TP8 overlay) that showed the pattern | rhys101 (SG17) | Idea |
| L2 prefetch in collective windows; target-side vocab-parallel argmax; FP8 draft head | knapcio, GLM-5.3-Flash-4x-DGX-Spark-TP4 @770d115 (Apache-2.0 overlays, MIT draft head) | Re-implemented for the full GLM-5.3 in `verify_cap_overlay/glm_fast/`; FP8 head in the drafter |
| Vocab-parallel value/index reduction | vLLM `LogitsProcessor.get_top_tokens` (vllm#34049, zixi-qi) | Used for the target model's greedy verify |
| Prefill targets and the overlapped-collective idea | TensorFold GLM-5.3 TP4 stack, [drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark](https://github.com/drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark) | Its review started the prefill work; the micro-batch overlap was analysed and dropped, the shipped changes are listed under Performance |

Licenses and modified files are listed in [NOTICE](NOTICE).
