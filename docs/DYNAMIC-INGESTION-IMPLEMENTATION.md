# Dynamic checkpoint ingestion: implementation and Spark validation

**Deployment update:** the subsequent
[coalesced reader](COALESCED-LOADER-IMPLEMENTATION.md) is now deployed, reaching
114.68 seconds to healthy and 41.8–42.4 seconds for target weights. The results
below document the earlier Run:ai implementation.

The `nvme` vLLM loader now has a `stream` mode that consumes original checkpoint
shards without a prior full-content hashing pass, prepared rank files, or
artifact export. It is implemented as a default-loader iterator extension,
preserving native model construction, placement, quantization processing, and
engine warmup. This is the implementation of the
[dynamic ingestion investigation](DYNAMIC-CHECKPOINT-INGESTION.md).

**The Run:ai build was deployed on all four Sparks.** It reached a healthy API in
**131.27 seconds**, streamed target weights in **59–61 seconds per rank**, and
passed generation and durable-KV recovery. It requires no prepared weight files.
The complete report below distinguishes first and final builds.

The next optimization investigation is the
[boot roofline advancement plan](BOOT-ROOFLINE-ADVANCEMENTS.md), covering
redundant reads, placement traffic, startup overlap, and first-request readiness.

## Completed validation

The first full four-Spark activation reached a healthy API in **136.06 seconds**
and passed the count-to-100 generation check. Workers were already stopped after
GPU fixture testing, so this is controller launch-to-health time. Hosts were
already booted and compiler/driver caches persisted. It is not a cold host reboot
or an empty compiler-cache measurement.

| First full stream, rank 0 | Result |
|---|---:|
| Target source payload | 405.220 GB, 177,569 tensors |
| Target stream including staging and consumer backpressure | 66.372 s |
| Process storage-read counter delta during target stream | 405.206 GB |
| Storage reads / target stream wall time | 6.105 GB/s |
| Draft source payload | 4.919 GB, 96 tensors |
| Draft stream | 1.361 s |
| Target plus draft model loading and processing | 80.843 s |
| Engine KV allocation and warmup | 18.72 s |
| Controller launch to healthy API | 136.060 s |

The earlier native target load took 358.03 seconds. The new target stream is
about 5.4 times faster, while avoiding the subsequent artifact-export phase.
Prepared artifacts remain faster for repeated loads: the previous target restore
was about 8 seconds, with full serving restarts around 82–87 seconds.

The Linux per-process storage-read counter is distinct from bytes yielded to
the model. In this run they nearly agree, so the result is not simply reading
the checkpoint out of page cache. Counters can include other reads by the worker;
they are not a block-device utilization measurement. No synthetic storage
benchmark was run. The supplied 4–5 GB/s remains a planning specification,
not a reason to promise this observed rate on every device.

Evidence: [activation result](../results/nvme-loader/nvme-stream-1789317849.json),
[generation result](../results/nvme-loader/nvme-stream-1789317849-smoke.json),
[controller log](../results/nvme-loader/full-stream.log),
[first deployment image IDs](../results/nvme-loader/stream-first-deployed.json).

### Final build: separate CUDA upload stream

After exact GPU parameter/output parity passed again, the final four-node run
used a separate upload stream to reduce serialization with native GPU transforms:

| Final build | Result |
|---|---:|
| Target streaming across ranks | 59.10–60.67 s |
| Storage reads / streaming wall time across ranks | 6.68–6.86 GB/s |
| Draft streaming across ranks | 1.34–1.77 s |
| Target plus draft loading/processing, rank 0 | 74.56 s |
| Engine KV allocation and warmup | 19.06 s |
| Controller launch to healthy API | 131.27 s |
| Count-to-100 generation | Passed |
| First durability probe after restart | 1,664 external KV hits; zero GPU prefix hits |

Both full activations started after the fixture harness had stopped the workers;
the reported health times exclude fixture downtime. Target streaming is about
six times faster than the prior 358-second native load. These two runs support
the observed improvement, not a statistically isolated speedup attributable
solely to the separate stream.

Evidence: [all-rank results](../results/nvme-loader/stream-results.json),
[final activation](../results/nvme-loader/nvme-stream-1789318252.json),
[generation](../results/nvme-loader/nvme-stream-1789318252-smoke.json),
[durable KV](../results/nvme-loader/kv-stream-verify.json),
[deployed image/source hashes](../results/nvme-loader/stream-deployed.json).

## Correctness and failure handling

- **21 sandbox tests passed.** Tests cover original multi-file tensors, mixed
  dtypes, scalar/empty entries, storage aliases retained by consumers, invalid
  metadata, missing weights, source replacement including symlink redirection,
  cancellation, teardown failures, and the existing artifact/distributed gates.
- **Llama and GPT-2 GPU parity passed.** Every parameter hash, generated token,
  and token log probability exactly matched native loading in both CPU and GPU
  streaming modes. GPU mode was checked again after separating the upload stream.
  These tests ran without a configured artifact root or prepared source identity.
- **Mixed quantized GLM and DFlash loaded and generated on TP4/DCP2.** All selected
  nonempty checkpoint entries were accounted for before activation.
- **Automatic recovery was exercised.** The initial native fixture run generated
  correctly but its parameter-hash RPC hit a test-only serialization setting.
  The harness restored the retained serving stack and verified generation before
  the corrected test run. Only isolated fixture containers enable callback
  serialization; the production configuration does not.

The [fixture log](../results/nvme-loader/stream-model-checks-r2.log) records the
native/CPU/GPU comparisons. The [upload-stream fixture log](../results/nvme-loader/stream-upload-checks.log)
records the final CUDA placement comparison. Fable reviewed the
[implementation](../results/nvme-loader/fable-stream-implementation-review.md)
and [producer lifetime/cancellation](../results/nvme-loader/fable-stream-pump-review.md).
Reviews contain proposals as well as findings; observed results above take
precedence over their unmeasured timing or memory estimates.

## How the implementation works

`DefaultModelLoader` still selects canonical checkpoint files and handles
primary/secondary sources, prefixes, allowed patterns, expert filtering, and
native weight tracking. The plugin validates safetensors headers using the
official parser before starting its streamer. Unsupported formats, tensor-subclass
reconstruction, unavailable streaming dependencies, over-budget individual
tensors, and online quantization use the native path where detected before
stream consumption. A late error aborts the worker; it never reloads into
partially mutated model state.

Run:ai Model Streamer 0.16.1 supplies 32 concurrent reads across original files
with a shared 2 GiB CPU buffer budget. One producer thread advances that iterator
and creates owning output tensors, preparing at most one queued tensor ahead
of the native consumer. It never calls the model's mutable weight callbacks.
Completed source files receive `POSIX_FADV_DONTNEED` to avoid filling unified RAM
with a second checkpoint copy in page cache.

In GPU mode all tensors, including scales and scalars, go to the GPU. Uploads
finish before Run:ai can reuse the CPU buffer. The final implementation uses
a separate CUDA upload stream, waits for initial caller-stream work, and records
the consumer stream on returned tensors before native use. This lets native
transposes and copies use the GPU and preserves ownership across asynchronous
consumer work. CPU and pinned-CPU modes remain available.

A storage-level budget tracks aliases through C++ storage weak references.
The default owning budget is 6 GiB, separate from the read buffer, native
temporaries, and allocator caches. The DFlash loader legitimately retains its
entire 4.919 GB checkpoint before placement; the first full run confirmed that
exact owning peak. Target owning peak was about 2.054 GB. The memory guard
monitors the whole node rather than treating logical allocation budgets as
a physical RAM cap.

Missing zero-element tensors are synthesized because the installed streamer
omits them. Missing nonempty tensors, duplicates, unexpected shapes/dtypes,
source mutation, or incomplete consumption fail activation. The all-rank gate
compares source metadata and runtime configuration; streaming receipts explicitly
set `content_verified: false` and do not confuse header identity with a full
payload digest.

## Durable KV and deployment boundaries

The Spark stream profile uses a separate store under
`/var/tmp/nvme-loader/stream-kvcache`, capped at 30 GB per rank. Its salt is
stable only when optional, previously hashed immutable source inventories still
match; otherwise it is fresh per activation. The retained artifact service's
slabs are unchanged. Canonical model directories and model paths are never rewritten.
Unrelated models cannot reuse each other's KV.

This store policy is part of this cluster's deployment controller. Other
deployments must provide their own compatible KV identity policy. The stream
loader itself does not require the optional hashed source manifests.

The deployed engine remains vLLM 0.29.0 with the existing runtime overlays and
dependencies. The derivative image is `spark-vllm:0.29.0-nvme2`. Original and
last-good containers/images are retained. No GPU or microVM snapshot, in-process
architecture replacement, TP-selective reader, or cross-node weight scatter
has been implemented. This path reads full original tensors on each node and
finishes native processing/warmup before serving.

## Reproduce

```bash
PYTHONPATH=runtime/nvme_loader .venv/bin/python -m pytest \
  tests/test_nvme_streaming.py tests/test_nvme_loader.py -q
.venv/bin/python runtime/nvme_loader/build.py
.venv/bin/python runtime/nvme_loader/check_stream_models.py
.venv/bin/python runtime/nvme_loader/sparkctl.py stream
```

For an ordinary compatible vLLM installation with the wheel and streaming SDK:

```bash
NVME_LOADER_MODE=stream NVME_STREAM_DEVICE=cuda \
  vllm serve /models/example --load-format nvme
```

The controller remains specific to this cluster's GLM serving profile. The
loader accepts ordinary supported safetensors architectures; performance depends
on their size, tensor layout, and native loader behavior. The 30–60 second
end-to-end objective has not been reached.
