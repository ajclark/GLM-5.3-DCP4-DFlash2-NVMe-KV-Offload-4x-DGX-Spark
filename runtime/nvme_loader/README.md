# Concurrent local NVMe model loading

This package registers vLLM's `nvme` load format. A normal checkpoint load can
publish each worker's state **before native kernel postprocessing**. Subsequent
workers allocate the ordinary model, read only their local rank artifact, and
run the normal kernel postprocessing once. Model construction, quantization,
attention initialization and CUDA graph capture still belong to vLLM.

The four-Spark deployment is running this loader. Prepared-artifact restarts took
**82–87 seconds**, including shutdown and readiness polling; target weight
restoration took **about 8 seconds per rank**. See the
[implementation and results report](../../docs/NVME-LOADER-IMPLEMENTATION.md).
The current coalesced deployment streams original shards in **41.8–42.4 seconds
for target weights**, reaches a healthy API in **114.68 seconds**, and produces
first streamed output at **115.20 seconds**. Hosts were already booted and
compiler caches present. Generation and durable-KV recovery passed. See the
[coalesced implementation report](../../docs/COALESCED-LOADER-IMPLEMENTATION.md).
The [earlier Run:ai stream](../../docs/DYNAMIC-INGESTION-IMPLEMENTATION.md)
took 59–61 seconds for target weights and 131 seconds to healthy.

The artifact reader uses 32 concurrent `preadv` calls, 4 MiB extents, aligned pinned
staging, and `O_DIRECT`. Its staging allocation is approximately 128 MiB per
worker. Reads from one rank file can all run concurrently; file count no longer
limits parallelism. Each staging slot is fenced by the CUDA copy completion
event before reuse. Every extent is SHA256-verified before copying. This is
local storage I/O on each worker; weights do not traverse the NCCL ring.

## Artifact contract

Artifacts contain physical storage bytes and a schema describing names, shapes,
strides, dtypes, offsets and aliases, including nonpersistent buffers. Restore
requires an exact constructor schema match. The manifest includes source content
identity, model/config, precision, topology, global rank, CUDA capability and
runtime versions. Data and metadata are fsynced and published by directory
rename. A corrupted, incomplete or incompatible artifact cannot activate.

Preparation hashes all canonical checkpoint bytes once. At startup, a size/mtime
inventory checks that those **immutable canonical files** have not been replaced.
Do not edit checkpoint bytes while preserving their metadata; publish a new
source identity after any change. The model's original path remains the vLLM
model path, preserving tokenizer/config and existing durable-KV identity.

This first implementation intentionally stores pre-kernel state. Marlin packing
and other native post-load transforms still take time. Postprocessed artifacts
need explicit quantizer adapters and are a later extension. Online quantization
that changes storage during loading uses the native path. An architecture
allowlist and a runtime-attribute change guard restrict the prepared path;
unrecognized architectures remain available through native loading. DFlash's derived
context-KV buffers are rebuilt after restore. Models with additional custom
`load_weights` side effects need an adapter and output-parity validation.

## Usage

### First load directly from original checkpoints

The new coalesced backend reads original safetensors ranges with `O_DIRECT`,
a persistent read pool, and two reusable pinned staging tiles. A 64 MiB byte
batch normally becomes one GPU upload, then native placement receives typed
views into owning storage. Large tensors are assembled through bounded tiles;
they need not fit the host staging budget. No prepared artifact is required.

```bash
NVME_LOADER_MODE=stream NVME_STREAM_BACKEND=coalesced NVME_STREAM_DEVICE=cuda \
  vllm serve /models/example --load-format nvme
```

The library's `NVME_STREAM_BATCH_BYTES` defaults to 64 MiB; the Spark controller
selects the tested 128 MiB profile. Host tile allocations together
are capped by `NVME_STREAM_MEMORY_BYTES`; with defaults they occupy about
128 MiB (about 256 MiB in the Spark profile). `NVME_STREAM_OWNED_BYTES` charges entire backing batches, including
storage held by retained tensor views. Default output budget is 6 GiB. The
producer reads concurrently within a tile and pipelines batches with native
consumption; it does not keep arbitrary numbers of files/batches resident.
Unsupported direct-I/O filesystems or tensor alignments select native loading
before consumption. Real read errors and source changes abort activation.

The original Run:ai backend remains the default and is selectable explicitly
with `NVME_STREAM_BACKEND=runai`. The Spark controller defaults to coalesced/128 MiB:

```bash
python runtime/nvme_loader/sparkctl.py stream --stream-backend coalesced --observe-boot
```

`--observe-boot` captures host CPU, NVMe, memory-availability, and network counters
through the first validated streamed count-to-100 response. These counters do
not directly measure GPU or DRAM bandwidth utilization.

`stream` reads original safetensors shards concurrently without a source-hashing
pass, prepared rank files, or artifact export:

```bash
NVME_LOADER_MODE=stream NVME_STREAM_DEVICE=cuda \
  vllm serve /models/example --load-format nvme
```

This mode requires `runai-model-streamer` (validated with 0.16.1). It preserves
default vLLM source selection, secondary-source prefixes, and native model
placement/postprocessing. Unsupported formats and online quantization keep
their native path. Completion order within a source follows vLLM's existing
Run:ai streaming interface. CPU and GPU streaming have exact parameter and
token/logprob parity on the unrelated Llama/GPT2 fixtures.

`NVME_STREAM_CONCURRENCY` defaults to 32. The SDK's shared CPU read buffer is
bounded by `NVME_STREAM_MEMORY_BYTES` (default 2 GiB); each original tensor must
fit. `NVME_STREAM_OWNED_BYTES` defaults to 6 GiB and bounds live owning output
storage, including aliases and tensors retained by native loaders. These are
logical allocation budgets, not a cap on allocator caches, native temporary
tensors, or total process memory. The DFlash draft retains its entire 4.92 GB
source, which is why its owning budget exceeds the read buffer.

`NVME_STREAM_DEVICE` is `cpu` by default; `cuda` stages every tensor on the GPU,
and `pinned` returns owning pinned CPU tensors. GPU mode allows native tensor
transposes to run on the GPU. A background producer prepares one tensor ahead;
the native consumer remains on its original thread. GPU uploads complete before
source buffers are reused, and outputs record the consumer's CUDA stream.
Empty tensors omitted by the SDK are synthesized; missing nonempty tensors,
schema changes, or source replacement abort the fresh worker.

Streaming receipts distinguish header-based source consistency from verified
payload content. This mode does not cryptographically hash all checkpoint bytes.
Generic durable-KV deployments must isolate unknown sources until their identity
is established; a header digest is not a durable-KV content identity. The Spark
controller uses a separate 30 GB/rank streaming KV store and a fresh activation
salt unless optional, previously hashed immutable source inventories still
match. It never writes into the canonical model directories.

```bash
python runtime/nvme_loader/build.py
python runtime/nvme_loader/check_stream_models.py
python runtime/nvme_loader/sparkctl.py stream
```

The GPU fixture harness stops the retained stack, automatically restores it on
failure, and leaves workers stopped on success for the full activation above.
The image defaults to `spark-vllm:0.29.0-nvme4`; `NVME_IMAGE` overrides it.
See [dynamic ingestion design](../../docs/DYNAMIC-CHECKPOINT-INGESTION.md).

### Prepared artifact loading

Install the wheel in an existing compatible vLLM environment, or build the Spark
derivative image with `python runtime/nvme_loader/build.py`. The derivative keeps
the original engine and dependencies unchanged.

Prepare a content identity outside the canonical model directory:

```bash
python -m spark_nvme.identity /models/example /nvme-artifacts/sources/example.json
```

Use these environment variables with ordinary `vllm serve` model arguments:

```bash
NVME_ARTIFACT_ROOT=/nvme-artifacts \
NVME_RUNTIME_ID=<pinned-engine-and-overlay-identity> \
NVME_LOADER_MODE=prepare vllm serve /models/example --load-format nvme
```

Restart with `NVME_LOADER_MODE=restore` to require the prepared path. `auto`
allows native fallback before writes begin; failures after writes begin abort
activation and quarantine the artifact for a clean native retry on restart.
`native` is an explicit bypass. `NVME_READ_DEPTH` changes
the bounded queue depth (1–128). `NVME_DIRECT=0` selects buffered I/O for filesystems
without direct-I/O support. A startup receipt reports bytes, verified transfer
time/rate, content identity and per-rank artifact identity. The CPU distributed
group compares common activation identities before post-load initialization.

An unrelated model needs its own canonical directory; artifact restore also
needs that model's own prepared artifacts. Streaming has no such prerequisite.
Restart its workers with its ordinary vLLM configuration. Durable KV is specific
to a model and compatible runtime; it cannot make the new model serve before its
weights are ready. This package does not provide in-process architecture swaps.
In-place `reload_weights` is rejected before any destination is written. DP
placement beyond DP1, separate `model_weights` and explicit quantization
overrides require additional adapters; `auto` can use native loading.

## Spark rollout and validation

The test controller saves exact original containers and checks real generation:

```bash
python runtime/nvme_loader/build.py
python runtime/nvme_loader/prepare.py
python runtime/nvme_loader/sparkctl.py snapshot
python runtime/nvme_loader/sparkctl.py stop
python runtime/nvme_loader/check_models.py
python runtime/nvme_loader/sparkctl.py prepare
python runtime/nvme_loader/sparkctl.py restore
# Restore the retained original images, commands, mounts and environment:
python runtime/nvme_loader/sparkctl.py rollback
```

The controller is for this cluster's existing GLM profile; the loader itself uses
ordinary vLLM configuration. It persists the vLLM, TorchInductor and FlashInfer
caches and the CUDA driver cache (4 GiB cap), retains the existing Triton cache
and durable-KV mounts, monitors memory, and rolls back
on a failed full-model activation. Small-model checks require the fixtures made
by `make_fixtures.py` to be staged under `/var/tmp/nvme-loader/fixtures` on rank 0.
Never run these GPU checks alongside the full serving model on a 128 GB Spark.
The launcher uses a 90% initial memory admission fraction while retaining the
explicit 6 GB KV allocation. Shutdown grace defaults to two seconds and can be
set with `NVME_STOP_SECONDS`. Automatic recovery selects the most recently
tested containers; the explicit `rollback` command selects the original ones.

CPU correctness tests:

```bash
PYTHONPATH=runtime/nvme_loader .venv/bin/python -m pytest tests/test_nvme_loader.py -q
```

Original-shard tests additionally require `runai-model-streamer==0.16.1` and
`safetensors`; run `tests/test_nvme_streaming.py` alongside the artifact tests.

Functional results and actual full-model startup times are recorded under
`results/nvme-loader/`. No storage benchmark is required for deployment. The
30–60 second objective is an end-to-end target, not a claim established by the
small CUDA transport check.
