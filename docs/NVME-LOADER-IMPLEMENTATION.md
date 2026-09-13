# Concurrent NVMe loader: implementation and Spark results

The loader is implemented and running on all four Sparks. It replaces repeated
canonical checkpoint scans with concurrent reads of each worker's prepared
state. Verified cache-enabled restarts reached serving in **82–87 seconds**;
the final deployed build took **87.07 seconds**, including shutdown and
readiness polling. **The 30–60 second end-to-end target
has not yet been reached.** Weight I/O is now a small part of startup.

For first loads without prepared artifacts, the
[dynamic ingestion implementation](DYNAMIC-INGESTION-IMPLEMENTATION.md) consumes
original checkpoint shards directly:
original-shard target streaming takes 59–61 seconds, with a 131-second serving
activation. The results below describe the earlier prepared-artifact path.

## Results

| Check | Result |
|---|---|
| Target native checkpoint load during preparation, rank 0 | 358.03 s |
| Target artifact restore, rank 0, final build | 8.03 s for 101.39 GB |
| Draft artifact restore, rank 0, final build | 0.258 s for 3.08 GB |
| First full artifact restart, including shutdown | 122.65 s; generation passed |
| Restart with persistent CUDA driver cache and shorter shutdown | 81.55 s; generation passed |
| Final build with adapter registry and revised admission gate | 87.07 s; generation passed |
| Target plus draft model construction/loading/postprocessing, final build | 20.36 s |
| Engine KV allocation, warmup and graph capture, final build | 17.48 s |
| Same-model durable KV after the restart | 1,664 tokens recovered from external storage; expected output |
| Unrelated tiny Llama and GPT2 fixtures | Exact generated tokens and log probabilities match native loading |
| Local CPU tests | 11 passed, including two-process failure propagation |
| Spark CUDA transport | Byte-for-byte equality, asynchronous copies, multiple slots and a tail chunk |

These are actual loader and serving observations, not device specifications.
The supplied 4,000–5,000 MB/s planning input gives a 21–26 second transfer budget
for the combined 104.48 GB rank image. The observed transfer times on these
Sparks are faster than that estimate; they should not be promised for an
arbitrary drive with that stated bandwidth. No fio run or storage tuning sweep
was needed to build the implementation.

The one-time full preparation took 673.08 seconds to health: a native load,
writing the target and draft artifacts, and normal engine initialization.
Canonical content hashing was a separate preparation step. Subsequent loads of
a prepared model do not repeat either operation.

## What was built

The installable package is under [runtime/nvme_loader](../runtime/nvme_loader/README.md).
It registers `--load-format nvme` through vLLM's plugin entry point and uses the
installed vLLM 0.29 loader interface. The transport is independent of the model
architecture, tensor names, expert counts and quantization format.

1. **Concurrent reads within a rank file.** Thirty-two native `preadv` calls can
   be outstanding against one file. File count no longer limits concurrency.
   Four-MiB aligned extents and `O_DIRECT` avoid filling the page cache with
   another model-sized copy on the unified-memory Spark.
2. **Bounded ownership.** Approximately 128 MiB of pinned staging per worker;
   slot reuse waits for the corresponding CUDA copy completion event. All
   copies drain before postprocessing or error propagation. Every extent is
   SHA256-verified before it reaches its destination.
3. **Immutable rank-local artifacts.** The format stores physical storage,
   shapes, dtypes, strides, offsets and alias relationships, including
   nonpersistent buffers. Data and manifest are fsynced and published by
   directory rename. Export periodically flushes and discards its page-cache
   window to limit memory pressure.
4. **Explicit compatibility.** Full checkpoint content hashes are prepared on
   every node and checked for agreement. Startup verifies the immutable source
   inventory and matches model/config, precision, relevant topology, rank,
   device capability, engine/runtime identity and the constructor's exact
   tensor schema. Source identities and artifacts live outside the model
   directory, preserving the original tokenizer/config and durable-KV paths.
5. **Native kernel processing.** Artifacts are captured *before* native
   postprocessing. Restoration fills ordinary constructor allocations, then
   vLLM performs quantization finalization, kernel packing and attention
   initialization exactly once. It does not replay packing on already packed
   tensors. Model-specific loader side effects live in an explicit adapter
   registry; DFlash rebuilds its derived context-KV buffers before warmup.
6. **Coordinated activation.** Every rank contributes either a successful
   receipt or its error to the already-created CPU process group. Identity or
   error disagreement prevents activation. A failed read never exposes a
   partially loaded model to requests.
7. **Persistent caches and recoverable deployment.** The controller preserves
   vLLM, Triton, FlashInfer, TorchInductor and CUDA driver caches. CUDA's cache
   cap is 4 GiB. Nodes launch concurrently, with a configurable two-second
   shutdown grace period. Original containers remain retained; failed
   activations restore the most recently tested containers when available.

The derivative image is `spark-vllm:0.29.0-nvme1`. The existing GLM TP4/DCP2,
DFlash2 K=7, 180,224-token context, 12 sequences, 6 GB KV per rank and 150 GB
durable slab per rank are retained. No NCCL, attention-kernel or model-weight
changes were needed for this loader.

A validation restart encountered the native startup admission gate with only
150 MiB less free memory than its 91% threshold required. Automatic rollback
restored the retained fast stack and passed generation. The final launcher uses
a 90% admission fraction while retaining the explicit 6 GB KV allocation; that
change passed full activation and durable-KV verification.

## Generic model support and boundaries

The GPU checks switched between unrelated Llama and GPT2 architectures using
the same storage format and loader. The production check covers GLM's mixed
integer quantization/MoE layout and the replicated DFlash2 draft. Architecture
adapters also identify the conventional Qwen and DeepSeek loader families;
those additional variants have not all received deployment tests here.

An unrecognized architecture, online quantization, DP placement beyond DP1,
explicit quantization overrides, separate `model_weights`, changed tensor
schemas or unhandled loader side effects require native loading or an adapter.
`auto` provides native fallback before destination writes begin. After writes
begin, a failure aborts that worker generation; automatic mode quarantines the
artifact for a clean native retry on restart. `restore` is strict.

Prepared artifacts support **fresh workers**, not in-place `reload_weights` on
an already postprocessed model. Unrelated models get new workers and their
ordinary vLLM configurations. The supplied controller manages this cluster's
GLM profile; a general model-selection gateway and profile orchestrator are
separate work. Durable KV remains model-specific. The current single-profile
slab is preserved; a global multi-model KV retention/quota manager is not part
of this implementation.

Canonical checkpoint directories are treated as immutable. Startup's quick
inventory check detects ordinary replacement through file size/mtime changes;
it does not rehash hundreds of gigabytes on every switch. Any checkpoint edit
requires a newly prepared content identity.

## Remaining startup time

The CUDA cache fixed a substantial first-forward initialization cost. Fable's
review established that deleting the mixed warmup would move that cost to the
first request and omit useful initialization. Required mixed, sampling and
graph warmups remain enabled.

The cache-enabled run spent about 39 seconds before model loading began, then
about 20 seconds constructing/restoring/postprocessing the model, followed by
16–17 seconds of engine initialization and final API readiness.

A simple preloaded-fork supervisor was checked but not deployed: importing the
serving modules starts a native CUDA driver thread even while PyTorch reports
CUDA as uninitialized. The deployed loader uses fresh worker processes.

## Reproduce and operate

```bash
# Local correctness and package build
PYTHONPATH=runtime/nvme_loader .venv/bin/python -m pytest tests/test_nvme_loader.py -q
uv build --wheel --out-dir dist runtime/nvme_loader

# Build/stage the derivative image; does not stop serving
.venv/bin/python runtime/nvme_loader/build.py

# Restart this GLM profile through the prepared loader
.venv/bin/python runtime/nvme_loader/sparkctl.py restore
.venv/bin/python runtime/nvme_loader/kv_probe.py verify

# Explicitly restore the original retained containers
.venv/bin/python runtime/nvme_loader/sparkctl.py rollback
```

First preparation, fixture generation and generic `vllm serve` usage are in the
[package README](../runtime/nvme_loader/README.md). The primary evidence is under
[results/nvme-loader](../results/nvme-loader/): per-rank logs and activation JSON,
`model-checks.log`, `gpu-transport.log`, `kv-record.json`, `kv-verify.json`, and
three Fable implementation reviews. Private Docker inventories remain ignored
and mode 0600.
