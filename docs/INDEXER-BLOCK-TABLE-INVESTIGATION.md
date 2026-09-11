# Independent investigation: indexer block-table width mismatch

2026-09-11. The mixed-length exception is independently reproduced. The original
report gets the immediate cause right, but its uniform-path analysis and proposed
fix are incomplete. The follow-up [deployment record](INDEXER-BLOCK-TABLE-DEPLOYMENT.md)
tracks the authorized backport and live validation.

**Attribution correction:** fetching the exact upstream commit `ab6660699` proves
that its `indexer.py` differs from our fork baseline by just one line: the fork
adds the `+1`. The original report's description of this slack as upstream code is
wrong. Equality to a fork baseline did not establish equality to upstream. See the
[saved diff](../results/indexer-block-table-investigation/upstream-to-fork-indexer.diff).

## Evidence and root cause

I read the saved worker log, traced both producer and consumers, and executed the
actual baseline and overlay functions separately. This does not rely on the
original report's source comparison.

The raw evidence is
[`pre-spark-06c4.log`](../results/rollout-recover20260911-20260911-051330/pre-spark-06c4.log),
lines 3822–3882. It identifies the V2 runner (`worker/gpu/model_runner.py`), the
04:57:10 failure at `indexer.py:538`, and exactly:

```text
Target sizes: [21, 1409]. Tensor sizes: [21, 1408]
num_scheduled_tokens: 5, 8, 8
two requests have seven scheduled draft placeholders; the five-token request has none
```

The scheduled request with five tokens has zero output tokens. A prefill tail is
consistent with that evidence; its exact lifecycle is not established by the dump
alone. The three lengths are a multiset: the dump's dictionary ordering is not a
proof of the metadata row ordering. The failure is invariant to that ordering.

The concrete failure chain is:

1. [V2 runner allocation](../overlay/vllm/v1/worker/gpu/model_runner.py),
   `initialize_kv_cache`, computes the group's block capacity and aligns it.
   For this full-attention group, `ceil(180224 / (64 * 2)) = 1408`; alignment to
   two columns leaves 1408 unchanged.
2. [BlockTables](../overlay/vllm/v1/worker/gpu/block_table.py) allocates that width
   and returns row slices of it. Kernel-block splitting would multiply the width;
   there is no such multiplication in the observed 1408-column table.
3. [DeepseekV32IndexerMetadataBuilder](../overlay/vllm/v1/attention/backends/mla/indexer.py)
   independently allocates `ceil(max_model_len / (block_size * CP)) + 1`, or 1409.
4. The same file's initialization enables flattening on this platform at `next_n=8`.
   The worker log independently records `use_flattening=True` at line 1939.
   The reorder threshold is eight, so all three requests enter decode preparation.
5. `_prepare_decode_tensors` selects its variable-length branch, repeats three
   source rows into 21 rows, and assigns all 1408 columns into a 1409-column slice.
   Torch rejects the incompatible shapes before this assignment can succeed.

This is inconsistent metadata capacity, not a token-position off-by-one near the
context limit. None of the observed requests was close to 180224 tokens. Long
contexts, three requests, DCP, and a real model are unnecessary to reproduce the
tensor failure. Two unequal lengths on the flattening path suffice. Native decode
does not enter this expansion path, so mixed draft counts alone are not sufficient
on every platform/configuration.

## A second bug: uniform batches read beyond the source row

The original report says two strides absorb the width mismatch. They only locate
the row starts. The actual uniform kernel uses the **destination** stride as both
its loop bound and its source load mask:

```python
src = block_table_ptr + req_id * block_table_stride
for i in tl.range(0, expanded_bt_stride, BLOCK_SIZE):
    off = i + tl.arange(0, BLOCK_SIZE)
    mask = off < expanded_bt_stride
    src_block = tl.load(src + off, mask=mask)
```

With source stride 1408 and destination stride 1409, every expanded row reads
source column 1408. For contiguous rows, this is the next row's first element.
At the final logical request it reads beyond the logical table; it can still be
inside the runner's larger backing allocation. At the end of that allocation it
would be an out-of-allocation read. A single request avoids the Torch exception,
but does **not** make this kernel correct.

The repro uses three four-column rows backed by an additional allocated guard row.
The real Triton kernel, interpreted on CPU, returns these fifth-column values:

```text
source rows: [100..103], [104..107], [108..111]
allocated guard row starts with 112
lengths: [2,2,2]
extra output column: [104,104,108,108,112,112]
```

Thus the extra column is neither unused by the copy nor reliably zero. This test
does not deliberately access unallocated CPU memory, and is not a CUDA sanitizer
result. It proves the kernel's erroneous logical reads. There is no evidence here
that this secondary bug caused the observed exception or altered generated tokens.

## Corrections to the initial analysis

| Initial claim | Independent finding |
|---|---|
| The runner formula always gives a one-column gap | True for the reported parameters, not all V2 configurations. V2 alignment and kernel-block splitting matter. |
| Uniform batches safely absorb different widths | False: source loads use the destination width. |
| `sparse_utils.py` consumes this expanded view | Not in the traced call chain. The expanded table feeds paged MQA logits; sparse attention conversion receives the original common block table. |
| Slicing also handles a narrower destination | False. A slice cannot enlarge storage; the copy still fails. Width-one sources can even broadcast silently. |
| Narrowing the returned view changes no shape/layout | False. A `[N,W]` view of `[N,W+1]` keeps row stride `W+1` and is normally noncontiguous. |
| Fork baseline equality proves a current upstream bug | It proves the defect predates these overlay changes. Current upstream needs separate inspection. |
| All previous benchmarks were single-stream | The repository README and concurrency sweep results document concurrent benchmarks. Mixed-path coverage, rather than concurrency alone, is the missing regression. |

For example, with V2, block size 64, CP2, and `max_model_len=180096`, the unaligned
width is 1407. The runner rounds to 1408, matching the indexer's 1407+1. Removing
`+1` universally would break that case. At smaller block sizes the alignment gap
can be larger and can put the runner ahead of the indexer. Such arithmetic examples
are not claims that every corresponding model/backend combination is supported.

The original 27-hour uptime is also not established by the examined log: its most
recent initialization is September 10 around 12:43–12:45, roughly 16 hours before
the exception. The recovery snapshot's ~114 GiB available on the dead leader is
post-exit memory, not proof of abundant memory immediately before the crash. The
shape error is independently sufficient to explain the worker failure.

## Consumer trace and the extra column

The expanded view is stored in `DeepseekV32IndexerDecodeMetadata`, passed by
[`sparse_attn_indexer.py`](../overlay/vllm/model_executor/layers/sparse_attn_indexer.py)
to `fp8_fp4_paged_mqa_logits`. Conversely,
[`flashmla_sparse.py`](../overlay/vllm/v1/attention/backends/mla/flashmla_sparse.py)
stores `cm.block_table_tensor` in its own attention metadata and passes that
original table to `triton_convert_req_index_to_global_index`.

Both Spark paged-logits kernels in
[`sm12x_mqa.py`](../stage/glm-triton/sm12x_mqa.py) accept explicit row/column strides.
The rowwise kernel masks table loads by each token's context length. With valid
context lengths within the runner capacity, the extra logical column is not needed
for its attention calculation. The generic kernel instead masks table loads by
the logits extent, and only masks subsequent KV loads by context length. Therefore
one cannot claim that *all* table loads are sequence-length masked. DCP's padded
logits extent can extend into the slack block. Retaining initialized slack is a
conservative choice for this pinned stack.

The scheduler also caps scheduled input positions by `max_model_len` in
[`scheduler.py`](../overlay/vllm/v1/core/sched/scheduler.py), lines 489–496. Its
lookahead allocations should not be confused with valid attention context lengths.
The MTP allocation comment alone is not evidence that a live decode may safely
index arbitrary columns beyond the runner table.

## Reproduction and checked candidate

Run from the repository root:

```bash
.venv/bin/python bench/repro/indexer_block_table_width.py --baseline-only
.venv/bin/python bench/repro/check_indexer_block_table_fix.py \
  --source overlay/vllm/v1/attention/backends/mla/indexer.py --device cpu
```

The original [standalone repro](../bench/repro/indexer_block_table_width.py) AST-extracts
the unmodified builder method and Triton kernel from **each** source tree. It
supplies CPU workspaces instead of invoking the model-dependent constructor.
It uses Torch operations and the actual Triton kernel through `TRITON_INTERPRET=1`;
it does not substitute a Python implementation of the copy kernel. The baseline
and overlay definitions compare equal independently of line numbers. Source hashes
and results are recorded in [repro.json](../results/indexer-block-table-investigation/repro.json).

Results on Torch 2.14.0+cpu / Triton 3.8.0:

- Both original trees raise the exact incident exception at CP2 and equivalent
  errors at CP1/CP4 for lengths `[5,8,8]` and widths 2816/1408/704.
- Both original kernels copy neighboring-row/guard values on uniform batches.
- A width-one source demonstrates silent broadcasting on the original mixed path.
- The proposed correction passes **162 cases per tree, 324 total**: actual incident
  widths and tiny widths, equal or wider buffers, noncontiguous source rows,
  mixed/uniform lengths, a single request, zero-length padding requests, padded
  output rows, reused poisoned buffers, native/plain bypass, and rejection of
  undersized destinations. Uniform and forced-variable outputs agree for the same
  inputs, including sequence lengths and decode lengths.

In the original investigation the candidate was patched into temporary files for execution. The supplied
[candidate.patch](../results/indexer-block-table-investigation/candidate.patch)
is the historical candidate diff. The subsequently applied implementation adds a
once-per-process mixed-branch diagnostic; see the deployment record.
These checks cover the implicated metadata routines, not full builder construction,
CUDA compilation, graph replay, model output, or a stock vLLM server.

## Suggested fix

For a small backport to the pinned Spark stack, fix **both** copying paths:

1. In the variable path, copy into `[:actual_expanded, :source_width]`, then zero
   the remaining destination columns. Zero complete padding rows on reuse.
2. Pass `source_width` separately to the uniform kernel. Mask source loads by
   `off < source_width`, use `other=0`, and retain destination-width stores.
3. Preserve the preallocated contiguous destination and its returned width. This
   avoids changing consumer layout and keeps defined slack for padded logits loads.
4. Reject an input wider than the workspace before launching either path. This is
   a diagnostic guard, **not** a solution for configurations that require a larger
   workspace. The pinned incident configuration has sufficient capacity.

This is what the checked candidate implements. It introduces no new tensor
allocation beyond the existing `repeat_interleave` temporaries, but clearing the
tail may add a Torch kernel launch on the mixed path; GPU overhead is unmeasured.
It does not prohibit legitimate mixed scheduling or reduce the draft count.

For the durable fix, make the builder use the runner's authoritative per-group,
post-alignment, post-kernel-splitting width at initialization, before CUDA graph
capture. Keep logical copy width distinct from row stride in kernels. Do not
independently rederive two supposedly equal capacities or globally add/remove one.

Upstream **v0.27.0**, published 2026-08-10, contains the shared-sizing fix
[PR #50302](https://github.com/vllm-project/vllm/pull/50302), commit
`a0cd2b69b3dac2b43be02fc16ff940b856d6791b` (merged July 31).
The GitHub compare API confirms that commit is an ancestor of the v0.27.0 tag;
[release evidence](../results/indexer-block-table-investigation/release-provenance.json)
records the comparison. This fixes a broader alignment mismatch, not the addition
of our fork's `+1`: that addition is absent even in the image's exact upstream base
and upstream v0.23.0. Do not describe v0.27.0 as the release that removed this
fork-only line.

Current upstream has the relevant sizing direction: the
[indexer constructor](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/indexer.py)
requires `block_table_width`, and
[AttentionGroup.create_metadata_builders](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/utils.py)
supplies it using the shared block-table sizing helper. These are moving `main`
sources inspected on 2026-09-11, not the incident revision and not a tested upgrade.
The old fork baseline remains demonstrably affected. The uniform kernel still
assumes matching widths; the upstream fix establishes that invariant through sizing
rather than introducing the explicit source-width load mask used by our backport.

Before deployment, run this metadata repro in the serving image on CUDA, add a
compute-sanitizer run with an exactly sized source allocation, and check graph
capture/replay. Then run an isolated integration gate with instrumented branch
coverage: two or more requests with different scheduled decode lengths, plus
uniform/speculative and context-boundary cases. Confirm actual mixed-path execution,
completion, and deterministic output against a known-good control. Concurrent API
requests alone do not guarantee the scheduler will form the triggering batch.
The initial investigation performed no production crash attempt or rollout. The
subsequently authorized deployment and its results are recorded separately.
