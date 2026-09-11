# DSA indexer: expanded block-table buffer is one column wider than the runner's block table

**2026-09-11. Engine-killing crash, reproduced once in production, root-caused, fix proposed, not yet applied.**
Written for a reader who was not present. Everything below is from the crashed
container's logs and from source comparison against `baseline/` (the pristine fork
snapshot in this repo); no stock-vLLM run was made — see [§4 caveat](#what-was-not-verified).

---

## 1. Summary

A batch containing requests with **different decode lengths** (some speculating, some
not) makes `FlashMLASparseIndexerMetadataBuilder._prepare_decode_tensors` assign a
`[N, 1408]` tensor into a `[N, 1409]` destination slice. Torch raises, the worker dies,
`EngineCore` follows it, and the API server shuts down cleanly (exit 0). The three
worker ranks stay "Up" but orphaned, so the whole stack needs a relaunch.

The two widths come from two different formulas that are *supposed* to agree:

| Who | Formula | Value here |
|---|---|---|
| Runner's block table | `cdiv(max_model_len, block_size * cp_world)` | 1408 |
| Indexer's expanded buffer | `cdiv(max_model_len, block_size * cp_world) **+ 1**` | 1409 |

The `+ 1` is upstream slack for MTP ("spec tokens can extend a request one block past
max_model_len"); nothing adds the matching column on the runner side. The gap is
therefore **always exactly one, at every context-parallel size including cp=1**.

Both formulas, and the assignment that trips over them, are **byte-identical to the
fork baseline** — this is not a defect introduced by the DCP/DFlash overlays in this
repo. Our configuration is what reaches the code path. See [§4](#4-provenance-is-this-ours).

---

## 2. The incident

| | |
|---|---|
| When | 2026-09-11 04:57:10 UTC, after 27 h uptime |
| Stack | GLM-5.3 743B, 4× DGX Spark, TP4 + DCP2, DFlash2 drafter K=7, `max_model_len` 180224, block size 64, image `vllm-glm52-b12x:dflash2-port2` (vLLM `v0.23.1rc1.dev190+gab6660699.d20260830`) |
| Symptom | `Worker_TP0_DCP0` raised → `EngineCore encountered a fatal error` → `EngineDeadError` on all in-flight streams → `Shutting down`, container `Exited (0)` |
| Not implicated | No OOM (`OOMKilled=false`, 114 GB free), no CUDA fault, no NCCL error, all IB ports `ACTIVE` on all four nodes, no host reboot |
| Recovery | `SKIP_PREFLIGHT=1 ./rollout_dcp.sh <label>` — the normal rollout refuses to start when production is unhealthy, which is exactly the recovery case |

Worker traceback (trimmed to the relevant frames):

```
vllm/v1/worker/gpu/model_runner.py:1233   execute_model
vllm/v1/worker/gpu/model_states/default.py:181   prepare_attn
vllm/v1/worker/gpu/attn_utils.py:452   build_attn_metadata
vllm/v1/attention/backends/mla/indexer.py:697   build
vllm/v1/attention/backends/mla/indexer.py:538   _prepare_decode_tensors
    self.expanded_block_table_buffer[:actual_expanded] = (...)
RuntimeError: The expanded size of the tensor (1409) must match the existing size (1408)
  at non-singleton dimension 1.  Target sizes: [21, 1409].  Tensor sizes: [21, 1408]
```

Scheduler state at the crash (from `dump_input`):

```
3 running requests, num_computed_tokens = [56940, 49176, 49224]
num_scheduled_tokens  = {req_a: 5, req_b: 8, req_c: 8}   total = 21
scheduled_spec_decode_tokens = {req_b: [-1]*7, req_c: [-1]*7}     # req_a: none
num_spec_tokens_to_schedule = 7
```

Two requests were speculating (1 + 7 = 8 tokens each); the third had just finished
prefill and was scheduled 5 tokens with no draft. That is the whole trigger:
`decode_lens = [5, 8, 8]`, i.e. `min_decode_len != max_decode_len`.

---

## 3. Root cause

### 3.1 The two sizings

Allocation — `indexer.py` (`baseline` L311-322, deployed overlay L360-372), unmodified
by this repo:

```python
max_num_blocks_per_req = cdiv(
    self.vllm_config.model_config.max_model_len,
    self.kv_cache_spec.block_size * get_total_cp_world_size(),
) + 1  # MTP spec tokens can extend a request one block past max_model_len
self.expanded_block_table_buffer = torch.zeros(
    (scheduler_config.max_num_batched_tokens, max_num_blocks_per_req),
    dtype=torch.int32, device=self.device,
)
```

The block table the runner actually builds — `gpu_model_runner.py` (`baseline` L6975,
overlay L6990):

```python
max_num_blocks_per_req = cdiv(max_model_len, block_size * cp_world_size)   # no + 1
```

Only `MambaSpec` groups get extra speculative blocks added; full-attention groups
(GLM-5.3's) never do. So:

| cp world size | runner block table | indexer buffer | gap |
|---|---:|---:|---:|
| 1 (no DCP) | 2816 | 2817 | **1** |
| 2 (this deployment) | 1408 | 1409 | **1** |
| 4 | 704 | 705 | **1** |

The mismatch is structural and independent of DCP.

### 3.2 Why it only fires sometimes

`_prepare_decode_tensors` has two branches:

* **Uniform** (`min_decode_len == max_decode_len`) — a Triton kernel
  (`_prepare_uniform_decode_kernel`) fills the buffer and is passed **both** strides
  explicitly, so the width difference is absorbed. No exception.
* **Variable** (`min != max`) — a plain torch assignment:

  ```python
  self.expanded_block_table_buffer[:actual_expanded] = (
      torch.repeat_interleave(block_table, decode_lens, dim=0, output_size=actual_expanded)
  )
  ```

  Broadcasting requires the trailing dimensions to match exactly. 1408 ≠ 1409 → raise.

A batch reaches the variable branch whenever concurrent requests carry different draft
counts: a request fresh out of prefill next to speculating ones, a drafter that emitted
fewer than K tokens, or an invalidated speculation. With a single request in flight the
uniform branch always wins, which is why this survived 27 hours of production and every
guarded experiment (all single-stream) before it.

### 3.3 Secondary observation (unverified, worth a look)

Both branches return `self.expanded_block_table_buffer[:num_decode_tokens]` — a view
**1409 columns wide** — to the caller, even though the source block table has 1408. In
`sparse_utils.py` the downstream code derives `max_num_blocks_per_req = block_table.shape[1]`
from exactly that view. On the uniform path this does not raise, so if the extra column
matters, the effect would be silent rather than fatal. I have not traced whether the
trailing column is ever read (seq-len bounds may make it unreachable). **Worth
confirming while fixing the crash**, since it is the same off-by-one.

---

## 4. Provenance: is this ours?

The overlays in this repo modify `indexer.py` heavily (30 KB → 40 KB), so "we broke it"
was the first hypothesis. Four checks, all against `baseline/vllm/...` (the pristine
fork snapshot committed in this repo):

1. **The crashing function is byte-identical to the fork.** `_prepare_decode_tensors`
   extracted by AST from both files: 5888 bytes each, equal. `git log --follow` on
   `overlay/.../indexer.py` shows a single commit (`8d35c2c`, the original import) — the
   function has never been edited here.
2. **The allocation line is byte-identical**, including the `+ 1` and its MTP comment.
3. **The runner-side width is arithmetically identical for this model.** Our overlay
   does edit that line — it asks per-KV-cache-group whether the group is token-sharded
   or replicated (`cp_world_size_for_kv_cache_spec`) instead of always using
   `get_total_cp_world_size()`. For a full-attention group it returns the same value, so
   the same 1408. The variant only diverges for *replicated* groups (sliding window
   under DCP), which this model does not have in this configuration.
4. **Our edits to this file do not feed the crash.** Of the six functions we modified
   (`split_indexer_prefill_chunks`, `get_max_prefill_buffer_size`,
   `build_prefill_chunk_metadata`, `_build_prefill_chunk_metadata_kernel`, `__init__`,
   `build`), a diff of `build()` filtered for lines mentioning `decode_lens` or
   `block_table` returns **nothing**. Our changes there concern prefill chunk splitting,
   workspace sizing, and DCP-local decode sequence lengths.

**Conclusion:** the defect is in the fork/upstream. What *is* attributable to this
deployment is reaching it: a replicated DFlash2 drafter at K=7 plus three concurrent
requests produced the mixed decode lengths. A stock deployment needs MTP plus the same
concurrency shape, which is plausibly why it has stayed latent.

### What was not verified

Source was compared; stock vLLM was **not run** to reproduce. A standalone repro that
instantiates the metadata builder with these shapes and no overlay files loaded would
settle it conclusively and is cheap (local, no cluster) — see §6.

---

## 5. Proposed fix

Defensive slicing: never assume the two widths agree, and hand back a view whose width
matches the caller's block table.

```python
# vllm/v1/attention/backends/mla/indexer.py, _prepare_decode_tensors, variable branch
# (baseline L431-440 / overlay L538-547)
     num_blocks = block_table.shape[1]
-    self.expanded_block_table_buffer[:actual_expanded] = (
+    self.expanded_block_table_buffer[:actual_expanded, :num_blocks] = (
         torch.repeat_interleave(
             block_table, decode_lens, dim=0, output_size=actual_expanded
         )
     )
     if actual_expanded < num_decode_tokens:
         self.expanded_block_table_buffer[actual_expanded:num_decode_tokens, 0] = 0
-    block_table = self.expanded_block_table_buffer[:num_decode_tokens]
+    block_table = self.expanded_block_table_buffer[:num_decode_tokens, :num_blocks]
```

Apply the same `:num_blocks` narrowing to the uniform branch's return (baseline L396 /
overlay L503) if §3.3 turns out to matter.

Properties: correct whether the buffer is wider, equal, or (defensively) narrower;
leaves the MTP slack column allocated but unused; no allocation, no shape change visible
to any consumer; no behavioural change on the uniform path.

**Alternative considered — add `+ 1` on the runner side instead.** Rejected: it widens
the block table for every backend and every model, costs memory proportional to
`max_num_reqs`, and would need matching changes wherever the width is re-derived. The
indexer created the asymmetry; the indexer should absorb it.

---

## 6. Validation plan

1. **Local repro, no cluster** — construct `block_table` `[B, W]`, `decode_lens` with
   `min != max`, and a buffer of width `W + 1`; assert the current code raises and the
   patched code does not. Then assert the returned table equals
   `repeat_interleave(block_table, decode_lens, dim=0)` row-for-row.
2. **Equivalence with the uniform path** — for a batch where all decode lengths are
   equal, the variable branch (forced) and the uniform kernel must produce identical
   `block_table`, `seq_lens` and `decode_lens`.
3. **Cluster gate** (guarded experiment, not a production rollout): boot with the patch,
   run the count smoke and confirm the token ids are byte-identical to a stock boot, then
   drive **≥3 concurrent requests with deliberately mixed draft counts** at ~50k context
   — the shape that crashed — and confirm no raise plus unchanged output.
4. Regression test alongside the existing indexer tests in `tests/`.

---

## 7. Upstreaming

This is the cleanest upstream contribution found in this project so far: a two-line
defensive fix, independent of the DCP/DFlash work, in a function this repo has never
modified, with a deterministic repro that needs no multi-node setup. It reproduces at
cp=1, so it is not conditioned on context parallelism. Suggested report title:

> DSA sparse-MLA indexer: `expanded_block_table_buffer` is allocated one block wider
> than the runner's block table, raising on batches with mixed decode lengths

---

## 8. Open questions for the reader

1. Does the uniform path's oversized return view (§3.3) ever produce a wrong read, or is
   the trailing column always masked out by sequence-length bounds?
2. Under stock MTP, is there a path that widens the runner's block table by
   `num_speculative_blocks` for non-Mamba specs — i.e. was the `+ 1` correct for some
   configuration that has since changed?
3. Should the scheduler avoid mixing speculating and non-speculating requests in one
   batch, independent of this bug? It would narrow the blast radius of any similar
   width assumption, at some throughput cost.

---

## Appendix: reproduction conditions

```
model            GLM-5.3 (DSA sparse MLA), 743B
parallelism      TP4 + DCP2 (cp_world_size = 2)
speculation      DFlash2, K = 7, replicated drafter group
max_model_len    180224        block_size 64
=> runner block table width   cdiv(180224, 64*2)     = 1408
=> indexer buffer width       cdiv(180224, 64*2) + 1 = 1409

trigger          >= 2 concurrent requests whose scheduled decode lengths differ
                 (observed: [5, 8, 8] — one post-prefill request beside two speculating)
frequency        once in 27 h of single-user production; never in single-stream benchmarks
```
