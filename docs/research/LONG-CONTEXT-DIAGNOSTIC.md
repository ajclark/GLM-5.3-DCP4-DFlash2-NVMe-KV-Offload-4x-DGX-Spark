# Long-context DFlash2 diagnostic: a replicated-cache table sizing defect

**2026-09-09; local source audit and CPU experiments.** The pinned V2 worker
sizes the replicated DFlash2 group's block table as if DCP shards its tokens.
For the deployed 180224-token limit, DCP2 and 64-token blocks, it allocates
1408 logical columns where the replicated group needs 2816. The DFlash input
kernel then silently clamps absolute positions beyond 90111 to the final
column. This is a concrete runtime geometry defect and a strong explanation
for good acceptance at 32k and collapsed acceptance at 100k/170k.

The CPU reproduction establishes the defect, **not the amount of Spark
performance recovery**. No Spark inference, SSH, deployment change, or runtime
overlay edit was performed by this diagnostic task. The parent task owns the
runtime repair and subsequent guarded validation. This document describes the
captured pre-repair sources; the working overlay may subsequently be repaired.

## Evidence connecting the symptom to the defect

The existing R4 fixed-seven request records show the following. Fractions are
first-position accepted-token Prometheus deltas divided by draft-cycle deltas.
They include request-boundary accounting and are deliberately distinguished
from the filtered eligible cycles used to fit controller priors.

| Prompt tokens | Task | First token accepted / draft cycles | Second accepted | Decode tok/s |
|---:|---|---:|---:|---:|
| 4097 | Coding | 69/95 (72.6%) | 41 | 16.93 |
| 4090 | Prose | 68/93 (73.1%) | 39 | 17.26 |
| 32142 | Coding | 65/78 (83.3%) | 46 | 20.42 |
| 32135 | Prose | 68/99 (68.7%) | 39 | 16.12 |
| 100164 | Coding | 3/252 (1.19%) | 0 | 6.44 |
| 100157 | Prose | 1/254 (0.39%) | 0 | 6.53 |
| 170132 | Coding | 9/245 (3.67%) | 1 | 6.49 |
| 170125 | Prose | 1/254 (0.39%) | 0 | 6.31 |

These are one-repeat development runs, not independent held-out estimates.
The [measurement README](../../results/adaptive-spec/README.md) also records
zero accepted proposals in an **unchanged-runtime** 100055-token counting
control: 455 draft tokens and 7.05 tok/s. That control predates adaptive
verification changes. The 4k and 32k repository prefixes were already cached,
so their TTFT cannot serve as cold-prefill comparisons.

The source chain is specific:

1. The captured [V2 model runner](../../baseline/vllm/v1/worker/gpu/model_runner.py),
   `initialize_kv_cache`, computes every group's maximum columns from
   `ceil(max_model_len / (spec.block_size * self.dcp_size))`, with alignment.
   It does not use the per-group CP ownership helper at this point.
2. Captured [V2 BlockTables](../../baseline/vllm/v1/worker/gpu/block_table.py)
   allocates exactly those supplied widths. Its optional conversion from
   allocation blocks to kernel blocks does not repair the missing replicated
   CP factor. Concrete live block/kernel sizes should be logged at repaired
   startup; the numerical reproduction uses 64/64, the expected draft layout.
3. The [existing per-group rule](../../overlay/vllm/v1/kv_cache_interface.py),
   `cp_world_size_for_kv_cache_spec`, returns 1 for sliding-window groups.
   The coordinator and [FlashAttention builder/implementation](../../overlay/vllm/v1/attention/backends/flash_attn.py)
   already honor that rule. The drafter's KV must contain every position on
   each rank's head shard. The legacy worker had a per-group table fix;
   the active V2 worker did not inherit it.
4. Captured [DFlash input preparation](../../tests/fixtures/spec_runtime/dflash_speculator.py)
   loads the shared draft table with `absolute_position // block_size`, then
   clamps the column to `block_table_stride - 1`. At the reduced width, the
   last legal logical block is 1407 and the first overflow position is 90112.
   An eight-query proposal begins crossing that boundary before the committed
   prefix itself reaches 90112.
5. The captured [staged writer](../../baseline/vllm/v1/worker/gpu/buffer_utils.py)
   validates nonnegative row/start values but does not check that the written
   logical list fits the row. Its kernel stores at `row * stride + start + i`.
   The captured gather kernel likewise copies `num_blocks` without checking
   that this count fits the allocated row. Thus the defect is not limited to
   harmless truncation: an oversized logical list can overwrite another
   request's table row. The CPU test demonstrates precisely that case inside
   a safely sized CPU allocation; it does not assert that a particular live
   allocation was corrupted in the historical runs.

A sliding window reduces which physical pages must remain live. It does not
make an absolute-indexed block table a ring with half as many logical columns.
Freed/null earlier columns still occupy logical positions in this layout.

## Concrete sandbox results

[tests/test_spec_long_context_diagnostic.py](../../tests/test_spec_long_context_diagnostic.py)
passes **170 tests** in the existing CPU/Triton interpreter
environment. The tests exercise the actual pinned DFlash kernel, not a
substitute mathematical implementation.

- An AST extraction executes the committed original V2 sizing loop with a
  sliding-window spec and obtains 1408 columns. The existing per-group helper
  requires CP=1 and 2816 columns for that spec. The same extraction against
  the parent's repaired overlay obtains 2816 columns and feeds that result
  into the real DFlash kernel successfully at 100055.
- At positions 90108, 100055 and 170125, the real DFlash kernel silently maps
  overflowing query positions into the final physical block of a 1408-column
  table. Keeping the same positions, request state and trained eight-query
  width while supplying 2816 columns removes every address mismatch.
- The captured staged-write kernel receives 1564 entries for row 0 of a
  two-row, 1408-column CPU table. It writes 156 entries into row 1. The entire
  write stays inside the CPU allocation; this is a safe reproduction of
  cross-row corruption rather than an out-of-allocation experiment.
- All acceptance lengths for caps 1/3/5/7 pass at absolute positions around
  2048, 32768, 65536, 90112, 100k, 131072, 170k and the configured end boundary.
  The correct accepted-prefix anchor, query positions, context positions,
  slots, sampling positions, padding and bonus token survive rollback.
- A 257-token context chunk plus a second request exercises multiple Triton
  programs, permuted request-state indices and the no-sample prefill branch.
- Synthetic physical slot addresses above signed int32 remain correct: the
  DFlash kernel casts block IDs to int64 before multiplication. No corresponding
  KV pages are allocated.

All 170 tests use committed baseline/overlay sources, the source fixture and
the standalone diagnostic. The original staged writer is pinned with the
baseline sources so its cross-row reproduction also works in a fresh checkout. The baseline runner, baseline BlockTables and DFlash fixture
SHA-256 hashes match their original Spark captures exactly. Reproduction:

```sh
.venv/bin/pytest -q tests/test_spec_long_context_diagnostic.py
.venv/bin/python bench/spec_long_context_diagnostic.py
```

The second command produces the table's raw counters and source filenames as
JSON without exposing prompt or response contents. It reads locally restored
R4 request records. If those records are absent, restore the published
measurement archive as documented in the measurement README; an empty record
list is not a new benchmark.

## Hypotheses narrowed by the audit

| Hypothesis | Evidence and present conclusion |
|---|---|
| Wrong CP ownership/table width | **Reproduced.** Correct before 90k for this configuration, invalid above it; fix first. The exact recovery still needs Spark measurement. |
| Signed 16-bit/65536 position wrap | Rejected in the exercised input/slot kernels. Positions are int64, counters int32, and the boundary tests pass. This does not certify every attention or RoPE kernel. |
| Rejection/rollback shifts the next anchor | Rejected for the tested DFlash input kernel across every accepted length and cap at long positions. Async scheduler state and GPU execution still need their separate integration checks. |
| Ring rotation or 2048-token wrap in input preparation | No such rotation exists in the captured DFlash preparation path: it uses absolute block indices. Interpreting the undersized table as a deliberate ring is inconsistent with that code. Attention's window masking is a separate concern. |
| Generic prefix-cache failure alone | Weaker explanation: 32k cached requests accept well, and long-context failures occur in cold/warm checks and the unchanged-runtime control. Cache reuse may amplify bad table state; repaired cold/warm/disk checks remain required. |
| DFlash2 grouped convolution uses global position incorrectly | The captured convolution uses offsets within the contiguous eight-token query block, not the absolute sequence position. Preserve that trained block. This audit found no long-position threshold in this operation. |
| Checkpoint context limit or RoPE generalization | Not established as the cause. The actual config supports 1048576 positions, default RoPE theta 1000000, six noncausal sliding-window layers of width 2048. Fix address corruption before evaluating model generalization. |
| Quantized target hidden features or fused context-KV math | Not ruled out by CPU geometry. If poor acceptance remains with verified addresses, compare bounded context-KV/reference calculations and candidate coverage at matched positions. |

The actual [DFlash2 config](../../results/adaptive-next/runtime-inventory/draft-config.json)
uses target hidden-layer IDs `[5, 19, 33, 47, 61, 75]`, selector top-k 16,
selector rank 256, convolution taps 2 and block size 8. Captured
[qwen3_dflash2.py](../../results/adaptive-next/runtime-inventory/model_executor/models/qwen3_dflash2.py)
inherits attention and context-KV precomputation from
[qwen3_dflash.py](../../results/adaptive-next/runtime-inventory/model_executor/models/qwen3_dflash.py).
That precomputation repeats **absolute** context positions across six layers
for RoPE and writes them through the supplied slot map. It does not repair
a bad logical table. The model config's advertised positional capacity is
therefore insufficient evidence that the runtime accesses that capacity safely.

## Repair contract and bounded Spark confirmation

The parent task is implementing the repair separately. It should use one
per-group CP rule for table sizing, generic slot mapping and attention metadata;
preserve the target's DCP sharding; preserve the replicated drafter's complete
positions; and reject invalid host staging ranges before GPU writes. A gather
bound is defense in depth, not a replacement for rejecting invalid allocations.
For differing allocation/kernel block sizes, verify expansion and use the
matching units at each table lookup. Preserve seven draft proposals, eight
query tokens, fixed-width proposal buffers and exact greedy verifier rules.

For one 64-token draft group and max-seqs 12, doubling the two int32 GPU table
copies from 1408 to 2816 columns adds **135168 bytes (132 KiB) per rank**.
This is table metadata, not a second drafter or a larger KV pool. Actual extra
buffers and captured-graph allocations still need measurement. A startup-only
geometry fix needs new table addresses and graph capture; do not resize live
captured buffers in place.

Before any further long inference on Sparks, require the repaired sandbox
tests plus a read-only startup audit of every rank: group type, CP factor,
allocation block size, kernel block size, table shape/stride, maximum valid
logical length and active draft group ID. Add capture of the active draft
attention backend/version and its sliding-window metadata. The existing
startup log confirms the model and serving limit but does not print all of
these per-group details. Do not run a known-invalid original long-context
configuration merely to obtain a cleaner A/B reproduction.

Then run a bounded sequence on the repaired instance:

1. Observe available memory, swap deltas and PSI on all four ranks before
   each experiment. Reuse the established loaded-model pressure guard and
   measured headroom policy; do not demand an arbitrary large free-memory
   reserve or enlarge the 6 GB KV pool. Run C1 only, one request at a time.
2. Start with short fixed-seven controls, then a single 64-output-token
   counting request with actual tokenized context just below the old boundary
   (about 89k), and one above it (about 92k). Keep the prompt's final task and
   local suffix matched; record exact token counts, first-position acceptance,
   target outputs and the compact geometry snapshot. Stop if bounds, pressure
   or correctness checks fail. No original unsafe long control is needed.
3. If geometry is correct, run 100k coding and prose at fixed7, 128 output
   tokens each, cold then warm. These measure whether the acceptance collapse
   disappears. GPU prefix-cache and disk-reload states must be identified
   separately. Keep the existing prompt-only disk-storage contract.
4. Only after these pass, repeat fixed-cap calibration at 100k and optionally
   170k, then an independent adaptive comparison. Use the existing graph
   shapes and measure J/token alongside tok/s. This repair alone does not
   establish a change in loaded idle power.

Go/no-go is geometric first: every live query position is addressable by the
correct request row, staged/gathered lengths fit, no unexpected alias appears,
and greedy fixed-logit/high-margin checks pass. Acceptance recovery is the
performance hypothesis. If it remains poor, collect a small sampled candidate
set/selected-path score diagnostic to distinguish missing target token from
selector error, then investigate bounded context-window/KV reconstruction.
Do not declare a training problem solely from remaining low acceptance.

The old 100k/170k acceptance priors and K1 recommendation were learned while
this defect existed. If the repair restores draft quality, **recalibrate those
anchors and version them with the repaired runtime hash**. Historical long
adaptive gains remain measured mitigation of the old behavior; they are not
proof of the best cap with repaired cache addressing. Device energy savings
also need remeasurement because useful accepted work per cycle changes.

## Offline snapshot schema

[bench/spec_long_context_diagnostic.py](../../bench/spec_long_context_diagnostic.py)
accepts `--snapshot PATH`. Supply these integer arrays for one sampled step:
`target_positions`, `target_query_start_loc`, `num_rejected`, `block_table`
(gathered batch rows, not request-state rows), `block_size`, and optionally
`num_query_per_req` (default 8). Include optional `observed` arrays named
`context_positions`, `context_slots`, `query_positions`, `query_slots`,
`sample_positions`, or `seq_lens`; include active entries only, without graph
padding. The tool computes expected values independently and returns status 2
for geometry faults. It never launches inference or reads KV contents.

A minimal valid example is:

```json
{
  "target_positions": [60, 61, 62, 63, 64, 65, 66, 67],
  "target_query_start_loc": [0, 8],
  "num_rejected": [7],
  "block_size": 64,
  "block_table": [[10, 11]]
}
```

Capture must occur after the step's accepted/rejected counts and gathered
table are available. Keep this opt-in and sampled; do not introduce a GPU to
CPU synchronization into every decode step. Static table capacity checks can
run on the host without that dependency. The diagnostic checks metadata
addressability, not whether a table entry points to the intended live KV page;
allocator ownership and cached contents require separate validation.

Captured source identifiers used for this audit (SHA-256):

| Source | SHA-256 |
|---|---|
| Original V2 model runner | `184a43d10ed89e6c7f20fc25ea510d1849c0b863ad5de7afdba7e42049faecf4` |
| Original V2 BlockTables | `5337965e7914c4e466d5a66f086cd616f81e81e3941f61ad4518fefafc62ce69` |
| DFlash input-kernel fixture | `69f5d0bb9bb36724cfa36dc40e1ed2509d20540c69fbb9075e4bd482e8882623` |
| Loaded draft config | `1261b2f5a3a62be348fb7abdc15a2b00c1b456ce0930a6e8b97e15c5573d063f` |
| Captured qwen3_dflash.py | `f92925e47c09e032fcd15bd8d2b8cb4c9e6947fd279a8797b54f6351d93fa159` |
| Captured qwen3_dflash2.py | `c141daa4b2059c0098224ac36471c2197b7052c100bef0a4dbc2ca79b627053f` |

## Subsequent repaired-runtime Spark validation

The parent experiment implemented the repair in `8f684a4` and validated it on
all four Sparks. Each rank logs target CP2/1408 columns and draft CP1/2816
columns. The added table metadata is 132 KiB per rank. The 6 GB/rank KV pool,
198551-token reported capacity and 180224-token model limit are unchanged.
This addresses the metadata allocation, without adding a second model.

Fixed-K7 probes at 89055 and 92056 prompt tokens emitted identical 60-token
outputs, retained the marker and counted correctly. They accepted 56 of 70
drafted tokens across ten cycles in each request, at 42.77 and 43.00 tok/s.
See the [boundary report](../../results/adaptive-next/cache-width-r1/boundary-report.json).
These are bounded diagnostic outputs, not a workload benchmark.

New full-cycle calibration measures caps 1/3/5/7 at short, 100k and 170k
contexts with real FULL graph dispatch. The
[repaired curve](../../results/adaptive-next/cache-width-r1/costs-repaired-curve.json)
contains actual points near 221, 100289 and 170166 tokens; intermediate
contexts are interpolation, not separately measured experiments. Historical
long-context priors and their apparent 35–42% adaptive gains remain excluded
because their baseline was affected by the sizing defect.

Two-repeat repaired-runtime comparisons use one coding and one prose prompt
per context, 256 output tokens per request:

| Context | Coding fixed7 → adaptive tok/s | Paired coding ratio | Prose fixed7 → adaptive tok/s | Paired prose ratio |
|---|---:|---:|---:|---:|
| 100k | 16.76 → 17.53 | 1.0459 | 16.92 → 18.62 | 1.1005 |
| 170k | 16.90 → 17.56 | 1.0413 | 16.26 → 17.94 | 1.1016 |

These are exploratory screens: one prompt per domain/context cannot provide
an independent prompt-level interval. Complete-function checks also uncovered
a target-output repeatability problem under both fixed K7 and requests
labelled adaptive that actually stayed at K7. The
[repeatability diagnostic](REPEATABILITY-DIAGNOSTIC.md) retains the functional
failures and target logprob evidence. Better addressability and recovered
acceptance do not constitute a clean end-to-end quality promotion.
