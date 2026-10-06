# Step 4: the host critical path before each verify (2026-10-01)

Overview: `docs/DECODE-SPEEDUPS.md`, step 4, re-scoped after step 3 (`results/step3-roce-20261001/REPORT.md`, "Notes").
Files are in
`runtime/vllm029/verify_cap_overlay/`.

Measured on an abliterated graft of GLM-5.3 with the same quantization envelope as GLM-5.3 Int4-Int8Mix (Int4
group-128 routed experts, Int8 group-128 elsewhere, identical shapes).

## Design

**The problem.** The verifycap8 log shows three host costs:
- the gloo K broadcast, 0.6–0.9 ms per decision;
- re-preparation after a mispredicted K, 0.45–0.68 ms, on 20–52% of decisions;
- a decision wait of "0.00 ms", meaning the draft is always complete when the host reaches the decision.

So the host sits between the end of the draft and the verify launch, and every host millisecond there is GPU idle time.

### (a) Rank-local decision (`VLLM_VERIFY_CAP_LOCAL_DECIDE=1`)
Every TP rank runs the unchanged float64 decision, so there is no per-step host collective.

**Inputs that are identical on every rank by construction:**
- batch, request ids, slots and context, which come from the scheduler output;
- candidate ids and unary logits, which are all-gathered;
- biases and λ, which are functions of these. λ is not timing-based.

**p_sel (the draft confidence) is not identical across ranks** (see (c)). Right after drafting, rank 0's `[n, 8]` buffer
(7 p_sel + the sampled count) is therefore broadcast in place:
- one NCCL broadcast on the TP PyNccl communicator;
- on the current stream, with no host sync.

**Control file and cost table.**
- They travel at sync points, one every `VLLM_VERIFY_CAP_SYNC_EVERY` (32) decisions.
- A sync point runs in `after_launch()`, right after a verify is queued, so it is off the critical path.
- Each sync is a gloo all_gather of [crc32, decision count, control sequence, drift counters]. A
  `broadcast_object_list` follows only when rank 0 changed something.
- Changes apply from the next decision on every rank. The initial state is handed out the same way at construction.

**Rank check.**
- The crc32 covers the decision index, n, K, context, bias and λ, plus each request's bias and p_sel.
- A mismatch logs an ERROR on every rank, and all ranks fall back together to rank-0 decide + broadcast.

Logging, periods and the `periods.json` writes move to `after_launch()`.

### (b) The rest of the gap
- **`VLLM_VERIFY_CAP_EARLY=1`** decides in `pre_trim` when the draft event is already complete (a non-blocking query).
  The step is then prepared once, with the actual K.
- **A static audit found no hidden GPU→CPU sync** between the draft and the verify launch. `VLLM_GPU_SYNC_CHECK=warn`
  in a test boot would confirm it.
- **Not done:** prepare-8-then-fixup, and extra graphs (rank-0 memory).

### (c) A drafter determinism fix found by this review (`VLLM_DFLASH_DET_CONV`, default 1)
**The cause.** The int8 drafter's conv `kernel_projection` (12 layers; per rank n=1536, k=6144) is a `ReplicatedLinear`.
Its output enters the drafter's residual stream with no all-reduce after it.
- `VLLM_MARLIN_USE_ATOMIC_ADD=1` makes Marlin use atomic-add split-K for n < 2048 and k ≥ 2048. The summation order of
  atomic-add split-K varies from run to run.
- So each TP rank carries a slightly different drafter residual. That affects p_sel, and on a near-tie the selector's
  choice of draft token too.
- Ranks drafting different tokens would verify different input ids. This cannot happen with the BF16 drafter: cuBLAS
  is deterministic and the inputs are identical.
- **It is a candidate cause of the open item from step 2**: agent-turn tokens per cycle 3.01 → 2.89–2.91 after the
  switch to the int8 drafter.

**The fix** (`qwen3_dflash2.py`): these GEMMs call `ops.marlin_gemm` with `use_atomic_add=False`, Marlin's deterministic
global reduce. That restores bit-identical drafter state on all ranks. `DFLASH_DET_CONV=0` reverts.

### Instrumentation
The existing 60 s log line gains:
- host prep, post and pre_trim→launch times;
- the draft-launch→verify-launch window;
- decide mode and early %;
- device-broadcast and sync cost;
- rank-check status;
- per-rank p_sel drift counts. These should read 0 with `DET_CONV=1`.

With `VLLM_VERIFY_CAP_GAP_EVENTS=1` it also shows, from rank-0 CUDA events, the GPU's verify end → draft end and draft
end → verify start.

### Files
- `verify_cap.py`.
- `model_runner.py`: the `before_launch` / `after_launch` hooks, 6 lines.
- `test_verify_cap.py`: a pytest entry point.
- `test_verify_cap_local.py` (new). It includes a 4-process gloo run with a noisy per-rank p_sel, early decisions at
  random, a mid-run control change and an injected drift.
- `qwen3_dflash2.py`: the det-conv fix.

### Switches
`VERIFY_CAP_LOCAL_DECIDE` 0, `VERIFY_CAP_SYNC_EVERY` 32, `VERIFY_CAP_EARLY` 0, `VERIFY_CAP_GAP_EVENTS` 0,
`DFLASH_DET_CONV` 1.

### Keep criteria
- count100 matches `LOCAL_DECIDE=0`.
- Probe outputs CLEAN.
- The rank check passes with 0 mismatches over the soak.
- The broadcast reads 0.00 ms, and re-prepared is ~0% with EARLY.
- C1 prose/code ≥ +1%, or a measurable drop in the GPU idle gap.

Expected: ~1 ms per ~100 ms cycle (+1–1.5%). For det-conv, the p_sel drift counter reads 0, and agent tokens per cycle
are compared against step 3.

## Tests
CPU (`python -m pytest test_verify_cap.py test_verify_cap_local.py` in `runtime/vllm029/verify_cap_overlay/`):
9 passed.

## Results (image verifycap10-stack; details in `../step456-integration-20261001/REPORT.md`)

**Rank-local decision.**
- **Rank check:** ok 1258/1258 sync points over the boot-B series.
- **Device broadcast:** 0.07 ms on the GPU. The sync point costs 0.14 ms on the host after launch.
- **GPU gap, draft end → verify start:**

  | `LOCAL_DECIDE` | median | max |
  |---|---:|---:|
  | 0 (boot C) | 1.34–1.47 ms | 2.9 |
  | 1 (boot B) | **0.38–0.40 ms** | 0.75 |

- **C1, paired B vs C:** prose +2.0% (13/15 faster), code +3.9% (13/15), code+think +6.9%.

**Early decision:** it never triggered (0%). Under async scheduling the host is 80–100 ms ahead, so the draft is never
done at `pre_trim`. Re-preparations (25–32%, 0.39 ms) happen off the GPU's critical path. Off by default.

**`DFLASH_DET_CONV=1`:**
- p_sel differed from rank 0's on **0 of 59,280** drafted rows per rank (counter from the device broadcast).
- Agent tokens per cycle on the paired turns: 3.357 at step 7 → 3.395 (+1.1%).
- No boot measured the drift with the fix off, because the counter only runs with local decide. The fix restores the
  BF16-era invariant that every rank drafts from identical state.

**Fixed-K cycle cost** with everything on: unchanged (boot A off arms 124.0 ms vs step 7's 124.4 ms at 1 × K=7).

## Decision
**Keep:** `VERIFY_CAP_LOCAL_DECIDE=1` (launcher default) and `DFLASH_DET_CONV=1` (default). `VERIFY_CAP_EARLY` and
`VERIFY_CAP_GAP_EVENTS` exist but are off.
