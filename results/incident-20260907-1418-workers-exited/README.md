# 2026-09-07 14:18 PDT: all four workers exited (Triton JIT during inference on rank 0)

Timeline (PDT):
- 13:59:30 a request arrives (watcher: min_idle 0). Adapters on, plugins active; no spin-down had happened (last cycle 10:52-10:54, the deployment test).
- 13:59:45 rank 0's worker raises `RuntimeError: Triton Error [CUDA]: operation not permitted` in `triton/compiler/compiler.py:_init_handles` -> `driver.active.utils.load_binary` while JIT-compiling and loading a Triton kernel during inference (`rank0-traceback.txt`). Ranks 1-3 are in `_ALLGATHER_BASE` (8,093,696 elements: a prefill-sized DCP all-gather, PG 3) waiting for rank 0.
- 14:09:45 PyTorch's ProcessGroupNCCL watchdog times out the collective (600 s) on rank 3 first, then all ranks; flight-recorder dump.
- 14:17:45 the watchdog terminates the worker processes; 14:18 the executors shut down "gracefully" (containers on ranks 1-3 exit 0; rank 0's API server keeps running and `/health` stays 200).
- 14:18:10 the watcher sees all four plugin endpoints unreachable and holds (correct: nothing powered off, nothing to wake).

Not the hot-plug plugin and not the idle logic: no prepare/commit/resume ran between 10:54 and the hang, the adapters were up, and the plugin logs are clean. This is the known Triton-JIT-during-inference wedge (memory note "GLM JIT hang risk"): a request with a new shape compiles a kernel at inference time; here the CUDA module load failed on one rank, which is fatal for a TP ring.

Watcher observation: between 13:59:30 and 14:18:10 it logged nothing because its status line did not change (the NCCL proxy kept retrying, so `idle=` stayed 0). A hung stack therefore looks like "busy" to the watcher; only the process exit made it visible. Recovery: relaunch (`dcp2-hotplug-3`).

## Root cause (Codex review `results/codex-review-triton-crash.md`, upstream vllm-project/vllm#52877)

Two levels.

1. **Platform.** On DGX Spark (GB10, driver 580.159.03) a CUDA module load
   performed hours into a long-running process can fail with
   `CUDA_ERROR_NOT_PERMITTED` (800). Upstream issue #52877 shows the identical
   Triton `load_binary` failure on three unrelated stacks (Nemotron, Qwen3.8,
   ours) after 1.5 to 3 days of uptime, with no Xid or NVRM message; the exact
   CUDA call that fails and why is still unproven (Triton's loader does not
   name it). Both of our crashes had the nodes deep in swap, the strongest
   correlate, but not a proven mechanism.
2. **Amplifier, ours.** The vendored sm12x indexer kernels (tonyd2wild's
   overlay, `stage/glm-triton/sm12x_mqa.py`) declared `num_q`, `seq_len_kv`
   and every stride as `tl.constexpr`, so each new prompt or context length
   produced a new binary, compiled and loaded at inference. vLLM's warmup never
   launches the indexer (its profiling path returns early), so those loads
   happened all day long, hours after start, which is exactly the exposure the
   platform bug needs. That is why we hit it in hours where others hit it in
   days, and why it started when the lanes grew memory-hungry (LMCache on the
   4th, the DCP=2 180k lane now).

## Fix (commit ee30068)

- `do_not_specialize` for the varying dimensions and strides of the three
  mqa-logits kernels: one binary each, loaded once. Structural constants
  (heads, head_dim, block sizes, next_n) stay compile-time.
- Every launch site logs uptime, MemAvailable, swap, RSS and capture state
  before re-raising, so the next failure yields the trace the review asked
  for.
- `bench/warm_kernels.py` runs in `rollout_dcp.sh` after the generation probe:
  a 12k-token prefill, two odd-length prompts, six concurrent requests. Every
  request path compiles and loads its Triton kernels (including the DFlash
  prepare and chunk-metadata kernels we do not own) while the process is
  fresh.
- Not changed: memory headroom of the lane. Still the lever if the platform
  bug turns out to be pressure-driven.

## Reproduction attempts (2026-09-07 evening)

- The persisted Triton cache on spark-06c4 holds 7,383 binaries of `_fp8_mqa_logits_kernel`
  (143 of the rowwise decode kernel): the amplifier, quantified.
- `bench/repro/cuda_module_load_stress.py`: 60,000 module loads in one process, no failure.
  Module count alone is not it.
- `bench/repro/indexer_kernel_specialization_stress.py` (the real kernel, real
  specializations) needs a node without the serving stack; scheduled for the next window.
- The fix does not change results (count100 hash identical) and, contrary to my first
  reading, does not change decode speed either: the broad version (all strides runtime) and
  the narrowed one (per-request arguments only, 487b91f, lane `dcp2-hotplug-5`) both measure
  181 ms/cycle, and so did the plugin lane before any kernel change (this afternoon's count100
  on `dcp2-hotplug-1`: 200 tokens in 4.9 s). The 181 vs 144 ms gap is between the plugin lane
  and the builtin-backend lane, under investigation (`dcp2-builtin-check` relaunch, then a
  profiler trace of the plugin lane). GPU clocks are locked at 1995 MHz on all nodes.
