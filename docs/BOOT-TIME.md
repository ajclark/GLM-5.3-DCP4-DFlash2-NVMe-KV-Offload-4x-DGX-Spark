# Serving-stack boot time: anatomy and how to cut it

Investigation by a Claude sub-agent, 2026-09-05 19:10, read-only on the cluster (one direct-read benchmark of a single shard on spark-365c). Nothing below has been tried yet; the first experiment is at the end.

## Boot anatomy (rank 0 spark-06c4, `results/rollout-dcp2-dflash-180k-prod2-20260905-182340`)

`docker run` 18:24:21 → "Starting vLLM server" 18:32:46 = **505 s** (rollout.log says 520 s because `wait_healthy` polls every 20 s, `rollout_dcp.sh:83`).

| Phase | Seconds | Evidence |
|---|---|---|
| Python/engine-core/worker spawn, CUDA + NCCL init (4-rank comm init itself is 0.40 s) | 0-50 | first worker line `Loading model from scratch` 18:25:11 (`up-spark-06c4.log:1493`); 09-03 full log: API args at +9 s, EngineCore init at +28 s |
| Param allocation (96 GB) | 50-54 | mem.log avail 116 GB → 15.5 GB by 18:25:17 |
| **282-shard safetensors loop** | **54-337 (283 s)** | `Loading weights took 283.11 seconds` 18:29:58; rank 1 278.40 s; ~1.0 s/shard |
| Marlin repack + MLA dequant, draft load (10.6 s for 4.6 GB = 0.45 GB/s), draft repack | 337-366 | `Model loading took 96.32 GiB and 316.76 s` 18:30:27 |
| torch.compile (Dynamo 15.2 s + inductor 22.1 s + dflash head 6.3 s + selector 1 s + warmup 8.5 s) | 366-432 | `up-spark-365c.log:880-888`; engine: `compilation: 55.01 s` |
| KV alloc 5.59 GiB, slab open | 432-434 | 18:31:33-34 |
| **FlashInfer autotune dummy run** | **434-471 (37.4 s)** | `Autotuning process starts` 18:31:34.667 → `ends` 18:32:12.018 (`up-spark-365c.log:3508,3996`) |
| CUDA graph capture (24 piecewise + 12 full + 12 dflash), 1.42 GiB | 471-485 | `Graph capturing finished in 14 secs` |
| Engine done 18:32:27 → scheduler slab index (5 s) → tokenizer/chat template → server | 485-505 | `init engine took 119.23 s`; `Starting vLLM server` 18:32:46 |

First request (18:33:12) still JITs `_fp8_mqa_logits_kernel` (~1 s).

**Bottleneck: the kernel buffered/mmap read path, not the NVMe and not obviously the CPU.** On spark-365c, one 1.4 GB shard (`model-00002-of-00282.safetensors`): `dd iflag=direct` **10.3 GB/s**; Python mmap cold sequential **0.53 GB/s**, strided (192 B of every 768 B, the row-parallel int4 pattern) 0.68 GB/s, `read()` 0.8-0.9 GB/s, mmap *warm* only 1.12 GB/s; buffered scales to 2.2 GB/s at 4 threads, 4.6 at 8; O_DIRECT single-thread `preadv` 9.7 GB/s. safetensors 0.8 `get_tensor` is zero-copy over mmap, so the weight_loader's `narrow().copy_()` faults pages: w2 experts are dim-1-sharded so every rank faults every page (~170 GB/rank) ÷ 0.6 GB/s ≈ 283 s. Second-order term: **177,569 tensors** (58,992 `weight_packed` + scales + shapes) → up to 1.6 ms/tensor of Python is hidden inside that 283 s; its true share only becomes visible once I/O is fixed. Memory side effect today: page-cache growth pushed rank 0 into swap during the load (pswpout +611k pages = 2.3 GB between 18:25:41 and 18:26:06) despite `cache_flusher.sh`. Note the 4K page size, `read_ahead_kb=128`, kernel 6.17-nvidia; `nvidia_fs` not loaded (no GDS).

The fastsafetensors backup diff is empty because `.bak-fastsafetensors-20260902-191502` is the pre-experiment snapshot; the failure is documented in `spark-cluster-experiments/PLAN-40-TOKS.md:520-533` (71 shards staged in pinned memory + broadcast; took 06c4 down).

## Ranked options

1. **Persist the torch.compile/AOT cache — flags only.** `VLLM_CACHE_ROOT` defaults to `~/.cache/vllm` inside the container and dies with it (`envs.py:34`). AOT compile is on (torch ≥ 2.10, `envs.py:325-335`; log: `saved AOT compiled function to /root/.cache/vllm/torch_compile_cache/torch_aot_compile/<hash>/rank_1_0/model`). A hit returns before Dynamo runs (`compilation/decorators.py:565-575`) and validates traced source files (`_verify_source_unchanged`), so overlay edits invalidate correctly; the Inductor cache lives under the same dir. Key = `vllm_config.compute_hash()` (includes max_model_len, DCP size, kv-transfer config), so each lane pays once. Saves ~55-60 s. Needs: `-e VLLM_CACHE_ROOT=/cache/huggingface/.vllmcache` (already-mounted NVMe, root-written like `.tritoncache`). Risk: none for memory; ~1 GB disk per config. Validate: log shows `Directly load AOT compilation from path` and `init engine ... (compilation: ~0 s)`.

2. **`--load-format instanttensor` — flags (+ optional 1-line overlay).** Package `instanttensor 0.1.9` is in the image (with `cufile`), wired in `default_loader.py:272` / `model_loader/__init__.py:56`. It reads with io_uring/AIO **direct I/O** into a bounded GPU ring buffer and yields full tensors on-device; vLLM slices as today. Expected: I/O 377 GB/node at ~9 GB/s ≈ 40 s; weight phase 283 → ~60-120 s depending on the Python per-tensor share; draft 10.6 → ~1 s. Peak-memory calc: default buffer = 8 MB × concurrency 1 × io_depth 128 × world 4 = 4 GiB; cap it: `INSTANTTENSOR_BUFFER_SIZE=2147483648`, `INSTANTTENSOR_MAX_FREE_MEM_USAGE=0.25` (fraction of `cuda.mem_get_info` free, MIN-reduced across ranks), `INSTANTTENSOR_IO_DEPTH=32`. Peak ≈ 2 GiB buffer + ≤1.9 GiB tensor clone (embed 154,880×6144 bf16) + NCCL scratch ≈ 4.5 GiB of the ~15 GiB free at that point — and **zero page-cache growth**, i.e. less pressure than today's path that already swaps. Docker 29.2.1's builtin seccomp blocks io_uring: set `INSTANTTENSOR_BACKEND=AIO` explicitly (the library probes `backend_available` and falls back, but be explicit). Caveat: `weight_utils.py:1113` passes the world `device_group`, so each rank reads 1/4 and `ncclAllGather`s the rest — NCCL in the load path is what bit fastsafetensors. Prefer a one-line overlay (`process_group=None`) for local-only reads; add distributed mode later. Risk: a weight_loader that assumes CPU tensors → boot failure, not OOM; rollout auto-restore covers it. Validate: `Loading weights took`, `INSTANTTENSOR_DEBUG=1` throughput line, mem.log avail ≥ 8 GB on 06c4 throughout.

3. **Kill the 37 s autotune window.** The launcher's `VLLM_DISABLE_FLASHINFER_AUTOTUNE=1` is dead: not referenced in `envs.py`; the knob is `kernel_config.enable_flashinfer_autotune`, True at the O2 default (`config/vllm.py:235,364`), and the 09-03 config dump confirms `enable_flashinfer_autotune=True`. `kernel_warmup.py:73-80,137-143` runs a 2048-token dummy run under `fi_utils.autotune()`. This stack's hot kernels are Marlin MoE, Triton sparse MLA and FLASH_ATTN, so FlashInfer tactics are unlikely to matter. Add `--kernel-config '{"enable_flashinfer_autotune": false}'`. If the window stays ~37 s, it is Triton's in-process autotune inside the dummy run (not persisted across processes) — then the fix is pinning configs in the sm12x overlays. The persistent FlashInfer cache exists but is compiled out (`_FLASHINFER_USE_PERSISTENT_CACHE = False`, `kernel_warmup.py:118`) — 1-line overlay alternative. Risk: throughput regression only; validate with the usual post-boot decode/prefill check.

4. **Pre-sharded, post-processed per-rank checkpoint (the real "instant loader").** 95 GB/rank contiguous, ~2.4k fused tensors, no TP slicing, no marlin repack, read with O_DIRECT → ~15-25 s. Not flags-only: `sharded_state` and the `save_sharded_state` RPC exist (`sharded_state_loader.py`, `gpu_worker.py:1022`), but marlin's `process_weights_after_loading` **replaces parameters with different shapes** (`compressed_tensors_moe_wna16_marlin.py:379-428`) and `base_loader.py:64-80` re-runs it after `load_weights`; MLA `W_UK_T/W_UV` are plain attributes outside `state_dict`. Needs a ~300-line loader plugin (`register_model_loader`, `model_loader/__init__.py:69`): init → PWAL on empty weights → O_DIRECT-read the rank file into params → re-run only the MLA PWAL; plus a loader-side chunked save on a dedicated boot with KVBYTES lowered. Disk 4×95 GB (1.2 TB free). Validate with the byte-identical count100 hash check. Do after #2 reveals the Python residue.

5. **Weights resident across restarts.** Sleep mode is out: level 1 needs a 95 GB CPU copy (2× on UMA), level 2 reloads from disk, neither survives `docker rm`. A holder process exporting post-PWAL tensors via CUDA IPC (torch.multiprocessing reductions) with a meta-device loader adopting them would make the load a ~seconds remap at zero extra memory, but it is gated on CUDA IPC support on GB10 (a CUDA-context probe — only in a maintenance window), a daemon that must never die, and ~500 lines. Floor with everything: ~50 s pre-worker + AOT load + 14 s capture (never cacheable) + ~20 s server tail ≈ 100-130 s. "Well under a minute" is not reachable while the process restarts.

6. **Minor:** warm `_fp8_mqa_logits_kernel` in warmup (1 s at first request); rollout poll 20 → 5 s (reported time only); keep the pre-launch cache drop (admission check has 1.5 GiB margin: 112.27 GiB free vs 0.91×121.7) but with direct I/O the 60 s periodic drops during load become unnecessary.

7. **Not recommended:** fastsafetensors (ruled out), `enable_multithread_load` (`load_file` materializes whole shards: up to 4×5 GB on a 15 GB budget, still the 0.9 GB/s path), `--safetensors-load-strategy prefetch/eager` (page-cache path, each node prefetches only `files[rank::4]`, fights the flusher), `runai_streamer` (installed, but buffered).

## First experiment (one boot via `rollout_dcp.sh`, standard config)

In `launch-glm53big-dcp.sh`, add env flags:
```
-e VLLM_CACHE_ROOT=/cache/huggingface/.vllmcache \
-e INSTANTTENSOR_BACKEND=AIO -e INSTANTTENSOR_BUFFER_SIZE=2147483648 \
-e INSTANTTENSOR_MAX_FREE_MEM_USAGE=0.25 -e INSTANTTENSOR_IO_DEPTH=32 -e INSTANTTENSOR_DEBUG=1 \
```
and to `vllm serve`: `--load-format instanttensor --kernel-config '{"enable_flashinfer_autotune": false}'`. Optionally mount an overlay of `model_executor/model_loader/weight_utils.py` with `process_group=None` in `instanttensor_weights_iterator` (local reads, no NCCL). Expect ~300-350 s on this boot, ~240-290 s on the next same-config boot (compile hit); #4 then targets ~150 s. Check `Loading safetensors using InstantTensor loader`, `Loading weights took`, `Skipping FlashInfer autotune`, min avail on 06c4 in mem.log, and a real generation plus the hash comparison before adopting.
