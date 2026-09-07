# 2026-09-07 14:18 PDT: all four workers exited (Triton JIT during inference on rank 0)

Timeline (PDT):
- 13:59:30 a request arrives (watcher: min_idle 0). Adapters on, plugins active; no spin-down had happened (last cycle 10:52-10:54, the deployment test).
- 13:59:45 rank 0's worker raises `RuntimeError: Triton Error [CUDA]: operation not permitted` in `triton/compiler/compiler.py:_init_handles` -> `driver.active.utils.load_binary` while JIT-compiling and loading a Triton kernel during inference (`rank0-traceback.txt`). Ranks 1-3 are in `_ALLGATHER_BASE` (8,093,696 elements: a prefill-sized DCP all-gather, PG 3) waiting for rank 0.
- 14:09:45 PyTorch's ProcessGroupNCCL watchdog times out the collective (600 s) on rank 3 first, then all ranks; flight-recorder dump.
- 14:17:45 the watchdog terminates the worker processes; 14:18 the executors shut down "gracefully" (containers on ranks 1-3 exit 0; rank 0's API server keeps running and `/health` stays 200).
- 14:18:10 the watcher sees all four plugin endpoints unreachable and holds (correct: nothing powered off, nothing to wake).

Not the hot-plug plugin and not the idle logic: no prepare/commit/resume ran between 10:54 and the hang, the adapters were up, and the plugin logs are clean. This is the known Triton-JIT-during-inference wedge (memory note "GLM JIT hang risk"): a request with a new shape compiles a kernel at inference time; here the CUDA module load failed on one rank, which is fatal for a TP ring.

Watcher observation: between 13:59:30 and 14:18:10 it logged nothing because its status line did not change (the NCCL proxy kept retrying, so `idle=` stayed 0). A hung stack therefore looks like "busy" to the watcher; only the process exit made it visible. Recovery: relaunch (`dcp2-hotplug-3`).
