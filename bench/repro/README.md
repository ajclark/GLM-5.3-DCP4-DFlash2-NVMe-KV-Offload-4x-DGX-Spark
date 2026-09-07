# Reproducers for "Triton Error [CUDA]: operation not permitted" on DGX Spark

Symptom (ours twice, upstream vllm-project/vllm#52877 on two other stacks): hours or days into
a vLLM process on GB10 / driver 580.159.03, the lazy first load of a Triton kernel binary
(`compiler.py:_init_handles -> driver.active.utils.load_binary`) fails with
`CUDA_ERROR_NOT_PERMITTED` (800). No Xid, no NVRM message. The rank dies, its peers wait in a
collective, the NCCL watchdog kills the job ten minutes later.

Amplifier on this stack: the vendored sm12x indexer kernels declared every dimension and
stride `tl.constexpr`, so every new prompt or context length was a new binary compiled and
loaded at inference; the persisted Triton cache on one node held **7,383** binaries of
`_fp8_mqa_logits_kernel` (and 143 of the rowwise decode variant) accumulated over the launches
of one week. Fixed in `stage/glm-triton/sm12x_mqa.py` (per-request arguments are runtime
values) plus `bench/warm_kernels.py` in the rollout.

## cuda_module_load_stress.py

Loads one small cubin through `cuModuleLoadData` in a loop, never unloading (Triton never
unloads), reporting host memory as it goes; on the first failure unloads everything and tries
again. Run 1 (spark-a218, serving stack resident, 2026-09-07): **60,000 loads in 0.6 s, no
failure**, about 10 KB of host memory per module. Raw module count at that scale is not the
trigger.

## indexer_kernel_specialization_stress.py

The faithful path: the pre-fix kernel file, thousands of distinct (num_q, seq_len_kv) shapes in
one process, each a compile plus a lazy first load of a new binary. Needs the node to itself:
with the serving stack resident it stops at its memory guard after 25 shapes (about 0.1 s of
compile per shape, so 7,000 shapes take a quarter of an hour). Planned for the next window
with the stack down, then again under `--pressure` (pinned host memory) to test the memory
correlation both crashes showed (MemAvailable under 1 GB and 2.5 GB, swap in use).

Both scripts run inside the serving image:
```
docker run --rm --gpus all -v $PWD/cuda_module_load_stress.py:/repro.py:ro vllm-glm52-b12x:dflash2-port2 python3 /repro.py --max 60000
docker run --rm --gpus all -e TRITON_CACHE_DIR=/tcache -v $HOME/repro-tritoncache:/tcache -v $PWD/indexer_kernel_specialization_stress.py:/repro.py:ro -v $PWD/sm12x_mqa_old.py:/old/sm12x_mqa.py:ro vllm-glm52-b12x:dflash2-port2 python3 /repro.py --kernel-file /old/sm12x_mqa.py --max 7000
```
