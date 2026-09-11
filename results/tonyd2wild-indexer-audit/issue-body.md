The published **80K fp8 DFlash2 lane has a reproducible indexer width mismatch, even without DCP**. Mixed scheduled lengths can raise a Torch exception; the uniform Triton path also reads beyond each logical source row. This concerns the serving metadata, not the quantized weights.

### Exact provenance

Checked GLM-5.3 revision `a1806cb82493aa6f28709f77acf59c1937bdf756`:

- [`launch-glm53-dflash2-WORKING.sh`](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/blob/a1806cb82493aa6f28709f77acf59c1937bdf756/launch/launch-glm53-dflash2-WORKING.sh) uses `vllm-glm52-b12x:dflash2-port2`, max-model-len 80000, K=7 and no DCP.
- The DFlash2 build inherits the GLM-5.2 modded base and forces the V2 runner.
- GLM-5.2's [`patches/fix-indexer-mtp-overhang.py`](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark--36tok-s/blob/162d7a766e4bc9de1809ff724339538380c43b0c/patches/fix-indexer-mtp-overhang.py) adds `+1` to the indexer workspace width.

Applying that public patch to exact upstream `ab666069935c1f23e8ef56038b4659ac9e8f19f8` reconstructs our original image's entire indexer file byte-for-byte: SHA-256 `2aa896c467f44c3a65e7d04b07ada866dfa23fb23ee8ec245e5d8c89780cef6b`. Source was extracted without our DCP overlays. This does not assume your current private image digest matches ours.

### Reproduction and scope

At block size 64 and CP1, the V2 runner allocates 1250 columns for 80000 tokens (already aligned to 128 tokens). The patched indexer allocates 1251. GB10 with K=7 enters flattening. Executing the original method with scheduled lengths `[5,8,8]` produces:

```text
The expanded size of the tensor (1251) must match the existing size (1250)
at non-singleton dimension 1. Target sizes: [21, 1251]. Tensor sizes: [21, 1250]
```

The assignment is a whole-row copy of `repeat_interleave(block_table, decode_lens)`. It fails regardless of context position; a short request/prefill tail scheduled alongside speculative decodes can supply the unequal lengths. Concurrent API requests alone do not guarantee that exact scheduling.

The uniform kernel's load mask also uses the **destination** stride. With `[8,8,8]`, it copies the next source row's first element into the extra column. The repro retains an allocated guard row to demonstrate the logical overread safely; an out-of-allocation access depends on the backing allocation. We have not established corrupted model output from this secondary defect.

This is configuration-dependent: the published 270000-token NVFP4 DFlash lane's corresponding widths both equal 4220, so that sizing does **not** reproduce the mismatch. An 80064-token control also matches. We are not claiming every MTP lane fails.

Model-free repro, in an environment with PyTorch and Triton:

```bash
git clone https://github.com/ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark.git
cd GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark
git checkout df02e8e
python3 bench/repro/repro_original_glm_indexer.py \
  --source results/tonyd2wild-indexer-audit/image-indexer.py
```

The harness AST-loads the actual method/kernel, uses CPU Torch and Triton interpretation, and supplies workspaces sized from the inspected allocation code. It does not boot the original model or run its constructor. [Recorded results and source provenance](https://github.com/ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark/tree/df02e8e/results/tonyd2wild-indexer-audit).

### Repair

Our [backport](https://github.com/ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark/commit/dbf2100) copies only the source width, zeroes workspace slack/padding, and gives the uniform Triton load an explicit source-column bound with `other=0`. It preserves the contiguous workspace and rejects insufficient capacity. All four CP1 repro configurations and the uniform case pass with it.

On our four-Spark DCP deployment, the final source additionally passed 158 CUDA cases, four graph replays and compute-sanitizer with zero errors. Live mixed-branch execution was confirmed on every worker; count100 matched all 299 pre-deployment token IDs, and concurrent 50K-context generation passed. These live checks were on our derivative deployment, not your unmodified launcher. [Deployment evidence](https://github.com/ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark/blob/dbf2100/docs/INDEXER-BLOCK-TABLE-DEPLOYMENT.md).

For the durable fix, share the runner's authoritative aligned width. Removing `+1` unconditionally breaks other alignment cases, and fixing only the Torch assignment leaves the uniform read defect. Upstream [vLLM #50302](https://github.com/vllm-project/vllm/pull/50302), first included in v0.27.0, implements shared sizing; the particular `+1` above is a fork addition, absent from the exact upstream base.
