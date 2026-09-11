# Original GLM recipe: indexer defect confirmed

2026-09-11. The original **80,000-token DFlash2 configuration is affected without
DCP**, when mixed scheduled lengths enter the flattening path. This is established
from its published patch chain and execution of the actual image source, not by
assuming our DCP overlay behaves like the original.

## Provenance

- GLM-5.3 recipe at `a1806cb82493aa6f28709f77acf59c1937bdf756`:
  [working DFlash launcher](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/blob/a1806cb82493aa6f28709f77acf59c1937bdf756/launch/launch-glm53-dflash2-WORKING.sh)
  uses `vllm-glm52-b12x:dflash2-port2`, 80000 context, K=7, no DCP.
- Its [build recipe](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/blob/a1806cb82493aa6f28709f77acf59c1937bdf756/dflash2-port/build_node.sh)
  inherits `vllm-node-tf5-glm52-b12x:probe-modded` and forces V2 for DFlash2.
- The GLM-5.2 dependency at `162d7a766e4bc9de1809ff724339538380c43b0c`
  publishes the exact addition in
  [fix-indexer-mtp-overhang.py](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark--36tok-s/blob/162d7a766e4bc9de1809ff724339538380c43b0c/patches/fix-indexer-mtp-overhang.py).
- Applying that script's `OLD` → `NEW` replacement to upstream
  `ab666069935c1f23e8ef56038b4659ac9e8f19f8` reconstructs our unmounted original
  image's indexer **byte-for-byte**. It also equals our original baseline. SHA-256:
  `2aa896c467f44c3a65e7d04b07ada866dfa23fb23ee8ec245e5d8c89780cef6b`.

The image source was extracted using a temporary container with no GPU, network,
model or overlay mounts. No production container was changed. We did not inspect
Tony's running machines or claim his private image digest equals ours; the public
patch reconstructing the entire file is the independent provenance evidence.

## Reproduction

On GB10, K=7 enables flattening (`next_n=8`). For block size 64 and CP1, the V2
runner aligns capacity to two columns while the patched indexer adds one column.

| Context limit | Runner width | Indexer width | Actual mixed-source result |
|---|---:|---:|---|
| 80000, published fp8 DFlash lane | 1250 | 1251 | Torch shape exception |
| 80064, arithmetic control | 1252 | 1252 | Success |
| 180224, CP1 control | 2816 | 2817 | Torch shape exception |
| 270000, published NVFP4 DFlash lane's sizing | 4220 | 4220 | Success |

The actual extracted method with scheduled lengths `[5,8,8]` raises:

```text
The expanded size of the tensor (1251) must match the existing size (1250)
at non-singleton dimension 1. Target sizes: [21, 1251]. Tensor sizes: [21, 1250]
```

The actual uniform Triton kernel, interpreted on CPU with `[8,8,8]`, copies
neighboring-row values 1350 / 2600 / 3850 into the extra column, eight times each.
The last value comes from an allocated guard row outside the logical source table;
the repro never deliberately reads unallocated CPU memory. A larger production
backing allocation may absorb the final logical overread. No corrupted model output
is attributed to this secondary defect by this test.

The source-width backport passes all four mixed configurations and the 80k uniform
case. Source and results are in
[results/tonyd2wild-indexer-audit](../results/tonyd2wild-indexer-audit).

```bash
.venv/bin/python bench/repro/repro_original_glm_indexer.py \
  --source results/tonyd2wild-indexer-audit/image-indexer.py
```

The test AST-loads the original method and kernel and supplies workspaces; it does
not boot the original full model or execute its constructor. The dimensions come
from separately inspected allocation code. Live mixed-branch coverage, exact
count100 parity, long-context generation and CUDA memcheck were verified on our
repaired DCP deployment, as described in the
[deployment record](INDEXER-BLOCK-TABLE-DEPLOYMENT.md).

This finding does not imply every MTP lane or context limit fails. Fixing only the
mixed Torch assignment leaves the uniform source-bound defect. Removing `+1`
universally also breaks alignment cases such as 80064. Use shared authoritative
sizing, or the explicit source-width backport with zeroed workspace slack.
