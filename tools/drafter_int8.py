#!/usr/bin/env python3
"""Re-encode the DFlash2 drafter's large linears as compressed-tensors W8A16 int8
(symmetric, group 128, pack-quantized): the same format and quantizer as the target's
dense layers (ct_common.quantize_int8 / pack_int8), so vLLM serves it on the same Marlin
kernel.

Quantized: every decoder q/k/v/o/gate/up/down projection, fc, and the conv
kernel_projection of every layer (vLLM needs the verify_cap_overlay qwen3_dflash*.py to
build the conv projections with the quant config and to dequantize the context-KV
weights). Kept as-is: norms, conv base kernels, the candidate selector (codebooks are
gathered row by row; hidden_projection is 3 MB).

    python3 tools/drafter_int8.py SRC_DIR OUT_DIR    (from the repository root)
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ct_common import CT_CONVENTION, dequant_int8, pack_int8, quantize_int8

GROUP = 128
QUANT = re.compile(r"^(?:layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj"
                   r"|(?:attention|mlp)_conv\.kernel_projection)|fc)\.weight$")
QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "ignore": [],
    "packed_modules_mapping": {"qkv_proj": ["q_proj", "k_proj", "v_proj"],
                               "gate_up_proj": ["gate_proj", "up_proj"]},
    "config_groups": {
        "w8a16_drafter": {
            "targets": [
                "re:.*self_attn[.](?:qkv_proj|q_proj|k_proj|v_proj|o_proj)$",
                "re:.*mlp[.](?:gate_up_proj|gate_proj|up_proj|down_proj)$",
                "re:(?:.*[.])?fc$",
                "re:.*_conv[.]kernel_projection$",
            ],
            "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "group",
                        "group_size": GROUP, "dynamic": False},
        }
    },
}


def main(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tensors, report = {}, []
    for f in sorted(src.glob("*.safetensors")):
        with safe_open(str(f), "pt") as sf:
            for name in sf.keys():
                w = sf.get_tensor(name)
                if not QUANT.match(name):
                    tensors[name] = w.contiguous()
                    continue
                q, scale = quantize_int8(w, GROUP, CT_CONVENTION)
                packed = pack_int8(q)
                base = name[: -len(".weight")]
                tensors[base + ".weight_packed"] = packed
                tensors[base + ".weight_scale"] = scale.to(torch.bfloat16).contiguous()
                tensors[base + ".weight_shape"] = torch.tensor(list(w.shape), dtype=torch.int64)
                back = dequant_int8(packed, scale, GROUP)
                rel = float((back - w.float()).norm() / w.float().norm())
                report.append({"name": name, "shape": list(w.shape), "rel_frobenius_error": rel})
                print(f"{name:55s} {list(w.shape)} rel err {rel:.5f}", flush=True)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization_config"] = QUANT_CONFIG
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for extra in src.iterdir():
        if extra.is_file() and extra.suffix not in (".safetensors",) and extra.name not in ("config.json",) \
                and not extra.name.startswith(".") and extra.name != "model.safetensors.index.json":
            shutil.copy2(extra, out / extra.name)
    (out / "int8-report.json").write_text(json.dumps({"convention": CT_CONVENTION, "group_size": GROUP,
                                                      "tensors": report}, indent=1))
    print(f"wrote {out} ({len(report)} quantized tensors)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
