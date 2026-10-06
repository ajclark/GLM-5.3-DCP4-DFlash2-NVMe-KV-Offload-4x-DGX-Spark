"""Compressed-tensors int8 helpers for tools/drafter_int8.py (W8A16, symmetric, group 128). CPU-only."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

TORCH_TO_ST = {}
for st, dt in {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
    "I64": torch.int64, "I32": torch.int32, "I8": torch.int8, "U8": torch.uint8,
}.items():
    TORCH_TO_ST[str(dt)] = st

NP_DTYPE = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32,
            "I64": np.int64, "I32": np.int32, "I8": np.int8, "U8": np.uint8}
TORCH_DTYPE = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
               "I64": torch.int64, "I32": torch.int32, "I8": torch.int8, "U8": torch.uint8}




CT_CONVENTION = {"recipe": "compressed-tensors-rtn-bf16-scale-v1",
                 "divisor": 127.5, "lo": -128, "rounding": "half_even",
                 "order": "le", "packing": "offset_binary",
                 "scale_dtype": "bfloat16"}
CONVENTIONS = [CT_CONVENTION]


def quantize_int8(w, group_size, conv):
    """Symmetric per-group Int8. w: [out, in] float. Returns (q, scale).

    q: int8 [out, in]; scale: float32 [out, in//g] (g = group_size, or the
    full input dim for channelwise).
    """
    if conv != CT_CONVENTION:
        raise ValueError(f"unsupported quantization convention: {conv}")
    from compressed_tensors.quantization import QuantizationArgs
    from compressed_tensors.quantization.utils.helpers import calculate_qparams
    from compressed_tensors.quantization.lifecycle.forward import quantize
    w = w.float()
    if w.ndim != 2 or not torch.isfinite(w).all():
        raise ValueError("quantization requires a finite 2D weight matrix")
    out_f, in_f = w.shape
    g = min(group_size, in_f)
    if g <= 0 or in_f % g:
        raise ValueError(f"input dim {in_f} not divisible by group size {g}")
    args = QuantizationArgs(num_bits=8, type="int", symmetric=True,
                            strategy="channel" if g == in_f else "group",
                            group_size=-1 if g == in_f else g, dynamic=False)
    wg = w.reshape(out_f, in_f // g, g)
    scale, zp = calculate_qparams(wg.amin(-1), wg.amax(-1), args)
    scale = scale.to(torch.bfloat16).float()
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("invalid scale after BF16 rounding")
    q = quantize(w, scale, zp, args, dtype=torch.int8)
    return q, scale


def pack_int8(q, order="le"):
    """[out, in] int8 -> [out, in//4] int32, 4 values per word."""
    if order != "le":
        raise ValueError("compressed-tensors requires little-endian packing")
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
    return pack_to_int32(q, 8, packed_dim=1).contiguous()


def unpack_int32(packed):
    # Independent decoding formula, checked against the library and golden
    # words in tests. The stored byte is q + 128, not q cast to uint8.
    u = packed.numpy().astype(np.uint32)
    q = np.stack([u & 0xFF, (u >> 8) & 0xFF, (u >> 16) & 0xFF, (u >> 24) & 0xFF],
                 axis=-1).astype(np.int16) - 128
    return torch.from_numpy(q.astype(np.int8).reshape(u.shape[0], -1))


def dequant_int8(packed, scale, group_size):
    q = unpack_int32(packed).float()
    out_f, in_f = q.shape
    g = min(group_size, in_f)
    s = scale.float()
    return (q.reshape(out_f, in_f // g, g) * s.unsqueeze(-1)).reshape(out_f, in_f)


def infer_group_size(packed_shape, scale_shape):
    """Group size from the baseline's own layout: in_f = packed*4, g = in_f/k."""
    in_f = packed_shape[1] * 4
    k = scale_shape[1]
    if in_f % k:
        raise ValueError(f"cannot infer group size: in={in_f} k={k}")
    return in_f // k
