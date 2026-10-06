#!/usr/bin/env python3
"""The drafter-diet context-KV dequantizer (_dense_rows in qwen3_dflash.py) must equal the
reference decoder in tools/ct_common.py on packed W8A16 int8 weights (group 128).

    python3 test_drafter_int8.py    (needs torch)
"""
import ast
import sys
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tools"))
from ct_common import CT_CONVENTION, dequant_int8, pack_int8, quantize_int8  # noqa: E402


def load_dense_rows():
    src = (HERE / "vllm/model_executor/models/qwen3_dflash.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_dense_rows")
    ns = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "qwen3_dflash.py", "exec"), ns)
    return ns["_dense_rows"]


def main():
    dense_rows = load_dense_rows()
    torch.manual_seed(0)
    w = torch.randn(96, 512) * 0.02
    q, scale = quantize_int8(w, 128, CT_CONVENTION)
    lin = types.SimpleNamespace(weight_packed=pack_int8(q), weight_scale=scale.to(torch.bfloat16))
    ref = dequant_int8(lin.weight_packed, scale, 128)
    got = dense_rows(lin, 32)
    assert got.dtype == torch.bfloat16 and got.shape == (64, 512)
    assert torch.equal(got, ref[32:].to(torch.bfloat16)), (got - ref[32:]).abs().max()
    # a BF16 layer passes through untouched
    lb = types.SimpleNamespace(weight=w.to(torch.bfloat16))
    assert torch.equal(dense_rows(lb, 32), lb.weight[32:])
    print("drafter int8 tests passed")


if __name__ == "__main__":
    main()
