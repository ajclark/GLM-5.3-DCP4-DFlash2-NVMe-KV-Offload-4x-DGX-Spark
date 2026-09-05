# SPDX-License-Identifier: Apache-2.0
"""Load the patched vLLM overlay kernels without importing vLLM.

The overlay modules import a large part of vLLM (custom ops, platforms, the
config system), none of which exists in this sandbox and none of which the
DCP arithmetic depends on. So instead of stubbing all of that, we parse each
overlay file and exec only the top-level definitions we need into a namespace
that provides torch / triton / a few helpers.

That keeps the tests bound to the *real patched source*: if a kernel is edited
the tests see the edit, and if a needed definition disappears the extraction
fails loudly.
"""

from __future__ import annotations

import ast
import os
import pathlib

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OVERLAY = ROOT / "overlay" / "vllm"
BASELINE = ROOT / "baseline" / "vllm"


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def extract(path: pathlib.Path, names: list[str], extra_globals: dict | None = None):
    """Exec the named top-level defs (and their decorators) from `path`."""
    src = path.read_text()
    tree = ast.parse(src)
    wanted = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in names:
                wanted[node.name] = node
    missing = set(names) - set(wanted)
    if missing:
        raise AssertionError(f"{path.name}: missing definitions {sorted(missing)}")

    ns: dict = {
        "torch": torch,
        "triton": triton,
        "tl": tl,
        "cdiv": cdiv,
    }
    ns.update(extra_globals or {})
    module = ast.Module(body=[wanted[n] for n in names], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), ns)  # noqa: S102
    return ns


# --------------------------------------------------------------------------
# Reference implementations of *unmodified* vLLM behaviour, transcribed from
# the base commit so the tests can check the patched code against it.
# --------------------------------------------------------------------------


def ref_local_seq_lens(seq_len: int, rank: int, world: int, interleave: int) -> int:
    """vllm/v1/attention/backends/utils.py::get_dcp_local_seq_lens, scalar."""
    base = seq_len // interleave // world * interleave
    remainder = seq_len - base * world
    remainder = min(max(remainder - rank * interleave, 0), interleave)
    return base + remainder


def ref_owner_and_local(pos: int, world: int, interleave: int) -> tuple[int, int]:
    """vllm/v1/worker/block_table.py::_compute_slot_mapping_kernel, scalar.

    Which rank stores global position `pos`, and at which local offset.
    """
    owner = (pos // interleave) % world
    local = (pos // (world * interleave)) * interleave + pos % interleave
    return owner, local


def ref_physical_slot(
    block_table_row, pos: int, block_size: int, world: int, interleave: int, rank: int
):
    """Physical cache slot for `pos` on `rank`, or None if another rank owns it."""
    owner, local = ref_owner_and_local(pos, world, interleave)
    if owner != rank:
        return None
    return int(block_table_row[local // block_size]) * block_size + local % block_size


def ref_lse_merge(outs: list[torch.Tensor], lses: list[torch.Tensor]) -> torch.Tensor:
    """What cp_lse_ag_out_rs computes: all_gather(lse), rescale, reduce_scatter.

    outs[r]: [T, H, D] rank r's softmax over its own subset (already
    normalised). lses[r]: [T, H] natural-log sum-exp of that subset.
    Returns the merged [T, H, D] (before the head-dim scatter, i.e. summed).
    """
    stacked = torch.stack(lses, dim=0)  # [N, T, H]
    stacked = torch.where(torch.isnan(stacked), torch.full_like(stacked, -torch.inf), stacked)
    lse_max = stacked.max(dim=0).values
    lse_max = torch.where(torch.isinf(lse_max), torch.zeros_like(lse_max), lse_max)
    lse_global = torch.log(torch.exp(stacked - lse_max).sum(dim=0)) + lse_max
    acc = torch.zeros_like(outs[0], dtype=torch.float32)
    for r in range(len(outs)):
        factor = torch.exp(lses[r] - lse_global)
        factor = torch.nan_to_num(factor, nan=0.0, posinf=0.0, neginf=0.0)
        acc = acc + outs[r].float() * factor.unsqueeze(-1)
    return acc


def extract_methods(
    path: pathlib.Path,
    class_name: str,
    names: list[str],
    extra_globals: dict | None = None,
):
    """Exec named methods of a class as plain functions (self passed explicitly)."""
    tree = ast.parse(path.read_text())
    cls = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            cls = node
            break
    if cls is None:
        raise AssertionError(f"{path.name}: no class {class_name}")
    wanted = {
        n.name: n
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    }
    missing = set(names) - set(wanted)
    if missing:
        raise AssertionError(f"{path.name}:{class_name}: missing {sorted(missing)}")
    for node in wanted.values():
        node.decorator_list = [
            d for d in node.decorator_list if not (isinstance(d, ast.Name) and d.id == "staticmethod")
        ]
    ns: dict = {"torch": torch, "triton": triton, "tl": tl, "cdiv": cdiv}
    ns.update(extra_globals or {})
    module = ast.Module(body=[wanted[n] for n in names], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), ns)  # noqa: S102
    return ns
