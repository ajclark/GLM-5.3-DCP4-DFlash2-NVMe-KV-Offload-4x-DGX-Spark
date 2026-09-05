# SPDX-License-Identifier: Apache-2.0
"""Static checks on the overlay files that a CPU box can still enforce.

These catch the classes of mistake that would only show up minutes into a
cluster boot: a custom-op signature that no longer matches its fake, a name
used before assignment in __init__, or an accidental edit outside the DCP
change.
"""

import ast
import pathlib

import pytest

from harness import BASELINE, OVERLAY

FILES = [
    "v1/attention/backends/mla/sparse_utils.py",
    "v1/attention/backends/mla/flashmla_sparse.py",
    "v1/attention/backends/mla/indexer.py",
    "model_executor/layers/sparse_attn_indexer.py",
    "model_executor/layers/attention/mla_attention.py",
    "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
    "v1/attention/ops/deepseek_v4_ops/b12x_sparse_helpers.py",
]

INDEXER = OVERLAY / "model_executor/layers/sparse_attn_indexer.py"
SPARSE = OVERLAY / "v1/attention/backends/mla/flashmla_sparse.py"


@pytest.mark.parametrize("rel", FILES)
def test_overlay_parses_and_matches_baseline_shape(rel):
    """Every overlay must parse, and must define the same top-level names as
    the file it replaces plus whatever the patch adds (never fewer)."""
    over = ast.parse((OVERLAY / rel).read_text())
    base = ast.parse((BASELINE / rel).read_text())

    def names(tree):
        return {
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }

    removed = names(base) - names(over)
    assert not removed, f"{rel} drops {sorted(removed)}"


def test_custom_op_signature_matches_its_fake():
    tree = ast.parse(INDEXER.read_text())
    sigs = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in (
            "sparse_attn_indexer",
            "sparse_attn_indexer_fake",
        ):
            sigs[n.name] = [a.arg for a in n.args.args]
    assert len(sigs) == 2
    assert sigs["sparse_attn_indexer"] == sigs["sparse_attn_indexer_fake"]
    for tail in ("dcp_rank", "dcp_world_size", "cp_kv_cache_interleave_size"):
        assert tail in sigs["sparse_attn_indexer"]


def test_layer_passes_every_op_argument():
    """SparseAttnIndexer.forward_cuda must pass all op args positionally."""
    tree = ast.parse(INDEXER.read_text())
    op_args = None
    call_args = None
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "sparse_attn_indexer":
            op_args = [a.arg for a in n.args.args]
        if isinstance(n, ast.Call):
            f = n.func
            if (
                isinstance(f, ast.Attribute)
                and f.attr == "sparse_attn_indexer"
                and isinstance(f.value, ast.Attribute)
                and f.value.attr == "vllm"
            ):
                call_args = n.args
    assert op_args is not None and call_args is not None
    assert len(call_args) == len(op_args), (
        f"op takes {len(op_args)} args, forward_cuda passes {len(call_args)}"
    )


def _assigned_before_use(fn: ast.FunctionDef, attr: str) -> bool:
    """True if the first textual reference to self.<attr> in fn is a store."""
    refs = [
        (node.lineno, node.col_offset, isinstance(node.ctx, ast.Store))
        for node in ast.walk(fn)
        if isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ]
    if not refs:
        return True
    refs.sort()
    return refs[0][2]


@pytest.mark.parametrize(
    "attr", ["use_fp8_kv_cache", "fp8_use_mixed_batch", "dcp_world_size", "num_heads"]
)
def test_builder_init_assigns_before_reading(attr):
    """The DCP guard reads several attributes; each must already be set.

    (This test was written after the guard was first placed above the
    `use_fp8_kv_cache` assignment, which would have raised AttributeError on
    the first DCP boot.)
    """
    tree = ast.parse(SPARSE.read_text())
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "FlashMLASparseMetadataBuilder":
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                    assert _assigned_before_use(fn, attr), (
                        f"self.{attr} is read before it is assigned"
                    )
                    return
    pytest.fail("FlashMLASparseMetadataBuilder.__init__ not found")


def test_a2a_is_rejected():
    """The Spark ring cannot do all-to-all; the backend must refuse it."""
    src = SPARSE.read_text()
    assert "dcp_comm_backend != \"ag_rs\"" in src
    assert "NotImplementedError" in src


def test_dcp_lse_is_asserted_in_the_attention_layer():
    src = (OVERLAY / "model_executor/layers/attention/mla_attention.py").read_text()
    assert "assert lse is not None" in src
    # the old blanket fp8 refusal must be gone
    assert "DCP not support fp8 kvcache now" not in src
