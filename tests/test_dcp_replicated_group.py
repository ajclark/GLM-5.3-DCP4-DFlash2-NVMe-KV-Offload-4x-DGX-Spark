# SPDX-License-Identifier: Apache-2.0
"""DFlash under DCP: the drafter's sliding-window KV group is *replicated*.

The target's full-attention group is token-sharded across the DCP ranks; the
drafter's sliding-window group must behave exactly as at DCP=1 on every rank
(every position stored, plain attention path). One helper decides that per
spec, and the KV coordinator, block-size resolution, block tables and the
flash-attn builder all consult it. These tests drive the patched code.
"""

import ast
from types import SimpleNamespace

import pytest
import torch

from harness import OVERLAY, cdiv, extract, extract_methods

KVI = OVERLAY / "v1/kv_cache_interface.py"
KVU = OVERLAY / "v1/core/kv_cache_utils.py"
COORD = OVERLAY / "v1/core/kv_cache_coordinator.py"
FA = OVERLAY / "v1/attention/backends/flash_attn.py"
CPU = OVERLAY / "v1/worker/cp_utils.py"


# Minimal stand-ins mirroring the fork's spec class hierarchy:
# AttentionSpec <- FullAttentionSpec <- MLAAttentionSpec ; AttentionSpec <- SlidingWindowSpec
class KVCacheSpec:
    def __init__(self, block_size=64):
        self.block_size = block_size


class AttentionSpec(KVCacheSpec):
    pass


class FullAttentionSpec(AttentionSpec):
    pass


class MLAAttentionSpec(FullAttentionSpec):
    pass


class SlidingWindowSpec(AttentionSpec):
    pass


class MambaSpec(KVCacheSpec):
    pass


class UniformTypeKVCacheSpecs(KVCacheSpec):
    def __init__(self, specs, block_size=64):
        super().__init__(block_size)
        self.kv_cache_specs = specs


SPEC_GLOBALS = dict(
    KVCacheSpec=KVCacheSpec,
    FullAttentionSpec=FullAttentionSpec,
    SlidingWindowSpec=SlidingWindowSpec,
    UniformTypeKVCacheSpecs=UniformTypeKVCacheSpecs,
    MambaSpec=MambaSpec,
)

HELPER = extract(KVI, ["cp_world_size_for_kv_cache_spec"], SPEC_GLOBALS)[
    "cp_world_size_for_kv_cache_spec"
]


def _target_group():
    return UniformTypeKVCacheSpecs({"l0": MLAAttentionSpec(), "l0.idx": MLAAttentionSpec()})


def _drafter_group():
    return UniformTypeKVCacheSpecs({"d0": SlidingWindowSpec(), "d1": SlidingWindowSpec()})


# ------------------------------------------------------------------ the rule
@pytest.mark.parametrize("cp", [1, 2, 4])
def test_full_attention_is_sharded_sliding_window_is_replicated(cp):
    assert HELPER(FullAttentionSpec(), cp) == cp
    assert HELPER(MLAAttentionSpec(), cp) == cp
    assert HELPER(_target_group(), cp) == cp
    assert HELPER(SlidingWindowSpec(), cp) == 1
    assert HELPER(_drafter_group(), cp) == 1
    assert HELPER(MambaSpec(), cp) == 1


def test_mixed_group_is_rejected():
    mixed = UniformTypeKVCacheSpecs({"a": MLAAttentionSpec(), "b": SlidingWindowSpec()})
    with pytest.raises(AssertionError):
        HELPER(mixed, 4)


# --------------------------------------------------- block-size resolution
def _resolve(groups, dcp, pcp=1, prefix_caching=True, hash_override=None):
    ns = extract(
        KVU,
        ["resolve_kv_cache_block_sizes"],
        {
            **SPEC_GLOBALS,
            "math": __import__("math"),
            "cp_world_size_for_kv_cache_spec": HELPER,
        },
    )
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=64,
            enable_prefix_caching=prefix_caching,
            hash_block_size=hash_override,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp, prefill_context_parallel_size=pcp
        ),
        kv_transfer_config=None,
    )
    kvc = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=g) for g in groups]
    )
    return ns["resolve_kv_cache_block_sizes"](kvc, cfg)


def test_block_sizes_glm_dflash_under_dcp4():
    """Target 64 * 4 = 256 tokens per block, drafter 64: scheduler aligns on
    256 and hashes at 64."""
    assert _resolve([_target_group(), _drafter_group()], dcp=4) == (256, 64)


def test_block_sizes_unchanged_without_dcp():
    assert _resolve([_target_group(), _drafter_group()], dcp=1) == (64, 64)
    assert _resolve([_target_group()], dcp=1) == (64, 64)
    assert _resolve([_target_group()], dcp=4) == (256, 256)


def test_pcp_with_hybrid_groups_still_rejected():
    with pytest.raises(ValueError):
        _resolve([_target_group(), _drafter_group()], dcp=1, pcp=2)


# ------------------------------------------------------ static invariants
def test_coordinator_passes_per_group_dcp_and_uses_manager_block_sizes():
    src = COORD.read_text()
    assert 'assert dcp_world_size == 1, "DCP not support hybrid attn now."' not in src
    assert "cp_world_size_for_kv_cache_spec(" in src
    # both hit lookups forward the group's own dcp size
    assert src.count("dcp_world_size=group_manager.dcp_world_size") == 2
    # no length arithmetic on the raw spec block size remains in the hybrid lookups
    tree = ast.parse(src)
    hybrid = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "HybridKVCacheCoordinator"
    )
    for fn in hybrid.body:
        if isinstance(fn, ast.FunctionDef) and fn.name.startswith("find_longest_cache_hit"):
            body_src = ast.get_source_segment(src, fn)
            assert "spec.block_size" not in body_src, fn.name


def test_sliding_window_specs_no_longer_assert_dcp():
    assert 'assert vllm_config.parallel_config.decode_context_parallel_size == 1' not in KVI.read_text()
    assert 'assert vllm_config.parallel_config.decode_context_parallel_size == 1' not in KVU.read_text()


def test_flash_attn_impl_override_precedes_dcp_dtype_capture():
    """The impl's replicated override must run before `_dcp_dtype` is derived
    from dcp_world_size, or the DCP combine would still be armed."""
    src = FA.read_text()
    override = src.index("if sliding_window is not None and self.dcp_world_size > 1:")
    dtype = src.index("self._dcp_dtype: torch.dtype | None = None")
    assert override < dtype
    # and the builder cross-checks the group against the impls
    assert "cp_world_size_for_kv_cache_spec(" in src
    assert "impl_replicated != (group_cp == 1)" in src


def test_cp_compatibility_check_exempts_replicated_impls():
    ns = extract(
        CPU,
        ["check_attention_cp_compatibility"],
        {
            "get_layers_from_vllm_config": lambda cfg, t: cfg.layers,
            "cast": lambda t, v: v,
            "AttentionLayerBase": object,
            "Any": object,
        },
    )
    check = ns["check_attention_cp_compatibility"]
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=4,
            cp_kv_cache_interleave_size=1,
        ),
        speculative_config=None,
        layers={},
    )
    sharded_ok = SimpleNamespace(impl=SimpleNamespace(dcp_world_size=4, need_to_return_lse_for_decode=True, supports_pcp=False))
    replicated = SimpleNamespace(impl=SimpleNamespace(dcp_world_size=1, need_to_return_lse_for_decode=False, supports_pcp=False))
    sharded_bad = SimpleNamespace(impl=SimpleNamespace(dcp_world_size=4, need_to_return_lse_for_decode=False, supports_pcp=False))

    cfg.layers = {"t": sharded_ok, "d": replicated}
    check(cfg)  # must not raise: the drafter's impl is replicated
    cfg.layers = {"t": sharded_bad}
    with pytest.raises(AssertionError):
        check(cfg)
    # Dense MLA impls carry dcp_world_size = -1 until their first forward;
    # that placeholder is NOT "replicated" and must still be checked.
    placeholder_bad = SimpleNamespace(impl=SimpleNamespace(dcp_world_size=-1, need_to_return_lse_for_decode=False, supports_pcp=False))
    cfg.layers = {"t": placeholder_bad}
    with pytest.raises(AssertionError):
        check(cfg)
