# SPDX-License-Identifier: Apache-2.0
"""Speculative-decode verification under DCP: the target-side pieces.

With any drafter (DFlash in production, MTP if ever re-enabled) each request
contributes K+1 query tokens per verify step through the sharded target. The
indexer expands the request-level context length into a per-token causal
bound *before* sharding it onto the DCP rank; the tests here pin down why that
order is the only correct one, that the localizer handles both the 1-D
(flatten) and 2-D (native) layouts the fork produces, that the static logits
width always covers the local bound, and that the top-k merge collective is
issued before spec-decode unpacking on every rank.
"""

import ast
import pathlib
from types import SimpleNamespace

import pytest
import torch

from harness import OVERLAY, cdiv, extract, extract_methods, ref_local_seq_lens

# The unmodified helper the localizer relies on, read from the image's tree.
FORK = pathlib.Path("~/lmcache-mg/spark-src/vllm").expanduser()
UTILS = extract(
    FORK / "v1/attention/backends/utils.py", ["get_dcp_local_seq_lens"]
)
get_dcp_local_seq_lens = UTILS["get_dcp_local_seq_lens"]

LOCALIZE = extract_methods(
    OVERLAY / "v1/attention/backends/mla/indexer.py",
    "DeepseekV32IndexerMetadataBuilder",
    ["_dcp_localize_decode_seq_lens"],
    {"get_dcp_local_seq_lens": get_dcp_local_seq_lens},
)["_dcp_localize_decode_seq_lens"]

INDEXER_SRC = (OVERLAY / "v1/attention/backends/mla/indexer.py").read_text()
OP_SRC = (OVERLAY / "model_executor/layers/sparse_attn_indexer.py").read_text()
SPARSE_SRC = (OVERLAY / "v1/attention/backends/mla/flashmla_sparse.py").read_text()


def _builder(world, rank, interleave=1, buf_len=64):
    return SimpleNamespace(
        dcp_world_size=world,
        dcp_rank=rank,
        cp_kv_cache_interleave_size=interleave,
        decode_seq_lens_buffer=torch.zeros(buf_len, dtype=torch.int32),
    )


def test_helper_matches_scalar_reference():
    """Sanity: the fork's get_dcp_local_seq_lens equals the scalar reference."""
    lens = torch.arange(0, 40, dtype=torch.int32)
    for world in (1, 2, 4):
        for interleave in (1, 2):
            for rank in range(world):
                got = get_dcp_local_seq_lens(lens, world, rank, interleave)
                exp = [ref_local_seq_lens(int(L), rank, world, interleave) for L in lens]
                assert got.tolist() == exp, (world, interleave, rank)


@pytest.mark.parametrize("world", [2, 4])
def test_expand_then_localize_is_exact_and_the_other_order_is_not(world):
    """Verify step with K+1 = 3 tokens for a request of global length 10.

    Token j (j = 0, 1, 2) attends to 8, 9, 10 global positions. The correct
    per-token local bound is local(8), local(9), local(10). Localizing the
    request length first and then subtracting offsets in local space gives
    local(10) - 2, local(10) - 1, local(10), which is wrong whenever the
    sharding is uneven, e.g. rank 1 of 2: [4, 4, 5] vs [3, 4, 5].
    """
    L, next_n = 10, 3
    per_token_global = torch.tensor([L - next_n + j + 1 for j in range(next_n)])
    for rank in range(world):
        expand_then_localize = get_dcp_local_seq_lens(
            per_token_global, world, rank, 1
        ).tolist()
        naive = [ref_local_seq_lens(int(g), rank, world, 1) for g in per_token_global]
        assert expand_then_localize == naive

        local_L = ref_local_seq_lens(L, rank, world, 1)
        localize_then_expand = [local_L - next_n + j + 1 for j in range(next_n)]
        # The wrong order under-counts on at least one rank whenever N > 1.
        if world == 2 and rank == 1:
            assert localize_then_expand == [3, 4, 5]
            assert naive == [4, 4, 5]
        assert any(
            wrong < right for wrong, right in zip(localize_then_expand, naive)
        ) or localize_then_expand == naive


@pytest.mark.parametrize("world", [2, 4])
def test_localizer_handles_2d_native_bounds_in_place(world):
    """Native MTP path: (B, next_n) view of decode_seq_lens_buffer."""
    B, next_n = 3, 4
    b = _builder(world, rank=world - 1)
    view = b.decode_seq_lens_buffer[: B * next_n].view(B, next_n)
    global_bounds = torch.tensor(
        [[17 - next_n + j + 1 for j in range(next_n)],
         [5 - next_n + j + 1 for j in range(next_n)],
         [64 - next_n + j + 1 for j in range(next_n)]],
        dtype=torch.int32,
    )
    view.copy_(global_bounds)

    out = LOCALIZE(b, view, B, seq_lens_is_buffer_view=True)

    assert out.data_ptr() == view.data_ptr(), "must localize in place"
    assert out.shape == (B, next_n)
    expected = [
        [ref_local_seq_lens(int(g), world - 1, world, 1) for g in row]
        for row in global_bounds
    ]
    assert out.tolist() == expected


@pytest.mark.parametrize("world", [2, 4])
def test_localizer_handles_1d_flatten_bounds_in_place(world):
    """Flatten path (what sm12x uses at K=4): 1-D per-token bounds in the buffer."""
    b = _builder(world, rank=0)
    n = 7
    view = b.decode_seq_lens_buffer[:n]
    view.copy_(torch.tensor([8, 9, 10, 3, 4, 5, 6], dtype=torch.int32))
    out = LOCALIZE(b, view, num_decodes=3, seq_lens_is_buffer_view=True)
    assert out.data_ptr() == view.data_ptr()
    assert out.tolist() == [
        ref_local_seq_lens(g, 0, world, 1) for g in [8, 9, 10, 3, 4, 5, 6]
    ]


def test_localizer_never_mutates_shared_seq_lens():
    """Plain decode: seq_lens aliases common_attn_metadata and must survive."""
    world, rank = 4, 2
    b = _builder(world, rank)
    shared = torch.tensor([13, 7, 100], dtype=torch.int32)
    before = shared.clone()
    out = LOCALIZE(b, shared, num_decodes=3, seq_lens_is_buffer_view=False)
    assert torch.equal(shared, before), "shared metadata was mutated"
    assert out.data_ptr() == b.decode_seq_lens_buffer.data_ptr()
    assert out.tolist() == [ref_local_seq_lens(int(g), rank, world, 1) for g in before]


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("next_n", [1, 2, 4, 5, 6, 8, 9])
def test_static_logits_width_covers_every_local_bound(world, next_n):
    """dcp_logits_max_len must exceed the largest per-token local bound any
    rank can see, including a spec-decode request one block past
    max_model_len. Brute force over every global length."""
    max_model_len, block_size = 1000, 64
    bound = cdiv(max_model_len + block_size, world) + next_n
    for L in range(0, max_model_len + block_size + 1):
        for rank in range(world):
            assert ref_local_seq_lens(L, rank, world, 1) <= bound


def _decode_block(src: str) -> str:
    start = src.index("    if has_decode:")
    end = src.index("    return topk_indices_buffer", start)
    return src[start:end]


def test_merge_precedes_spec_decode_unpack_on_the_decode_path():
    """Padded rows must join the all_gather; unpacking first would change the
    row count on ranks whose padding differs."""
    block = _decode_block(OP_SRC)
    merge = block.index("_merge_dcp_topk_global(")
    unpack = block.index("unpack_seq_triton(")
    assert merge < unpack


def test_indexer_builder_derives_locality_from_fresh_seq_lens():
    """The proposer forwards a STALE dcp_local_seq_lens from the base batch
    into every draft step while advancing seq_lens. The builder must therefore
    localize from seq_lens itself and never consume dcp_local_seq_lens."""
    tree = ast.parse(INDEXER_SRC)
    reads = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr == "dcp_local_seq_lens"
    ]
    assert not reads, "indexer builder must not read common_attn_metadata.dcp_local_seq_lens"
    assert "get_dcp_local_seq_lens(" in INDEXER_SRC


def test_spec_decode_reorder_threshold_survives_dcp():
    """Without supports_dcp_with_varlen the base clamps the threshold to 1 and
    MTP verify rows would be routed to the prefill path."""
    assert "supports_dcp_with_varlen=" in SPARSE_SRC
    assert "self.cp_kv_cache_interleave_size == 1" in SPARSE_SRC


def test_launcher_defaults_to_dflash_and_mounts_every_overlay():
    root = OVERLAY.parent.parent
    launcher = (root / "launch-glm53big-dcp.sh").read_text()
    assert 'SPEC_MODE="${2:-dflash}"' in launcher
    assert '"method":"dflash"' in launcher
    assert "mtp)" in launcher and "dropped" in launcher  # refused with a reason
    # every staged overlay file is bind-mounted, and preflighted
    staged = sorted(p.name for p in (root / "stage/glm-dcp").glob("*.py"))
    for name in staged:
        assert f'$DCP_DIR/{name}:' in launcher, f"{name} not mounted"
        if name == "multinode.py":  # KVTIER lane: preflighted inside its own block
            assert 'grep -q "class MultiNodeOffloadingConnector" "$DCP_DIR/multinode.py"' in launcher
            continue
        if name == "offloading_scheduler.py":  # KVTIER lane: the connector-scheduler fix
            assert 'grep -q "DCP overlay: eagle trailing block is revisited" "$DCP_DIR/offloading_scheduler.py"' in launcher
            continue
        assert name in launcher.split("DCP_FILES=(")[1].split(")")[0], f"{name} not preflighted"
