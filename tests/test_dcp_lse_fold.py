"""Lever H: folding the DCP LSE all-gather into the reduce-scatter must equal
the all-gather + correction + reduce-scatter merge (bf16 rounding aside)."""
import torch

from harness import OVERLAY, extract, ref_lse_merge

NS = extract(OVERLAY / "model_executor/layers/attention/mla_attention.py",
             ["dcp_fold_pack", "dcp_fold_unpack", "dcp_fold_lse_rs"],
             {"DCP_LSE_FOLD_MAX": 80.0, "DCP_LSE_FOLD_PAD": 8})
PACK, UNPACK, FOLD = NS["dcp_fold_pack"], NS["dcp_fold_unpack"], NS["dcp_fold_lse_rs"]


def rank_inputs(world, T=8, H=32, D=512, seed=0, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    outs, lses = [], []
    for r in range(world):
        outs.append(torch.randn(T, H, D, generator=g).to(dtype))
        lses.append(torch.randn(T, H, generator=g) * 6 + 12.0)
    # Rank 1 owns nothing for token 3; every rank is empty for token 5 head 7;
    # a NaN LSE (backend convention for an empty row) on rank 0 token 6.
    lses[1][3, :] = float("-inf")
    for r in range(world):
        lses[r][5, 7] = float("-inf")
    lses[0][6, :] = float("nan")
    return outs, lses


def simulate(world, outs, lses):
    """What the ranks compute: pack locally, reduce-scatter (sum + head slice), unpack."""
    packed = [PACK(outs[r], lses[r]) for r in range(world)]
    total = torch.stack([p.to(torch.float32) for p in packed]).sum(0).to(packed[0].dtype)
    per = total.shape[1] // world
    D = outs[0].shape[-1]
    merged = [UNPACK(total[:, r * per:(r + 1) * per], D, outs[0].dtype) for r in range(world)]
    return torch.cat(merged, dim=1), packed


def test_fold_matches_reference_merge_at_dcp2_and_dcp4():
    for world in (2, 4):
        outs, lses = rank_inputs(world, seed=world)
        want = torch.nan_to_num(ref_lse_merge(outs, lses), nan=0.0)
        got, _ = simulate(world, outs, lses)
        got = got.to(torch.float32)
        assert got.shape == want.shape
        scale = want.abs().amax().clamp(min=1.0)
        assert torch.allclose(got, want, atol=2e-2 * scale, rtol=2e-2), (got - want).abs().max()
        assert torch.isfinite(got).all()
        assert got[5, 7].abs().max() == 0  # every rank empty -> zeros, never NaN


def test_fold_is_exact_in_fp32_and_payload_layout():
    outs, lses = rank_inputs(2, seed=9, dtype=torch.float32)
    want = torch.nan_to_num(ref_lse_merge(outs, lses), nan=0.0)
    got, packed = simulate(2, outs, lses)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)
    assert packed[0].shape == (8, 32, 512 + 8) and packed[0][..., 513:].abs().max() == 0
    assert packed[1][3, :, 512].abs().max() == 0  # empty rows carry zero weight


def test_composed_function_uses_the_group_reduce_scatter():
    outs, lses = rank_inputs(2, seed=3)
    lses[0][0, 0] = 200.0  # would overflow bf16 unclamped
    packed_all = [PACK(outs[r], lses[r]) for r in range(2)]

    class Group:
        def reduce_scatter(self, packed, dim):
            assert dim == 1 and torch.equal(packed, packed_all[0])
            total = (packed_all[0].float() + packed_all[1].float()).to(packed.dtype)
            return total[:, :16]
    got = FOLD(outs[0], lses[0], Group())
    assert got.shape == (8, 16, 512) and torch.isfinite(got.float()).all()
