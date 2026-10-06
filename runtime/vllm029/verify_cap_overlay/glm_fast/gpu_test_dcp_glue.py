# SPDX-License-Identifier: Apache-2.0
"""Single-GPU bit-identity test of the DCP combine glue (item 2 E1). Run in the image, GPU free enough:

    docker run --rm --gpus all --entrypoint python3 -e VLLM_DCP_GLUE=1 <image> /opt/glm-fast/glm_fast/gpu_test_dcp_glue.py

Emulates the two DCP ranks on one GPU and checks, for prefill and decode sizes, with junk (NaN, inf, random)
in the rows whose local shard is empty:
 1. stock path (backend masked_fill_ of empty rows, then the in-place correction kernel) == glue path (no
    masked_fill_, head-major correction kernel), per rank, byte for byte;
 2. the two-rank reduce-scatter over heads: stock (movedim().contiguous(), add, movedim().contiguous()) == glue
    (head-major input, torch.add into a strided view of the output layout);
 3. the 2-D last-dim gather view == torch.cat along the middle dim, for the query [T,16,576] bf16 and the
    indexer candidates [T,2048,2] fp32;
 4. the patched module function is installed (VLLM_DCP_GLUE=1) and its __wrapped__ is the stock one.
Prints PASS/FAIL lines and exits non-zero on any failure.
"""
from __future__ import annotations

import sys

import torch


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.contiguous().view(-1).view(torch.uint8), b.contiguous().view(-1).view(torch.uint8))


def main() -> int:
    from vllm.v1.attention.ops import dcp
    from glm_fast import dcp_glue

    dev = torch.device("cuda", 0)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name, flush=True)

    check("patched cp_lse_ag_out_rs installed", getattr(dcp.cp_lse_ag_out_rs, "_glm_dcp_glue", False))
    check("stock kept as __wrapped__", getattr(dcp.cp_lse_ag_out_rs, "__wrapped__", None) is not None
          and not getattr(dcp.cp_lse_ag_out_rs.__wrapped__, "_glm_dcp_glue", False))

    g = torch.Generator(device="cpu").manual_seed(20261006)
    H, D = 32, 512
    for T in (2, 8, 32, 96, 1024, 2048):
        for base_e in (True, False):
            outs, lses_l, empties = [], [], []
            for r in range(2):
                out = torch.randn(T, H, D, generator=g).to(torch.bfloat16).to(dev)
                lse = (torch.randn(T, H, generator=g) * 4).to(dev)
                empty = torch.rand(T, generator=g) < 0.15
                empty[0] = r == 1  # row 0 is empty on rank 1 (the interleave's first-token case)
                empty = empty.to(dev)
                # junk in empty rows: NaN, inf and random values, as an unwritten kernel output may hold
                junk = torch.full_like(out, float("nan"))
                junk[:, ::3] = float("inf")
                junk[:, 1::3] = out[:, 1::3]
                out = torch.where(empty[:, None, None], junk, out)
                lse = lse.masked_fill(empty[:, None], float("-inf"))  # the backend's lse fill (kept)
                outs.append(out)
                lses_l.append(lse)
                empties.append(empty)
            lses = torch.stack(lses_l, 0)  # what the LSE all-gather returns: [N, B, H]
            stock_corr, glue_corr = [], []
            for r in range(2):
                s = outs[r].clone()
                s.masked_fill_(empties[r][:, None, None], 0)  # stock backend
                s, s_lse = dcp.correct_attn_out(s, lses, r, None, is_lse_base_on_e=base_e)
                gl, g_lse = dcp_glue.correct_attn_out_head_major(outs[r], lses, r, is_lse_base_on_e=base_e)
                check(f"T={T} base_e={base_e} rank {r}: correction output identical", bitwise_equal(s, gl))
                check(f"T={T} base_e={base_e} rank {r}: lse identical", bitwise_equal(s_lse, g_lse))
                check(f"T={T} rank {r}: glue output is a head-major view",
                      gl.permute(1, 0, 2).is_contiguous())
                check(f"T={T} rank {r}: no NaN/inf in corrected output", bool(torch.isfinite(gl.float()).all()))
                stock_corr.append(s)
                glue_corr.append(gl)
            half = H // 2
            for r in range(2):
                # stock reduce-scatter (glm_roce path without glue, = vLLM's layout handling)
                xs = [c.movedim(0, 1).contiguous() for c in stock_corr]
                stock_rs = (xs[r][r * half:(r + 1) * half] + xs[1 - r][r * half:(r + 1) * half]).movedim(0, 1).contiguous()
                # glue: head-major input -> contiguous() is a no-op; add straight into the output layout
                xg = [c.movedim(0, 1).contiguous() for c in glue_corr]
                check(f"T={T} rank {r}: head-major input needs no copy",
                      xg[r].data_ptr() == glue_corr[r].data_ptr())
                res = torch.empty((T, half, D), dtype=torch.bfloat16, device=dev)
                torch.add(xg[r][r * half:(r + 1) * half], xg[1 - r][r * half:(r + 1) * half], out=res.movedim(1, 0))
                check(f"T={T} base_e={base_e} rank {r}: reduce-scatter result identical", bitwise_equal(stock_rs, res))

    for shape, dtype in (((2048, 16, 576), torch.bfloat16), ((2048, 2048, 2), torch.float32),
                         ((8, 16, 576), torch.bfloat16)):
        x0 = torch.randn(shape, generator=g).to(dtype).to(dev)
        x1 = torch.randn(shape, generator=g).to(dtype).to(dev)
        ref = torch.cat((x0, x1), dim=1)
        lead = shape[0]
        out2 = torch.cat((x0.view(lead, -1), x1.view(lead, -1)), dim=1)  # what the last-dim gather writes
        got = out2.view(shape[:1] + (2 * shape[1],) + shape[2:])
        check(f"2-D last-dim gather == cat(dim=1) for {shape} {dtype}", bitwise_equal(ref, got))

    print("RESULT", "ok" if ok else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
