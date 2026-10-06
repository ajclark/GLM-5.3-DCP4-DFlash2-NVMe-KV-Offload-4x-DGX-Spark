"""Two-rank RoCEnante test of the DCP collectives through the real vLLM path.

Run on one DCP pair (two directly cabled Sparks) with the model stopped, by
``run_dcp_pair_test.sh``. Each rank initializes vLLM's distributed state with TP=2 and
DCP=2, so the DCP group's ``CudaCommunicator`` gets the shim (``GLM_ROCE_GROUPS=dcp``) via
the image's ``glm_roce.pth`` hook. Checks, with our production shapes:

 1. wiring: hook loaded, the DCP communicator has an enabled adapter, the TP one does not,
    ``verify_targets`` passes;
 2. eager, for T in decode token counts: query all-gather [T,16,576] bf16 along dim 1,
    LSE all-gather [T,32] fp32 along dim 0, indexer merge [T,2048,2] fp32 along dim 1,
    output reduce-scatter [T,32,512] bf16 along dim 1. Each is byte-compared with the
    stock NCCL path (the unwrapped method) and with a CPU reference (concatenation, or
    x0 + x1 with one rounding);
 3. CUDA graphs: a 4-layer decode-like sequence captured inside vLLM's graph_capture(),
    replayed with fresh inputs, eager collectives interleaved, every 25th replay checked;
 4. latency: 78 layers x (query gather + LSE gather + reduce-scatter) captured once on
    RoCE and once on NCCL, median microseconds per collective;
 5. health: check_health and runtime stats;
 8. pipe stress (``--pipe-stress N``, ``GLM_ROCE_PIPE=1``, item 2 E2 pipelining): N rounds of the prefill-size
    query gather, indexer merge and output reduce-scatter with fresh inputs every round (each rank regenerates
    both ranks' inputs from shared seeds, so the expected bytes need no communication), uneven token counts so
    chunks split unevenly, interleaved with decode-size gathers on the vendored runtime; every result is
    byte-compared;
 7. glue (``VLLM_DCP_GLUE=1`` in the container, item 2 E1): the patched ``cp_lse_ag_out_rs`` (head-major
    correction, no masked_fill_ of empty rows) against the stock one and a CPU reference, eager at decode
    and prefill sizes and replayed in a CUDA graph, with junk (NaN/inf) in rows whose local shard is empty;
 6. prefill (``--prefill-tokens``, results/prefill-item2-20261006/REPORT.md): the same four
    collectives at prefill chunk sizes, eager (prefill runs eager), byte-compared with NCCL and
    the CPU reference, plus eager milliseconds per op on RoCE and on NCCL. Run with
    ``GLM_ROCE_GATHER_MAX_SIZE`` large enough (36 MiB covers 2048 tokens) or they stay on NCCL.

Prints one ``RESULT {json}`` line per rank and exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import traceback

import torch
import torch.distributed as dist

TOKENS = (2, 8, 16, 32, 48, 96)


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def cpu_gather(t: torch.Tensor, group) -> list[torch.Tensor]:
    raw = t.detach().contiguous().cpu().view(-1).view(torch.uint8)
    parts = [torch.empty_like(raw) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, raw, group=group)
    return [p.view(t.dtype).view(t.shape) for p in parts]


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def prefill_checks(args, rank, comm, make, ops, refs, roce_ag, roce_rs, nccl_ag_f, nccl_rs_f) -> dict:
    """Prefill-size collectives: byte equality (RoCE vs NCCL vs CPU) and eager ms per op."""
    out = {"rows": [], "timing": {}}
    for T in [int(t) for t in args.prefill_tokens.split(",") if t.strip()]:
        x = make(T)
        ops0 = comm.stats().get("ops_posted", 0)
        y = ops(x, roce_ag, roce_rs)
        torch.cuda.synchronize()
        routed = comm.stats().get("ops_posted", 0) - ops0
        yn = ops(x, nccl_ag_f, nccl_rs_f)
        torch.cuda.synchronize()
        ref = refs(x)
        row = {"T": T, "ops_on_roce": routed,
               "bytes": {k: x[k].numel() * x[k].element_size() for k in x}}
        for k in y:
            row[k] = {"vs_ref": torch.equal(y[k].cpu(), ref[k]), "vs_nccl": torch.equal(y[k].cpu(), yn[k].cpu())}
            assert row[k]["vs_ref"] and row[k]["vs_nccl"], f"prefill {k} T={T}: {row[k]}"
        out["rows"].append(row)
        for name, ag, rs in (("roce", roce_ag, roce_rs), ("nccl", nccl_ag_f, nccl_rs_f)):
            for k, fn in (("q", lambda: ag(x["q"], 1)), ("idx", lambda: ag(x["idx"], 1)),
                          ("out", lambda: rs(x["out"], 1))):
                fn(); torch.cuda.synchronize()
                times = []
                for _ in range(args.prefill_iters):
                    dist.barrier(group=comm.group)
                    t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
                out["timing"][f"{name}_{k}_T{T}"] = round(statistics.median(times) * 1e3, 3)
        comm.check_health()
    return out


def pipe_stress(args, rank, dcp, comm, dev) -> dict:
    """Fresh-input rounds through the pipe runtime (and decode-size gathers through the vendored one)."""
    assert comm.pipe is not None, "--pipe-stress needs GLM_ROCE_PIPE=1 and a gather limit above the pipe threshold"

    def gen(shape, dtype, seed, r):
        g = torch.Generator(device=dev).manual_seed(seed * 7919 + r)
        return torch.randn(shape, generator=g, device=dev, dtype=torch.float32).to(dtype)

    bad = {"q": 0, "idx": 0, "out": 0, "small": 0}
    pipe_ops0 = comm.pipe.stats().get("ops_posted", 0)
    main_ops0 = comm._runtime.stats().get("ops_posted", 0)
    token_counts = (2048, 1999, 1024, 1537, 2048)
    for i in range(args.pipe_stress):
        T = token_counts[i % len(token_counts)]
        seed = 1_000_003 * (i + 1)
        q = [gen((T, 16, 576), torch.bfloat16, seed, r) for r in range(2)]
        idx = [gen((T, 2048, 2), torch.float32, seed + 1, r) for r in range(2)]
        out = [gen((T, 32, 512), torch.bfloat16, seed + 2, r) for r in range(2)]
        small = [gen((8, 16, 576), torch.bfloat16, seed + 3, r) for r in range(2)]
        got_q = dcp.all_gather(q[rank], 1)
        got_small = dcp.all_gather(small[rank], 1)  # decode-size: vendored runtime, between pipe ops
        got_idx = dcp.all_gather(idx[rank], 1)
        got_out = dcp.reduce_scatter(out[rank], 1)
        exp_out = (out[0] + out[1]).chunk(2, dim=1)[rank]
        bad["q"] += not torch.equal(got_q, torch.cat(q, dim=1))
        bad["small"] += not torch.equal(got_small, torch.cat(small, dim=1))
        bad["idx"] += not torch.equal(got_idx, torch.cat(idx, dim=1))
        bad["out"] += not torch.equal(got_out, exp_out)
    torch.cuda.synchronize()
    comm.check_health()
    res = {"rounds": args.pipe_stress, "mismatches": bad,
           "pipe_ops": comm.pipe.stats().get("ops_posted", 0) - pipe_ops0,
           "vendored_ops": comm._runtime.stats().get("ops_posted", 0) - main_ops0}
    assert sum(bad.values()) == 0, f"pipe stress mismatches: {res}"
    assert res["pipe_ops"] >= 3 * args.pipe_stress, f"large gathers did not take the pipe runtime: {res}"
    return res


def glue_checks(args, rank, dcp, dev) -> dict:
    """Patched vs stock DCP combine over the real pair (RoCE), byte for byte, plus a CPU reference."""
    from vllm.v1.attention.ops import dcp as dcp_ops

    glued = dcp_ops.cp_lse_ag_out_rs
    assert getattr(glued, "_glm_dcp_glue", False), "VLLM_DCP_GLUE=1 but cp_lse_ag_out_rs is not patched"
    stock = glued.__wrapped__
    cpu = dcp.cpu_group
    out = {"rows": [], "graph": None}

    def inputs(T, seed):
        gi = torch.Generator(device="cpu").manual_seed(seed + 7919 * rank)
        o = torch.randn(T, 32, 512, generator=gi).to(torch.bfloat16)
        lse = torch.randn(T, 32, generator=gi) * 4
        empty = torch.rand(T, generator=gi) < 0.15
        empty[0] = rank == 1
        junk = torch.full_like(o, float("nan"))
        junk[:, ::3] = float("inf")
        o = torch.where(empty[:, None, None], junk, o)
        lse = lse.masked_fill(empty[:, None], float("-inf"))
        return o.to(dev), lse.to(dev), empty.to(dev)

    def stock_combine(o, lse, empty, base_e):
        s = o.clone()
        s.masked_fill_(empty[:, None, None], 0)
        return stock(s, lse.clone(), dcp, is_lse_base_on_e=base_e)

    def reference(o, lse, empty, base_e):
        s = o.clone()
        s.masked_fill_(empty[:, None, None], 0)
        lses = torch.stack(cpu_gather(lse, cpu), 0).to(dev)
        corr, _ = dcp_ops.correct_attn_out(s, lses, rank, None, is_lse_base_on_e=base_e)
        torch.cuda.synchronize()
        parts = cpu_gather(corr, cpu)
        full = (parts[0].float() + parts[1].float()).to(torch.bfloat16)
        return full.chunk(2, dim=1)[rank].contiguous()

    for T in (2, 8, 32, 96, 1024, 2048):
        for base_e in (True, False):
            o, lse, empty = inputs(T, 100 * T + int(base_e))
            g = glued(o, lse.clone(), dcp, is_lse_base_on_e=base_e)
            st = stock_combine(o, lse, empty, base_e)
            torch.cuda.synchronize()
            ref = reference(o, lse, empty, base_e)
            row = {"T": T, "base_e": base_e, "glue_vs_stock": torch.equal(g.cpu(), st.cpu()),
                   "glue_vs_ref": torch.equal(g.cpu(), ref), "contiguous": g.is_contiguous()}
            assert row["glue_vs_stock"] and row["glue_vs_ref"] and row["contiguous"], f"glue combine T={T}: {row}"
            out["rows"].append(row)
    # CUDA graph (decode): capture the glued combine at T=8 and replay with fresh inputs.
    from vllm.distributed import parallel_state as ps

    T = 8
    o, lse, empty = inputs(T, 4242)
    graph = torch.cuda.CUDAGraph()
    with ps.graph_capture(device=dev) as ctx:
        y = glued(o, lse, dcp, is_lse_base_on_e=True)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=ctx.stream):
            y = glued(o, lse, dcp, is_lse_base_on_e=True)
    torch.cuda.synchronize()
    verified = 0
    for i in range(60):
        o2, lse2, empty2 = inputs(T, 5000 + i)
        o.copy_(o2); lse.copy_(lse2)
        graph.replay()
        if i % 10 == 9:
            torch.cuda.synchronize()
            ref = reference(o2, lse2, empty2, True)
            assert torch.equal(y.cpu(), ref), f"glue combine graph replay {i} mismatch"
            verified += 1
    out["graph"] = {"T": T, "replays": 60, "verified": verified}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--port", type=int, default=29681)
    ap.add_argument("--replays", type=int, default=300)
    ap.add_argument("--prefill-tokens", default="", help="comma list, e.g. 1024,2048")
    ap.add_argument("--prefill-iters", type=int, default=20)
    ap.add_argument("--skip-decode", action="store_true", help="only the wiring and prefill checks")
    ap.add_argument("--pipe-stress", type=int, default=0, help="rounds of the pipe stress test (0 = off)")
    args = ap.parse_args()
    rank = args.rank
    result: dict = {"rank": rank, "ok": False, "checks": {}}
    checks = result["checks"]
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    try:
        boot = sys.modules.get("glm_roce.boot")
        checks["hook_loaded"] = bool(boot is not None and boot._finder is not None)
        assert checks["hook_loaded"], "glm_roce.pth hook not active (GLM_ROCE_ALLREDUCE=1?)"
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import parallel_state as ps

        with set_current_vllm_config(VllmConfig()):
            ps.init_distributed_environment(
                world_size=2, rank=rank, local_rank=0,
                distributed_init_method=f"tcp://{args.master}:{args.port}", backend="nccl")
            ps.ensure_model_parallel_initialized(2, 1, decode_context_model_parallel_size=2)
            dcp = ps.get_dcp_group()
            tp = ps.get_tp_group()
            dc = dcp.device_communicator
            comm = getattr(dc, "glm_roce_comm", None)
            checks["dcp_adapter_enabled"] = bool(comm is not None and not comm.disabled)
            checks["tp_adapter"] = getattr(tp.device_communicator, "glm_roce_comm", None) is not None
            assert checks["dcp_adapter_enabled"], f"DCP communicator has no enabled adapter: {comm!r}"
            assert not checks["tp_adapter"], "TP communicator must stay on NCCL"
            from glm_roce.install import verify_targets

            problems = verify_targets()
            checks["verify_targets"] = problems
            assert not problems, problems
            cpu = dcp.cpu_group
            stats0 = comm.stats()
            result["runtime"] = {k: stats0.get(k) for k in ("hcas", "max_size", "max_gather_bytes", "slot_bytes", "spin_limit")}
            log(rank, f"adapter up: {result['runtime']}")
            cls = type(dc)
            nccl_ag = cls.all_gather.__wrapped__
            nccl_rs = cls.reduce_scatter.__wrapped__

            g = torch.Generator(device="cpu").manual_seed(1000 + rank)

            def make(T):
                return {
                    "q": torch.randn(T, 16, 576, generator=g).to(torch.bfloat16).to(dev),
                    "lse": torch.randn(T, 32, generator=g).to(dev),
                    "idx": torch.randn(T, 2048, 2, generator=g).to(dev),
                    "out": torch.randn(T, 32, 512, generator=g).to(torch.bfloat16).to(dev),
                }

            def ops(x, ag, rs):
                return {"q": ag(x["q"], 1), "lse": ag(x["lse"], 0), "idx": ag(x["idx"], 1),
                        "out": rs(x["out"], 1)}

            def refs(x):
                r = {}
                for k, dim in (("q", 1), ("lse", 0), ("idx", 1)):
                    r[k] = torch.cat(cpu_gather(x[k], cpu), dim=dim)
                parts = cpu_gather(x["out"], cpu)
                full = (parts[0].float() + parts[1].float()).to(torch.bfloat16)
                r["out"] = full.chunk(2, dim=1)[rank].contiguous()
                return r

            roce_ag = lambda t, d: dcp.all_gather(t, d)  # noqa: E731
            roce_rs = lambda t, d: dcp.reduce_scatter(t, d)  # noqa: E731
            nccl_ag_f = lambda t, d: nccl_ag(dc, t, d)  # noqa: E731
            nccl_rs_f = lambda t, d: nccl_rs(dc, t, d)  # noqa: E731

            import os as _os

            if args.pipe_stress:
                checks["pipe_stress"] = pipe_stress(args, rank, dcp, comm, dev)
                log(rank, f"pipe stress ok: {checks['pipe_stress']}")
            if _os.environ.get("VLLM_DCP_GLUE", "0") not in ("", "0"):
                checks["glue"] = glue_checks(args, rank, dcp, dev)
                log(rank, f"glue ok: {len(checks['glue']['rows'])} eager cases, graph {checks['glue']['graph']}")
            if args.prefill_tokens:
                checks["prefill"] = prefill_checks(args, rank, comm, make, ops, refs, roce_ag, roce_rs,
                                                   nccl_ag_f, nccl_rs_f)
                log(rank, f"prefill ok {checks['prefill']['timing']}")
            eager = []
            ops0 = comm.stats().get("ops_posted", 0)
            for T in (() if args.skip_decode else TOKENS):
                x = make(T)
                y = ops(x, roce_ag, roce_rs)
                yn = ops(x, nccl_ag_f, nccl_rs_f)
                torch.cuda.synchronize()
                ref = refs(x)
                row = {"T": T}
                for k in y:
                    row[k] = {"vs_ref": torch.equal(y[k].cpu(), ref[k]), "vs_nccl": torch.equal(y[k].cpu(), yn[k].cpu())}
                    assert row[k]["vs_ref"] and row[k]["vs_nccl"], f"{k} T={T}: {row[k]}"
                eager.append(row)
            routed_ops = comm.stats().get("ops_posted", 0) - ops0
            checks["eager"] = eager
            checks["eager_ops_on_roce"] = routed_ops
            if not args.skip_decode:
                assert routed_ops >= 4 * len(TOKENS), f"only {routed_ops} RoCE ops for {4 * len(TOKENS)} collectives"
            log(rank, f"eager ok ({routed_ops} RoCE ops)")

            # CUDA graphs: 4 layers, T=8 (one request) and T=32 (four)
            graphs = []
            for T in (() if args.skip_decode else (8, 32)):
                xs = [make(T) for _ in range(4)]
                graph = torch.cuda.CUDAGraph()
                with ps.graph_capture(device=dev) as ctx:
                    outs = [ops(x, roce_ag, roce_rs) for x in xs]
                    torch.cuda.synchronize()
                    with torch.cuda.graph(graph, stream=ctx.stream):
                        outs = [ops(x, roce_ag, roce_rs) for x in xs]
                torch.cuda.synchronize()
                verified = 0
                for i in range(args.replays):
                    gi = torch.Generator(device="cpu").manual_seed(10_000 * (i + 1) + rank)
                    for x in xs:
                        for t in x.values():
                            t.copy_(torch.randn(t.shape, generator=gi).to(t.dtype))
                    graph.replay()
                    if i % 10 == 5:
                        e = dcp.all_gather(torch.full((4, 32), float(rank + 1), device=dev), 0)
                        torch.cuda.synchronize()
                        assert e[:4].eq(1).all() and e[4:].eq(2).all(), "eager gather between replays is wrong"
                    if i % 25 == 0 or i == args.replays - 1:
                        torch.cuda.synchronize()
                        for x, o in zip(xs, outs):
                            ref = refs(x)
                            for k in o:
                                assert torch.equal(o[k].cpu(), ref[k]), f"graph replay {i} T={T} {k} mismatch"
                        verified += 1
                comm.check_health()
                graphs.append({"T": T, "replays": args.replays, "verified": verified})
            checks["graphs"] = graphs
            log(rank, f"graphs ok {graphs}")

            # Latency: 78 layers x 3 collectives (query gather, LSE gather, reduce-scatter)
            lat = {}
            for name, ag, rs in (("roce", roce_ag, roce_rs), ("nccl", nccl_ag_f, nccl_rs_f)):
                for T in (() if args.skip_decode else (8, 32)):
                    x = make(T)

                    def layer_seq():
                        return [(ag(x["q"], 1), ag(x["lse"], 0), rs(x["out"], 1)) for _ in range(78)]

                    graph = torch.cuda.CUDAGraph()
                    with ps.graph_capture(device=dev) as ctx:
                        layer_seq()
                        torch.cuda.synchronize()
                        with torch.cuda.graph(graph, stream=ctx.stream):
                            layer_seq()
                    torch.cuda.synchronize()
                    times = []
                    for _ in range(30):
                        dist.barrier(group=cpu)
                        t0 = time.perf_counter()
                        graph.replay()
                        torch.cuda.synchronize()
                        times.append(time.perf_counter() - t0)
                    lat[f"{name}_T{T}"] = {"ms_per_78_layers": round(statistics.median(times) * 1e3, 3),
                                           "us_per_collective": round(statistics.median(times) * 1e6 / (78 * 3), 1)}
            checks["latency"] = lat
            log(rank, f"latency {lat}")
            comm.check_health()
            stats = comm.stats()
            result["stats"] = {k: stats.get(k) for k in ("ops_posted", "writes_completed", "error_seq", "epoch")}
            assert stats.get("error_seq") == 0, stats
            result["ok"] = True
    except Exception:  # noqa: BLE001
        result["error"] = traceback.format_exc()
    print("RESULT " + json.dumps(result, default=str), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
