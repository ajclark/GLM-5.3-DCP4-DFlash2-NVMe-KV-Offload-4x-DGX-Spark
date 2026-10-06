"""Four-rank test of the TP ring all-reduce (glm_roce.ring) through the real vLLM path.

Run on all four Sparks with the model stopped, by ``run_tp_ring_test.sh``. Each rank
initializes vLLM's distributed state with TP=4 and DCP=2 as in production, with
``GLM_ROCE_GROUPS=tp,dcp`` and ``GLM_ROCE_TP_RING=1``, so the TP communicator gets the
ring runtime and the DCP pairs keep RoCEnante. Checks:

 1. wiring: the TP adapter runs the ring transport, the DCP adapter is enabled, links;
 2. eager all-reduce of decode-sized activations ([T, 6144] bf16 for T up to 85, fp32,
    odd sizes): equal to the fixed-order fp32 reference ((x0 + x1) + x2) + x3 rounded
    once, identical on all ranks; the difference from NCCL is reported (not zero: NCCL
    sums in another order with bf16 intermediate roundings);
 3. CUDA graphs: 78 layers x (TP all-reduce + DCP all-gather) with compute between,
    captured inside vLLM's graph_capture(), 300 replays with fresh inputs, eager
    collectives interleaved, every 25th replay checked;
 4. latency: 156 back-to-back TP all-reduces captured once on the ring and once on
    NCCL, median microseconds per all-reduce at T = 1, 2, 4, 8, 16, 32;
 5. health and runtime stats.

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

HIDDEN = 6144


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def cpu_gather(t: torch.Tensor, group) -> list[torch.Tensor]:
    raw = t.detach().contiguous().cpu().view(-1).view(torch.uint8)
    parts = [torch.empty_like(raw) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, raw, group=group)
    return [p.view(t.dtype).view(t.shape) for p in parts]


def ref_sum(x: torch.Tensor, group) -> torch.Tensor:
    parts = cpu_gather(x, group)
    acc = parts[0].float()
    for p in parts[1:]:
        acc = acc + p.float()
    return acc.to(x.dtype)


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--port", type=int, default=29691)
    ap.add_argument("--replays", type=int, default=300)
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
                world_size=4, rank=rank, local_rank=0,
                distributed_init_method=f"tcp://{args.master}:{args.port}", backend="nccl")
            ps.ensure_model_parallel_initialized(4, 1, decode_context_model_parallel_size=2)
            tp = ps.get_tp_group()
            dcp = ps.get_dcp_group()
            tc = tp.device_communicator
            comm = getattr(tc, "glm_roce_comm", None)
            dcomm = getattr(dcp.device_communicator, "glm_roce_comm", None)
            checks["tp_ring"] = bool(comm is not None and not comm.disabled and getattr(comm, "ring", False))
            checks["dcp_adapter_enabled"] = bool(dcomm is not None and not dcomm.disabled)
            assert checks["tp_ring"], f"TP communicator has no enabled ring adapter: {comm!r}"
            assert checks["dcp_adapter_enabled"], "DCP communicator lost its adapter"
            from glm_roce.install import verify_targets

            problems = verify_targets()
            assert not problems, problems
            cpu = tp.cpu_group
            st = comm.stats()
            result["runtime"] = {k: st.get(k) for k in ("algorithm", "links", "hcas", "split", "chunks", "blocks", "max_size",
                                                        "max_gather_bytes", "slot_bytes", "spin_limit")}
            log(rank, f"ring up: {result['runtime']}")
            cls = type(tc)
            nccl_ar = cls.all_reduce.__wrapped__
            ring_ar = lambda t: tp.all_reduce(t)  # noqa: E731
            nccl_ar_f = lambda t: nccl_ar(tc, t)  # noqa: E731
            g = torch.Generator(device="cpu").manual_seed(1000 + rank)

            # 2. eager
            cases = [((T, HIDDEN), torch.bfloat16) for T in (1, 2, 4, 8, 16, 24, 32, 48, 64, 85)]
            cases += [((2, HIDDEN), torch.float32), ((8, HIDDEN), torch.float32), ((5, 1000), torch.bfloat16),
                      ((7, 3), torch.bfloat16), ((200, HIDDEN), torch.bfloat16)]
            eager = []
            ops0 = comm.stats().get("ops_posted", 0)
            expect_routed = 0
            for shape, dtype in cases:
                x = (torch.randn(shape, generator=g) * 4).to(dtype).to(dev)
                routed = comm.should_custom_ar(x)
                expect_routed += int(routed)
                y = ring_ar(x.clone())
                yn = nccl_ar_f(x.clone())
                torch.cuda.synchronize()
                ref = ref_sum(x, cpu)
                digests = [None] * 4
                dist.all_gather_object(digests, digest(y), group=cpu)
                same_all = len(set(digests)) == 1
                diff = (y.float().cpu() - yn.float().cpu()).abs()
                row = {"shape": list(shape), "dtype": str(dtype).replace("torch.", ""), "routed": routed,
                       "vs_ref": torch.equal(y.cpu(), ref), "same_on_all_ranks": same_all,
                       "vs_nccl_max_abs": float(diff.max()), "vs_nccl_frac_diff": float((diff > 0).float().mean())}
                assert same_all, f"{shape} {dtype}: ranks differ {digests}"
                if routed:
                    assert row["vs_ref"], f"{shape} {dtype}: ring result differs from the fixed-order reference"
                eager.append(row)
            routed_ops = comm.stats().get("ops_posted", 0) - ops0
            checks["eager"] = eager
            checks["eager_ops_on_ring"] = routed_ops
            assert routed_ops == expect_routed, f"{routed_ops} ring ops for {expect_routed} eligible all-reduces"
            log(rank, f"eager ok ({routed_ops} ring ops; non-routed cases on NCCL)")

            # 3. graphs: 78 layers x (TP all-reduce + DCP gather) with compute in between
            w = (torch.randn(HIDDEN, 64, generator=g) / 64).to(torch.bfloat16).to(dev)
            graphs = []
            for T in (8, 32):
                xs = [(torch.randn(T, HIDDEN, generator=g)).to(torch.bfloat16).to(dev) for _ in range(78)]
                qs = [(torch.randn(T, 16, 576, generator=g)).to(torch.bfloat16).to(dev) for _ in range(78)]

                def seq():
                    outs, gath = [], []
                    for x, q in zip(xs, qs):
                        h = (x @ w).sum(dim=1, keepdim=True) * 0  # compute that keeps x intact
                        outs.append(ring_ar(x + h.to(x.dtype)))
                        gath.append(dcp.all_gather(q, 1))
                    return outs, gath

                graph = torch.cuda.CUDAGraph()
                with ps.graph_capture(device=dev) as ctx:
                    seq()
                    torch.cuda.synchronize()
                    with torch.cuda.graph(graph, stream=ctx.stream):
                        outs, gath = seq()
                torch.cuda.synchronize()
                verified = 0
                for i in range(args.replays):
                    gi = torch.Generator(device="cpu").manual_seed(10_000 * (i + 1) + rank)
                    for x in xs:
                        x.copy_(torch.randn(x.shape, generator=gi).to(x.dtype))
                    for q in qs:
                        q.copy_(torch.randn(q.shape, generator=gi).to(q.dtype))
                    graph.replay()
                    if i % 10 == 5:
                        e = ring_ar(torch.full((4, HIDDEN), float(rank + 1), dtype=torch.bfloat16, device=dev))
                        torch.cuda.synchronize()
                        assert e.eq(10).all(), "eager all-reduce between replays is wrong"
                    if i % 25 == 0 or i == args.replays - 1:
                        torch.cuda.synchronize()
                        for j in (0, 1, 38, 77):
                            assert torch.equal(outs[j].cpu(), ref_sum(xs[j], cpu)), f"replay {i} T={T} layer {j}"
                            dref = torch.cat(cpu_gather(qs[j], dcp.cpu_group), dim=1)
                            assert torch.equal(gath[j].cpu(), dref), f"replay {i} T={T} dcp gather {j}"
                        verified += 1
                comm.check_health()
                dcomm.check_health()
                graphs.append({"T": T, "replays": args.replays, "verified": verified})
            checks["graphs"] = graphs
            log(rank, f"graphs ok {graphs}")

            # 4. latency: 156 back-to-back all-reduces per graph
            lat = {}
            for name, ar in (("ring", ring_ar), ("nccl", nccl_ar_f)):
                for T in (1, 2, 4, 8, 16, 32):
                    x = torch.randn(T, HIDDEN, generator=g).to(torch.bfloat16).to(dev)

                    def chain():
                        y = x
                        for _ in range(156):
                            y = ar(y) * 0.25
                        return y

                    graph = torch.cuda.CUDAGraph()
                    with ps.graph_capture(device=dev) as ctx:
                        chain()
                        torch.cuda.synchronize()
                        with torch.cuda.graph(graph, stream=ctx.stream):
                            chain()
                    torch.cuda.synchronize()
                    times = []
                    for _ in range(30):
                        dist.barrier(group=cpu)
                        t0 = time.perf_counter()
                        graph.replay()
                        torch.cuda.synchronize()
                        times.append(time.perf_counter() - t0)
                    med = statistics.median(times)
                    lat[f"{name}_T{T}"] = {"ms_per_156": round(med * 1e3, 3), "us_per_allreduce": round(med * 1e6 / 156, 1)}
            checks["latency"] = lat
            log(rank, f"latency {lat}")
            comm.check_health()
            stats = comm.stats()
            result["stats"] = {k: stats.get(k) for k in ("ops_posted", "forwards", "forwards_done",
                                                         "writes_completed", "error_seq", "epoch",
                                                         "bytes_posted_per_hca")}
            assert stats.get("error_seq") == 0, stats
            result["ok"] = True
    except Exception:  # noqa: BLE001
        result["error"] = traceback.format_exc()
    print("RESULT " + json.dumps(result, default=str), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
