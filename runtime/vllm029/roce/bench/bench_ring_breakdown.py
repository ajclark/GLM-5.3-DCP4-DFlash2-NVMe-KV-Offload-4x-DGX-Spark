"""Four-rank timeline of the TP ring all-reduce, from inside the proxy (trace build).

Runs like test_tp_ring.py (TP=4, DCP=2 through vLLM's groups, GLM_ROCE_TP_RING=1) but with
GLM_ROCE_RING_TRACE=1, so the proxy timestamps every op on CLOCK_MONOTONIC:
  t_db        doorbell first seen          t_own_post   own payload+flag posted (both HCAs)
  t_own_cqe   own flag write acked (cw/ccw) t_arr[src]   each peer's flag first seen here
  t_fwd_post  forward posted (cw)          t_fwd_cqe    forward acked
For T = 1..32 tokens it replays a graph of 156 chained all-reduces (y = ar(y) * 0.25), then
reduces the trace of the last replays to per-op medians relative to this rank's doorbell. The
period between consecutive doorbells is the whole per-op cost on the GPU's timeline; the part
after the last peer flag arrives (period - last arrival) is GPU work: flag detection, reduce,
the * 0.25 kernel, the next launch and stage.

--pin big|little|none pins the hot proxy threads (found by CPU time) to a big (X925) or little
(A725) core first, to see what the CPU contributes.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback

import torch
import torch.distributed as dist

HIDDEN = 6144


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def cpu_capacity() -> dict[int, int]:
    caps = {}
    for cpu in range(os.cpu_count() or 0):
        try:
            caps[cpu] = int(open(f"/sys/devices/system/cpu/cpu{cpu}/cpu_capacity").read())
        except OSError:
            pass
    return caps


def thread_times() -> dict[int, tuple[int, int]]:
    """tid -> (utime+stime ticks, last cpu)."""
    out = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            f = open(f"/proc/self/task/{tid}/stat").read().rsplit(")", 1)[1].split()
            out[int(tid)] = (int(f[11]) + int(f[12]), int(f[36]))
        except (OSError, IndexError, ValueError):
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--port", type=int, default=29791)
    ap.add_argument("--pin", choices=("none", "big", "little"), default="none")
    args = ap.parse_args()
    rank = args.rank
    result: dict = {"rank": rank, "ok": False, "pin": args.pin}
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    try:
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import parallel_state as ps

        with set_current_vllm_config(VllmConfig()):
            ps.init_distributed_environment(world_size=4, rank=rank, local_rank=0,
                                            distributed_init_method=f"tcp://{args.master}:{args.port}",
                                            backend="nccl")
            ps.ensure_model_parallel_initialized(4, 1, decode_context_model_parallel_size=2)
            tp = ps.get_tp_group()
            comm = tp.device_communicator.glm_roce_comm
            assert comm is not None and comm.ring, "TP ring adapter missing"
            rt = comm._runtime
            proxy = rt._proxy
            cpu = tp.cpu_group
            caps = cpu_capacity()
            result["cpu_capacity"] = caps

            def chain_graph(T):
                x = torch.randn(T, HIDDEN).to(torch.bfloat16).to(dev)

                def chain():
                    y = x
                    for _ in range(156):
                        y = tp.all_reduce(y) * 0.25
                    return y

                g = torch.cuda.CUDAGraph()
                with ps.graph_capture(device=dev) as ctx:
                    chain()
                    torch.cuda.synchronize()
                    with torch.cuda.graph(g, stream=ctx.stream):
                        chain()
                torch.cuda.synchronize()
                return g

            # warm up, find the hot proxy threads (ring + DCP) by CPU time during a burst
            g8 = chain_graph(8)
            t0 = thread_times()
            for _ in range(30):
                g8.replay()
            torch.cuda.synchronize()
            t1 = thread_times()
            hot = sorted(((t1[t][0] - t0.get(t, (0, 0))[0], t) for t in t1), reverse=True)[:4]
            result["hot_threads_before_pin"] = [{"tid": t, "ticks": d, "cpu": t1[t][1],
                                                 "capacity": caps.get(t1[t][1])} for d, t in hot]
            if args.pin != "none":
                big = max(caps.values())
                want = [c for c, v in caps.items() if (v == big) == (args.pin == "big")]
                main_tid = os.getpid()
                targets = [t for d, t in hot if d > 0 and t != main_tid][:1]
                for i, t in enumerate(targets):
                    os.sched_setaffinity(t, {want[-1 - i]})
                result["pinned"] = {t: want[-1 - i] for i, t in enumerate(targets)}

            out = {}
            for T in (1, 2, 4, 8, 16, 32):
                g = chain_graph(T)
                for _ in range(5):
                    g.replay()
                torch.cuda.synchronize()
                dist.barrier(group=cpu)
                seq0 = proxy.stats()["last_seq"]
                times = []
                for _ in range(20):
                    dist.barrier(group=cpu)
                    a = time.perf_counter()
                    g.replay()
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - a)
                seq1 = proxy.stats()["last_seq"]
                time.sleep(0.05)
                tr = proxy.trace()
                lo = seq1 - 156 * 15
                recs = {int(r["seq"]): r for r in tr if lo < int(r["seq"]) <= seq1 and r["t_db"]}
                nxt, prv, opp = (rank + 1) % 4, (rank + 3) % 4, (rank + 2) % 4
                m = {k: [] for k in ("db_to_post", "post_to_cqe_cw", "post_to_cqe_ccw", "arr_prev", "arr_next",
                                     "arr_opp", "last_arr", "fwd_react", "fwd_post_to_cqe", "period", "gpu_part",
                                     "arr_prev_to_opp", "first_prev", "first_opp")}
                for s, r in recs.items():
                    db = int(r["t_db"])
                    us = lambda v: (int(v) - db) / 1e3  # noqa: E731
                    arr = [int(r["t_arr"][i]) for i in (prv, nxt, opp)]
                    if not all(arr) or not r["t_own_post"] or not r["t_fwd_post"]:
                        continue
                    m["db_to_post"].append(us(r["t_own_post"]))
                    if r["t_own_cqe"][0]:
                        m["post_to_cqe_cw"].append((int(r["t_own_cqe"][0]) - int(r["t_own_post"])) / 1e3)
                    if r["t_own_cqe"][1]:
                        m["post_to_cqe_ccw"].append((int(r["t_own_cqe"][1]) - int(r["t_own_post"])) / 1e3)
                    m["arr_prev"].append(us(arr[0]))
                    m["arr_next"].append(us(arr[1]))
                    m["arr_opp"].append(us(arr[2]))
                    m["last_arr"].append(us(max(arr)))
                    m["fwd_react"].append((int(r["t_fwd_post"]) - max(int(r["t_first"][prv]) or arr[0], int(r["t_own_post"]))) / 1e3)
                    m["arr_prev_to_opp"].append((arr[2] - arr[0]) / 1e3)
                    if r["t_first"][prv] and r["t_first"][opp]:
                        m["first_prev"].append(us(r["t_first"][prv]))
                        m["first_opp"].append(us(r["t_first"][opp]))
                    if r["t_fwd_cqe"]:
                        m["fwd_post_to_cqe"].append((int(r["t_fwd_cqe"]) - int(r["t_fwd_post"])) / 1e3)
                    n = recs.get(s + 1)
                    if n is not None and n["t_db"]:
                        per = (int(n["t_db"]) - db) / 1e3
                        if per < 2000:
                            m["period"].append(per)
                            m["gpu_part"].append(per - us(max(arr)))
                out[T] = {"ops": len(recs), "us_per_allreduce_graph": round(statistics.median(times) * 1e6 / 156, 1),
                          **{k: round(statistics.median(v), 2) for k, v in m.items() if v}}
                log(rank, f"T={T} {json.dumps(out[T])}")
            t2 = thread_times()
            hot2 = sorted(((t2[t][0] - t1.get(t, (0, 0))[0], t) for t in t2), reverse=True)[:4]
            result["hot_threads_after"] = [{"tid": t, "ticks": d, "cpu": t2[t][1],
                                            "capacity": caps.get(t2[t][1])} for d, t in hot2]
            result["breakdown"] = out
            rt.check_health()
            result["ok"] = True
    except Exception:  # noqa: BLE001
        result["error"] = traceback.format_exc()
    print("RESULT " + json.dumps(result, default=str), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
