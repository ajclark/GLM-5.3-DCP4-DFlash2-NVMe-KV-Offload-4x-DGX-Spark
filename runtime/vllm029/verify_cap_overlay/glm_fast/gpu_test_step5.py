#!/usr/bin/env python3
"""Step-5 GPU validation for glm_fast (one GPU, serving container stopped; nothing distributed).

    rsync -a --delete runtime/vllm029/verify_cap_overlay/glm_fast/ spark-06c4.local:glm-fast-test/glm_fast/
    ssh spark-06c4.local 'docker run --rm --gpus all --ipc=host --entrypoint python3 \
        -v $HOME/glm-fast-test:/w spark-vllm:0.29.0-nvme4-stridefix-kvtier-verifycap8-roce-20261001 \
        /w/glm_fast/gpu_test_step5.py' | tee step5-gpu.log

Sections (each prints PASS/FAIL/INFO lines; exit code 1 if any FAIL):
  A  argmax exactness: the Triton local-key kernel on 4 simulated vocab shards (154880 / 4) vs
     torch.argmax on the full row, random and adversarial (ties at shard / block edges, -inf,
     +inf, -0/+0, equal rows, subnormals); NaN rows must stay in range.
  B  argmax vs the production sampler: the image's rejection_sample() (overlay kernels) and
     gumbel_sample() at temperature 0 vs glm_fast's greedy verify on the same logits.
  C  argmax timing (informational): stock local work at L = 8 / 32 vs the fast path (the NCCL
     gather itself is not measurable on one GPU; the DCP2 trace gives 184 us at L = 8).
  D  L2 prefetch kernel: nvcc .so and Triton builds load and run; graph capture with a
     side-stream fork/join replays and gives bit-identical outputs.
  E  L2 prefetch value: a GEMV over a weight-sized buffer after a simulated collective window
     (torch.cuda._sleep on the main stream while the side stream prefetches), cold vs
     prefetched, eager and inside a CUDA graph; bf16 cuBLAS and int8 Marlin (o_proj /
     fused_qkv_a shapes, if vLLM's Marlin test helpers import).

--cpu-smoke runs A and B under TRITON_INTERPRET=1 against the overlay files (development only).
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

ap = argparse.ArgumentParser()
ap.add_argument("--cpu-smoke", action="store_true")
ap.add_argument("--cases", type=int, default=300)
ARGS = ap.parse_args()
if ARGS.cpu_smoke:
    os.environ["TRITON_INTERPRET"] = "1"
os.environ.setdefault("VLLM_MARLIN_USE_ATOMIC_ADD", "1")  # as launch.sh

import torch  # noqa: E402

from glm_fast import l2_prefetch as l2  # noqa: E402
from glm_fast import vocab_argmax as va  # noqa: E402

DEV = "cpu" if ARGS.cpu_smoke else "cuda"
V, TP = 154880, 4
VS = V // TP
RESULTS = {"PASS": 0, "FAIL": 0}


def report(ok: bool | None, name: str, detail: str = "") -> None:
    tag = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    if ok is not None:
        RESULTS[tag] += 1
    print(f"{tag} {name}{(': ' + detail) if detail else ''}", flush=True)


def section(fn):
    def run():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            report(False, fn.__name__, f"raised {exc!r}")
    run.__name__ = fn.__name__
    return run


def ref_argmax(x):
    f = x.float()
    return torch.where(torch.isnan(f), torch.full_like(f, float("-inf")), f).argmax(-1)


def shard_keys(x):
    return torch.cat([va.local_keys(x[:, r * VS:(r + 1) * VS], r * VS)[0] for r in range(TP)])


def adversarial(g):
    rows = []

    def base():
        return (torch.randn(V, generator=g, device=DEV) * 3).to(torch.bfloat16)
    r = base(); r[5] = 30; r[VS * 2 + 3] = 30; rows.append(r)
    r = base(); r[VS + 10] = 30; r[VS + 9] = 30; rows.append(r)
    r = base(); r[VS - 1] = 30; r[VS] = 30; rows.append(r)
    rows.append(torch.full((V,), float("-inf"), dtype=torch.bfloat16, device=DEV))
    r = torch.full((V,), float("-inf"), dtype=torch.bfloat16, device=DEV); r[V - 1] = -1e30; rows.append(r)
    r = base(); r[VS * 3 - 1] = float("inf"); r[VS * 3] = float("inf"); rows.append(r)
    r = torch.zeros(V, dtype=torch.bfloat16, device=DEV); r[: VS + 7] = -0.0; rows.append(r)
    r = torch.full((V,), -5.0, dtype=torch.bfloat16, device=DEV); r[VS * 2] = -0.0; r[VS * 3 + 1] = 0.0; rows.append(r)
    rows.append(torch.full((V,), 1.5, dtype=torch.bfloat16, device=DEV))
    r = base(); r[V - 1] = 40; rows.append(r)
    r = base(); r[0] = 40; rows.append(r)
    r = base(); r[VS * 2] = 40; rows.append(r)
    rows.append((torch.randn(V, generator=g, device=DEV) * 1e-38).to(torch.bfloat16))
    for edge in (8192 * 3, 8192 * 5, 1024, 1024 * 77):
        r = base(); r[edge - 1] = 33; r[edge] = 33; rows.append(r)
    rows.append(torch.randint(-2, 3, (V,), generator=g, device=DEV).to(torch.bfloat16))
    return torch.stack(rows)


# ---------------------------------------------------------------------------------------------
@section
def A_argmax_exact():
    g = torch.Generator(device=DEV).manual_seed(1234)
    bad = total = kbad = 0
    sizes = (1, 2, 8, 16, 32, 96) if not ARGS.cpu_smoke else (2, 8)
    for L in sizes:
        for trial in range(20 if not ARGS.cpu_smoke else 1):
            x = (torch.randn(L, V, generator=g, device=DEV) * (1 + trial)).to(torch.bfloat16)
            if trial % 3 == 1:
                x = torch.randint(-3, 4, (L, V), generator=g, device=DEV).to(torch.bfloat16)  # many ties
            ids = va.reduce_keys(shard_keys(x), TP)
            bad += int((ids != ref_argmax(x)).sum())
            total += L
            if trial == 0:
                for r in range(TP):
                    sh = x[:, r * VS:(r + 1) * VS]
                    kbad += int((va.local_keys(sh, r * VS)[0] != va.local_keys_torch(sh, r * VS)).sum())
    x = adversarial(g)
    ids = va.reduce_keys(shard_keys(x), TP)
    abad = int((ids != ref_argmax(x)).sum())
    report(bad == 0, "A1 argmax random rows vs torch.argmax", f"{bad} mismatches / {total} rows")
    report(abad == 0, "A2 argmax adversarial rows vs torch.argmax", f"{abad} mismatches / {x.shape[0]} rows")
    report(kbad == 0, "A3 Triton keys bitwise == torch reference keys", f"{kbad} differing keys")
    xn = adversarial(g)[:6].clone()
    xn[0, 17] = float("nan"); xn[1, :] = float("nan"); xn[2, ::2] = float("nan")
    xn[3, VS * 2 + 5] = float("nan"); xn[3, VS * 2 + 6] = 1e4; xn[4, VS:] = float("nan")
    ids = va.reduce_keys(shard_keys(xn), TP)
    in_range = bool(((ids >= 0) & (ids < V)).all())
    report(in_range and bool((ids == ref_argmax(xn)).all()), "A4 NaN rows: in range, NaN treated as -inf",
           f"ids={ids.tolist()}")


def _case(g, ns, acc, inject=None):
    L = sum(ns)
    x = (torch.randn(L, V, generator=g, device=DEV) * 3).to(torch.bfloat16)
    if inject is not None:
        inject(x)
    am = ref_argmax(x)
    draft = torch.randint(0, V, (L,), generator=g, device=DEV)
    cu = [0]
    for n in ns:
        cu.append(cu[-1] + n)
    for r, n in enumerate(ns):
        s = cu[r]
        for i in range(n - 1):
            if i < acc[r]:
                draft[s + i + 1] = am[s + i]
            elif i == acc[r]:
                draft[s + i + 1] = (am[s + i] + 1) % V if (r + i) % 3 else -1
    return x, draft, torch.tensor(cu, dtype=torch.int32, device=DEV)


def _stock(rs_mod, x, draft, cu, K):
    nreq = cu.numel() - 1
    idx = torch.arange(nreq, dtype=torch.int32, device=DEV)
    counts = (cu[1:] - cu[:-1]).to(torch.int64)
    exp_idx = torch.repeat_interleave(idx, counts)
    loc = torch.cat([torch.arange(int(c), dtype=torch.int32, device=DEV) + (K + 1 - int(c)) for c in counts.tolist()])
    temp = torch.zeros(nreq, dtype=torch.float32, device=DEV)
    seed = torch.zeros(nreq, dtype=torch.int64, device=DEV)
    pos = torch.arange(x.shape[0], dtype=torch.int64, device=DEV)
    return rs_mod.rejection_sample(x, None, draft, cu, pos, idx, exp_idx, loc, temp, seed, K)


def _stock_modules():
    if not ARGS.cpu_smoke:
        from vllm.v1.worker.gpu.sample import gumbel as g_mod
        from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as rs_mod
        return g_mod, rs_mod
    sys.path.insert(0, str(HERE.parent))
    import test_glm_fast as t  # the CPU tests' loader of the overlay kernel files
    return t.stock_kernels()


@section
def B_argmax_vs_stock_sampler():
    import random
    g_mod, rs_mod = _stock_modules()
    g = torch.Generator(device=DEV).manual_seed(99)
    rnd = random.Random(5)
    K = 7
    adv = adversarial(g)
    bad = cases = reqs = 0
    n_cases = ARGS.cases if not ARGS.cpu_smoke else 6
    first_bad = None
    for c in range(n_cases):
        nreq = rnd.choice([1, 1, 2, 3, 4, 12])
        k = rnd.choice([1, 3, 5, 7])
        ns = [k + 1] * nreq
        if c % 5 == 4:
            ns[rnd.randrange(nreq)] = 1  # a prefill-tail row in the batch
        acc = [rnd.randint(0, n - 1) for n in ns]

        def inject(x, c=c):
            if c % 2:
                for _ in range(3):
                    x[rnd.randrange(x.shape[0])] = adv[rnd.randrange(adv.shape[0])]
        x, draft, cu = _case(g, ns, acc, inject)
        s_ref, n_ref = _stock(rs_mod, x, draft, cu, K)
        s_f, n_f = va.greedy_verify(shard_keys(x), TP, draft, cu, nreq, K + 1)
        ok = torch.equal(n_f, n_ref)
        if ok:
            for r in range(nreq):
                m = int(n_ref[r])
                ok &= torch.equal(s_f[r, :m], s_ref[r, :m])
        ok &= bool(((s_f >= 0) & (s_f < V)).all())
        if not ok:
            bad += 1
            first_bad = first_bad or (ns, acc, n_f.tolist(), n_ref.tolist())
        cases += 1
        reqs += nreq
    report(bad == 0, "B1 greedy verify vs stock rejection_sample", f"{bad} bad / {cases} batches ({reqs} requests)"
           + (f"; first {first_bad}" if first_bad else ""))
    x = torch.cat([(torch.randn(12, V, generator=g, device=DEV) * 3).to(torch.bfloat16), adv])
    L = x.shape[0]
    ref = g_mod.gumbel_sample(x, torch.zeros(L, dtype=torch.int32, device=DEV), torch.zeros(1, device=DEV),
                              torch.zeros(1, dtype=torch.int64, device=DEV), torch.arange(L, device=DEV),
                              apply_temperature=False, is_drafting=False)
    got = va.reduce_keys(shard_keys(x), TP)
    report(torch.equal(got, ref), "B2 plain sampler path vs stock gumbel_sample(temp 0)",
           f"{int((got != ref).sum())} mismatches / {L} rows")
    # NaN rows: informational agreement with the stock kernels, but always in range
    agree = tot = 0
    in_range = True
    for c in range(20 if not ARGS.cpu_smoke else 2):
        def inject(x, c=c):
            x[rnd.randrange(x.shape[0]), rnd.randrange(V)] = float("nan")
            if c % 2:
                x[rnd.randrange(x.shape[0])] = float("nan")
        x, draft, cu = _case(g, [8, 8], [7, 3], inject)
        s_f, n_f = va.greedy_verify(shard_keys(x), TP, draft, cu, 2, K + 1)
        in_range &= bool(((s_f >= 0) & (s_f < V)).all())
        try:
            s_ref, n_ref = _stock(rs_mod, x, draft, cu, K)
        except Exception:  # noqa: BLE001 - the CPU interpreter rejects all-NaN reductions
            continue
        for r in range(2):
            m = int(n_ref[r])
            agree += int(torch.equal(n_f[r], n_ref[r]) and torch.equal(s_f[r, :m], s_ref[r, :m]))
            tot += 1
    report(in_range, "B3 NaN-containing verify steps: every emitted id in range")
    report(None, "B3 NaN-containing requests identical to stock (not required)", f"{agree}/{tot}")


def _time(fn, n=50, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)


@section
def C_argmax_timing():
    if ARGS.cpu_smoke:
        return
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as rs_mod
    g = torch.Generator(device=DEV).manual_seed(3)
    for nreq in (1, 4):
        ns = [8] * nreq
        x, draft, cu = _case(g, ns, [3] * nreq)
        shards = [x[:, r * VS:(r + 1) * VS].contiguous() for r in range(TP)]
        L = x.shape[0]

        def stock():
            # all_gather(dim=-1) output handling: gathered [TP, L, VS] -> movedim -> [L, V]
            full = torch.stack(shards).movedim(0, 1).reshape(L, V)
            _stock(rs_mod, full, draft, cu, 7)

        keys_all = shard_keys(x)

        def fast():
            k, _ = va.local_keys(shards[0], 0)
            kk = keys_all.clone()
            kk[:L] = k
            va.greedy_verify(kk, TP, draft, cu, nreq, 8)
        ts, tf = _time(stock), _time(fast)
        report(None, f"C argmax local work at L={L}", f"stock gather-copy+sampler {ts:.1f} us, fast {tf:.1f} us "
               f"(plus NCCL: stock gathers {L * VS * 2 * (TP - 1) / 1e6:.2f} MB/rank, fast {L * 8 * (TP - 1)} B)")
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        d = va.source_digest(GPUModelRunner.sample)
        report(None, "C GPUModelRunner.sample digest in this image",
               f"{d} ({'reviewed' if d in va.KNOWN_SAMPLE_DIGESTS else 'NOT the reviewed one: re-check'})")
    except Exception as exc:  # noqa: BLE001
        report(None, "C GPUModelRunner.sample digest", f"import failed: {exc!r}")


# ---------------------------------------------------------------------------------------------
def _cfg(impl):
    c = l2.Cfg()
    c.impl = impl
    c.so = "/nonexistent/libglm_l2pf.so"
    c.cache = "/tmp/glm_fast_test"
    c.ctas = 8
    return c


BACKENDS = {}


@section
def D_l2_kernel_and_graph():
    if ARGS.cpu_smoke:
        return
    for impl in ("cuda", "triton"):
        try:
            t0 = time.time()
            be = l2.make_backend(_cfg(impl))
            buf = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
            tbl = torch.tensor([buf.data_ptr(), buf.numel()], dtype=torch.int64, device="cuda")
            be.launch(tbl, 1, torch.cuda.current_stream())
            torch.cuda.synchronize()
            BACKENDS[impl] = be
            report(True, f"D1 {impl} prefetch kernel builds, loads and runs", f"{time.time() - t0:.1f} s, "
                   f"{getattr(be, 'path', 'jit')}")
        except Exception as exc:  # noqa: BLE001
            report(False, f"D1 {impl} prefetch kernel builds, loads and runs", repr(exc))
    w = torch.randn(1024, 6144, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(8, 6144, dtype=torch.bfloat16, device="cuda")
    segs = l2.take(l2.dense_runs(w), 12 << 20)
    tbl = torch.tensor([v for s in segs for v in s], dtype=torch.int64, device="cuda")
    ref = torch.nn.functional.linear(x, w)
    for impl, be in BACKENDS.items():
        try:
            side = torch.cuda.Stream()
            flag = torch.ones(1, dtype=torch.int32, device="cuda")
            out = torch.empty_like(ref)
            graph = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):  # warm-up outside capture (cuBLAS handles, workspaces)
                    side.wait_stream(s)
                    be.launch(tbl, len(segs), side, flag)
                    out.copy_(torch.nn.functional.linear(x, w))
                    s.wait_stream(side)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, stream=s):
                cur = torch.cuda.current_stream()
                side.wait_stream(cur)                 # fork
                be.launch(tbl, len(segs), side, flag)
                torch.cuda._sleep(20000)
                out.copy_(torch.nn.functional.linear(x, w))
                cur.wait_stream(side)                 # join
            ok = True
            for i in range(20):
                flag.fill_(i % 2)                     # runtime on/off flag read by the captured kernel
                out.zero_()
                graph.replay()
                torch.cuda.synchronize()
                ok &= torch.equal(out, ref)
            report(ok, f"D2 {impl}: captured fork/prefetch/join graph replays (flag on and off), "
                       "output bit-identical")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            report(False, f"D2 {impl}: graph capture with side-stream prefetch", repr(exc))


class _Holder(torch.nn.Module):
    pass


def _flush_buf():
    return torch.empty(256 << 20, dtype=torch.uint8, device="cuda")


def _cycles_per_us():
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(100000)
    a.record()
    torch.cuda._sleep(2_000_000)
    b.record()
    b.synchronize()
    return 2_000_000 / (a.elapsed_time(b) * 1000)


def _measure(gemv, be, tbl, nseg, window_us, cpu_us, flush, side, n=40):
    """Median GEMV time (us) after an L2 flush and a `window_us` main-stream sleep, with or
    without a side-stream prefetch issued at the start of the window."""
    ts = []
    cur = torch.cuda.current_stream()
    for _ in range(n + 3):
        flush.zero_()
        if be is not None:
            side.wait_stream(cur)
            be.launch(tbl, nseg, side)
        if window_us:
            torch.cuda._sleep(int(window_us * cpu_us))
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        gemv()
        b.record()
        cur.wait_stream(side)
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts[3:])


def _warm(gemv, n=40):
    ts = []
    for _ in range(n):
        gemv()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        gemv()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)


def _marlin_gemv(K, N):
    """An int8 W8A16 group-128 Marlin GEMV at M=8 as served (vLLM's Marlin test helpers)."""
    from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
    from vllm.scalar_type import scalar_types
    w = torch.randn(K, N, dtype=torch.bfloat16, device="cuda") * 0.02
    _w_ref, q_w, s, _g, _sort, _perm = marlin_quantize(w, scalar_types.uint8b128, 128, act_order=False)
    dev = torch.device("cuda")
    empty = mu.marlin_make_empty_g_idx(dev)
    ws = mu.marlin_make_workspace_new(dev)
    x = torch.randn(8, K, dtype=torch.bfloat16, device="cuda")
    holder = _Holder()
    holder.weight_packed = torch.nn.Parameter(q_w, requires_grad=False)
    holder.weight_scale = torch.nn.Parameter(s, requires_grad=False)

    def gemv():
        return mu.apply_gptq_marlin_linear(x, q_w, s, empty, empty, empty, ws, scalar_types.uint8b128,
                                           output_size_per_partition=N, input_size_per_partition=K,
                                           is_k_full=True)
    return gemv, holder


@section
def E_l2_value():
    if ARGS.cpu_smoke or not BACKENDS:
        return
    props = torch.cuda.get_device_properties(0)
    report(None, "E device", f"{props.name}, L2 {getattr(props, 'L2_cache_size', 0) / 2**20:.1f} MiB, "
           f"{props.multi_processor_count} SMs")
    cpu_us = _cycles_per_us()
    report(None, "E torch.cuda._sleep calibration", f"{cpu_us:.0f} cycles/us")
    flush = _flush_buf()
    side = torch.cuda.Stream()
    cases = []
    w = torch.randn(1024, 6144, dtype=torch.bfloat16, device="cuda")  # 12.6 MB
    xb = torch.randn(8, 6144, dtype=torch.bfloat16, device="cuda")
    hb = _Holder()
    hb.weight = torch.nn.Parameter(w, requires_grad=False)
    cases.append(("bf16 cuBLAS GEMV 6144x1024 (12.6 MB)", lambda: torch.nn.functional.linear(xb, w), hb,
                  (12,)))
    try:
        g1, h1 = _marlin_gemv(4096, 6144)   # o_proj per rank: 25.2 MB int8
        g1()
        cases.append(("int8 Marlin o_proj 4096x6144 (25.2 MB)", g1, h1, (8, 12)))
        g2, h2 = _marlin_gemv(6144, 2624)   # fused_qkv_a (replicated): 16.1 MB int8
        g2()
        cases.append(("int8 Marlin fused_qkv_a 6144x2624 (16.1 MB)", g2, h2, (12, 16)))
        g3, h3 = _marlin_gemv(6144, 1024)   # shared expert gate_up per rank: 6.3 MB int8
        g3()
        cases.append(("int8 Marlin shared gate_up 6144x1024 (6.3 MB)", g3, h3, (8,)))
    except Exception as exc:  # noqa: BLE001
        report(None, "E Marlin cases skipped", repr(exc))
    best = {}
    for name, gemv, holder, budgets in cases:
        os.environ["GLM_FAST_TEST_CPU_TABLES"] = "0"
        runs = l2.module_tensors(holder)
        cold = _measure(gemv, None, None, 0, 80, cpu_us, flush, side)
        warm = _warm(gemv)
        line = [f"cold {cold:.1f} us", f"L2-warm (re-read) {warm:.1f} us"]
        for impl, be in BACKENDS.items():
            for mb in budgets:
                segs = l2.take(runs, mb << 20)
                tbl = torch.tensor([v for s in segs for v in s], dtype=torch.int64, device="cuda")
                for win in (20, 40, 80):
                    t = _measure(gemv, be, tbl, len(segs), win, cpu_us, flush, side)
                    line.append(f"{impl} {mb}MiB/{win}us {t:.1f}")
                    key = (name, impl)
                    if win == 80 and (key not in best or t < best[key][0]):
                        best[key] = (t, cold, mb)
        report(None, f"E {name}", "; ".join(line))
    ok = False
    for (name, impl), (t, cold, mb) in best.items():
        gain = 1 - t / cold
        ok |= gain >= 0.15
        report(None, f"E best at an 80 us window: {name} [{impl}]", f"{cold:.1f} -> {t:.1f} us "
               f"(-{cold - t:.1f} us, {gain * 100:.0f}%) with {mb} MiB")
    report(ok, "E1 a prefetched GEMV is >= 15% faster than cold after an 80 us window")
    # inside a CUDA graph: [sleep 80 us -> GEMV] vs [fork prefetch | sleep 80 us -> GEMV, join]
    name, gemv, holder, budgets = next((c for c in cases if "fused_qkv_a" in c[0]), cases[0])
    impl, be = next(iter(BACKENDS.items()))
    segs = l2.take(l2.module_tensors(holder), max(budgets) << 20)
    tbl = torch.tensor([v for s in segs for v in s], dtype=torch.int64, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    graphs = {}
    with torch.cuda.stream(s):
        gemv()
    torch.cuda.synchronize()
    flag = torch.ones(1, dtype=torch.int32, device="cuda")
    for kind in ("cold", "prefetch"):
        gph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph, stream=s):
            cur = torch.cuda.current_stream()
            if kind == "prefetch":
                side.wait_stream(cur)
                be.launch(tbl, len(segs), side, flag)
            torch.cuda._sleep(int(80 * cpu_us))
            gemv()
            if kind == "prefetch":
                cur.wait_stream(side)
        graphs[kind] = gph
    torch.cuda.synchronize()
    res = {}
    for kind, gph, fl in (("cold", graphs["cold"], 1), ("prefetch", graphs["prefetch"], 1),
                          ("prefetch-flag-off", graphs["prefetch"], 0)):
        flag.fill_(fl)
        ts = []
        for _ in range(43):
            flush.zero_()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            gph.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1000)
        res[kind] = statistics.median(ts[3:])
    report(res["prefetch"] < res["cold"], f"E2 graph replay [{impl}] {name}: 80 us window + GEMV",
           f"cold {res['cold']:.1f} us, prefetched {res['prefetch']:.1f} us (-{res['cold'] - res['prefetch']:.1f} us)")
    report(abs(res["prefetch-flag-off"] - res["cold"]) < max(5.0, 0.05 * res["cold"]),
           "E3 runtime flag off: the captured prefetch graph runs like the cold one",
           f"{res['prefetch-flag-off']:.1f} us vs cold {res['cold']:.1f} us")


def main():
    print(f"glm_fast step-5 GPU test: torch {torch.__version__}, device {DEV}"
          + (f" {torch.cuda.get_device_name(0)}" if DEV == "cuda" else ""), flush=True)
    try:
        import triton
        print(f"triton {triton.__version__}", flush=True)
    except Exception:  # noqa: BLE001
        pass
    for fn in (A_argmax_exact, B_argmax_vs_stock_sampler, C_argmax_timing, D_l2_kernel_and_graph, E_l2_value):
        print(f"== {fn.__name__}", flush=True)
        fn()
    print(f"SUMMARY pass={RESULTS['PASS']} fail={RESULTS['FAIL']}", flush=True)
    return 1 if RESULTS["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
