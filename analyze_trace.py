#!/usr/bin/env python3
"""Per-verify-pass breakdown of a DCP decode trace (torch profiler, one rank).
Cuts the trace at the CPU-side CUDA-graph replay annotations of the 8-token
verify pass and buckets every GPU kernel inside each pass."""
import gzip, json, sys, collections, statistics as st

def load(p):
    return json.load(gzip.open(p, "rt"))["traceEvents"]

def bucket(name):
    n = name.lower()
    if "ncclDevKernel_AllReduce" in name: return "nccl AllReduce (TP)"
    if "ncclDevKernel_AllGather" in name: return "nccl AllGather"
    if "ncclDevKernel_ReduceScatter" in name: return "nccl ReduceScatter"
    if "nccl" in n: return "nccl other"
    if "sparse_mla" in n or "attention" in n or "mqa" in n or "flash" in n or "gather_dequant" in n: return "attention/indexer kernels"
    if "marlin" in n or "moe" in n or "grouped_topk" in n: return "MoE / marlin GEMM"
    if "topk" in n or "radixsort" in n: return "top-k"
    if "gemm" in n or "cutlass" in n or "cublas" in n or "mm" in n: return "dense GEMM"
    return "other kernels"

def main(p):
    ev = load(p)
    # the verify pass = the repeated CUDA-graph replay annotation with the
    # longest median duration (graph numbering differs between boots)
    ann = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") == "user_annotation" and e.get("name", "").startswith("execute_context_"):
            ann[e["name"]].append(e)
    cands = {n: st.median(x["dur"] for x in xs) for n, xs in ann.items() if len(xs) >= 8}
    name = max(cands, key=cands.get)
    print(f"verify-pass annotation: {name} (median {cands[name]/1e3:.1f} ms, {len(ann[name])} replays); others: " +
          ", ".join(f"{n} x{len(ann[n])} ~{cands.get(n, 0)/1e3:.1f} ms" for n in ann if n != name))
    passes = sorted((e["ts"], e["ts"] + e["dur"]) for e in ann[name])
    kernels = sorted((e["ts"], e["dur"], e["name"]) for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e)
    print(f"{len(passes)} verify-pass windows; wall per window (ms): " + " ".join(f"{(b-a)/1e3:.0f}" for a, b in passes))
    rows = []
    for a, b in passes:
        ks = [k for k in kernels if a <= k[0] < b]
        by = collections.defaultdict(float); cnt = collections.Counter()
        ag_small = ag_big = 0.0; n_small = n_big = 0
        for ts, dur, name in ks:
            bk = bucket(name); by[bk] += dur; cnt[bk] += 1
            if bk == "nccl AllGather":
                if dur < 50: ag_small += dur; n_small += 1
                else: ag_big += dur; n_big += 1
        # union busy
        ivs = sorted((ts, ts + dur) for ts, dur, _ in ks)
        busy = 0.0
        if ivs:
            cs, ce = ivs[0]
            for s, e in ivs[1:]:
                if s > ce: busy += ce - cs; cs, ce = s, e
                else: ce = max(ce, e)
            busy += ce - cs
        rows.append(dict(wall=(b - a) / 1e3, busy=busy / 1e3, idle=(b - a - busy) / 1e3, by={k: v / 1e3 for k, v in by.items()}, cnt=dict(cnt),
                         ag_small=ag_small / 1e3, n_small=n_small, ag_big=ag_big / 1e3, n_big=n_big))
    # drop the first (JIT/warm) and report medians over the rest
    body = rows[1:] if len(rows) > 2 else rows
    med = lambda f: st.median(f(r) for r in body)
    print(f"\nmedian over {len(body)} passes: wall {med(lambda r: r['wall']):.1f} ms, GPU busy {med(lambda r: r['busy']):.1f} ms, idle {med(lambda r: r['idle']):.1f} ms")
    keys = sorted({k for r in body for k in r["by"]}, key=lambda k: -st.median(r["by"].get(k, 0) for r in body))
    for k in keys:
        print(f"  {k:28s} median {med(lambda r: r['by'].get(k, 0)):6.1f} ms  n={int(med(lambda r: r['cnt'].get(k, 0)))}")
    print(f"  AllGather split: <50us (LSE-size) median {med(lambda r: r['ag_small']):.1f} ms n={int(med(lambda r: r['n_small']))}; >=50us (query/indexer) median {med(lambda r: r['ag_big']):.1f} ms n={int(med(lambda r: r['n_big']))}")
    # duration histogram of the big all-gathers and reduce-scatters in one median pass
    mid = body[len(body) // 2]
    a, b = passes[rows.index(mid)]
    ag = sorted(dur for ts, dur, name in kernels if a <= ts < b and "AllGather" in name)
    rs = sorted(dur for ts, dur, name in kernels if a <= ts < b and "ReduceScatter" in name)
    q = lambda xs, f: xs[int(len(xs) * f)] if xs else 0
    print(f"  one pass: AllGather n={len(ag)} p10/p50/p90 {q(ag,.1):.0f}/{q(ag,.5):.0f}/{q(ag,.9):.0f} us; ReduceScatter n={len(rs)} p10/p50/p90 {q(rs,.1):.0f}/{q(rs,.5):.0f}/{q(rs,.9):.0f} us")

if __name__ == "__main__":
    main(sys.argv[1])
