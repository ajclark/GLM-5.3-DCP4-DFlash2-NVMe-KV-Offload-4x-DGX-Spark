#!/usr/bin/env python3
"""Reproducer, faithful path: drive the vendored sm12x indexer kernel (the file that crashed,
pre-fix: every dimension and stride tl.constexpr) through many distinct (num_q, seq_len_kv)
shapes in ONE process. Each new shape is a new Triton specialization: compile, then the lazy
first load (cuModuleLoadData + cuModuleGetFunction + cuFuncGetAttribute/SetAttribute) that
failed on the serving stack with "Triton Error [CUDA]: operation not permitted".
Standalone: one GPU, torch + triton, no vLLM. One JSON line per event; stops at the first
launch failure (exit 2), at --max shapes, or when host MemAvailable < --min-avail-mb.
Usage: indexer_kernel_specialization_stress.py --kernel-file /old/sm12x_mqa.py [--max 3000] [--num-heads 64 --head-dim 128] [--report 50]"""
import argparse, importlib.util, json, os, random, sys, time

def meminfo():
    mi = {}
    with open("/proc/meminfo") as f:
        for l in f:
            mi[l.split(":")[0]] = int(l.split()[1])
    st = {}
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith(("VmRSS", "VmSwap")): st[l.split(":")[0]] = int(l.split()[1])
    return {"MemAvailable_MB": mi["MemAvailable"] // 1024, "SwapUsed_MB": (mi["SwapTotal"] - mi["SwapFree"]) // 1024, "RSS_MB": st.get("VmRSS", 0) // 1024}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-file", required=True); ap.add_argument("--max", type=int, default=3000)
    ap.add_argument("--num-heads", type=int, default=64); ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--report", type=int, default=50); ap.add_argument("--min-avail-mb", type=int, default=3000); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    import torch, triton
    say = lambda **kw: print(json.dumps(kw), flush=True)
    spec = importlib.util.spec_from_file_location("sm12x_mqa", a.kernel_file); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    say(event="start", torch=torch.__version__, triton=triton.__version__, gpu=torch.cuda.get_device_name(0), pid=os.getpid(), kernel_file=a.kernel_file, **meminfo())
    rng = random.Random(a.seed); t0 = time.time(); seen = set(); n = 0
    while n < a.max:
        num_q = rng.choice([1, 8, 16, 33, 64, 128, 257, 512, 1024, 2048, 4096]) + rng.randint(0, 7)
        seq_len_kv = rng.randint(64, 20000)
        if (num_q, seq_len_kv) in seen: continue
        seen.add((num_q, seq_len_kv)); n += 1
        q = torch.randn(num_q, a.num_heads, a.head_dim, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        k = torch.randn(seq_len_kv, a.head_dim, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale = torch.rand(seq_len_kv, device="cuda", dtype=torch.float32)
        w = torch.rand(num_q, a.num_heads, device="cuda", dtype=torch.float32)
        ks = torch.zeros(num_q, device="cuda", dtype=torch.int32); ke = torch.full((num_q,), seq_len_kv, device="cuda", dtype=torch.int32)
        t1 = time.time()
        try:
            out = mod.fp8_mqa_logits_triton(q, (k, scale), w, ks, ke, clean_logits=True)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            say(event="launch_failed", shape_index=n, num_q=num_q, seq_len_kv=seq_len_kv, error=str(e)[:200], elapsed_s=round(time.time() - t0, 1), **meminfo())
            raise SystemExit(2)
        del q, k, scale, w, ks, ke, out
        if n % a.report == 0:
            m = meminfo(); say(event="progress", shapes=n, last_compile_s=round(time.time() - t1, 2), elapsed_s=round(time.time() - t0, 1), **m)
            if m["MemAvailable_MB"] < a.min_avail_mb:
                say(event="stopped_for_memory", shapes=n, **m); raise SystemExit(3)
    say(event="limit_not_reached", shapes=n, elapsed_s=round(time.time() - t0, 1), **meminfo())

if __name__ == "__main__":
    main()
