#!/usr/bin/env python3
"""Reproducer for "Triton Error [CUDA]: operation not permitted" (CUDA_ERROR_NOT_PERMITTED, 800)
on DGX Spark GB10 / driver 580: does repeated CUDA module loading in one long-lived process
eventually get refused?  Standalone: one GPU, no vLLM.

What it does (phase by phase, one JSON line per event):
  1. compiles one tiny Triton kernel once and takes its cubin bytes;
  2. loads that SAME cubin again and again through the driver API (cuModuleLoadData), like a
     process that keeps JIT-loading new kernel specializations, never unloading (Triton never
     unloads). Reports the CUresult of every failure, the count of live modules, host
     MemAvailable and swap, and the process RSS, every --report loads;
  3. optionally (--pressure GB) first pins that much host memory with cudaHostAlloc, to see
     whether host-memory pressure on the unified pool changes when the refusal happens;
  4. on the first failure, unloads every module and tries one more load: does the context
     recover, or is it wedged (that is what a serving rank would want to know)?
Usage: cuda_module_load_stress.py [--max 200000] [--report 1000] [--pressure 0] [--big] [--min-avail-mb 3000]
--big uses a large kernel (more code per module) to test whether the limit is code bytes or
module count.  Exit 0 if the limit was reached cleanly, 2 if a load failed.
"""
import argparse, ctypes, json, os, sys, time

def meminfo():
    mi = {}
    with open("/proc/meminfo") as f:
        for l in f:
            k, v = l.split(":")[0], int(l.split()[1]); mi[k] = v
    st = {}
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith(("VmRSS", "VmSwap")): st[l.split(":")[0]] = int(l.split()[1])
    return {"MemAvailable_MB": mi["MemAvailable"] // 1024, "SwapUsed_MB": (mi["SwapTotal"] - mi["SwapFree"]) // 1024,
            "RSS_MB": st.get("VmRSS", 0) // 1024, "VmSwap_MB": st.get("VmSwap", 0) // 1024}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=200000); ap.add_argument("--report", type=int, default=1000)
    ap.add_argument("--pressure", type=float, default=0.0); ap.add_argument("--big", action="store_true")
    ap.add_argument("--min-avail-mb", type=int, default=3000, help="stop before the host runs out of memory (unified memory: the serving stack shares it)")
    a = ap.parse_args()
    import torch, triton, triton.language as tl
    torch.cuda.init(); dev = torch.cuda.current_device()
    say = lambda **kw: print(json.dumps(kw), flush=True)
    say(event="start", gpu=torch.cuda.get_device_name(dev), torch=torch.__version__, triton=triton.__version__,
        driver=os.popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader").read().strip(), pid=os.getpid(), **meminfo())

    if a.big:
        @triton.jit
        def k(x_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0); offs = pid * BLOCK + tl.arange(0, BLOCK); m = offs < n
            v = tl.load(x_ptr + offs, mask=m, other=0.0)
            for i in tl.static_range(0, 64):
                v = tl.sin(v * (1.0 + i * 0.01)) + tl.cos(v)
            tl.store(x_ptr + offs, v, mask=m)
    else:
        @triton.jit
        def k(x_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0); offs = pid * BLOCK + tl.arange(0, BLOCK); m = offs < n
            tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=m, other=0.0) + 1.0, mask=m)
    x = torch.zeros(1024, device="cuda")
    k[(1,)](x, 1024, BLOCK=1024); torch.cuda.synchronize()
    compiled = next(iter(k.device_caches[dev][0].values())) if hasattr(k, "device_caches") else next(iter(k.cache[dev].values()))
    cubin = compiled.asm["cubin"]
    say(event="compiled", cubin_bytes=len(cubin), kernel="big" if a.big else "tiny")

    pinned = None
    if a.pressure > 0:
        n = int(a.pressure * (1 << 30))
        pinned = torch.empty(n, dtype=torch.uint8, pin_memory=True); pinned.fill_(1)
        say(event="pinned", GB=a.pressure, **meminfo())

    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
    cuda.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    def errstr(rc):
        s = ctypes.c_char_p(); cuda.cuGetErrorString(rc, ctypes.byref(s)); return (s.value or b"?").decode()
    mods = []; t0 = time.time(); buf = ctypes.create_string_buffer(cubin, len(cubin))
    rc_fail = None
    for i in range(1, a.max + 1):
        h = ctypes.c_void_p()
        rc = cuda.cuModuleLoadData(ctypes.byref(h), buf)
        if rc != 0:
            rc_fail = rc
            say(event="load_failed", at_load=i, live_modules=len(mods), CUresult=rc, error=errstr(rc), elapsed_s=round(time.time() - t0, 1), **meminfo())
            break
        mods.append(h)
        if i % a.report == 0:
            m = meminfo(); say(event="progress", loads=i, elapsed_s=round(time.time() - t0, 1), **m)
            if m["MemAvailable_MB"] < a.min_avail_mb:
                say(event="stopped_for_memory", loads=i, **m); break
    else:
        say(event="limit_not_reached", loads=a.max, elapsed_s=round(time.time() - t0, 1), **meminfo())
    if rc_fail is not None:
        # does the context recover once modules are released?
        for h in mods: cuda.cuModuleUnload(h)
        mods.clear()
        h = ctypes.c_void_p(); rc = cuda.cuModuleLoadData(ctypes.byref(h), buf)
        say(event="after_unload_all", CUresult=rc, error=errstr(rc) if rc else "ok", **meminfo())
        try:
            k[(1,)](x, 1024, BLOCK=1024); torch.cuda.synchronize(); say(event="kernel_after_unload", ok=True)
        except Exception as e:  # noqa: BLE001
            say(event="kernel_after_unload", ok=False, error=str(e)[:160])
        raise SystemExit(2)
    for h in mods: cuda.cuModuleUnload(h)
    raise SystemExit(0)

if __name__ == "__main__":
    main()
