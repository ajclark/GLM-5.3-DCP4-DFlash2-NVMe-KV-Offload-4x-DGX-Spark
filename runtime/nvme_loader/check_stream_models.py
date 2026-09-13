#!/usr/bin/env python3
"""Native/stream GPU parameter and output parity with automatic stack recovery."""
import json
import argparse
from pathlib import Path
import shlex
import subprocess
import time

import sparkctl as ctl
from build import IMAGE

OUT = ctl.OUT
HOST = "spark-06c4.local"


def check(name, mode, device, guard, backend="runai"):
    label = f"{name}-{mode}-{device}" + ("-coalesced" if backend == "coalesced" and mode == "stream" else "")
    result_name = label + ".json"
    command = ["docker", "run", "--rm", "--name", "nvme-stream-fixture", "--gpus", "all",
        "--ipc", "host", "--network", "host", "--entrypoint", "python3",
        "-v", "/var/tmp/nvme-loader/fixtures:/fixtures:ro",
        "-v", "/var/tmp/nvme-loader:/nvme-artifacts",
        "-v", "/home/napta2k/spark-nvme-build:/test:ro",
        "-v", "/var/tmp/nvme-loader/cuda-cache:/root/.nv/ComputeCache",
        "-v", "/var/tmp/nvme-loader/vllm-cache:/root/.cache/vllm",
        "-v", "/var/tmp/nvme-loader/torchinductor-cache:/tmp/torchinductor_root",
        "-e", "CUDA_CACHE_PATH=/root/.nv/ComputeCache",
        "-e", "CUDA_CACHE_MAXSIZE=4294967296",
        # Only this isolated test serializes our parameter-hash callback over
        # vLLM's internal RPC. Production containers do not enable this flag.
        "-e", "VLLM_ALLOW_INSECURE_SERIALIZATION=1",
        "-e", "NVME_LOADER_MODE=" + mode, "-e", "NVME_STREAM_DEVICE=" + device,
        "-e", "NVME_STREAM_BACKEND=" + backend,
        "-e", "NVME_STREAM_MEMORY_BYTES=8388608",
        "-e", "NVME_STREAM_OWNED_BYTES=16777216", "-e", "OMP_NUM_THREADS=1",
        IMAGE, "/test/gpu_check.py", "--model", "/fixtures/" + name,
        "--output", "/nvme-artifacts/" + result_name]
    try:
        ctl.r.guarded_process(["ssh", HOST, shlex.join(command)], OUT / (label + ".log"), guard, 240)
    finally:
        subprocess.run(["ssh", HOST, "docker rm -f nvme-stream-fixture >/dev/null 2>&1 || true"],
                       capture_output=True, timeout=20)
    for suffix in ("", ".weights.json"):
        subprocess.run(["scp", HOST + ":/var/tmp/nvme-loader/" + result_name + suffix,
                        str(OUT / (result_name + suffix))], check=True, capture_output=True)
    print(label, "complete", flush=True)
    return [json.loads((OUT / (result_name + suffix)).read_bytes())
            for suffix in ("", ".weights.json")]


def check_transport(guard):
    command = ["docker", "run", "--rm", "--name", "nvme-stream-fixture", "--gpus", "all",
               "--ipc", "host", "--entrypoint", "python3",
               "-v", "/var/tmp/nvme-loader:/nvme-artifacts",
               "-v", "/home/napta2k/spark-nvme-build:/test:ro",
               IMAGE, "/test/coalesced_gpu_check.py"]
    try:
        ctl.r.guarded_process(["ssh", HOST, shlex.join(command)], OUT / "coalesced-gpu-transport.log", guard, 180)
    finally:
        subprocess.run(["ssh", HOST, "docker rm -f nvme-stream-fixture >/dev/null 2>&1 || true"],
                       capture_output=True, timeout=20)
    print("Coalesced CUDA ownership/large-tensor transport passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-only", action="store_true",
                        help="compare to existing native baselines after an upload-stream change")
    parser.add_argument("--backend", choices=("runai", "coalesced"), default="runai")
    args = parser.parse_args()
    fallback = json.loads(ctl.LAST_GOOD.read_bytes())
    guard = ctl.r.MemoryGuard(OUT / "stream-model-check-memory.jsonl", loading=True).start()
    try:
        guard.preflight(seconds=0)
        running = ctl.r.parallel(lambda h: ctl.r.ssh(h, ["docker", "ps", "-aq", "--filter",
                                                        "name=^/" + ctl.r.NAME + "$"]).strip())
        if any(running.values()):
            if not all(running.values()):
                raise RuntimeError("partial serving topology before fixture checks")
            ctl.stop_current("stream-fixtures-" + str(int(time.time())))
        if args.backend == "coalesced":
            check_transport(guard)
        for name in ("tiny-llama", "tiny-gpt2"):
            native = ([json.loads((OUT / (name + "-native-cpu.json" + suffix)).read_bytes())
                       for suffix in ("", ".weights.json")] if args.cuda_only else
                      check(name, "native", "cpu", guard))
            for device in (("cuda",) if args.cuda_only else ("cpu", "cuda")):
                streamed = check(name, "stream", device, guard, args.backend)
                assert native == streamed, name + " parameter/token/logprob parity failure: " + device
                print(name, device, "exact parameters, tokens and logprobs passed", flush=True)
    except BaseException as exc:
        print("Fixture check failed before recovery:", repr(exc), flush=True)
        guard.close()
        ctl.rollback(fallback)
        raise
    else:
        # The full-stack stream test follows immediately; leave workers stopped.
        print("All fixture checks passed; workers stopped for full activation", flush=True)
    finally:
        guard.close()
