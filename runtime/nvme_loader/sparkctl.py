#!/usr/bin/env python3
"""Restart-based all-Spark activation with exact-container rollback."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime/vllm029"))
import rollout as r
from boot_observer import BootSamples, timed_smoke

OUT = ROOT / "results/nvme-loader"
STATE = OUT / "original.private.json"
LAST_GOOD = OUT / "last-good.private.json"
RUNTIME_ID = hashlib.sha256((ROOT / "runtime/vllm029/manifest.json").read_bytes()).hexdigest()


def stop_current(generation):
    def stop(host):
        current = json.loads(r.ssh(host, ["docker", "inspect", r.NAME]))[0]
        r.ssh(host, ["docker", "stop", "-t", os.getenv("NVME_STOP_SECONDS", "2"), r.NAME])
        r.ssh(host, ["docker", "rename", r.NAME, r.NAME + "_" + generation])
    r.parallel(stop)


def rollback(state):
    def stop_new(host):
        try:
            current = json.loads(r.ssh(host, ["docker", "inspect", r.NAME]))[0]
        except RuntimeError:
            return
        if current["Id"] == state[host]["Id"]:
            return
        if not current["Config"].get("Labels", {}).get("spark.nvme.generation"):
            raise RuntimeError("refusing to stop unexpected container on " + host)
        r.ssh(host, ["docker", "stop", "-t", "5", r.NAME])
        r.ssh(host, ["docker", "rename", r.NAME, r.NAME + "_nvme_failed_" + str(int(time.time()))])
    r.parallel(stop_new)
    for host in reversed(r.HOSTS):
        old = json.loads(r.ssh(host, ["docker", "inspect", state[host]["Id"]]))[0]
        if old["Name"] != "/" + r.NAME:
            r.ssh(host, ["docker", "rename", old["Id"], r.NAME])
        r.ssh(host, "$HOME/glm53big/start-flusher.sh")
        if not old["State"]["Running"]:
            r.ssh(host, ["docker", "start", old["Id"]])
    r.wait_health(OUT, None)
    (OUT / "rollback-smoke.json").write_text(json.dumps(r.smoke(), indent=2))
    r.parallel(lambda h: r.ssh(h, "pkill -f '[c]ache_flusher.sh' || true"))
    print("Retained serving stack restored; generation passed", flush=True)


def snapshot():
    if STATE.exists():
        return json.loads(STATE.read_bytes())
    r.request()
    state = r.parallel(lambda h: json.loads(r.ssh(h, ["docker", "inspect", r.NAME]))[0])
    with os.fdopen(os.open(STATE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
        json.dump(state, f)
    (OUT / "native-count100.json").write_text(json.dumps(r.smoke(), indent=2))
    # Preserve compiled/autotune artifacts before replacing the container.
    def preserve_caches(host):
        for src, name in (("/root/.cache/vllm", "vllm-cache"),
                          ("/tmp/torchinductor_root", "torchinductor-cache"),
                          ("/root/.cache/flashinfer", "flashinfer-cache"),
                          ("/root/.nv/ComputeCache", "cuda-cache")):
            r.ssh(host, ["mkdir", "-p", "/var/tmp/nvme-loader/" + name])
            r.ssh(host, ["docker", "cp", state[host]["Id"] + ":" + src + "/.",
                         "/var/tmp/nvme-loader/" + name + "/"], timeout=180)
    r.parallel(preserve_caches)
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("snapshot", "stop", "prepare", "restore", "stream", "rollback"))
    p.add_argument("--stream-device", choices=("cpu", "pinned", "cuda"), default="cuda")
    p.add_argument("--stream-backend", choices=("runai", "coalesced"), default="coalesced")
    p.add_argument("--stream-batch-mib", type=int, default=128)
    p.add_argument("--observe-boot", action="store_true")
    a = p.parse_args()
    if not 1 <= a.stream_batch_mib <= 1024:
        p.error("stream batch size must be 1..1024 MiB")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = snapshot()
        fallback = json.loads(LAST_GOOD.read_bytes()) if LAST_GOOD.exists() else state
        if a.mode == "snapshot":
            print("Original containers and generation saved", flush=True); return
        if a.mode == "rollback":
            rollback(state); return
        generation = "nvme-" + a.mode + "-" + str(int(time.time()))
        if a.mode == "stop":
            stop_current(generation); return
        def interrupted(signum, frame):
            raise RuntimeError("activation interrupted")
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        guard = r.MemoryGuard(OUT / (generation + "-memory.jsonl"), loading=True).start()
        start = time.monotonic()
        samples = None
        try:
            guard.preflight(seconds=0)
            if a.observe_boot:
                samples = BootSamples(r.HOSTS, OUT, generation)
            # The GPU check stage may already have stopped/renamed originals.
            running = r.parallel(lambda h: r.ssh(h, ["docker", "ps", "-aq", "--filter", "name=^/" + r.NAME + "$"]).strip())
            if any(running.values()):
                if not all(running.values()):
                    raise RuntimeError("partial serving topology before activation")
                stop_current(generation)
            def launch(host):
                r.ssh(host, ["python3", "spark-nvme-build/container.py", state[host]["Id"],
                             a.mode, generation, RUNTIME_ID,
                             "--image", os.getenv("NVME_IMAGE", "spark-vllm:0.29.0-nvme4"),
                             "--stream-device", a.stream_device,
                             "--stream-backend", a.stream_backend,
                             "--stream-batch-mib", str(a.stream_batch_mib)])
                if a.mode == "prepare":
                    r.ssh(host, "$HOME/glm53big/start-flusher.sh")
                r.ssh(host, ["docker", "start", r.NAME])
            r.parallel(launch)
            r.wait_health(OUT, guard, timeout=1200)
            healthy = time.monotonic() - start
            smoke_started = time.monotonic() - start
            smoke = timed_smoke(r.BASE) if a.observe_boot else r.smoke()
            (OUT / (generation + "-smoke.json")).write_text(json.dumps(smoke, indent=2))
            r.logs(OUT, generation)
            result = {"mode": a.mode, "generation": generation, "health_seconds": healthy,
                      "generation_passed": True, "stream_backend": a.stream_backend,
                      "stream_batch_mib": a.stream_batch_mib,
                      "stopped_existing_workers": any(running.values())}
            if a.observe_boot:
                result.update(first_content_seconds=smoke["first_content_seconds"],
                              launch_to_first_content_seconds=smoke_started + smoke["first_content_seconds"],
                              launch_to_validated_completion_seconds=smoke_started + smoke["completion_seconds"])
            (OUT / (generation + ".json")).write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
            latest = r.parallel(lambda h: json.loads(r.ssh(h, ["docker", "inspect", r.NAME]))[0])
            with os.fdopen(os.open(LAST_GOOD, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
                json.dump(latest, f)
        except BaseException:
            r.logs(OUT, generation + "-failed")
            guard.close()
            guard = None
            rollback(fallback)
            raise
        finally:
            if samples is not None:
                samples.close()
            if guard is not None:
                guard.close()
            r.parallel(lambda h: r.ssh(h, "pkill -f '[c]ache_flusher.sh' || true"))


if __name__ == "__main__":
    main()
