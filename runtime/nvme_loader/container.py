"""Create a test container from the exact prior configuration using Docker API.

Only image, loader args/env, labels and two persistent mounts change. Credentials
and the original inventory stay local with mode 0600 and are never logged.
"""
import argparse
import http.client
import json
import hashlib
import os
from pathlib import Path
import socket
from urllib.parse import quote


class Docker(http.client.HTTPConnection):
    def __init__(self):
        super().__init__("localhost", timeout=90)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect("/var/run/docker.sock")


def call(method, path, body=None):
    c = Docker()
    c.request(method, "/v1.47" + path, body=json.dumps(body) if body is not None else None,
              headers={"Content-Type": "application/json"})
    r = c.getresponse(); data = r.read(); c.close()
    if not 200 <= r.status < 300:
        raise RuntimeError(f"Docker {method} {path}: {r.status} {data.decode()}")
    return json.loads(data) if data else None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("original"); p.add_argument("mode", choices=("prepare", "restore", "auto", "stream"))
    p.add_argument("generation"); p.add_argument("runtime_id")
    p.add_argument("--image", default="spark-vllm:0.29.0-nvme4")
    p.add_argument("--stream-device", choices=("cpu", "pinned", "cuda"), default="cuda")
    p.add_argument("--stream-backend", choices=("runai", "coalesced"), default="coalesced")
    p.add_argument("--stream-batch-mib", type=int, default=128)
    a = p.parse_args()
    if not 1 <= a.stream_batch_mib <= 1024:
        p.error("stream batch size must be 1..1024 MiB")
    original = call("GET", "/containers/" + quote(a.original, safe="") + "/json")
    cfg = original["Config"]
    cfg["Image"] = a.image
    cfg["Env"] = [e for e in cfg["Env"] if not e.startswith(("NVME_", "CUDA_CACHE_PATH=", "CUDA_CACHE_MAXSIZE=",
        "VLLM_NO_USAGE_STATS=", "VLLM_DO_NOT_TRACK=", "DO_NOT_TRACK="))] + [
        "VLLM_NO_USAGE_STATS=1", "VLLM_DO_NOT_TRACK=1", "DO_NOT_TRACK=1",
        "NVME_ARTIFACT_ROOT=/nvme-artifacts", "NVME_LOADER_MODE=" + a.mode,
        "NVME_GENERATION=" + a.generation, "NVME_RUNTIME_ID=" + a.runtime_id,
        "CUDA_CACHE_PATH=/root/.nv/ComputeCache", "CUDA_CACHE_MAXSIZE=4294967296"]
    if a.mode == "stream":
        cfg["Env"] += ["NVME_STREAM_DEVICE=" + a.stream_device,
                       "NVME_STREAM_BACKEND=" + a.stream_backend,
                       "NVME_STREAM_BATCH_BYTES=" + str(a.stream_batch_mib << 20)]
    cfg["Labels"] = {**(cfg.get("Labels") or {}), "spark.nvme.generation": a.generation}
    if "--load-format" in cfg["Cmd"]:
        i = cfg["Cmd"].index("--load-format"); cfg["Cmd"][i + 1] = "nvme"
    else:
        cfg["Cmd"] += ["--load-format", "nvme"]
    # With an explicit KV byte budget this fraction is an initial admission
    # gate, not the KV allocation size. Leave room for normal OS/cache variance
    # on the shared-memory Spark; 0.91 rejected a healthy node by only 150 MiB.
    if "--gpu-memory-utilization" in cfg["Cmd"]:
        i = cfg["Cmd"].index("--gpu-memory-utilization")
        cfg["Cmd"][i + 1] = "0.90"
    host = original["HostConfig"]
    host["Binds"] = list(host.get("Binds") or []) + [
        "/var/tmp/nvme-loader:/nvme-artifacts",
        "/var/tmp/nvme-loader/vllm-cache:/root/.cache/vllm",
        "/var/tmp/nvme-loader/torchinductor-cache:/tmp/torchinductor_root",
        "/var/tmp/nvme-loader/flashinfer-cache:/root/.cache/flashinfer",
        "/var/tmp/nvme-loader/cuda-cache:/root/.nv/ComputeCache"]
    if a.mode == "stream":
        # Keep experimental slabs separate from the retained artifact service.
        # Reuse is allowed only if optional, previously hashed immutable source
        # inventories still match. Otherwise each activation gets a fresh salt.
        content_ids = []
        for name, folder in (("glm-5.3", "GLM-5.3-Int4-Int8Mix"),
                             ("dflash2-draft", "GLM-5.3-DFlash2-draft")):
            try:
                identity = json.loads(Path("/var/tmp/nvme-loader/sources", name + ".json").read_bytes())
                root = Path("/var/tmp/models", folder)
                inventory = [{"name": str(p.relative_to(root)), "size": p.stat().st_size,
                              "mtime_ns": p.stat().st_mtime_ns}
                             for p in sorted(root.rglob("*")) if p.is_file() and
                             p.suffix in (".safetensors", ".json", ".bin", ".pt", ".pth", ".gguf")]
                if inventory != identity["inventory"]:
                    raise ValueError("source inventory changed")
                if hashlib.sha256(json.dumps(identity["files"], sort_keys=True,
                        separators=(",", ":")).encode()).hexdigest() != identity["content_id"]:
                    raise ValueError("source identity manifest checksum mismatch")
                content_ids.append(identity["content_id"])
            except (OSError, ValueError, KeyError):
                content_ids = []
                break
        kv_id = "|".join(content_ids) if content_ids else a.generation
        Path("/var/tmp/nvme-loader/stream-kvcache").mkdir(parents=True, exist_ok=True)
        host["Binds"] = [b for b in host["Binds"] if b.split(":")[1] != "/kvcache"]
        host["Binds"].append("/var/tmp/nvme-loader/stream-kvcache:/kvcache")
        i = cfg["Cmd"].index("--kv-transfer-config")
        kv = json.loads(cfg["Cmd"][i + 1])
        extra = kv["kv_connector_extra_config"]
        extra["disk_bytes_per_rank"] = 30_000_000_000
        extra["slab_salt"] = hashlib.sha256((extra["slab_salt"] + "|stream|" + kv_id).encode()).hexdigest()
        cfg["Cmd"][i + 1] = json.dumps(kv)
    host["RestartPolicy"] = {"Name": "no", "MaximumRetryCount": 0}
    cfg["HostConfig"] = host
    result = call("POST", "/containers/create?name=vllm_glm53big", cfg)
    print(result["Id"], flush=True)
