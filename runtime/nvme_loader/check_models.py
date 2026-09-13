#!/usr/bin/env python3
"""Llama -> GPT2 -> Llama restart parity using the public vLLM loader plugin."""
import json
from pathlib import Path
import subprocess

HOST = "spark-06c4.local"
OUT = Path(__file__).resolve().parents[2] / "results/nvme-loader"


def run(name, mode):
    filename = name + "-" + mode + ".json"
    command = ("docker run --rm --gpus all --ipc host --network host --entrypoint python3 "
        "-v /var/tmp/nvme-loader:/nvme-artifacts -v \"$HOME/spark-nvme-build:/test:ro\" "
        "-v /var/tmp/nvme-loader/vllm-cache:/root/.cache/vllm "
        "-e NVME_ARTIFACT_ROOT=/nvme-artifacts -e NVME_RUNTIME_ID=fixture-v1 "
        f"-e NVME_LOADER_MODE={mode} spark-vllm:0.29.0-nvme1 "
        f"/test/gpu_check.py --model /nvme-artifacts/fixtures/{name} "
        f"--output /nvme-artifacts/{filename}")
    with (OUT / (name + "-" + mode + ".log")).open("w") as log:
        result = subprocess.run(["ssh", HOST, command], stdout=log, stderr=subprocess.STDOUT, timeout=240)
    if result.returncode:
        raise RuntimeError("GPU model check failed: " + str(log.name))
    subprocess.run(["scp", HOST + ":/var/tmp/nvme-loader/" + filename, str(OUT / filename)], check=True)
    print(name, mode, "generation complete", flush=True)


if __name__ == "__main__":
    for name in ("tiny-llama", "tiny-gpt2"):
        command = ("docker run --rm --entrypoint python3 -v /var/tmp/nvme-loader:/nvme-artifacts "
            f"spark-vllm:0.29.0-nvme1 -m spark_nvme.identity /nvme-artifacts/fixtures/{name} "
            f"/nvme-artifacts/sources/{name}.json")
        subprocess.run(["ssh", HOST, command], check=True)
        run(name, "prepare")
    for name in ("tiny-gpt2", "tiny-llama"):
        run(name, "restore")
        native = json.loads((OUT / (name + "-prepare.json")).read_bytes())
        prepared = json.loads((OUT / (name + "-restore.json")).read_bytes())
        assert native == prepared, name + " token/logprob parity failure"
        print(name, "exact token and logprob parity passed", flush=True)
