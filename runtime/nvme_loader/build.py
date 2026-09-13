#!/usr/bin/env python3
"""Stage a derivative image on all Sparks; no serving/GPU changes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess

HOSTS = ("spark-06c4.local", "spark-365c.local", "spark-ddbf.local", "spark-a218.local")
ROOT = Path(__file__).resolve().parent
IMAGE = os.getenv("NVME_IMAGE", "spark-vllm:0.29.0-nvme4")


def build(host):
    subprocess.run(["rsync", "-az", "--exclude=__pycache__", "--exclude=*.egg-info", str(ROOT) + "/",
                    host + ":spark-nvme-build/"], check=True)
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", host,
        "docker build -t " + IMAGE + " ~/spark-nvme-build"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(host + "\n" + result.stdout + result.stderr)
    print(host + " image ready", flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(build, HOSTS))
