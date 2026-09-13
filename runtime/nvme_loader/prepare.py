#!/usr/bin/env python3
"""Hash immutable canonical checkpoints once, on each worker's local NVMe."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from build import HOSTS, IMAGE

MODELS = {"glm-5.3": "GLM-5.3-Int4-Int8Mix", "dflash2-draft": "GLM-5.3-DFlash2-draft"}


def prepare(host):
    identities = {}
    for name, folder in MODELS.items():
        command = ("mkdir -p /var/tmp/nvme-loader/sources && docker run --rm --memory=1536m "
            "--entrypoint python3 -v /var/tmp/nvme-loader:/nvme-artifacts "
            f"-v /var/tmp/models/{folder}:/models/{name}:ro "
            f"{IMAGE} -m spark_nvme.identity /models/{name} /nvme-artifacts/sources/{name}.json")
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", host, command],
                                capture_output=True, text=True, timeout=1800)
        if result.returncode:
            raise RuntimeError(host + result.stderr)
        identities[name] = result.stdout.strip().splitlines()[-1]
        print(host, name, identities[name], flush=True)
    return identities


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        identities = dict(zip(HOSTS, pool.map(prepare, HOSTS)))
    if any(v != identities[HOSTS[0]] for v in identities.values()):
        raise RuntimeError("canonical checkpoint bytes differ across Sparks")
    out = Path(__file__).resolve().parents[2] / "results/nvme-loader/source-identities.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(identities, indent=2) + "\n")
