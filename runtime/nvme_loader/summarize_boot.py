#!/usr/bin/env python3
"""Extract loader receipts and host resource counters from one activation."""
import argparse
import json
from pathlib import Path
import re
import statistics


def resources(path):
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = []
    for old, new in zip(samples, samples[1:]):
        elapsed = new["monotonic"] - old["monotonic"]
        if elapsed <= 0:
            continue
        core_use = []
        for core, after in new["cpu"].items():
            before = old["cpu"].get(core)
            if before is None:
                continue
            total = after["total"] - before["total"]
            idle = after["idle"] + after["iowait"] - before["idle"] - before["iowait"]
            core_use.append((total - idle) / total if total else 0.)
        read_bytes = sum(d["read_bytes"] - old["nvme"].get(n, d)["read_bytes"]
                         for n, d in new["nvme"].items())
        write_bytes = sum(d["write_bytes"] - old["nvme"].get(n, d)["write_bytes"]
                          for n, d in new["nvme"].items())
        devices = {name: {
            "busy_fraction": (d["io_ms"] - old["nvme"].get(name, d)["io_ms"]) / elapsed / 1000,
            "mean_queue_depth": (d["weighted_io_ms"] - old["nvme"].get(name, d)["weighted_io_ms"]) / elapsed / 1000,
            "read_GB_s": (d["read_bytes"] - old["nvme"].get(name, d)["read_bytes"]) / elapsed / 1e9,
        } for name, d in new["nvme"].items()}
        rows.append({"wall_time": new["wall_time"], "seconds": elapsed,
                     "nvme_read_GB_s": read_bytes / elapsed / 1e9,
                     "nvme_write_GB_s": write_bytes / elapsed / 1e9,
                     "nvme_devices": devices,
                     "busy_cores": sum(core_use), "per_core_busy_fraction": core_use,
                     "mem_available_bytes": new["memory"]["MemAvailable"]})
    return {"samples": len(samples), "intervals": rows,
            "mean_busy_cores": statistics.mean(r["busy_cores"] for r in rows) if rows else None,
            "peak_busy_cores": max((r["busy_cores"] for r in rows), default=None),
            "min_mem_available_bytes": min((r["mem_available_bytes"] for r in rows), default=None),
            "gpu_and_dram_utilization": "not sampled; host counters do not establish these rooflines"}


def summarize(output, generation):
    result = json.loads((output / f"{generation}.json").read_text())
    result["ranks"] = {}
    for host in ("spark-06c4.local", "spark-365c.local", "spark-ddbf.local", "spark-a218.local"):
        rows = (output / f"{generation}-{host}.log").read_text().splitlines()
        sources = []
        for line in rows:
            if "NVME_LOAD " in line:
                receipt = json.JSONDecoder().raw_decode(line.split("NVME_LOAD ", 1)[1])[0]
                if receipt.get("error"):
                    raise ValueError(receipt["error"])
                sources.extend(receipt.get("sources", []))
        model = next((line for line in rows if "Model loading took" in line), None)
        engine = next((line for line in rows if "init engine (" in line), None)
        item = {"sources": sources, "model_loading": model, "engine_initialization": engine}
        if model:
            item["model_loading_seconds"] = float(re.search(r"([\d.]+) seconds", model)[1])
        if engine:
            item["engine_initialization_seconds"] = float(re.search(r"took ([\d.]+) s", engine)[1])
        resource_path = output / f"{generation}-{host}-resources.jsonl"
        if resource_path.exists():
            item["resources"] = resources(resource_path)
        result["ranks"][host] = item
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation")
    parser.add_argument("--results", type=Path, default=Path("results/nvme-loader"))
    args = parser.parse_args()
    result = summarize(args.results, args.generation)
    path = args.results / f"{args.generation}-analysis.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)
    print(json.dumps({k: v for k, v in result.items() if k != "ranks"}, indent=2))
    for host, rank in result["ranks"].items():
        print(host, [(s.get("backend"), s.get("seconds"), s.get("bytes")) for s in rank["sources"]])
