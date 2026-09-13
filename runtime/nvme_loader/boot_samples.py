#!/usr/bin/env python3
"""Low-overhead host counters during a functional boot; stdin EOF stops sampling."""
import json
from pathlib import Path
import re
import select
import sys
import time


def snapshot():
    cpu = {}
    for line in Path("/proc/stat").read_text().splitlines():
        words = line.split()
        if re.fullmatch(r"cpu\d+", words[0]):
            values = list(map(int, words[1:9]))  # exclude guest time already in user/nice
            cpu[words[0]] = {"total": sum(values), "idle": values[3], "iowait": values[4]}
    disk = {}
    for line in Path("/proc/diskstats").read_text().splitlines():
        w = line.split()
        if re.fullmatch(r"nvme\d+n\d+", w[2]):
            disk[w[2]] = {"read_ios": int(w[3]), "read_bytes": int(w[5]) * 512,
                          "write_bytes": int(w[9]) * 512, "inflight": int(w[11]),
                          "io_ms": int(w[12]), "weighted_io_ms": int(w[13])}
    memory = {w[0].rstrip(":"): int(w[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if (w := line.split())[0] in ("MemAvailable:", "MemTotal:", "SwapFree:")}
    net = {}
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        name, values = line.split(":")
        w = values.split()
        if name.strip() != "lo":
            net[name.strip()] = {"rx_bytes": int(w[0]), "tx_bytes": int(w[8])}
    return {"wall_time": time.time(), "monotonic": time.monotonic(),
            "cpu": cpu, "nvme": disk, "memory": memory, "net": net}


if __name__ == "__main__":
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        print(json.dumps(snapshot(), separators=(",", ":")), flush=True)
        ready, _, _ = select.select([sys.stdin], [], [], .5)
        if ready and not sys.stdin.buffer.read(1):
            break
