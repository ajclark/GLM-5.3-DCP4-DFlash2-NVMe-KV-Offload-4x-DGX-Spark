"""Read-only four-node memory guard; no CUDA imports, allocator probes or NIC changes."""
import json
from collections import deque
import shlex
import subprocess
import threading
import time
from pathlib import Path

HOSTS = ("spark-06c4.local", "spark-365c.local", "spark-ddbf.local", "spark-a218.local")
NODE_READER = '''
import json, os, time
from pathlib import Path
boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
page_mib = os.sysconf('SC_PAGE_SIZE') / 1048576
while True:
    mem = {s.split()[0].rstrip(':'): int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
    vm = {s.split()[0]: int(s.split()[1]) for s in Path('/proc/vmstat').read_text().splitlines()}
    psi = {s.split()[0]: {k:float(v) for k,v in (p.split('=') for p in s.split()[1:])} for s in Path('/proc/pressure/memory').read_text().splitlines()}
    print(json.dumps(dict(time=time.time(), boot=boot, available_mib=mem['MemAvailable']/1024,
        swap_used_mib=(mem['SwapTotal']-mem['SwapFree'])/1024,
        swap_in_mib=vm['pswpin']*page_mib, swap_out_mib=vm['pswpout']*page_mib,
        oom_kill=vm.get('oom_kill',0), full=psi['full'], some=psi['some'])), flush=True)
    time.sleep(2)
'''


def pressure_reason(current, previous=None, minimum_mib=512, loading=False):
    if current["available_mib"] < minimum_mib:
        return f"available memory {current['available_mib']:.0f} MiB < {minimum_mib} MiB"
    # Loading a 377 GiB checkpoint reclaims file cache and can briefly stall
    # with >15 GiB available. PSI alone is not evidence of imminent exhaustion.
    if current["available_mib"] < 2048 and current["full"]["avg10"] > (20 if loading else 10):
        return "sustained full memory stalls with low headroom"
    if previous:
        if current["boot"] != previous["boot"]:
            return "node rebooted"
        if current["oom_kill"] > previous["oom_kill"]:
            return "OOM counter increased"
        elapsed_us = max(1, (current["time"]-previous["time"])*1e6)
        if current["available_mib"] < 1024:
            if (current["swap_out_mib"]-previous["swap_out_mib"])/(elapsed_us/1e6) > 256:
                return "rapid swapping with less than 1 GiB headroom"
            if (current["full"]["total"]-previous["full"]["total"])/elapsed_us > 0.5:
                return "major full-stall interval with less than 1 GiB headroom"
    return None


class PressureTracker:
    def __init__(self, minimum_mib=512):
        self.minimum_mib = minimum_mib
        self.history = deque(maxlen=6)

    def observe(self, row, loading=False):
        previous = self.history[-1] if self.history else None
        reason = pressure_reason(row, previous, self.minimum_mib, loading)
        self.history.append(row)
        if len(self.history) == 6 and row['available_mib'] < 2048:
            first = self.history[0]
            if row['swap_out_mib']-first['swap_out_mib'] > 512 and row['full']['avg10'] > 2:
                return 'sustained swapping and stalls with low headroom'
        return reason


class MemoryGuard:
    def __init__(self, path, minimum_mib=512, loading=False):
        self.path = Path(path)
        self.minimum_mib = minimum_mib
        self.loading = loading
        self.trackers = {h:PressureTracker(minimum_mib) for h in HOSTS}
        self.latest, self.received, self.previous = {}, {}, {}
        self.reason = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.processes, self.threads = [], []
        self.out = self.path.open("a", buffering=1)

    def start(self):
        for host in HOSTS:
            cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host,
                   "python3 -u -c " + shlex.quote(NODE_READER)]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.processes.append(proc)
            thread = threading.Thread(target=self._read, args=(host,proc), daemon=True)
            thread.start()
            self.threads.append(thread)
        return self

    def _read(self, host, proc):
        try:
            for line in proc.stdout:
                row = json.loads(line)
                with self.lock:
                    why = self.trackers[host].observe(row, self.loading)
                    if why and not self.reason:
                        self.reason = f"{host}: {why}"
                    self.latest[host], self.received[host] = row, time.monotonic()
                    self.out.write(json.dumps({"host":host, **row, "reason":why})+"\n")
            if not self.stop.is_set():
                self.reason = f"{host}: memory monitor disconnected: {proc.stderr.read()[:300]}"
        except Exception as exc:
            self.reason = f"{host}: memory monitor failed: {exc}"

    def check(self):
        with self.lock:
            if self.reason:
                raise RuntimeError(self.reason)
            for host in HOSTS:
                if host not in self.latest or time.monotonic()-self.received[host] > 8:
                    raise RuntimeError(f"{host}: missing or stale memory observation")

    def preflight(self, seconds=10, minimum_mib=768):
        deadline = time.monotonic()+10
        while len(self.latest) < 4 and time.monotonic() < deadline and not self.reason:
            time.sleep(0.1)
        self.check()
        with self.lock:
            before = self.latest.copy()
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            self.check()
            time.sleep(0.2)
        with self.lock:
            for host,row in self.latest.items():
                why = pressure_reason(row, before[host], minimum_mib, self.loading)
                if why:
                    raise RuntimeError(f"preflight {host}: {why}")
                if not self.loading and (row['swap_out_mib']-before[host]['swap_out_mib'] > 64 or row['full']['avg10'] > 2):
                    raise RuntimeError(f"preflight {host}: active swapping")
            return {h:round(r['available_mib']) for h,r in self.latest.items()}

    def close(self):
        self.stop.set()
        for proc in self.processes:
            proc.terminate()
        for proc in self.processes:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for thread in self.threads:
            thread.join(timeout=3)
        self.out.close()
