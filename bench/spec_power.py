#!/usr/bin/env python3
"""Record NVIDIA-reported device power and clocks via NVML's nvidia-smi CLI.

This is NOT whole-node wall power. No CUDA contexts or model imports are made.
Runs in the sandbox; each node runs one lightweight nvidia-smi process.
"""
import argparse
import csv
import json
import signal
import subprocess
import threading
import time
from pathlib import Path

from spec_memory import HOSTS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seconds',type=int,default=3600)
    args = ap.parse_args()
    if not 1 <= args.seconds <= 7200:
        ap.error('duration must be 1..7200 seconds')
    stop, lock = threading.Event(), threading.Lock()
    processes, threads, errors = [], [], []
    signal.signal(signal.SIGTERM,lambda *a:stop.set())
    signal.signal(signal.SIGINT,lambda *a:stop.set())
    with args.out.open('x',buffering=1) as out:
        def read(host,proc):
            try:
                for fields in csv.reader(proc.stdout):
                    if len(fields)!=3:
                        continue
                    def number(s):
                        try:return float(s.strip())
                        except ValueError:return None
                    row = dict(host=host,received_at=time.time(),power_w=number(fields[0]),
                               graphics_mhz=number(fields[1]),temperature_c=number(fields[2]),
                               source='nvidia-smi; not wall power')
                    with lock:
                        out.write(json.dumps(row)+'\n')
                if not stop.is_set():
                    errors.append(host+': '+proc.stderr.read()[:300])
                    stop.set()
            except Exception as exc:
                errors.append(str(exc))
                stop.set()
        try:
            for host in HOSTS:
                cmd = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,
                       'nvidia-smi --query-gpu=power.draw,clocks.gr,temperature.gpu --format=csv,noheader,nounits --loop-ms=2000']
                proc = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                processes.append(proc)
                thread = threading.Thread(target=read,args=(host,proc),daemon=True)
                thread.start()
                threads.append(thread)
            print('sampling NVIDIA-reported power and clocks on all four nodes',flush=True)
            deadline = time.monotonic()+args.seconds
            while not stop.wait(1) and time.monotonic()<deadline:
                pass
        finally:
            stop.set()
            for proc in processes:proc.terminate()
            for proc in processes:
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            for thread in threads:thread.join(timeout=3)
        if errors:
            raise RuntimeError('; '.join(errors))
    print('power sampler stopped',flush=True)


if __name__ == '__main__':
    main()
