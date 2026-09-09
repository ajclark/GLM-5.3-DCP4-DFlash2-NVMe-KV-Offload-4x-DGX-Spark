#!/usr/bin/env python3
"""Bounded memory monitor that cancels only the named experimental herdr agent."""
import argparse
import json
import subprocess
import time
from pathlib import Path
from spec_memory import MemoryGuard


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--agent', required=True)
    ap.add_argument('--seconds', type=int, default=600)
    args = ap.parse_args()
    if not 1 <= args.seconds <= 7200:
        ap.error('duration must be 1..7200 seconds')
    args.out.mkdir(parents=True, exist_ok=True)
    guard = MemoryGuard(args.out/'memory.jsonl').start()
    deadline = time.monotonic() + args.seconds
    minima = {}
    try:
        initial = guard.preflight()
        (args.out/'ready.json').write_text(json.dumps(initial, indent=2)+'\n')
        print('preflight ready', initial, flush=True)
        while time.monotonic() < deadline and not (args.out/'finish').exists():
            guard.check()
            with guard.lock:
                for host, row in guard.latest.items():
                    minima[host] = min(minima.get(host, float('inf')), row['available_mib'])
            time.sleep(1)
        if time.monotonic() >= deadline:
            subprocess.run(['herdr','agent','send-keys',args.agent,'esc'],capture_output=True,timeout=10)
            (args.out/'deadline').write_text('bounded monitor duration reached; cancellation sent\n')
    except Exception as exc:
        (args.out/'tripped').write_text(str(exc)+'\n')
        subprocess.run(['herdr','agent','send-keys',args.agent,'esc'],capture_output=True,timeout=10)
        raise
    finally:
        guard.close()
        (args.out/'monitor-summary.json').write_text(json.dumps({'minimum_available_mib':minima,
            'finished_at':time.time(),'pressure_trip':(args.out/'tripped').exists()},indent=2)+'\n')


if __name__ == '__main__':
    main()
