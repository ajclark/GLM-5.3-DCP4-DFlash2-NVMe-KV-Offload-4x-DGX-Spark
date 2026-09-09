#!/usr/bin/env python3
"""Read a named pi agent's herdr state and pane every 60 seconds, with a deadline."""
import argparse
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path


def call(*args):
    result = subprocess.run(['herdr', 'agent', *args], capture_output=True, text=True, timeout=15)
    try:
        payload = json.loads(result.stdout if result.returncode == 0 else result.stderr)
    except json.JSONDecodeError:
        payload = {'text': (result.stdout + result.stderr)[-16000:]}
    return {'returncode': result.returncode, 'response': payload}


def assess(state, screen):
    if state['returncode'] or screen['returncode']:
        return {'status': 'watch_error', 'assessment': 'monitor command failed'}
    info = state['response'].get('result', {}).get('agent', {})
    status = info.get('agent_status', 'unknown')
    pane = json.dumps(screen['response'])
    error = bool(re.search(r'(?i)(API error|request failed|connection refused|context.*exceed|rate limit)', pane))
    blocked = bool(re.search(r'(?i)(waiting for (?:approval|input|permission)|permission required|approval required|blocked on)', pane))
    assessment = ('error_candidate' if error or status in ('error', 'errored', 'failed')
                  else 'blocked_candidate' if blocked or status in ('blocked', 'needs_input', 'waiting')
                  else 'done_or_idle' if status == 'idle' else status)
    return {'status': status, 'assessment': assessment,
            'error_candidate': error, 'blocked_candidate': blocked}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--agent', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seconds', type=int, default=7200)
    ap.add_argument('--interval', type=int, default=60)
    args = ap.parse_args()
    if os.environ.get('HERDR_ENV') != '1':
        ap.error('must run inside a herdr-managed pane')
    if not 1 <= args.seconds <= 43200 or not 10 <= args.interval <= 60:
        ap.error('duration must be 1..43200 seconds; interval 10..60 seconds')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *unused: stop.set())
    signal.signal(signal.SIGINT, lambda *unused: stop.set())
    deadline = time.monotonic() + args.seconds
    with args.out.open('x', buffering=1) as stream:
        while not stop.is_set() and time.monotonic() < deadline:
            started = time.monotonic()
            try:
                state = call('get', args.agent)
                screen = call('read', args.agent, '--source', 'recent-unwrapped', '--lines', '80')
                # A textual error is a review candidate, never an automatic approval.
                row = {'time': time.time(), 'agent': args.agent, **assess(state, screen),
                       'state': state, 'pane': screen}
            except Exception as exc:
                row = {'time': time.time(), 'agent': args.agent, 'status': 'watch_error', 'error': str(exc)}
            stream.write(json.dumps(row) + '\n')
            latest = args.out.with_suffix('.latest.json')
            temp = latest.with_suffix('.tmp')
            temp.write_text(json.dumps(row, indent=2) + '\n')
            temp.replace(latest)
            print(json.dumps({k: row[k] for k in ('time', 'agent', 'status', 'assessment') if k in row}), flush=True)
            stop.wait(max(0, min(deadline - time.monotonic(), args.interval - (time.monotonic() - started))))


if __name__ == '__main__':
    main()
