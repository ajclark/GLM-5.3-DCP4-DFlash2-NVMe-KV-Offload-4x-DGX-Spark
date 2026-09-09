"""Bounded node-side clock/governor experiment, with independent restoration.

The original 2000 MHz GPU lock is documented in docs/HANDOVER.md; NVML does
not expose its configured bounds here. Admission requires the observed clock
to agree. CPU governor values are read and restored individually.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROFILES = {'baseline': (2000, None), 'gpu2200': (2200, None), 'gpu1800': (1800, None),
            'gpu1600': (1600, None), 'schedutil': (2000, 'schedutil'),
            'idle_auto': (None, 'schedutil')}


def command(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=20).stdout.strip()


def original(root):
    return json.loads((root / 'original.json').read_text())


def cpu_observations():
    """Read-only frequency and idle residency proxies; these are not watts."""
    frequencies, idle = {}, {}
    for path in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq'):
        try:
            frequencies[path.parent.name] = int(path.read_text())
        except (OSError, ValueError):
            continue
    for path in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpuidle/state*'):
        try:
            idle[str(path)] = {'name': (path / 'name').read_text().strip(),
                               'time_us': int((path / 'time').read_text()),
                               'usage': int((path / 'usage').read_text())}
        except (OSError, ValueError):
            continue
    return {'cpu_scaling_cur_freq_khz': frequencies, 'cpu_idle_counters': idle}


def apply_profile(snapshot, profile):
    clock, governor = PROFILES[profile]
    for path, values in snapshot['cpus'].items():
        chosen = governor or values['governor']
        if chosen not in values['available']:
            raise RuntimeError('governor unavailable: ' + chosen)
    if clock is None:
        command('nvidia-smi', '-rgc')
    else:
        command('nvidia-smi', '-lgc', f'{clock},{clock}')
    for path, values in snapshot['cpus'].items():
        (Path(path) / 'scaling_governor').write_text(governor or values['governor'])


def restore(root):
    apply_profile(original(root), 'baseline')
    (root / 'restored').write_text(str(time.time()))
    (root / 'disarm').touch()


def initialize(root):
    if root.exists():
        raise RuntimeError('refusing to replace existing rollback material')
    clock = float(command('nvidia-smi', '--query-gpu=clocks.gr', '--format=csv,noheader,nounits'))
    if not 1950 <= clock <= 2050:
        raise RuntimeError('observed clock differs from documented original 2000 MHz lock')
    cpus = {}
    for path in Path('/sys/devices/system/cpu/cpufreq').glob('policy*'):
        cpus[str(path)] = {'governor': (path / 'scaling_governor').read_text().strip(),
                          'available': (path / 'scaling_available_governors').read_text().split()}
    if not cpus or any('schedutil' not in row['available'] for row in cpus.values()):
        raise RuntimeError('CPU governor interface unavailable')
    root.mkdir(mode=0o700)
    snapshot = {'time': time.time(), 'cpus': cpus, 'original_gpu_lock_mhz': [2000, 2000],
                'observed_graphics_mhz': clock, 'gpu_lock_source': 'docs/HANDOVER.md; observed clock checked'}
    (root / 'original.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    (root / 'heartbeat').touch()
    with (root / 'watchdog.log').open('a') as out:
        subprocess.Popen([sys.executable, __file__, 'watchdog', str(root)],
                         stdout=out, stderr=out, stdin=subprocess.DEVNULL, start_new_session=True)
    return snapshot


def watchdog(root):
    deadline = time.monotonic() + 3600
    while not (root / 'disarm').exists():
        try:
            expired = time.time() - (root / 'heartbeat').stat().st_mtime > 45
            if expired or time.monotonic() > deadline:
                restore(root)
                print('restored after controller deadline', flush=True)
                return
        except Exception as exc:
            print('restore retry:', repr(exc), flush=True)
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('initialize', 'profile', 'heartbeat', 'restore', 'watchdog'))
    ap.add_argument('root', type=Path)
    ap.add_argument('--profile', choices=tuple(PROFILES))
    args = ap.parse_args()
    if os.geteuid() != 0:
        ap.error('node-side power controls require root')
    if args.action == 'initialize':
        print(json.dumps(initialize(args.root)))
    elif args.action == 'watchdog':
        watchdog(args.root)
    elif args.action == 'restore':
        restore(args.root)
        print('restored documented GPU lock and original CPU governors')
    else:
        if (args.root / 'disarm').exists():
            raise RuntimeError('power experiment already disarmed')
        (args.root / 'heartbeat').touch()
        if args.action == 'profile':
            if not args.profile:
                ap.error('profile required')
            apply_profile(original(args.root), args.profile)
            (args.root / 'profile').write_text(args.profile)
        print(json.dumps({'profile': (args.root / 'profile').read_text() if (args.root / 'profile').exists() else 'baseline',
                          'graphics_mhz': command('nvidia-smi', '--query-gpu=clocks.gr', '--format=csv,noheader,nounits'),
                          'governors': sorted({(Path(p) / 'scaling_governor').read_text().strip() for p in original(args.root)['cpus']}),
                          **cpu_observations()}))


if __name__ == '__main__':
    main()
