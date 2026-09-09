"""Guarded fixed-work Spark frequency/governor screen with exact restoration.

One experimental deployment must already be held by spec_experiment.py. This
controller adds a separate clock/governor watchdog; no model is allocated.
Run spec_power.py alongside for NVIDIA-reported device energy, not wall power.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import threading
import time

from adaptive_spec import generate, idle_check, request_body
from spec_experiment import BASE, HOSTS, parallel, ssh
from spec_memory import MemoryGuard
from spec_power_node import PROFILES

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--experiment', required=True)
    ap.add_argument('--profiles', default='baseline,gpu1800,baseline,gpu1600,baseline,schedutil,baseline')
    ap.add_argument('--cases', default='code_cache,prose_heat')
    ap.add_argument('--corpus', type=Path, default=ROOT / 'bench/spec-development.json')
    ap.add_argument('--idle-seconds', type=int, default=60)
    ap.add_argument('--idle-only', action='store_true')
    ap.add_argument('--tokens', type=int, default=256)
    args = ap.parse_args()
    profiles = args.profiles.split(',')
    corpus = json.loads(args.corpus.read_text())
    cases = args.cases.split(',')
    if any(p not in PROFILES for p in profiles) or any(c not in corpus for c in cases):
        ap.error('unknown profile or corpus case')
    if not 30 <= args.idle_seconds <= 180 or not 64 <= args.tokens <= 512:
        ap.error('idle seconds must be 30..180, tokens 64..512')
    if (len(profiles) > 12 or len(cases) > 6
            or not all(re.fullmatch(r'[a-z0-9-]{1,48}', x) for x in (args.experiment, args.out.name))):
        ap.error('bounded profile/case counts and safe experiment/output names required')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'declaration.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')
    remote_script = f'glm-spec/{args.experiment}/power-node.py'
    remote_root = f'glm-spec/{args.experiment}/power-{args.out.name}'
    prepared, errors = [], []
    stop = threading.Event()
    guard = MemoryGuard(args.out / 'memory.jsonl').start()

    def rpc(host, action, profile=None):
        words = ['sudo', '-n', 'python3', remote_script, action, remote_root]
        if profile: words += ['--profile', profile]
        return ssh(host, shlex.join(words), timeout=45)

    def heartbeat():
        while not stop.wait(10):
            try:
                status = parallel(lambda h: json.loads(rpc(h, 'heartbeat')))
                with (args.out / 'heartbeat.jsonl').open('a') as out:
                    out.write(json.dumps({'time': time.time(), 'nodes': status}) + '\n')
            except Exception as exc:
                errors.append(str(exc))
                stop.set()

    def check():
        guard.check()
        if errors: raise RuntimeError('; '.join(errors))

    monitor = threading.Thread(target=heartbeat, daemon=True)
    try:
        print('preflight:', guard.preflight(), flush=True)
        idle_check(BASE)
        def initialize(host):
            code = ('import json,subprocess;d=json.loads(subprocess.check_output('
                    '["docker","inspect","vllm_glm53big"]))[0];'
                    'print((d["Config"].get("Labels") or {}).get("glm.spec.experiment",""))')
            if ssh(host, shlex.join(['python3', '-c', code])).strip() != args.experiment:
                raise RuntimeError('unexpected deployment on ' + host)
            ssh(host, shlex.join(['tee', remote_script]), (ROOT / 'bench/spec_power_node.py').read_bytes())
            result = json.loads(rpc(host, 'initialize'))
            prepared.append(host)
            return result
        snapshots = parallel(initialize)
        (args.out / 'original-settings.json').write_text(json.dumps(snapshots, indent=2) + '\n')
        monitor.start()
        for index, profile in enumerate(profiles):
            check()
            idle_check(BASE)
            guard.preflight(seconds=4)
            started = time.time()
            states = parallel(lambda h: json.loads(rpc(h, 'profile', profile)))
            print('profile', index, profile, states, flush=True)
            settled = time.time() + args.idle_seconds
            while time.time() < settled:
                check()
                idle_check(BASE)
                stop.wait(min(2, max(0, settled - time.time())))
            with (args.out / 'idle-windows.jsonl').open('a') as out:
                out.write(json.dumps({'index': index, 'profile': profile,
                                      'started_at': started, 'settled_start': started + 20,
                                      'ended_at': time.time()}) + '\n')
            if args.idle_only:
                continue
            for case in cases:
                check()
                idle_check(BASE)
                guard.preflight(seconds=4)
                label = f'{args.out.name}-{index}-{profile}-{case}'
                result = generate(BASE, request_body(corpus[case], 7, label, args.tokens), guard)
                check()
                result.update(label=label, profile=profile, profile_index=index, case=case, cap=7, repeat=index, policy='fixed')
                (args.out / (label + '.json')).write_text(json.dumps(result, indent=2) + '\n')
                print(json.dumps({k: result[k] for k in ('label', 'ttft', 'decode_tps', 'token_sha256')}), flush=True)
    finally:
        stop.set()
        if monitor.is_alive(): monitor.join(timeout=30)
        # Restore each initialized node even when another node failed admission.
        restored = {}
        for host in prepared:
            try: restored[host] = rpc(host, 'restore')
            except Exception as exc: restored[host] = {'error': str(exc), 'watchdog_remains_armed': True}
        (args.out / 'restored.json').write_text(json.dumps(restored, indent=2) + '\n')
        guard.close()
        if any(isinstance(v, dict) for v in restored.values()):
            raise RuntimeError('clock/governor restoration requires checking restored.json')
        print('original clock/governor settings restored on', len(restored), 'nodes', flush=True)


if __name__ == '__main__':
    main()
