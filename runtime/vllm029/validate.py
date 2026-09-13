#!/usr/bin/env python3
"""Validate a successful release rollout, including durable reload after restart."""
import argparse
import json
from pathlib import Path
import sys
import time

from rollout import (ROOT, HOSTS, NAME, MemoryGuard, ssh, parallel, guarded_process,
                     wait_health, smoke, logs, restore)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    args = ap.parse_args()
    out = ROOT/'results/vllm029-upgrade'/args.label
    if not (out/'status.json').exists():
        raise SystemExit('A successful rollout is required before extended validation')
    state = json.loads((out/'original.private.json').read_text())
    def owned(host):
        row = json.loads(ssh(host, ['docker', 'inspect', NAME]))[0]
        if row['Config'].get('Labels', {}).get('spark.vllm.upgrade') != args.label:
            raise RuntimeError('running container does not belong to this rollout: '+host)
        return row['Id']
    ids = parallel(owned)
    tested_image = json.loads(ssh(HOSTS[3], ['docker', 'inspect', NAME]))[0]['Image']
    ext = out/'extended'
    ext.mkdir(exist_ok=True)
    guard = MemoryGuard(ext/'memory.jsonl').start()
    success = False
    try:
        guard.preflight(seconds=3)
        for phase in ('api', 'cold'):
            guarded_process([sys.executable, str(Path(__file__).with_name('service_checks.py')),
                             phase, '--out', str(ext)], ext/(phase+'.log'), guard, 3600)
        # Drain asynchronous stores before shutting the engine down.
        for _ in range(15):
            time.sleep(2)
            guard.check()
        logs(out, 'before-restart')
        print('Restarting the exact v0.29 containers to verify durable disk reload', flush=True)
        guard.loading = True
        parallel(lambda h: ssh(h, ['docker', 'stop', '-t', '30', ids[h]]))
        guarded_process(['ssh', HOSTS[3],
            'docker run --rm --name vllm029-memcheck --gpus all --ipc host '
            '-v /usr/local/cuda-13.0/compute-sanitizer:/opt/compute-sanitizer:ro '
            '-v "$HOME/glm-vllm029-build:/regression:ro" '
            '--entrypoint /opt/compute-sanitizer/compute-sanitizer '+tested_image+
            ' --tool memcheck --error-exitcode 99 python3 /regression/regression.py --cuda'],
            ext/'memcheck.log', guard, 1200)
        guarded_process(['ssh', HOSTS[3],
            'docker run --rm --name vllm029-memcheck --gpus all --ipc host '
            '-v /usr/local/cuda-13.0/compute-sanitizer:/opt/compute-sanitizer:ro '
            '-v "$HOME/glm-vllm029-build:/regression:ro" '
            '--entrypoint /opt/compute-sanitizer/compute-sanitizer '+tested_image+
            ' --tool memcheck --error-exitcode 99 python3 /regression/indexer_regression.py'],
            ext/'indexer-memcheck.log', guard, 1200)
        parallel(lambda h: ssh(h, '$HOME/glm53big/start-flusher.sh'))
        for host in reversed(HOSTS):
            ssh(host, ['docker', 'start', ids[host]])
        wait_health(out, guard)
        guard.loading = False
        parallel(lambda h: ssh(h, "pkill -f '[c]ache_flusher.sh' || true"))
        guarded_process([sys.executable, str(Path(__file__).with_name('service_checks.py')),
                         'reload', '--out', str(ext)], ext/'reload.log', guard, 3600)
        guarded_process([sys.executable, str(ROOT/'bench/repro/probe_indexer_mixed_decode.py'),
                         '--context-tokens', '50000', '--out', str(ext/'mixed50k')],
                        ext/'mixed50k.log', guard, 1200)
        (ext/'final-count100.json').write_text(json.dumps(smoke(), indent=2)+'\n')
        guarded_process([sys.executable, str(Path(__file__).with_name('service_checks.py')),
                         'bench', '--out', str(ext/'bench')], ext/'bench.log', guard, 1200)
        logs(out, 'validated')
        (ext/'status.json').write_text(json.dumps({'ok': True, 'restarted_ids': ids}, indent=2)+'\n')
        success = True
        print('Extended API, long-context and durable restart regressions passed', flush=True)
    finally:
        guard.close()
        if not success:
            try:
                logs(out, 'extended-failed')
            finally:
                try:
                    ssh(HOSTS[3], 'docker rm -f vllm029-memcheck >/dev/null 2>&1 || true')
                finally:
                    restore(out, state)


if __name__ == '__main__':
    main()
