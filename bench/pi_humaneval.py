#!/usr/bin/env python3
"""HumanEval file tasks through real pi RPC sessions, with bounded concurrency."""
import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'bench/pi_throughput'
BASE = 'http://spark-06c4.local:8000'
INSTRUCTION = ('Create a file named solution.py in the current directory containing a complete, '
               'working implementation of the following. Keep the exact function name and '
               'signature. Do not create any other files.\n\n')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def snapshot():
    with urllib.request.urlopen(BASE + '/metrics', timeout=8) as response:
        body = response.read().decode()
    result = {}
    for line in body.splitlines():
        if not line.startswith('vllm:'):
            continue
        metric, value = line.rsplit(None, 1)
        name = metric.split('{', 1)[0]
        if '_bucket' in name or '_created' in name:
            continue
        result[name] = result.get(name, 0) + float(value)
    return result


async def idle(timeout=30):
    deadline = time.monotonic() + timeout
    while True:
        row = await asyncio.to_thread(snapshot)
        if all(row.get('vllm:num_requests_' + k) == 0 for k in ('running', 'waiting')):
            return row
        if time.monotonic() > deadline:
            raise RuntimeError('Endpoint has other work or has not drained; refusing next cell')
        await asyncio.sleep(1)


def evaluate(problem, solution):
    if not solution.exists():
        return {'passed': False, 'result': 'missing solution.py'}
    code = ('import resource\nresource.setrlimit(resource.RLIMIT_CPU, (3, 3))\n'
            'resource.setrlimit(resource.RLIMIT_AS, (536870912, 536870912))\n'
            + solution.read_text() + '\n' + problem['test']
            + '\ncheck(' + problem['entry_point'] + ')\n')
    command = ['bwrap', '--unshare-all', '--die-with-parent', '--ro-bind', '/usr', '/usr',
               '--ro-bind', '/lib', '/lib', '--ro-bind', '/lib64', '/lib64',
               '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--chdir', '/tmp',
               '/usr/bin/python3', '-I', '-']
    try:
        p = subprocess.run(command, input=code, text=True, capture_output=True, timeout=8)
        return {'passed': p.returncode == 0, 'result': 'passed' if p.returncode == 0 else 'failed',
                'returncode': p.returncode, 'output': (p.stdout + p.stderr)[-4000:]}
    except subprocess.TimeoutExpired:
        return {'passed': False, 'result': 'timed out'}


def request_stats(path):
    requests, current = [], None
    for line in path.read_text().splitlines():
        event = json.loads(line)
        kind, t = event['event'], event['epoch_ms'] / 1000
        if kind == 'request':
            if current is not None:
                raise RuntimeError('Uncompleted provider request before another request')
            current = {'start': t, 'payload': event, 'delta_chars': {}, 'first_delta': None,
                       'first_text': None, 'last_delta': None, 'statuses': []}
        elif current is not None:
            if kind == 'response':
                current['statuses'].append(event['status'])
            elif kind == 'delta':
                current['first_delta'] = current['first_delta'] or t
                current['last_delta'] = t
                if event['kind'] != 'thinking_delta':
                    current['first_text'] = current['first_text'] or t
                chars = current['delta_chars']
                chars[event['kind']] = chars.get(event['kind'], 0) + event['chars']
            elif kind == 'completion':
                current.update(end=t, usage=event['usage'], stop_reason=event['stop_reason'],
                               error=event.get('error'))
                current['seconds'] = t - current['start']
                current['ttft_s'] = (current['first_delta'] - current['start']
                                     if current['first_delta'] is not None else None)
                current['post_first_s'] = (t - current['first_delta']
                                          if current['first_delta'] is not None else 0)
                requests.append(current)
                current = None
    if current:
        raise RuntimeError('Incomplete request observation')
    return requests


async def solve(problem, folder, evidence, config, barrier=None):
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise RuntimeError('Task workspace must be empty: ' + str(folder))
    evidence.mkdir(parents=True)
    observation = evidence / 'observations.jsonl'
    env = {k: v for k, v in os.environ.items() if not k.startswith(('HERDR_', 'PI_SPEC_'))}
    env.update(PI_BENCH_OBSERVATIONS=str(observation), PI_CODING_AGENT_DIR=str(config),
               PI_OFFLINE='1', PI_TELEMETRY='0')
    command = ['pi', '--mode', 'rpc', '--no-session', '--offline', '--provider', 'glm53',
               '--model', 'glm-5.3', '--thinking', os.environ.get('PI_HE_THINKING', 'high'), '--no-extensions',
               '--extension', str(ASSETS / 'observe.ts'), '--no-skills', '--no-prompt-templates',
               '--no-themes', '--no-context-files', '--tools', 'read,bash,edit,write']
    stderr = (evidence / 'stderr.log').open('w')
    events = (evidence / 'events.jsonl').open('w')
    proc = await asyncio.create_subprocess_exec(*command, cwd=folder, env=env,
               stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
               stderr=stderr, limit=16 * 1024 * 1024, start_new_session=True)
    async def send(row):
        proc.stdin.write((json.dumps(row) + '\n').encode())
        await proc.stdin.drain()
    async def read():
        line = await proc.stdout.readline()
        if not line:
            raise RuntimeError(f'pi exited early: {proc.returncode}; see {evidence}')
        event = json.loads(line)
        events.write(json.dumps({'received_at': time.time(), **event}) + '\n')
        return event
    start, end, errors = None, None, []
    try:
        await send({'id': 'ready', 'type': 'get_state'})
        async with asyncio.timeout(45):
            while True:
                event = await read()
                if event.get('id') == 'ready':
                    if not event.get('success'):
                        raise RuntimeError('pi readiness failed')
                    write_json(evidence / 'pi-state.json', event)
                    break
        if barrier:
            await barrier.wait()
        prompt = INSTRUCTION + problem['prompt']
        (evidence / 'prompt.txt').write_text(prompt)
        start = time.time()
        await send({'id': 'task', 'type': 'prompt', 'message': prompt})
        async with asyncio.timeout(float(os.environ.get('PI_HE_TASK_TIMEOUT', '600'))):
            while True:
                event = await read()
                if event['type'] == 'response' and not event.get('success'):
                    errors.append(event)
                if event['type'] in ('auto_retry_start', 'auto_compaction_start', 'extension_error'):
                    errors.append(event)
                if event['type'] == 'agent_settled':
                    end = time.time()
                    break
    finally:
        if barrier and start is None:
            await barrier.abort()
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), 8)
            except asyncio.TimeoutError:
                import signal
                os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()
        stderr.close()
        events.close()
    requests = request_stats(observation)
    if not requests:
        raise RuntimeError('No native pi requests recorded')
    for row in requests:
        if row['stop_reason'] in ('error', 'aborted') or row['error']:
            errors.append({'stop_reason': row['stop_reason'], 'error': row['error']})
        if row['usage']['output'] <= 0:
            errors.append({'missing_usage': row})
    solution = folder / 'solution.py'
    if solution.exists():
        shutil.copyfile(solution, evidence / 'solution.py')
    result = {'task_id': problem['task_id'], 'start': start, 'end': end,
              'seconds': end-start, 'requests': requests, 'errors': errors,
              'truncated_calls': sum(r['stop_reason']=='length' for r in requests),
              'output_tokens': sum(r['usage']['output'] for r in requests),
              'reasoning_tokens': sum(r['usage'].get('reasoning', 0) for r in requests),
              'input_tokens': sum(r['usage']['input'] + r['usage'].get('cacheRead', 0) for r in requests),
              'request_seconds': sum(r['seconds'] for r in requests),
              'post_first_seconds': sum(r['post_first_s'] for r in requests)}
    result['check'] = await asyncio.to_thread(evaluate, problem, solution)
    write_json(evidence / 'result.json', result)
    return result


def summarize(rows, c, lane, before, after, samples):
    elapsed = max(r['end'] for r in rows) - min(r['start'] for r in rows)
    output = sum(r['output_tokens'] for r in rows)
    request_s = sum(r['request_seconds'] for r in rows)
    post_first_s = sum(r['post_first_seconds'] for r in rows)
    d = {k: after.get(k, 0) - v for k, v in before.items()}
    backend_decode = d.get('vllm:request_decode_time_seconds_sum', 0)
    ttfts = [r['ttft_s'] for row in rows for r in row['requests'] if r['ttft_s'] is not None]
    calls = sum(len(row['requests']) for row in rows)
    flags = []
    if any(row['errors'] for row in rows): flags.append('pi errors/retries/truncation')
    if d.get('vllm:request_success_total') != calls: flags.append('server/client request count mismatch')
    if abs(d.get('vllm:generation_tokens_total', 0) - output) > max(2*calls, output*.01):
        flags.append('server/client output count mismatch')
    return {'lane': lane, 'concurrency': c, 'tasks': len(rows),
            'passed': sum(r['check']['passed'] for r in rows), 'errors': sum(bool(r['errors']) for r in rows),
            'truncated_calls':sum(r.get('truncated_calls',0) for r in rows),
            'output_tokens': output, 'reasoning_tokens': sum(r['reasoning_tokens'] for r in rows),
            'wall_s': elapsed, 'aggregate_tok_s': output / elapsed,
            'request_tok_s': output / request_s, 'post_first_tok_s': output / post_first_s,
            'server_decode_tok_s': d.get('vllm:generation_tokens_total', 0) / backend_decode if backend_decode else None,
            'median_task_s': statistics.median(r['seconds'] for r in rows),
            'median_ttft_s': statistics.median(ttfts) if ttfts else None,
            'max_running': max((s.get('vllm:num_requests_running', 0) for s in samples), default=0),
            'mean_running': statistics.mean(s.get('vllm:num_requests_running', 0) for s in samples) if samples else 0,
            'api_calls': calls, 'metric_delta': d, 'validity_flags': flags}


async def run_cell(problems, c, lane, out, work, config, check=lambda: None):
    before = await idle()
    cell = out / lane / f'c{c:02d}'
    cell.mkdir(parents=True)
    queue = asyncio.Queue()
    for p in problems: queue.put_nowait(p)
    barrier = asyncio.Barrier(min(c, len(problems)))
    rows, samples = [], []
    done = asyncio.Event()
    async def monitor():
        with (cell / 'metrics.jsonl').open('w') as file:
            while not done.is_set():
                check()
                row = await asyncio.to_thread(snapshot)
                samples.append(row)
                file.write(json.dumps({'time': time.time(), **row}) + '\n')
                file.flush()
                try: await asyncio.wait_for(done.wait(), 1)
                except asyncio.TimeoutError: pass
    async def worker():
        first = True
        while not queue.empty():
            p = queue.get_nowait()
            tid = p['task_id'].replace('/', '_')
            folder = work / f'c{c:02d}' / tid
            if folder.exists():
                # Only this run's generated task directory, owned by this runner.
                shutil.rmtree(folder)
            row = await solve(p, folder, cell / tid, config, barrier if first else None)
            first = False
            rows.append(row)
            print(f"{lane} C{c} {p['task_id']} {row['output_tokens']} tokens {row['seconds']:.1f}s {row['check']['result']}", flush=True)
    try:
        async with asyncio.TaskGroup() as group:
            watcher = group.create_task(monitor())
            workers = [group.create_task(worker()) for _ in range(min(c, len(problems)))]
            await asyncio.gather(*workers)
            done.set()
            await watcher
    finally:
        done.set()
    # vLLM exports completed-request counters on its periodic stats tick.
    await asyncio.sleep(6)
    after = await idle()
    result = summarize(rows, c, lane, before, after, samples)
    write_json(cell / 'summary.json', result)
    print('CELL ' + json.dumps({k: v for k, v in result.items() if k != 'metric_delta'}), flush=True)
    if result['validity_flags']:
        raise RuntimeError('Invalid benchmark cell: ' + str(result['validity_flags']))
    return result


def prepare_config(directory):
    source = Path.home() / '.pi/agent'
    directory.mkdir(mode=0o700)
    for name in ('models.json', 'settings.json', 'SYSTEM.md'):
        shutil.copyfile(source / name, directory / name)
        (directory / name).chmod(0o600)
    (directory / 'auth.json').write_text('{}\n')
    # Only the tested local provider is needed; keep its complete native settings.
    models = json.loads((directory / 'models.json').read_text())
    models['providers'] = {'glm53': models['providers']['glm53']}
    if os.environ.get('PI_HE_MAX_TOKENS'):
        for model in models['providers']['glm53']['models']:
            if model['id'] == 'glm-5.3': model['maxTokens'] = int(os.environ['PI_HE_MAX_TOKENS'])
    if os.environ.get('PI_HE_BASE_URL'):
        # Route the snapshot through a local recording proxy (capture_proxy.py);
        # server counters are still read from the real endpoint.
        models['providers']['glm53']['baseUrl'] = os.environ['PI_HE_BASE_URL']
    write_json(directory / 'models.json', models)
    settings = json.loads((directory / 'settings.json').read_text())
    settings['packages'] = []
    write_json(directory / 'settings.json', settings)
    return {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
            for name in ('models.json', 'settings.json', 'SYSTEM.md')}


async def run_sweep(lane, out, work, config, levels, check=lambda: None, limit=None):
    problems = json.loads((ASSETS / 'humaneval12.json').read_text())['tasks']
    if limit: problems = problems[:limit]
    # Identical, predeclared task order on both stacks for each concurrency.
    results = []
    for c in levels:
        saved = out / lane / f'c{c:02d}' / 'summary.json'
        if saved.exists():
            result = json.loads(saved.read_text())
            if result['validity_flags'] == ['pi errors/retries/truncation']:
                prior = [json.loads(p.read_text()) for p in saved.parent.glob('HumanEval_*/result.json')]
                errors = [e for r in prior for e in r['errors']]
                if (len(prior)==len(problems) and errors and
                        all(e.get('stop_reason')=='length' and not e.get('error') for e in errors)):
                    # A native token-limit completion has valid usage and is a
                    # model outcome, not broken measurement. Preserve originals.
                    write_json(saved.with_name('summary-before-length-classification.json'), result)
                    result.update(validity_flags=[], errors=0, truncated_calls=len(errors),
                                  classification_note='Native output-limit completions retained; original pi errors and summary preserved. Functional checks unchanged.')
                    write_json(saved, result)
            if result['tasks'] != len(problems) or result['validity_flags']:
                raise RuntimeError('Cannot resume from an invalid or different task set')
            results.append(result)
            print(f'Reusing completed {lane} C{c}', flush=True)
            continue
        order = problems.copy()
        random.Random(9000 + c).shuffle(order)
        result = await run_cell(order, c, lane, out, work, config, check)
        results.append(result)
        write_json(out / f'{lane}-summary.json', results)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--lane', default='deployed')
    ap.add_argument('--levels', default=','.join(map(str, range(1,13))))
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='pi-he-') as tmp:
        config = Path(tmp) / 'config'
        hashes = prepare_config(config)
        write_json(args.out / 'config-hashes.json', hashes)
        asyncio.run(run_sweep(args.lane, args.out, Path(tmp)/'work', config,
                             [int(x) for x in args.levels.split(',')], limit=args.limit))


if __name__ == '__main__': main()
