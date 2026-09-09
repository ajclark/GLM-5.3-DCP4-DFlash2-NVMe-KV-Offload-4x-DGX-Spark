#!/usr/bin/env python3
"""Node-side bounded speculation experiment operations. Invoked by spec_experiment.py.

The existing container and its bind mounts are retained for exact rollback.
The watchdog only kills a container carrying this experiment's label.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from spec_memory import NODE_READER, PressureTracker

NAME = "vllm_glm53big"


def command(*args, check=True, timeout=60):
    return subprocess.run(args, check=check, capture_output=True, text=True, timeout=timeout)


def inspect(name=NAME):
    result = command("docker", "inspect", name, check=False)
    return json.loads(result.stdout)[0] if result.returncode == 0 else None


def prepare(root, label):
    current = inspect()
    if not current or not current['State']['Running'] or current['State']['OOMKilled']:
        raise RuntimeError("current stack is not healthy/running")
    if (current['Config'].get('Labels') or {}).get('glm.spec.experiment'):
        raise RuntimeError("another speculation experiment is already installed")
    if (root/'original.json').exists():
        raise RuntimeError("experiment already prepared")
    cmd = current['Config']['Cmd']
    def option(name):
        return cmd[cmd.index(name)+1]
    expected = {'--decode-context-parallel-size':'2', '--max-model-len':'180224',
                '--max-num-batched-tokens':'2048', '--kv-cache-memory-bytes':'6000000000',
                '--max-num-seqs':'12', '--tensor-parallel-size':'4'}
    if any(option(k) != v for k,v in expected.items()):
        raise RuntimeError("current lane differs from the tested experiment lane")
    if shutil.disk_usage(root).free < 200_000_000_000:
        raise RuntimeError("less than 200 GB free for isolated experiment cache")
    manifest = json.loads((root/'expected-runtime.json').read_text())
    # Verify the source actually running on each rank before replacing any file.
    for rel, wanted in manifest.items():
        result = command('docker','exec',NAME,'sha256sum',
                         '/usr/local/lib/python3.12/dist-packages/vllm/'+rel)
        if result.stdout.split()[0] != wanted:
            raise RuntimeError('runtime source mismatch: '+rel)
    # All deployed overlays must match the vetted staging tree, including the
    # durable slab implementation (the repo's overlay copy is older than stage).
    mount_rows = []
    for mount in current['Mounts']:
        path = Path(mount['Source'])
        if path.suffix == '.py' and path.is_file():
            mount_rows.append({'source':str(path), 'destination':mount['Destination'],
                               'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (root/'mount-manifest.json').write_text(json.dumps(mount_rows,indent=2)+'\n')
    old_dcp = Path(next(m['Source'] for m in current['Mounts']
                       if m['Destination'].endswith('/v1/core/sched/scheduler.py'))).parent
    shutil.copytree(old_dcp, root/'dcp', ignore=shutil.ignore_patterns('__pycache__'))
    for path in (root/'changes').glob('*.py'):
        shutil.copyfile(path, root/'dcp'/path.name)
    (root/'kvcache').mkdir()
    (root/'original.json').write_text(json.dumps(current))
    (root/'original.json').chmod(0o600)
    print(json.dumps({'prepared':label, 'image':current['Image'], 'mounts':mount_rows}))


def original(root):
    return json.loads((root/'original.json').read_text())


def reuse_cache(root, label, source_label):
    """Reuse only a stopped, identical experiment's cache for durability checks."""
    if not re.fullmatch(r'[a-z0-9-]{1,48}',source_label or '') or source_label==label:
        raise RuntimeError('invalid source experiment label')
    source = root.parent/source_label
    current,previous = original(root),original(source)
    if current['Id']!=previous['Id'] or current['Image']!=previous['Image'] or current['Config']['Cmd']!=previous['Config']['Cmd']:
        raise RuntimeError('cache reuse requires the identical original lane and image')
    def hashes(directory):
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.py')}
    def mounted_hashes(directory):
        return {r['destination']:r['sha256'] for r in json.loads((directory/'mount-manifest.json').read_text())}
    if (hashes(root/'dcp')!=hashes(source/'dcp') or not hashes(root/'dcp')
            or mounted_hashes(root)!=mounted_hashes(source)
            or (root/'launch.sh').read_bytes()!=(source/'launch.sh').read_bytes()
            or (root/'expected-runtime.json').read_bytes()!=(source/'expected-runtime.json').read_bytes()):
        raise RuntimeError('cache reuse source or kernel identity mismatch')
    old = inspect(NAME+'_finished_'+source_label)
    if (not old or old['State']['Running'] or old['State'].get('OOMKilled')
            or (old['Config'].get('Labels') or {}).get('glm.spec.experiment')!=source_label
            or not (source/'disarm').exists() or (source/'tripped').exists()):
        raise RuntimeError('source experiment must be stopped, disarmed, and untripped')
    try:
        old_cmd=old['Config']['Cmd']
        previous_salt=json.loads(old_cmd[old_cmd.index('--kv-transfer-config')+1])['kv_connector_extra_config']['slab_salt']
        kernels=Path(next(m['Source'] for m in current['Mounts']
                          if m['Destination'].endswith('/sparse_mla_kernels.py'))).parent
        digest=hashlib.sha256()
        # Match the launcher's complete Python-source salt, including kernel
        # files which are present in the directory but not individually mounted.
        for path in sorted((root/'dcp').glob('*.py'))+sorted(kernels.glob('*.py')):
            digest.update(path.read_bytes())
        if digest.hexdigest()[:16]!=previous_salt:
            raise ValueError
    except (KeyError,ValueError,StopIteration,IndexError):
        raise RuntimeError('cache reuse must preserve the exact deployed slab salt') from None
    cache = source/'kvcache'
    mounted = next((m['Source'] for m in old['Mounts'] if m['Destination']=='/kvcache'),None)
    if cache.is_symlink() or not cache.is_dir() or mounted is None or Path(mounted).resolve()!=cache.resolve():
        raise RuntimeError('source cache mount is not the expected isolated directory')
    destination = root/'kvcache'
    if destination.is_symlink() or any(destination.iterdir()):
        raise RuntimeError('destination experiment cache must be empty')
    destination.rmdir()  # Empty directory only; no cache file is removed.
    destination.symlink_to(cache.resolve(),target_is_directory=True)
    (root/'cache-reuse.json').write_text(json.dumps({'source_label':source_label,
        'source_cache':str(cache.resolve()),'dcp_sha256':hashes(root/'dcp')},indent=2)+'\n')
    print('identical-source isolated cache attached from '+source_label)


def stop(root, label):
    old = original(root)
    current = inspect()
    if not current or current['Id'] != old['Id']:
        raise RuntimeError("current container changed since preparation")
    # Container stays on disk and retains the original image, args and mounts.
    command('docker','rename',NAME,NAME+'_backup_'+label)
    command('docker','stop','-t','10',old['Id'])
    if inspect(old['Id'])['State']['Running']:
        raise RuntimeError("old container still running")
    result = command(str(Path.home()/'glm53big/start-flusher.sh'))
    if 'STARTED' not in result.stdout:
        raise RuntimeError('boot cache flusher failed')
    (root/'heartbeat').touch()
    with (root/'watchdog.log').open('a') as out:
        subprocess.Popen([sys.executable, str(root/'spec_node.py'), 'watchdog', label],
                         stdout=out, stderr=out, stdin=subprocess.DEVNULL, start_new_session=True)
    print('stopped original and armed watchdog')


def launch(root, label, rank):
    if (root/'tripped').exists():
        raise RuntimeError((root/'tripped').read_text())
    old = original(root)
    if inspect(old['Id'])['State']['Running']:
        raise RuntimeError('refusing to overlap model allocations')
    if inspect():
        raise RuntimeError('container name occupied')
    env = os.environ.copy()
    env.update(DCP_IMAGE=old['Image'], DCP_DIR=str(root/'dcp'),
               KVTIER_DIR=str(root/'kvcache'), DCP_SIZE='2', MAXLEN='180224',
               KVBYTES='6000000000', MAXBATCHED='2048', MAXSEQS='12',
               NCCL_HOTPLUG='0', GLM_SPEC_POLICY='shadow',
               GLM_SPEC_TRACE='' if (root/'trace-disabled').exists() else '/kvcache/spec-trace.jsonl',
               GLM_SPEC_EXPERIMENT=label)
    env['GLM_SPEC_CONFIDENCE_TRACE'] = '1' if (root/'confidence-trace-enabled').exists() else '0'
    if (root/'kvcache/boot-costs.json').exists():
        env.update(GLM_SPEC_POLICY='adaptive', GLM_SPEC_COSTS='/kvcache/boot-costs.json')
    if (root/'kvcache/hint-priors.json').exists():
        env['GLM_SPEC_HINT_PRIORS'] = '/kvcache/hint-priors.json'
    with (root/'launch.log').open('a') as out:
        result = subprocess.run(['bash',str(root/'launch.sh'),str(rank),'dflash'],env=env,
                                stdout=out,stderr=out,timeout=60)
    if result.returncode:
        raise RuntimeError('launch failed; inspect launch.log')
    print('launched rank '+str(rank))


def watchdog(root, label):
    proc = subprocess.Popen([sys.executable,'-u','-c',NODE_READER],stdout=subprocess.PIPE,text=True)
    tracker = PressureTracker()
    try:
        for line in proc.stdout:
            if (root/'disarm').exists():
                break
            row = json.loads(line)
            reason = tracker.observe(row, loading=not (root/'settled').exists())
            if time.time()-(root/'heartbeat').stat().st_mtime > 60:
                reason = 'experiment controller heartbeat expired'
            print(line.strip(), flush=True)
            if reason:
                (root/'tripped').write_text(reason+'\n')
                current = inspect()
                if current and (current['Config'].get('Labels') or {}).get('glm.spec.experiment') == label:
                    command('docker','kill',NAME,check=False)
                break
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def status(root):
    (root/'heartbeat').touch()
    current = inspect()
    print(json.dumps({'running':bool(current and current['State']['Running']),
                      'oom':bool(current and current['State']['OOMKilled']),
                      'tripped':(root/'tripped').read_text() if (root/'tripped').exists() else None}))


def restore(root, label):
    (root/'disarm').touch()
    old = original(root)
    current = inspect()
    if current and current['Id'] != old['Id']:
        if (current['Config'].get('Labels') or {}).get('glm.spec.experiment') != label:
            raise RuntimeError('refusing to stop an unrelated container')
        with (root/'experiment.log').open('w') as out:
            subprocess.run(['docker','logs',NAME],stdout=out,stderr=out,timeout=30)
        command('docker','stop','-t','5',NAME,check=False)
        # Keep stopped experimental container too, for inspection.
        command('docker','rename',NAME,NAME+'_finished_'+label)
    old_state = inspect(old['Id'])
    if not old_state:
        raise RuntimeError('original container missing')
    if old_state['Name'] != '/'+NAME:
        command('docker','rename',old['Id'],NAME)
    if not old_state['State']['Running']:
        command(str(Path.home()/'glm53big/start-flusher.sh'))
        command('docker','start',old['Id'])
    print('original container restored: '+old['Id'][:12])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action',choices=['prepare','stop','launch','watchdog','status','restore','settled','reuse-cache'])
    ap.add_argument('--source-label')
    ap.add_argument('label')
    ap.add_argument('--rank',type=int)
    args = ap.parse_args()
    if not re.fullmatch(r'[a-z0-9-]{1,48}',args.label):
        ap.error('invalid experiment label')
    root = Path.home()/'glm-spec'/args.label
    if args.action == 'prepare': prepare(root,args.label)
    elif args.action == 'stop': stop(root,args.label)
    elif args.action == 'launch': launch(root,args.label,args.rank)
    elif args.action == 'watchdog': watchdog(root,args.label)
    elif args.action == 'reuse-cache': reuse_cache(root,args.label,args.source_label)
    elif args.action == 'status': status(root)
    elif args.action == 'restore': restore(root,args.label)
    elif args.action == 'settled':
        (root/'settled').touch()
        command('pkill','-f','[c]ache_flusher.sh',check=False)
        print('boot cache flusher stopped')


if __name__ == '__main__':
    main()
