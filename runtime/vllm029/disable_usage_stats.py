#!/usr/bin/env python3
"""Install vLLM's opt-out marker on a stopped Docker deployment and host users.

Run on each Spark with sudo after stopping vLLM. Existing container layers are
preserved, including rollback containers. New launches also set opt-out env vars.
Only selected opt-out settings are printed; Docker credentials are never logged.
"""
import http.client
import io
import json
import os
from pathlib import PurePosixPath, Path
import pwd
import re
import shutil
import socket
import subprocess
import sys
import tarfile
from urllib.parse import urlencode


class Docker(http.client.HTTPConnection):
    def __init__(self):
        super().__init__('localhost', timeout=60)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect('/var/run/docker.sock')


def request(method, path, data=None):
    client = Docker()
    client.request(method, '/v1.47' + path, body=data,
                   headers={'Content-Type': 'application/x-tar'} if data is not None else {})
    response = client.getresponse()
    body = response.read()
    client.close()
    if not 200 <= response.status < 300:
        raise RuntimeError(f'Docker {method} {path}: HTTP {response.status}')
    return body


def marker_archive(path):
    path = PurePosixPath(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('opt-out marker path must be absolute and normalized')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        entry = tarfile.TarInfo(str(path).lstrip('/'))
        entry.mode = 0o644
        archive.addfile(entry, io.BytesIO(b''))
    return buffer.getvalue()


def install():
    if os.geteuid() != 0:
        raise RuntimeError('run with sudo to cover root and host login users')
    containers = json.loads(request('GET', '/containers/json?all=1'))
    selected = []
    for row in containers:
        config = json.loads(request('GET', '/containers/' + row['Id'] + '/json'))
        info = config['Config']
        identity = ' '.join([config['Name'], info['Image'], *(info.get('Entrypoint') or []),
                             *(info.get('Cmd') or [])]).lower()
        if 'vllm' in identity:
            selected.append(config)
    if any(c['State']['Running'] for c in selected):
        raise RuntimeError('stop every vLLM container first; opt-out decisions may be cached')
    users = []
    for account in pwd.getpwall():
        if account.pw_uid != 0 and not (1000 <= account.pw_uid < 65534 and
                                       account.pw_shell.rsplit('/', 1)[-1] in ('bash', 'sh', 'zsh', 'fish')):
            continue
        home = Path(account.pw_dir)
        if not home.is_dir():
            continue
        marker = home / '.config/vllm/do_not_track'
        for folder in (home / '.config', marker.parent):
            if not folder.exists():
                folder.mkdir()
                os.chown(folder, account.pw_uid, account.pw_gid)
        marker.touch(exist_ok=True)
        os.chown(marker, account.pw_uid, account.pw_gid)
        users.append(account.pw_name)
    results = []
    for container in selected:
        config = container['Config']
        env = dict(e.split('=', 1) for e in config.get('Env', []) if '=' in e)
        user = config.get('User') or 'root'
        home = env.get('HOME')
        if not home:
            uid = user.split(':', 1)[0]
            if uid in ('root', '0'):
                home = '/root'
            else:
                payload = request('GET', '/containers/' + container['Id'] + '/archive?' + urlencode({'path':'/etc/passwd'}))
                with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
                    rows = archive.extractfile(archive.getmembers()[0]).read().decode().splitlines()
                matches = [r.split(':') for r in rows if r.split(':')[0] == uid or r.split(':')[2] == uid]
                if len(matches) != 1:
                    raise RuntimeError('cannot determine container home for ' + container['Name'])
                home = matches[0][5]
        config_root = env.get('VLLM_CONFIG_ROOT') or str(PurePosixPath(env.get('XDG_CONFIG_HOME') or str(PurePosixPath(home) / '.config')) / 'vllm')
        marker = str(PurePosixPath(config_root) / 'do_not_track')
        request('PUT', '/containers/' + container['Id'] + '/archive?path=/', marker_archive(marker))
        payload = request('GET', '/containers/' + container['Id'] + '/archive?' + urlencode({'path': marker}))
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            assert any(m.isfile() and m.name.endswith('do_not_track') for m in archive.getmembers())
        results.append({'name':container['Name'], 'id':container['Id'][:12], 'marker':marker, 'verified':True})
    return {'host_users':users, 'containers':results}


def harden_legacy_launchers():
    """Preserve and update older host launch scripts used by legacy recovery."""
    if os.geteuid() != 0:
        raise RuntimeError('run with sudo to preserve private host backups')
    keys = ('VLLM_NO_USAGE_STATS', 'VLLM_DO_NOT_TRACK', 'DO_NOT_TRACK')
    pattern = re.compile(r'(?m)^(\s*(?:(?:exec|sudo)\s+)*(?:docker|run_docker)\s+run)(?=\s)')
    flags = ''.join(' --env ' + key + '=1' for key in keys)
    results = []
    for account in pwd.getpwall():
        for path in (Path(account.pw_dir) / 'glm53big').glob('launch-glm53big-*.sh'):
            old = path.read_text()
            if not pattern.search(old):
                continue
            if all(key + '=1' in old for key in keys):
                results.append({'path':str(path), 'changed':False})
                continue
            if any(key + '=' in old for key in keys):
                raise RuntimeError('review conflicting existing opt-out flags in ' + str(path))
            backup_dir = Path('/var/tmp/vllm-privacy-backups') / str(account.pw_uid)
            backup_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(backup_dir.parent, 0o700)
            backup = backup_dir / path.name
            if not backup.exists():
                shutil.copy2(path, backup)
                os.chmod(backup, 0o600)
            path.write_text(pattern.sub(lambda match: match[1] + flags, old))
            checked = subprocess.run(['bash', '-n', str(path)], capture_output=True)
            if checked.returncode:
                path.write_text(old)
                raise RuntimeError('shell syntax failed; restored ' + str(path))
            results.append({'path':str(path), 'changed':True})
    return results


if __name__ == '__main__':
    if sys.argv[1:] == ['--launchers-only']:
        print(json.dumps(harden_legacy_launchers(), sort_keys=True))
    elif not sys.argv[1:]:
        print(json.dumps(install(), sort_keys=True))
    else:
        raise SystemExit('usage: disable_usage_stats.py [--launchers-only]')
