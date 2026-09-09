import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
import spec_power_node as node


def snapshot(tmp_path):
    result = {'cpus': {}}
    for name, governor in [('p0', 'performance'), ('p1', 'ondemand')]:
        p = tmp_path / name
        p.mkdir()
        (p / 'scaling_governor').write_text(governor)
        result['cpus'][str(p)] = {'governor': governor, 'available': ['performance', 'ondemand', 'schedutil']}
    return result


def test_profile_and_restore_preserve_each_original_governor(tmp_path, monkeypatch):
    original = snapshot(tmp_path)
    (tmp_path / 'original.json').write_text(json.dumps(original))
    calls = []
    monkeypatch.setattr(node, 'command', lambda *args: calls.append(args))
    node.apply_profile(original, 'idle_auto')
    assert calls[-1] == ('nvidia-smi', '-rgc')
    assert all((Path(p) / 'scaling_governor').read_text() == 'schedutil' for p in original['cpus'])
    node.restore(tmp_path)
    assert calls[-1] == ('nvidia-smi', '-lgc', '2000,2000')
    for p, values in original['cpus'].items():
        assert (Path(p) / 'scaling_governor').read_text() == values['governor']
    assert (tmp_path / 'disarm').exists()


def test_invalid_governor_fails_before_any_clock_or_cpu_change(tmp_path, monkeypatch):
    original = snapshot(tmp_path)
    list(original['cpus'].values())[1]['available'] = ['ondemand']
    calls = []
    monkeypatch.setattr(node, 'command', lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match='governor unavailable'):
        node.apply_profile(original, 'schedutil')
    assert calls == []
    for p, values in original['cpus'].items():
        assert (Path(p) / 'scaling_governor').read_text() == values['governor']


def test_failed_restore_remains_armed_for_watchdog_retry(tmp_path, monkeypatch):
    (tmp_path / 'original.json').write_text(json.dumps(snapshot(tmp_path)))
    def fail(*args): raise RuntimeError('driver unavailable')
    monkeypatch.setattr(node, 'command', fail)
    with pytest.raises(RuntimeError): node.restore(tmp_path)
    assert not (tmp_path / 'disarm').exists()
