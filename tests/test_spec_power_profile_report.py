import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from spec_power_profile_report import bracketed, cpu_window


def test_two_sided_control_uses_same_prompt_and_keeps_missing_energy():
    rows = [{'label': str(i), 'profile_index': i, 'profile': profile, 'case': 'code',
             'decode_tps': rate, 'decode_device_j_per_token': None,
             'request_device_j_per_token': energy}
            for i, profile, rate, energy in [(0, 'baseline', 20, 8), (1, 'gpu1800', 18, 6), (2, 'baseline', 25, 10)]]
    rows += [{'label': 'wrong-prompt', 'profile_index': 0, 'profile': 'baseline',
              'case': 'prose', 'decode_tps': 100}]
    result = bracketed(rows)[0]
    assert result['controls'] == ['0', '2']
    assert result['decode_tps_ratio'] == pytest.approx(18 / (20 * 25) ** .5)
    assert result['request_device_j_per_token_ratio'] == pytest.approx(6 / (8 * 10) ** .5)
    assert result['decode_device_j_per_token_ratio'] is None
    assert bracketed(rows[:2])[0]['excluded'] == 'missing before/after baseline'


def test_idle_residency_uses_elapsed_time_and_all_observed_cores():
    rows = []
    for timestamp, counter in [(10, 1_000_000), (20, 6_000_000)]:
        idle = {f'/sys/devices/system/cpu/cpu{cpu}/cpuidle/state0': {'time_us': counter}
                for cpu in range(2)}
        rows.append({'time': timestamp, 'nodes': {'host': {'cpu_idle_counters': idle}}})
    result = cpu_window(rows, {'settled_start': 10, 'ended_at': 20})['host']
    assert result['mean_cpu_idle_residency_fraction'] == .5
    assert result['median_reported_cpu_khz'] is None
