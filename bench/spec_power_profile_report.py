"""Summarize interleaved clock/governor screens using four-device power only."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from analyze_adaptive_spec import device_energy
from spec_energy_report import integrate


def geomean(values):
    return math.exp(statistics.mean(math.log(v) for v in values)) if values else None


def bracketed(rows):
    """Compare each treatment with both adjacent same-prompt baselines."""
    comparisons = []
    for row in rows:
        if row['profile'] == 'baseline':
            continue
        controls = [r for r in rows if r['profile'] == 'baseline' and r['case'] == row['case']]
        before = [r for r in controls if r['profile_index'] < row['profile_index']]
        after = [r for r in controls if r['profile_index'] > row['profile_index']]
        item = {'label': row['label'], 'profile': row['profile'], 'case': row['case']}
        if not before or not after:
            item['excluded'] = 'missing before/after baseline'
            comparisons.append(item)
            continue
        left = max(before, key=lambda r: r['profile_index'])
        right = min(after, key=lambda r: r['profile_index'])
        item['controls'] = [left['label'], right['label']]
        for metric in ('decode_tps', 'decode_device_j_per_token', 'request_device_j_per_token'):
            values = [r.get(metric) for r in (left, row, right)]
            item[metric + '_ratio'] = None if any(v is None or v <= 0 for v in values) else values[1] / math.sqrt(values[0] * values[2])
        comparisons.append(item)
    return comparisons


def cpu_window(heartbeats, window):
    selected = [r for r in heartbeats if window['settled_start'] <= r['time'] <= window['ended_at']]
    result = {}
    for host in sorted({h for r in selected for h in r['nodes']}):
        rows = [(r['time'], r['nodes'][host]) for r in selected if host in r['nodes']]
        frequencies = [f for _, r in rows for f in r.get('cpu_scaling_cur_freq_khz', {}).values()]
        item = {'samples': len(rows), 'median_reported_cpu_khz': statistics.median(frequencies) if frequencies else None}
        if len(rows) >= 2:
            t0, left = rows[0]
            t1, right = rows[-1]
            a, b = left.get('cpu_idle_counters', {}), right.get('cpu_idle_counters', {})
            keys = a.keys() & b.keys()
            cores = {Path(k).parent.parent.name for k in keys}
            delta = sum(b[k]['time_us'] - a[k]['time_us'] for k in keys)
            item['mean_cpu_idle_residency_fraction'] = (delta / ((t1 - t0) * 1e6 * len(cores))
                if cores and t1 > t0 and all(b[k]['time_us'] >= a[k]['time_us'] for k in keys) else None)
            item['idle_counter_interval_seconds'] = t1 - t0
        result[host] = item
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    ap.add_argument('--power', type=Path, required=True)
    args = ap.parse_args()
    samples = defaultdict(list)
    for line in args.power.read_text().splitlines():
        row = json.loads(line)
        samples[row['host']].append(row)
    for values in samples.values():
        values.sort(key=lambda r: r['received_at'])
    rows = []
    for path in sorted(args.directory.glob('*.json')):
        row = json.loads(path.read_text())
        if not isinstance(row, dict) or not {'profile', 'profile_index', 'case', 'chunks', 'token_ids'} <= row.keys():
            continue
        end = row['started_at'] + max(c['seconds'] for c in row['chunks'])
        request_energy = integrate(samples, row['started_at'], end)
        rows.append({**{k: row[k] for k in ('label', 'profile', 'profile_index', 'case', 'decode_tps', 'ttft', 'token_sha256')},
                     'decode_device_j_per_token': device_energy(row, samples),
                     'request_device_j_per_token': request_energy / len(row['token_ids']) if request_energy is not None else None})
    comparisons = bracketed(rows)
    summaries = {}
    for profile in sorted({r['profile'] for r in comparisons}):
        group = [r for r in comparisons if r['profile'] == profile and 'excluded' not in r]
        summaries[profile] = {'paired_requests': len(group)}
        for metric in ('decode_tps_ratio', 'decode_device_j_per_token_ratio', 'request_device_j_per_token_ratio'):
            values = [r[metric] for r in group if r[metric] is not None]
            summaries[profile][metric] = geomean(values)
            summaries[profile][metric + '_valid_pairs'] = len(values)
    heartbeats = [json.loads(line) for line in (args.directory / 'heartbeat.jsonl').read_text().splitlines()]
    idle = []
    for line in (args.directory / 'idle-windows.jsonl').read_text().splitlines():
        window = json.loads(line)
        energy = integrate(samples, window['settled_start'], window['ended_at'])
        idle.append({**window, 'four_device_mean_idle_watts': energy / (window['ended_at'] - window['settled_start']) if energy is not None else None,
                     'cpu_proxies': cpu_window(heartbeats, window)})
    result = {'sensor': 'NVIDIA-reported power summed across four devices; CPU counters are separate proxies, not watts.',
              'screening_only': True, 'whole_node_or_wall_power_measured': False,
              'method': 'Treatment divided by geometric mean of nearest same-prompt baseline before and after; fixed cap7, complete request energy includes prefill.',
              'limitations': 'Small development screen, no confidence interval or promotion claim. Sampling needs all four devices with bracketing coverage and gaps <=5s. Output hashes retained to expose changing work.',
              'profiles': summaries, 'requests': rows, 'comparisons': comparisons, 'idle_windows': idle}
    (args.directory / 'power-profile-report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'profiles': summaries, 'idle_watts': [(r['profile'], r['four_device_mean_idle_watts']) for r in idle]}, indent=2))


if __name__ == '__main__':
    main()
