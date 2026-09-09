#!/usr/bin/env python3
"""Development-only, leave-one-prompt-out screen for workload acceptance priors.

The cost screen compares decisions at the SAME recorded full-seven boundaries.
It is not a closed-loop rollout: shorter verification would create other future
boundaries. No end-to-end speedup is inferred from this counterfactual diagnostic.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

CAPS = (1, 3, 5, 7)


def conditional_prior(accepted):
    return [(sum(a > j for a in accepted) + .5) / (sum(a >= j for a in accepted) + 1)
            for j in range(7)]


def survival(prior):
    out, product = [], 1.0
    for value in prior:
        product *= value
        out.append(product)
    return out


def best_cap(prior, costs):
    p = survival(prior)
    return max(CAPS, key=lambda k: ((1 + sum(p[:k])) / costs[str(k)], k))


def brier(accepted, prior):
    p = survival(prior)
    return sum((value - (a > j)) ** 2 for a in accepted for j, value in enumerate(p)) / (7 * len(accepted))


def screen(cases, costs):
    rows = []
    for case, observed in sorted(cases.items()):
        domain = case.split('_')[0]
        all_train = [a for name, values in cases.items() if name != case for a in values]
        domain_train = [a for name, values in cases.items() if name != case and name.split('_')[0] == domain for a in values]
        opposite = [a for name, values in cases.items() if name.split('_')[0] != domain for a in values]
        if not domain_train or not opposite:
            raise ValueError('each domain needs at least two development prompts')
        priors = {'global': conditional_prior(all_train), 'hint': conditional_prior(domain_train),
                  'wrong_hint': conditional_prior(opposite)}
        caps = {name: best_cap(prior, costs) for name, prior in priors.items()}
        utility = {name: sum(1 + min(a, k) for a in observed) / (len(observed) * costs[str(k)])
                   for name, k in caps.items()}
        rows.append({'case': case, 'domain': domain, 'boundaries': len(observed),
            'caps': caps, 'brier': {name: brier(observed, prior) for name, prior in priors.items()},
            'same_boundary_hint_over_global_utility_ratio': utility['hint'] / utility['global'],
            'same_boundary_wrong_hint_over_global_utility_ratio': utility['wrong_hint'] / utility['global'],
            'train_priors': priors})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace', type=Path, default=Path('results/adaptive-spec/adaptive-v2-20260908-r4/calibration-trace.jsonl'))
    ap.add_argument('--corpus', type=Path, default=Path('bench/spec-development.json'))
    ap.add_argument('--costs', type=Path, default=Path('results/adaptive-spec/adaptive-v2-20260908-r4/costs-v3-curve.json'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    corpus = json.loads(args.corpus.read_text())
    cases = {name: [] for name in corpus}
    for line in args.trace.read_text().splitlines():
        row = json.loads(line)
        match = re.fullmatch(r'dev-adaptive-r4-(.+)-r\d+-fixed-k7', row.get('label', ''))
        if match and match[1] in cases and row.get('event') == 'verify' and row.get('learned') and row['scheduled_k'] == 7:
            cases[match[1]].append(row['accepted'])
    if not all(cases.values()):
        raise ValueError('missing development traces')
    curve = json.loads(args.costs.read_text())
    costs = curve['calibration_points'][0]['cycle_ms']
    rows = screen(cases, costs)
    aggregate = {}
    for domain in ('code', 'prose'):
        selected = [r for r in rows if r['domain'] == domain]
        aggregate[domain] = {'prompts': len(selected),
            'mean_brier': {mode: sum(r['brier'][mode] for r in selected) / len(selected)
                           for mode in ('global', 'hint', 'wrong_hint')},
            'same_boundary_hint_utility_geomean': math.exp(sum(math.log(r['same_boundary_hint_over_global_utility_ratio']) for r in selected) / len(selected))}
    priors = {domain: conditional_prior([a for name, values in cases.items() if name.startswith(domain + '_') for a in values]) for domain in ('code', 'prose')}
    report = {'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.trace, args.corpus, args.costs)},
        'data': '12 development prompts only; full-seven observations; held-out evaluation excluded',
        'method': 'Leave one prompt out of prior fitting. Costs remain the independently measured frozen short-context anchor.',
        'limitations': 'Same-boundary utility is not a rollout or measured tok/s. Domain labels are known corpus intent, not classifier predictions. Single run per prompt; only two coarse domains.',
        'aggregate': aggregate, 'per_prompt': rows,
        'fitted_development_priors': {'context_range': [0, 4096], 'prior_strength': 2, 'domains': priors},
        'decision': 'Use a bounded live development comparison to test cold-start hints; retain runtime feedback and abstention. No production promotion.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
