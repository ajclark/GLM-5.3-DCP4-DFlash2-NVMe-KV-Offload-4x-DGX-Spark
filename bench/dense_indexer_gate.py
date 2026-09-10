#!/usr/bin/env python3
"""CPU gate over paired final pre-top-k indexer scores (.npy, same prefix/mask)."""
import argparse
import json
from pathlib import Path

import numpy as np


def stable_topk(values, k):
    threshold = np.partition(values, len(values) - k)[-k]
    above = np.flatnonzero(values > threshold)
    ties = np.flatnonzero(values == threshold)[:k - len(above)]
    return np.concatenate((above, ties))


def score(reference, candidate, topk=2048, min_rows=512):
    if reference.shape != candidate.shape or reference.ndim != 2 or topk < 1:
        raise ValueError('paired query-by-key score matrices and positive topk required')
    divergence, overlap, valid_counts = [], [], []
    for a, b in zip(reference, candidate):
        if (np.isnan(a).any() or np.isnan(b).any() or np.isposinf(a).any() or np.isposinf(b).any()
                or not np.array_equal(np.isneginf(a), np.isneginf(b))):
            raise ValueError('nonfinite scores or different causal masks')
        mask = np.isfinite(a)
        x, y = np.asarray(a[mask], dtype=np.float64), np.asarray(b[mask], dtype=np.float64)
        if len(x) <= topk:
            raise ValueError('gate needs more eligible keys than topk; short contexts give vacuous overlap')
        x -= x.max()
        y -= y.max()
        logp = x - np.log(np.exp(x).sum())
        logq = y - np.log(np.exp(y).sum())
        divergence.append(max(0., float(np.sum(np.exp(logp) * (logp - logq)))))
        overlap.append(len(np.intersect1d(stable_topk(x, topk), stable_topk(y, topk))) / topk)
        valid_counts.append(len(x))
    if not divergence:
        raise ValueError('empty score corpus')
    metrics = {'rows': len(divergence), 'topk': topk, 'minimum_eligible_keys': min(valid_counts),
               'mean_kl': float(np.mean(divergence)), 'p95_kl': float(np.quantile(divergence, .95)),
               'mean_topk_overlap': float(np.mean(overlap)), 'p05_topk_overlap': float(np.quantile(overlap, .05))}
    checks = {'coverage': len(divergence) >= min_rows, 'mean_kl': metrics['mean_kl'] <= 1e-3,
              'p95_kl': metrics['p95_kl'] <= 1e-2, 'mean_topk_overlap': metrics['mean_topk_overlap'] >= .99,
              'p05_topk_overlap': metrics['p05_topk_overlap'] >= .98}
    return {'passed': all(checks.values()), 'checks': checks, 'metrics': metrics,
            'method': 'Per-query KL(softmax(reference)||softmax(candidate)), temperature one, and top-k set overlap; ties resolve by key index. Inputs must be final aggregated indexer scores on the same prefix. Deterministic gate, not a statistical proof of downstream quality. Apply separately to every layer/context stratum.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('reference', type=Path)
    ap.add_argument('candidate', type=Path)
    ap.add_argument('--topk', type=int, default=2048)
    ap.add_argument('--min-rows', type=int, default=512)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    result = score(np.load(args.reference, mmap_mode='r', allow_pickle=False),
                   np.load(args.candidate, mmap_mode='r', allow_pickle=False), args.topk, args.min_rows)
    text = json.dumps(result, indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    print(text, end='')
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
