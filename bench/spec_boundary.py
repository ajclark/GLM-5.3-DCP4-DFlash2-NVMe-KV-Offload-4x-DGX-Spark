"""Bounded post-repair probes around the former replicated-table boundary."""
import argparse
import hashlib
import json
import random
import urllib.request
from pathlib import Path

from adaptive_spec import generate, idle_check, request_body, spec_metrics
from spec_experiment import validate_count_smoke
from spec_memory import MemoryGuard


def build(base, target):
    words = 'river window garden stone paper summer chair path station orange cloud table copper bridge'.split()
    rng = random.Random(20260908 + target)
    data = [rng.choice(words) for _ in range(target)]
    count = target
    for _ in range(8):
        prompt = ('The following is arbitrary test data. The validation code is CINNAMON.\n'
                  + ' '.join(data[:count])
                  + '\nEnd of test data. Repeat the validation code, then count from 1 to 30, one number per line.')
        body = request_body(prompt, 7, 'boundary-tokenize', 64)
        payload = {k: body[k] for k in ('model', 'messages', 'chat_template_kwargs')}
        req = urllib.request.Request(base + '/tokenize', json.dumps(payload).encode(),
                                     {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as response:
            tokens = json.load(response)['count']
        if target - 128 <= tokens <= target + 128:
            return prompt, tokens
        count = max(1, min(len(data), round(count * (target - 32) / tokens)))
    raise RuntimeError('context token calibration failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base', default='http://spark-06c4.local:8000')
    parser.add_argument('--targets', default='89000,92000')
    parser.add_argument('--repeats', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    targets = [int(x) for x in args.targets.split(',')]
    if not targets or any(not 1024 <= x <= 170000 for x in targets):
        parser.error('targets must be 1024..170000')
    args.out.mkdir(parents=True, exist_ok=False)
    guard = MemoryGuard(args.out / 'memory.jsonl').start()
    try:
        print('preflight:', guard.preflight(), flush=True)
        for target in targets:
            idle_check(args.base)
            prompt, tokens = build(args.base, target)
            if tokens + 64 > 170256:
                raise RuntimeError('tokenized prompt exceeds bounded probe capacity')
            for repeat in range(args.repeats):
                idle_check(args.base)
                guard.preflight(seconds=4)
                label = f'{args.out.name}-{target}-r{repeat}'
                body = request_body(prompt, 7, label, 64)
                (args.out / (label + '-request.json')).write_text(json.dumps(body) + '\n')
                print('starting', label, 'prompt tokens', tokens, flush=True)
                before = spec_metrics(args.base)
                result = generate(args.base, body, guard, deadline_seconds=900)
                after = spec_metrics(args.base)
                result.update(label=label, target=target, repeat=repeat,
                              prompt_tokens=tokens, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                              spec_metric_delta={k: after[k] - v for k, v in before.items() if k in after})
                # Save even failed correctness outcomes as evidence.
                (args.out / (label + '.json')).write_text(json.dumps(result, indent=2) + '\n')
                validate_count_smoke(result['text'])
                if 'CINNAMON' not in result['text']:
                    raise RuntimeError('validation code not retained')
                print(json.dumps({k: result[k] for k in ('label', 'prompt_tokens', 'ttft', 'decode_tps', 'spec_metric_delta')}), flush=True)
    finally:
        guard.close()


if __name__ == '__main__':
    main()
