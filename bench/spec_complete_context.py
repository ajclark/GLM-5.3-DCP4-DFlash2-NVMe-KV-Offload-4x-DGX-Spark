"""Check complete pure functions using an already frozen repository prefix."""
import argparse
import json
from pathlib import Path
import urllib.request

from adaptive_spec import add_costs, generate, idle_check, request_body
from spec_code_check import PROMPT, check
from spec_memory import MemoryGuard


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus', type=Path, required=True)
    ap.add_argument('--costs', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--base', default='http://spark-06c4.local:8000')
    ap.add_argument('--variants', default='fixed7,fixed3,adaptive')
    ap.add_argument('--repeats', type=int, choices=(1, 2, 3), default=1)
    args = ap.parse_args()
    corpus = json.loads(args.corpus.read_text())
    reference = next(iter(corpus.values()))
    separator = '</repository_reference>\n\nTask: '
    if reference.count(separator) != 1:
        ap.error('expected a saved repository-context corpus')
    prompt = reference.split(separator)[0] + separator + PROMPT
    costs = json.loads(args.costs.read_text()) if args.costs else None
    allowed = {**{f'fixed{k}': ('fixed', k) for k in (1, 3, 5, 7)}, 'adaptive': ('adaptive', 7)}
    try:
        modes = [allowed[name] for name in args.variants.split(',')]
    except KeyError:
        ap.error('unknown verification variant')
    if len(modes) > 5 or any(m == 'adaptive' for m, _ in modes) and costs is None:
        ap.error('at most five variants; adaptive needs costs')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'declaration.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')
    guard = MemoryGuard(args.out / 'memory.jsonl').start()
    try:
        print('preflight:', guard.preflight(), flush=True)
        idle_check(args.base)
        sample = request_body(prompt, 7, 'complete-context-tokenize', 768)
        token_payload = {k: sample[k] for k in ('model', 'messages', 'chat_template_kwargs')}
        req = urllib.request.Request(args.base + '/tokenize', json.dumps(token_payload).encode(),
                                     {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as response:
            tokens = json.load(response)['count']
        if tokens + 768 > 174000:
            raise RuntimeError('bounded complete-function context exceeded')
        report = []
        schedule = [(repeat, mode, cap) for repeat in range(args.repeats)
                    for mode, cap in (modes if repeat % 2 == 0 else list(reversed(modes)))]
        for repeat, mode, cap in schedule:
            idle_check(args.base)
            guard.preflight(seconds=4)
            label = f'{args.out.name}-r{repeat}-{mode}-k{cap}'
            body = request_body(prompt, cap, label, 768)
            body['vllm_xargs']['spec_policy'] = mode
            if costs:
                add_costs(body, costs)
            (args.out / (label + '-request.json')).write_text(json.dumps(body) + '\n')
            result = generate(args.base, body, guard, deadline_seconds=900)
            path = args.out / (label + '.json')
            result.update(label=label, policy=mode, cap=cap, prompt_tokens=tokens)
            path.write_text(json.dumps(result, indent=2) + '\n')
            try:
                result['functional_check'] = check(result['text'])
            except (ValueError, SyntaxError, RuntimeError) as exc:
                result['functional_check'] = 'FAILED: ' + str(exc)
            path.write_text(json.dumps(result, indent=2) + '\n')
            report.append({k: result[k] for k in ('label', 'functional_check', 'token_sha256', 'usage', 'ttft', 'decode_tps')})
            print(json.dumps(report[-1]), flush=True)
        (args.out / 'report.json').write_text(json.dumps({'runs': report,
            'all_token_ids_identical': len({r['token_sha256'] for r in report}) == 1}, indent=2) + '\n')
        if any(row['functional_check'].startswith('FAILED') for row in report):
            raise RuntimeError('complete-function quality gate failed; all results preserved in report.json')
    finally:
        guard.close()


if __name__ == '__main__':
    main()
