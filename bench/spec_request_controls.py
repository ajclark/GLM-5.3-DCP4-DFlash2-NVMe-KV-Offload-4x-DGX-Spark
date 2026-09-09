"""Guarded paired controls over exact captured Pi request payloads.

Capture Pi through herdr first. Replays preserve messages and provider options;
only experiment metadata, stream accounting and the declared token bound vary.
Raw payloads can include private system context and must stay out of publication.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import re
import urllib.request

from adaptive_spec import generate, idle_check, spec_metrics
from spec_memory import MemoryGuard


def controlled_body(original, study, variant, label, tokens):
    body = copy.deepcopy(original)
    if body.get('model') != 'glm-5.3' or body.get('temperature', 0) != 0 or body.get('n', 1) != 1:
        raise ValueError('controls require the intended GLM model and C1 greedy request')
    body.update(temperature=0, stream=True, stream_options={'include_usage': True}, return_token_ids=True)
    key = 'max_completion_tokens' if 'max_completion_tokens' in body else 'max_tokens'
    prior_limit = body.get(key, tokens)
    if type(prior_limit) is not int or prior_limit <= 0:
        raise ValueError('invalid captured token bound')
    body[key] = min(prior_limit, tokens)
    xargs = body.setdefault('vllm_xargs', {})
    xargs.update(spec_label=label, spec_verify_cap=7, spec_confidence_trace=False,
                 spec_policy='adaptive', spec_use_hints=False)
    if study == 'confidence':
        if variant not in ('off', 'on'):
            raise ValueError('unknown confidence control')
        xargs.update(spec_policy='fixed', spec_confidence_trace=variant == 'on')
    elif study == 'hints':
        if variant not in ('off', 'on', 'wrong', 'fixed7'):
            raise ValueError('unknown hint control')
        workload = xargs.get('spec_workload')
        if workload not in ('code_generate', 'code_edit', 'code_review', 'prose'):
            raise ValueError('captured Pi hint abstained; cannot test a domain prior')
        if xargs.get('spec_phase') != 'user_turn' or xargs.get('spec_hint_strength') != 'weak':
            raise ValueError('hint study requires an actual eligible Pi user-turn hint')
        if variant in ('on', 'wrong'):
            xargs.update(spec_use_hints=True)
        if variant == 'wrong':
            xargs['spec_workload'] = 'code_generate' if workload == 'prose' else 'prose'
        if variant == 'fixed7':
            xargs['spec_policy'] = 'fixed'
    else:
        raise ValueError('unknown study')
    return body


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--requests', type=Path, required=True, help='JSON case-name to captured payload path mapping')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--study', choices=('hints', 'confidence'), required=True)
    ap.add_argument('--base', default='http://spark-06c4.local:8000')
    ap.add_argument('--repeats', type=int, choices=(1, 2, 3, 4), default=2)
    ap.add_argument('--tokens', type=int, default=256)
    args = ap.parse_args()
    paths = {case: Path(path) for case, path in json.loads(args.requests.read_text()).items()}
    if not 1 <= len(paths) <= 30 or not 64 <= args.tokens <= 512:
        ap.error('bounded study requires 1..30 cases and 64..512 output tokens')
    if any(not re.fullmatch(r'[a-z0-9_]{1,32}', case) for case in paths) or not re.fullmatch(r'[a-z0-9-]{1,40}', args.out.name):
        ap.error('safe bounded case and output names required')
    originals = {case: json.loads(path.read_text()) for case, path in paths.items()}
    variants = ['off', 'on', 'wrong', 'fixed7'] if args.study == 'hints' else ['off', 'on']
    # Validate all controls before the first request.
    for original in originals.values():
        for variant in variants:
            controlled_body(original, args.study, variant, 'validation', args.tokens)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'declaration.json').write_text(json.dumps({**vars(args), 'variants': variants,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()}}, default=str, indent=2) + '\n')
    guard = MemoryGuard(args.out / 'memory.jsonl').start()
    try:
        print('preflight:', guard.preflight(), flush=True)
        for repeat in range(args.repeats):
            for case_index, (case, original) in enumerate(originals.items()):
                order = variants.copy()
                if len(order) == 2:
                    if (case_index + repeat) % 2:
                        order.reverse()
                else:
                    random.Random(20260909 + 1009 * repeat + case_index).shuffle(order)
                for variant in order:
                    idle_check(args.base)
                    guard.preflight(seconds=4)
                    label = f'{args.out.name}-{case}-r{repeat}-{variant}'
                    if len(label) > 96:
                        raise ValueError('experiment label exceeds trace identity bound')
                    body = controlled_body(original, args.study, variant, label, args.tokens)
                    payload = {k: body[k] for k in ('model', 'messages', 'chat_template_kwargs', 'tools') if k in body}
                    request = urllib.request.Request(args.base + '/tokenize', json.dumps(payload).encode(), {'Content-Type': 'application/json'})
                    with urllib.request.urlopen(request, timeout=30) as response:
                        prompt_tokens = json.load(response)['count']
                    if prompt_tokens + args.tokens > 4096:
                        raise ValueError('request exceeds calibrated short-context range')
                    before = spec_metrics(args.base)
                    result = generate(args.base, body, guard)
                    after = spec_metrics(args.base)
                    result.update(label=label, case=case, repeat=repeat, variant=variant, study=args.study,
                                  cap=7, policy=body['vllm_xargs']['spec_policy'],
                                  request_sha256=hashlib.sha256(json.dumps(body).encode()).hexdigest(),
                                  message_sha256=hashlib.sha256(json.dumps(body['messages']).encode()).hexdigest(),
                                  experiment_xargs=body['vllm_xargs'],
                                  spec_metric_delta={k: after[k] - v for k, v in before.items() if k in after})
                    (args.out / (label + '.json')).write_text(json.dumps(result, indent=2) + '\n')
                    print(json.dumps({k: result[k] for k in ('label', 'decode_tps', 'ttft', 'usage', 'token_sha256')}), flush=True)
    finally:
        guard.close()


if __name__ == '__main__':
    main()
