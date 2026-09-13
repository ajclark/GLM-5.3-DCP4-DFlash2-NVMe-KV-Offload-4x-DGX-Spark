#!/usr/bin/env python3
"""Real API and long-context cache checks, run under validate.py's memory guard."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dcp_probe import chat, make_prompt_words, metrics, phase_bench
from rollout import BASE, request


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def api_checks(out):
    body = {'model': 'glm-5.3', 'temperature': 0, 'max_tokens': 256,
            'chat_template_kwargs': {'enable_thinking': False}}
    tool = {'type': 'function', 'function': {'name': 'get_weather',
        'description': 'Return current weather for a city.', 'parameters': {
            'type': 'object', 'properties': {'city': {'type': 'string'}},
            'required': ['city'], 'additionalProperties': False}}}
    result = request({**body, 'messages': [{'role': 'user', 'content':
        'Use get_weather to get the current weather in London.'}],
        'tools': [tool], 'tool_choice': 'auto'}, '/v1/chat/completions', 300)
    save(out/'tool-call.json', result)
    calls = result['choices'][0]['message'].get('tool_calls') or []
    assert len(calls) == 1 and calls[0]['function']['name'] == 'get_weather', result
    assert json.loads(calls[0]['function']['arguments'])['city'].lower().startswith('london'), result
    message = result['choices'][0]['message']
    follow = request({**body, 'messages': [
        {'role': 'user', 'content': 'Use get_weather to get the current weather in London.'},
        message, {'role': 'tool', 'tool_call_id': calls[0]['id'],
                  'content': '{"temperature_c":17,"condition":"sunny"}'}],
        'tools': [tool]}, '/v1/chat/completions', 300)
    save(out/'tool-result.json', follow)
    answer = (follow['choices'][0]['message']['content'] or '').lower()
    assert '17' in answer or 'seventeen' in answer, follow
    thinking = request({**body, 'messages': [{'role': 'user', 'content':
        'What is 17 times 23? Give the integer answer.'}],
        'max_tokens': 1024, 'chat_template_kwargs': {'enable_thinking': True}},
        '/v1/chat/completions', 300)
    save(out/'reasoning.json', thinking)
    msg = thinking['choices'][0]['message']
    assert '391' in (msg['content'] or ''), thinking
    assert msg.get('reasoning') or msg.get('reasoning_content'), thinking
    cache_metrics = {key: value for key, value in metrics(BASE).items() if 'kv_offload' in key}
    save(out/'api-cache-metrics.json', cache_metrics)
    failures = sum(cache_metrics.get(key, 0) for key in (
        'vllm:kv_offload_allocation_failure', 'vllm:kv_offload_allocation_failure_total'))
    assert failures == 0, cache_metrics
    print('Tool selection, tool-result continuation and reasoning API passed', flush=True)


def long_check(out, phase):
    # This exceeds the former incorrectly DCP-divided draft table's 90112 limit.
    prompt = ('Remember this validation code: QUARTZ-7294.\nThe following words are test data:\n'
              + make_prompt_words(97000, seed=29000911)
              + '\nEnd of test data. Reply with only the validation code given at the start.')
    before = metrics(BASE)
    result = chat(BASE, [{'role': 'user', 'content': prompt}], 32, timeout=3600, stream=True)
    after = metrics(BASE)
    result['metrics_delta'] = {key: after.get(key, 0)-value for key, value in before.items()
                               if 'cache_hit' in key or 'cache_quer' in key}
    save(out/(phase+'.json'), result)
    assert 90112 < result['prompt_tokens'] < 180000, result
    assert 'QUARTZ-7294' in result['content'], result
    if phase == 'reload':
        hits = result['metrics_delta'].get('vllm:external_prefix_cache_hits_total', 0)
        assert hits > result['prompt_tokens'] * 0.95, result
    print(f"{phase}: {result['prompt_tokens']} tokens, TTFT {result['ttft_s']:.2f}s, "
          f"answer {result['content']!r}, cache {result['metrics_delta']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['api', 'cold', 'reload', 'bench'])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == 'bench':
        phase_bench(BASE, str(args.out), 'vllm-upgrade', reps=2)
    elif args.phase == 'api':
        api_checks(args.out)
    else:
        long_check(args.out, args.phase)


if __name__ == '__main__':
    main()
