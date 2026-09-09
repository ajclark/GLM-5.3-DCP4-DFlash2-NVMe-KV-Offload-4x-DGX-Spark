"""Small live fallback, concurrency-transition and cancellation probes."""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def run(base, out, guard):
    from adaptive_spec import generate, idle_check, request_body
    from spec_experiment import validate_count_smoke

    def body(name, tokens=128):
        return request_body('Count from 1 to 200, one number per line.',1,out.name+'-'+name,tokens)

    def save(name, result):
        validate_count_smoke(result['text'])
        (out/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        print('integration passed:',name,flush=True)

    def preflight():
        idle_check(base)
        guard.preflight(seconds=4)

    preflight()
    fallback = body('temperature',64)
    fallback['temperature'] = 0.3
    save('temperature',generate(base,fallback,guard))
    preflight()
    fallback = body('penalty',64)
    fallback['repetition_penalty'] = 1.01
    save('penalty',generate(base,fallback,guard))
    preflight()
    # A begins at C1/K1, overlaps B at C2/K7, then returns to C1/K1.
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(generate,base,body('transition-a',192),guard)
        time.sleep(1.5)
        second = pool.submit(generate,base,body('transition-b',64),guard)
        save('transition-b',second.result())
        save('transition-a',first.result())
    preflight()
    cancelled = body('cancel',256)
    req = urllib.request.Request(base+'/v1/chat/completions',json.dumps(cancelled).encode(),
                                 {'Content-Type':'application/json'})
    tokens = []
    with urllib.request.urlopen(req,timeout=60) as response:
        for line in response:
            guard.check()
            if line.startswith(b'data: ') and line.strip()!=b'data: [DONE]':
                row = json.loads(line[6:])
                for choice in row.get('choices',[]):
                    tokens.extend(choice.get('token_ids') or [])
                if len(tokens)>=16:
                    break
    deadline = time.monotonic()+15
    while True:
        guard.check()
        try:
            idle_check(base)
            break
        except RuntimeError:
            if time.monotonic()>deadline:
                raise RuntimeError('cancelled request did not leave the scheduler')
            time.sleep(.2)
    (out/'cancel.json').write_text(json.dumps({'tokens_before_close':tokens,'idle_after_cancel':True})+'\n')
    preflight()
    save('after-cancel',generate(base,body('after-cancel',64),guard))
