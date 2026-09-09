"""Bounded long-context verification smoke, with CPU/API token counting first."""
import hashlib
import json
import random
import urllib.request


def run(base,out,guard,costs=None):
    from adaptive_spec import generate, idle_check, request_body, add_costs
    from spec_experiment import validate_count_smoke

    words = 'river window garden stone paper summer chair path station orange cloud table copper bridge'.split()
    for target in (32768,100000):
        idle_check(base)
        print('context preflight:',target,guard.preflight(),flush=True)
        rng = random.Random(20260908+target)
        # Calibration is CPU-only at the API server; never allocate a second model.
        all_words = [rng.choice(words) for _ in range(target)]
        count = target
        for _ in range(5):
            prefix = ' '.join(all_words[:count])
            prompt = ('The following is arbitrary test data. The validation code is CINNAMON.\n'
                      +prefix+'\nEnd of test data. Repeat the validation code, then count from 1 to 30, one number per line.')
            body = request_body(prompt,7,out.name+f'-{target}-k7',96)
            payload = {k:body[k] for k in ('model','messages','chat_template_kwargs')}
            req = urllib.request.Request(base+'/tokenize',json.dumps(payload).encode(),{'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=30) as response:
                token_count = json.load(response)['count']
            if target-512 <= token_count <= target+128:
                break
            count = max(1,min(len(all_words),int(count*(target-64)/token_count)))
        else:
            raise RuntimeError('could not calibrate bounded context size')
        if token_count+96>101000:
            raise RuntimeError('context exceeds experiment bound')
        (out/f'prompt-{target}.json').write_text(json.dumps({'tokens':token_count,'sha256':hashlib.sha256(prompt.encode()).hexdigest(),'seed':20260908+target})+'\n')
        variants = [('fixed',7),('fixed',3)] + ([('adaptive',7)] if costs else [])
        for mode,cap in variants:
            idle_check(base)
            guard.preflight(seconds=4)
            label = out.name+f'-{target}-{mode}-k{cap}'
            print('starting context request:',label,'prompt tokens',token_count,flush=True)
            body = request_body(prompt,cap,label,96)
            body['vllm_xargs']['spec_policy'] = mode
            if costs:
                add_costs(body,costs)
            result = generate(base,body,guard,deadline_seconds=900)
            validate_count_smoke(result['text'])
            if 'CINNAMON' not in result['text']:
                raise RuntimeError('long-context validation code not retained')
            result.update(label=label,case=f'context_{target}',cap=cap,repeat=0,policy=mode)
            (out/(label+'.json')).write_text(json.dumps(result,indent=2)+'\n')
            print('context passed:',label,'TTFT',round(result['ttft'],2),'tok/s',round(result['decode_tps'],2),flush=True)
