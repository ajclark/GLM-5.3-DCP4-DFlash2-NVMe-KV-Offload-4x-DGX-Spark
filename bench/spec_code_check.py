"""Executable checks for small, complete generated functions (sandbox only)."""
import ast
import json
import re
import subprocess
import sys

PROMPT = '''Return Python code only, without imports, for these four pure functions:
lower_bound(values, target): first index whose value is >= target in a sorted list, or len(values).
merge_intervals(intervals): merge overlapping or touching closed integer intervals into sorted tuples; do not mutate the input.
run_length(values): encode a list as a list of (value, consecutive_count) tuples; empty input returns [].
stable_unique(values): deduplicate hashable values, preserving first occurrence, accepting a generator.
Do not include tests or explanations. Use straightforward implementations.'''

CHECKS = '''
assert lower_bound([], 4) == 0
assert [lower_bound([1,2,2,5], x) for x in (0,1,2,3,5,6)] == [0,0,1,3,3,4]
data = [(5,7),(1,3),(3,4),(10,10),(6,8)]
before = list(data)
assert merge_intervals(data) == [(1,4),(5,8),(10,10)]
assert data == before and merge_intervals([]) == []
assert merge_intervals([(1,9),(2,3)]) == [(1,9)]
assert run_length([]) == []
assert run_length(['a','a','b','a']) == [('a',2),('b',1),('a',1)]
assert stable_unique(x for x in [3,1,3,2,1]) == [3,1,2]
assert stable_unique([]) == []
'''


def check(text):
    code = re.sub(r'^```(?:python)?\s*\n|\n```\s*$','',text.strip())
    tree = ast.parse(code)
    if any(not isinstance(n,ast.FunctionDef) or n.decorator_list for n in tree.body):
        raise ValueError('expected plain function definitions only')
    # Generated code executes only in a bounded local child, with no imports,
    # filesystem/network builtins, or access to the surrounding process.
    if any(isinstance(n,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal)) or
           (isinstance(n,ast.Name) and n.id.startswith('__')) or
           (isinstance(n,ast.Attribute) and n.attr.startswith('__')) for n in ast.walk(tree)):
        raise ValueError('unexpected operation in generated pure functions')
    worker = '''import json,resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(2,2))
resource.setrlimit(resource.RLIMIT_AS,(128*1024*1024,128*1024*1024))
allowed={k:__builtins__.__dict__[k] for k in ('len','range','list','tuple','set','dict','sorted','enumerate','zip','min','max','int','str','bool','float','abs','all','any','reversed')}
ns={'__builtins__':allowed}
exec(compile(sys.stdin.read(),'<generated>','exec'),ns)
print('functional checks passed')
'''
    result = subprocess.run([sys.executable,'-I','-c',worker],input=code+'\n'+CHECKS,
                            text=True,capture_output=True,timeout=5)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    return result.stdout.strip()


def run(base,out,guard,costs=None):
    from adaptive_spec import generate,idle_check,request_body,add_costs
    for mode,cap in [('fixed',7),('adaptive',7)] if costs else [('fixed',7),('fixed',3)]:
        idle_check(base)
        guard.preflight(seconds=4)
        label=out.name+f'-{mode}-k{cap}'
        body=request_body(PROMPT,cap,label,768)
        body['vllm_xargs']['spec_policy']=mode
        if costs:
            add_costs(body,costs)
        result=generate(base,body,guard)
        # Preserve failures as evidence, too.
        (out/(label+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        result['functional_check']=check(result['text'])
        (out/(label+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(label,result['functional_check'],flush=True)
