"""Curate follow-up evidence without mutating original measurements."""
import copy,hashlib,io,json,re,sys,tarfile
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'bench'))
from spec_request_report import prompt_digest
ROOT=Path.cwd()
OUT=Path('/tmp/adaptive-next-publication')
PI_SAFE={'wire-report.json','live-pi-report.json','pi-control-quality-r2.json','live-pi-trace.jsonl','pi-code-quality.json','pi-prose-output.json','intent-prompts.json','pi-events.jsonl','next-boot-declaration.json','requests.json','memory.jsonl','device-power.jsonl','ready.json','monitor-summary.json'}

def private_pi(path):
    return any(part.startswith('pi-') for part in path.parts) or 'activation' in path.parts

def choose(path):
    rel=path.relative_to(ROOT)
    if not path.is_file() or path.is_symlink() or '__pycache__' in rel.parts:
        return False
    if path.name in ('measurements.tar.gz','ARTIFACT-PROVENANCE.json') or path.suffix not in ('.json','.jsonl','.py','.log','.md','.txt','.sh'):
        return False
    if any(part in ('sessions','payloads') for part in rel.parts):
        return False
    if path.name in ('original.json','prepared.json'):
        return False
    if path.name=='herdr-monitor-report.json':
        return True
    if path.name.startswith('herdr-'):
        return 'watch' in path.name and path.suffix=='.jsonl'
    if any(p in ('pi-hints-r1','pi-wire-r1') for p in rel.parts):
        start=next(i for i,p in enumerate(rel.parts) if p in ('pi-hints-r1','pi-wire-r1'))
        suffix=rel.parts[start+1:]
        return (len(suffix)==1 and path.name in PI_SAFE) or (len(suffix)>1 and (suffix[0].startswith(('hint-controls','confidence-controls')) or suffix[0].startswith('live-pi-monitor')))
    return True

def remove_prompt_fields(value):
    if isinstance(value,dict):
        if 'messages' in value:
            raise ValueError('unexpected Pi request body in public selection')
        return {k:remove_prompt_fields(v) for k,v in value.items() if k not in ('prompt_token_ids','prompt_text')}
    if isinstance(value,list):
        return [remove_prompt_fields(v) for v in value]
    return value

def transform(path,raw):
    rel=path.relative_to(ROOT)
    changes=[]
    if path.name.startswith('herdr-') and path.suffix=='.jsonl':
        allowed=('time','agent','status','assessment','error_candidate','blocked_candidate')
        rows=[{k:r[k] for k in allowed if k in r} for r in map(json.loads,raw.splitlines())]
        raw=(''.join(json.dumps(r)+'\n' for r in rows)).encode();changes.append('native status/assessment only; pane and state payloads private')
    elif private_pi(rel) and path.suffix=='.json':
        row=json.loads(raw)
        if isinstance(row,dict) and {'chunks','token_ids'}<=row.keys():
            digest=prompt_digest(row)
            row=remove_prompt_fields(row)
            row.update(prompt_token_sha256=digest,private_prompt_ids_removed=True)
            assert prompt_digest(row)==digest
            raw=(json.dumps(row,indent=2)+'\n').encode();changes.append('private prompt token IDs/text removed; verified token digest retained')
        elif 'prompt_token_ids' in raw.decode():
            raise ValueError('unhandled private prompt tokens: '+str(rel))
    text=raw.decode()
    scrubbed=text.replace(str(Path.home()),'~')
    scrubbed=re.sub(r'192\.168\.(\d+)\.(\d+)',r'10.99.\1.\2',scrubbed)
    if scrubbed!=text:
        changes.append('personal home/private IP placeholders')
    return scrubbed.encode(),changes

def main():
    OUT.mkdir(exist_ok=False)
    paths=list((ROOT/'results/adaptive-next').rglob('*'))
    for label in ('cache-width-20260909-r1','hints-conf-20260909-r1','hints-conf-20260909-r2','atomic-off-20260909-r1'):
        paths.extend((ROOT/'results/adaptive-spec'/label).rglob('*'))
    manifest={}
    for path in sorted(set(paths)):
        if not choose(path):continue
        rel=path.relative_to(ROOT); original=path.read_bytes()
        published,changes=transform(path,original)
        dest=OUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(published)
        manifest[str(rel)]={'original_sha256':hashlib.sha256(original).hexdigest(),'published_sha256':hashlib.sha256(published).hexdigest(),'bytes':len(published),'transformations':changes}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'files':len(manifest),'bytes':sum(v['bytes'] for v in manifest.values()),'changed':sum(bool(v['transformations']) for v in manifest.values())}))
if __name__=='__main__':main()
