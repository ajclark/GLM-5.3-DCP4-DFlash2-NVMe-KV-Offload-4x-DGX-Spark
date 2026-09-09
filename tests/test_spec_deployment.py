"""Rollback and allocation separation are tested without access to Docker."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'bench'))
import spec_node as N


def setup(root):
    old = {'Id':'old-id', 'Image':'old-image'}
    (root/'original.json').write_text(json.dumps(old))
    return old


def test_launch_cannot_overlap_original_model(tmp_path,monkeypatch):
    setup(tmp_path)
    monkeypatch.setattr(N,'inspect',lambda name=N.NAME:{'State':{'Running':True}})
    with pytest.raises(RuntimeError,match='overlap'):
        N.launch(tmp_path,'label',0)


def test_pressure_trip_prevents_launch(tmp_path):
    (tmp_path/'tripped').write_text('memory pressure')
    with pytest.raises(RuntimeError,match='memory pressure'):
        N.launch(tmp_path,'label',0)


def test_restore_refuses_unrelated_container(tmp_path,monkeypatch):
    setup(tmp_path)
    monkeypatch.setattr(N,'inspect',lambda name=N.NAME:{'Id':'unrelated','Config':{'Labels':{}}})
    with pytest.raises(RuntimeError,match='unrelated'):
        N.restore(tmp_path,'label')


def test_restore_retains_and_restarts_exact_original(tmp_path,monkeypatch):
    setup(tmp_path)
    calls = []
    monkeypatch.setattr(N,'inspect',lambda name=N.NAME:
        {'Id':'experiment','Config':{'Labels':{'glm.spec.experiment':'label'}}} if name==N.NAME else
        {'Id':'old-id','Name':'/backup','State':{'Running':False}})
    monkeypatch.setattr(N,'command',lambda *args,**kw:calls.append(args) or NS(stdout=''))
    monkeypatch.setattr(N.subprocess,'run',lambda *args,**kw:NS(returncode=0))
    N.restore(tmp_path,'label')
    assert ('docker','stop','-t','5',N.NAME) in calls
    assert ('docker','rename',N.NAME,N.NAME+'_finished_label') in calls
    assert ('docker','rename','old-id',N.NAME) in calls
    assert ('docker','start','old-id') in calls
    assert not any('rm' in c for c in calls)


def test_stage_matches_changes_and_checksums():
    import hashlib
    for rel in ('v1/core/sched/scheduler.py','v1/spec_decode/adaptive.py',
                'v1/worker/gpu/model_runner.py','v1/worker/gpu/cudagraph_utils.py'):
        assert (ROOT/'overlay/vllm'/rel).read_bytes() == (ROOT/'stage/glm-dcp'/Path(rel).name).read_bytes()
    assert (ROOT/'overlay/vllm/v1/worker/gpu/block_table.py').read_bytes() == (
        ROOT/'stage/glm-dcp/v2_block_table.py').read_bytes()
    for line in (ROOT/'stage/SHA256SUMS').read_text().splitlines():
        digest,name = line.split(maxsplit=1)
        assert '__pycache__' not in name
        assert hashlib.sha256((ROOT/'stage'/name).read_bytes()).hexdigest() == digest


def test_smoke_catches_skipped_or_repeated_positions():
    from spec_experiment import validate_count_smoke
    valid = '\n'.join(str(n) for n in range(1,20))
    validate_count_smoke(valid)
    for bad in (valid.replace('\n8\n','\n9\n'), valid.replace('\n8\n','\n'), 'OK'):
        with pytest.raises(RuntimeError,match='count check'):
            validate_count_smoke(bad)


def cache_pair(tmp_path,monkeypatch):
    import hashlib
    source,destination=tmp_path/'source',tmp_path/'destination'
    kernels=tmp_path/'kernels';kernels.mkdir()
    (kernels/'sparse_mla_kernels.py').write_text('kernel implementation')
    old={'Id':'original','Image':'image','Config':{'Cmd':['serve','model']},
         'Mounts':[{'Source':str(kernels/'sparse_mla_kernels.py'),'Destination':'/ops/sparse_mla_kernels.py'}]}
    for root in (source,destination):
        root.mkdir();(root/'kvcache').mkdir();(root/'dcp').mkdir()
        (root/'dcp/model.py').write_text('identical model implementation')
        (root/'original.json').write_text(json.dumps(old))
        (root/'mount-manifest.json').write_text(json.dumps([{'destination':'kernel.py','sha256':'same'}]))
        (root/'launch.sh').write_text('same launch')
        (root/'expected-runtime.json').write_text('{"runner":"same"}')
    (source/'disarm').touch()
    (source/'kvcache/headers').write_bytes(b'durable cache data')
    previous={'State':{'Running':False,'OOMKilled':False},
              'Config':{'Labels':{'glm.spec.experiment':'source'},'Cmd':['serve','--kv-transfer-config',
                        json.dumps({'kv_connector_extra_config':{'slab_salt':hashlib.sha256(
                            b'identical model implementationkernel implementation').hexdigest()[:16]}})]},
              'Mounts':[{'Destination':'/kvcache','Source':str(source/'kvcache')}]}
    monkeypatch.setattr(N,'inspect',lambda name:previous)
    return source,destination,previous


def test_identical_stopped_cache_can_be_reloaded_without_copying_or_resalting(tmp_path,monkeypatch):
    source,destination,_=cache_pair(tmp_path,monkeypatch)
    N.reuse_cache(destination,'destination','source')
    assert (destination/'kvcache').is_symlink()
    assert (destination/'kvcache/headers').read_bytes()==b'durable cache data'
    assert (source/'kvcache/headers').read_bytes()==b'durable cache data'


@pytest.mark.parametrize('change',('source','kernel','unmounted_kernel','running','trip','occupied'))
def test_cache_reuse_rejects_changed_identity_active_writer_and_existing_data(tmp_path,monkeypatch,change):
    source,destination,previous=cache_pair(tmp_path,monkeypatch)
    if change=='source':
        (destination/'dcp/model.py').write_text('different model implementation')
    elif change=='kernel':
        (destination/'mount-manifest.json').write_text('[]')
    elif change=='unmounted_kernel':
        (tmp_path/'kernels/unmounted.py').write_text('also affects slab identity')
    elif change=='running':
        previous['State']['Running']=True
    elif change=='trip':
        (source/'tripped').write_text('pressure')
    elif change=='occupied':
        (destination/'kvcache/existing').write_text('keep this')
    with pytest.raises(RuntimeError):
        N.reuse_cache(destination,'destination','source')
    assert not (destination/'kvcache').is_symlink()
    assert (source/'kvcache/headers').read_bytes()==b'durable cache data'


@pytest.mark.parametrize('trace_off',(False,True))
def test_trace_control_keeps_policy_and_allocation_identical(tmp_path,monkeypatch,trace_off):
    setup(tmp_path)
    if trace_off:(tmp_path/'trace-disabled').touch()
    monkeypatch.setattr(N,'inspect',lambda name=N.NAME:
                        {'State':{'Running':False}} if name=='old-id' else None)
    calls=[]
    monkeypatch.setattr(N.subprocess,'run',lambda *args,**kw:calls.append(kw) or NS(returncode=0))
    N.launch(tmp_path,'label',0)
    env=calls[0]['env']
    assert env['GLM_SPEC_TRACE']==('' if trace_off else '/kvcache/spec-trace.jsonl')
    assert env['GLM_SPEC_POLICY']=='shadow'
    assert env['KVBYTES']=='6000000000' and env['MAXLEN']=='180224'
