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
    for rel, name in (('v1/worker/gpu/async_utils.py', 'v2_async_utils.py'),
                      ('v1/outputs.py', 'v1_outputs.py'),
                      ('v1/spec_decode/confidence_trace.py', 'confidence_trace.py'),
                      ('v1/worker/gpu/spec_decode/rejection_sampler_utils.py', 'v2_rejection_sampler_utils.py'),
                      ('v1/worker/gpu/spec_decode/rejection_sampler.py', 'v2_rejection_sampler.py'),
                      ('v1/worker/gpu/sample/states.py', 'v2_sample_states.py')):
        assert (ROOT/'overlay/vllm'/rel).read_bytes() == (ROOT/'stage/glm-dcp'/name).read_bytes()
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
    assert env['GLM_SPEC_CONFIDENCE_TRACE']=='0'
    assert env['KVBYTES']=='6000000000' and env['MAXLEN']=='180224'


def test_prepared_server_calibration_enables_hints_with_unchanged_capacity(tmp_path,monkeypatch):
    setup(tmp_path)
    (tmp_path/'kvcache').mkdir()
    for name in ('boot-costs.json','hint-priors.json'):
        (tmp_path/'kvcache'/name).write_text('{}')
    monkeypatch.setattr(N,'inspect',lambda name=N.NAME:
                        {'State':{'Running':False}} if name=='old-id' else None)
    calls=[]
    monkeypatch.setattr(N.subprocess,'run',lambda *args,**kw:calls.append(kw) or NS(returncode=0))
    N.launch(tmp_path,'label',0)
    env=calls[0]['env']
    assert env['GLM_SPEC_POLICY']=='adaptive'
    assert env['GLM_SPEC_COSTS']=='/kvcache/boot-costs.json'
    assert env['GLM_SPEC_HINT_PRIORS']=='/kvcache/hint-priors.json'
    assert (env['KVBYTES'],env['MAXLEN'],env['MAXSEQS'])==('6000000000','180224','12')


def test_confidence_diagnostics_are_explicit_and_do_not_change_policy(tmp_path,monkeypatch):
    setup(tmp_path)
    (tmp_path/'confidence-trace-enabled').touch()
    monkeypatch.setattr(N,'inspect',lambda name=N.NAME:
                        {'State':{'Running':False}} if name=='old-id' else None)
    calls=[]
    monkeypatch.setattr(N.subprocess,'run',lambda *args,**kw:calls.append(kw) or NS(returncode=0))
    N.launch(tmp_path,'label',0)
    env=calls[0]['env']
    assert env['GLM_SPEC_CONFIDENCE_TRACE']=='1'
    assert env['GLM_SPEC_POLICY']=='shadow'
    assert (env['KVBYTES'],env['MAXLEN'],env['MAXSEQS'])==('6000000000','180224','12')


def test_atomic_control_changes_only_the_explicit_isolated_environment_setting():
    from spec_experiment import experiment_launcher
    original = (ROOT/'launch-glm53big-dcp.sh').read_bytes()
    assert experiment_launcher(original) is original
    modified = experiment_launcher(original, True)
    assert modified.replace(b'VLLM_MARLIN_USE_ATOMIC_ADD=0', b'VLLM_MARLIN_USE_ATOMIC_ADD=1') == original
    assert len(modified) == len(original)
    with pytest.raises(ValueError, match='exactly one'):
        experiment_launcher(b'no explicit setting', True)
    with pytest.raises(ValueError, match='exactly one'):
        experiment_launcher(original + original, True)


def test_atomic_control_refuses_persisted_cache_reuse_before_packaging(tmp_path):
    from spec_experiment import prepare
    with pytest.raises(ValueError, match='fresh isolated cache'):
        prepare('control', tmp_path, reuse_cache_from='old', no_marlin_atomic_add=True)


def test_prepare_accepts_vetted_mounted_overlays_and_rejects_unknown(tmp_path,monkeypatch):
    """Production mounts the V2 overlay set; prepare must accept exactly those
    files (by hash) over the pristine runtime and refuse anything else."""
    import contextlib, hashlib, io
    pristine, overlay = b'pristine\n', b'overlay\n'
    src = tmp_path/'glm-dcp'; src.mkdir()
    (src/'model_runner.py').write_bytes(overlay)
    (src/'scheduler.py').write_bytes(overlay)
    root = tmp_path/'exp'; root.mkdir()
    (root/'changes').mkdir()
    (root/'expected-runtime.json').write_text(json.dumps({
        'v1/worker/gpu/model_runner.py': hashlib.sha256(pristine).hexdigest(),
        'v1/core/sched/scheduler.py': hashlib.sha256(pristine).hexdigest(),
        'v1/worker/gpu/sample/states.py': hashlib.sha256(pristine).hexdigest()}))
    vetted = json.dumps({hashlib.sha256(overlay).hexdigest():'model_runner.py'})
    (root/'vetted-overlays.json').write_text(vetted)
    base = '/usr/local/lib/python3.12/dist-packages/vllm/'
    cmd = ['--decode-context-parallel-size','2','--max-model-len','180224','--max-num-batched-tokens','2048',
           '--kv-cache-memory-bytes','6000000000','--max-num-seqs','12','--tensor-parallel-size','4']
    container = {'State':{'Running':True,'OOMKilled':False},'Config':{'Labels':{},'Cmd':cmd},'Image':'img',
                 'Mounts':[{'Source':str(src/'model_runner.py'),'Destination':base+'v1/worker/gpu/model_runner.py'},
                           {'Source':str(src/'scheduler.py'),'Destination':base+'v1/core/sched/scheduler.py'}]}
    observed = {base+'v1/worker/gpu/model_runner.py': overlay}
    def fake_command(*args, check=True):
        path = args[-1]
        return NS(stdout=hashlib.sha256(observed.get(path, pristine)).hexdigest()+'  '+path)
    monkeypatch.setattr(N,'command',fake_command)
    monkeypatch.setattr(N,'inspect',lambda name=None: container)
    monkeypatch.setattr(N.shutil,'disk_usage',lambda p: NS(free=10**12))
    monkeypatch.setattr(N.shutil,'copytree',lambda a,b,ignore=None: (root/'dcp').mkdir())
    def run():
        for leftover in ('original.json','mount-manifest.json'):
            (root/leftover).unlink(missing_ok=True)
        for d in ('kvcache','dcp'):
            if (root/d).exists(): (root/d).rmdir()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            N.prepare(root,'lbl')
        return json.loads(out.getvalue())
    # 1. vetted overlay mounted over model_runner: accepted and reported.
    assert run()['overlays_running'] == {'v1/worker/gpu/model_runner.py':'model_runner.py'}
    # 2. the same mounted bytes, but not in the vetted set: refused.
    (root/'vetted-overlays.json').write_text('{}')
    with pytest.raises(RuntimeError, match='runtime source mismatch'):
        run()
    # 3. a mismatching file whose mount source does not hash to what runs: refused.
    (root/'vetted-overlays.json').write_text(vetted)
    observed[base+'v1/core/sched/scheduler.py'] = b'edited in the container\n'
    with pytest.raises(RuntimeError, match='scheduler'):
        run()


def test_lane_overrides_are_validated_and_reach_the_launch(tmp_path,monkeypatch):
    import importlib.util, io, contextlib
    spec = importlib.util.spec_from_file_location('spec_experiment_lane', ROOT/'bench/spec_experiment.py')
    E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
    # validation
    assert E.validated_lane({'SPEC_MODE':'mtp','MTP_K':2,'KVBYTES':3_200_000_000,'MAXLEN':90112,'GLM_SPEC_POLICY':'off'}) == {
        'SPEC_MODE':'mtp','MTP_K':2,'KVBYTES':3_200_000_000,'MAXLEN':90112,'GLM_SPEC_POLICY':'off'}
    assert E.validated_lane({'MAXBATCHED':4096}) == {'SPEC_MODE':'dflash','MAXBATCHED':4096}
    assert E.validated_lane({'NCCL_IB_QPS_PER_CONNECTION':2}) == {'SPEC_MODE':'dflash','NCCL_IB_QPS_PER_CONNECTION':2}
    for bad in ({'SPEC_MODE':'mtp'}, {'SPEC_MODE':'mtp','MTP_K':4,'KVBYTES':3_200_000_000}, {'MAXBATCHED':3000}, {'NCCL_IB_QPS_PER_CONNECTION':3},
                {'SPEC_MODE':'mtp','MTP_K':2,'KVBYTES':3_200_000_000,'MAXLEN':180224},
                {'KVBYTES':1_000_000_000}, {'FOO':1}, {'GLM_SPEC_POLICY':'lossy'}):
        with pytest.raises(ValueError):
            E.validated_lane(bad)
    # launch honours lane.json and passes the spec mode to the launcher
    root = tmp_path/'exp'; root.mkdir()
    (root/'original.json').write_text(json.dumps({'Id':'orig','Image':'img'}))
    (root/'lane.json').write_text(json.dumps({'SPEC_MODE':'mtp','MTP_K':2,'KVBYTES':3_200_000_000,'MAXLEN':90112,'GLM_SPEC_POLICY':'off'}))
    (root/'launch.sh').write_text('#!/bin/bash\n')
    monkeypatch.setattr(N,'inspect',lambda name=None: {'State':{'Running':False}} if name else None)
    seen = {}
    def fake_run(cmd, env=None, **kw):
        seen['cmd'] = cmd; seen['env'] = dict(env)
        return NS(returncode=0)
    monkeypatch.setattr(N.subprocess,'run',fake_run)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        N.launch(root,'lbl',0)
    assert seen['cmd'][-1] == 'mtp'
    assert seen['env']['MTP_K'] == '2' and seen['env']['KVBYTES'] == '3200000000'
    assert seen['env']['MAXLEN'] == '90112' and seen['env']['GLM_SPEC_POLICY'] == 'off'
    assert seen['env']['GLM_SPEC_LOSSY'] == '0'
    # default lane unchanged
    (root/'lane.json').unlink()
    with contextlib.redirect_stdout(out):
        N.launch(root,'lbl',0)
    assert seen['cmd'][-1] == 'dflash' and seen['env']['KVBYTES'] == '6000000000' and seen['env']['GLM_SPEC_POLICY'] == 'shadow'
