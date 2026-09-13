#!/usr/bin/env python3
"""Release-runtime regression gates; run inside the built image.

CPU mode exercises real vLLM configuration, metadata aggregation and disk I/O.
--cuda also tests real sparse MLA LSE, graph replay and slab transfers.
"""
import argparse
import dataclasses
import hashlib
import json
import math
import tempfile
import time
from pathlib import Path

import torch
from vllm.v1.kv_offload import base as b
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig, OffloadingConfig, OffloadingGroupConfig,
    OffloadingModelConfig, OffloadingParallelConfig,
)
from vllm.v1.kv_offload.tiering import multinode as mn


def config(root):
    return OffloadingConfig(
        groups=(OffloadingGroupConfig(128, ('target',)), OffloadingGroupConfig(64, ('draft',))),
        worker_kv_bytes_per_block=128, enable_kv_cache_events=False,
        extra_config={'root_dir': str(root), 'disk_bytes_per_rank': 1024*1024,
                      'bounce_blocks': 2, 'content_identity': {'test': 'v029'}},
        engine_id='regression', model=OffloadingModelConfig('regression-model', 'fp8'),
        cache=OffloadingCacheConfig(64, 1),
        parallel=OffloadingParallelConfig(0, 4, 4, 1, 1, 2, 0, 1, None, False),
    )


def key(i, group=0):
    return b.make_offload_key(hashlib.sha256(str(i).encode()).digest(), group)


def cpu_checks(root):
    spec = mn.MultiNodeSlabOffloadingSpec(config(root))
    assert spec.tokens_per_block == (128, 64)
    assert spec.tokens_per_hash == 64
    mapper = mn.make_file_mapper(spec, str(root))
    other = mn.MultiNodeSlabOffloadingSpec(dataclasses.replace(config(root),
        parallel=dataclasses.replace(config(root).parallel, rank=3)))
    assert mn.make_file_mapper(other, str(root)).base_path == mapper.base_path
    changed = dataclasses.replace(config(root), extra_config={**config(root).extra_config,
                                                           'content_identity': {'test':'changed'}})
    assert mn.make_file_mapper(mn.MultiNodeSlabOffloadingSpec(changed), str(root)).base_path != mapper.base_path
    meta = mn.MultiNodeWorkerMetadata(completed_jobs={1:1}, failed_jobs={1})
    merged = meta.aggregate(mn.MultiNodeWorkerMetadata(completed_jobs={1:1, 2:1}))
    assert merged.completed_jobs == {1:2, 2:1} and merged.failed_jobs == {1}
    sizes, counts = mn.slab_geometry([128,64], 1024*1024)
    rank_dir = Path(mapper.base_path + '_r0')
    io = mn.SlabIO(str(rank_dir), sizes, counts, epoch=3)
    payload = memoryview(bytearray(range(128)))
    io.write(key(1), 0, [payload], seq=1)
    io.sync(0,[0])
    result = memoryview(bytearray(128))
    io.read(key(1),0,[result],epoch=3)
    assert result == payload
    try:
        io.read(key(1),0,[result],epoch=2)
    except OSError:
        pass
    else:
        raise AssertionError('stale epoch accepted')
    with open(rank_dir/'g0.slab','r+b') as f:
        f.seek(mn.SLAB_HEADER_BYTES+3);f.write(b'\xff');f.flush()
    try:
        io.read(key(1),0,[result],epoch=3)
    except OSError:
        pass
    else:
        raise AssertionError('corrupt payload accepted')
    io.close()
    # Real enum contract: False/None from the old API must never reach the scheduler.
    manager=spec.get_manager()
    assert manager.lookup(key(9),b.ReqContext('r')) is b.LookupResult.MISS
    manager.on_schedule_end(None)
    # Recovery indexes only the exact engine/content namespace, even when a
    # newer incompatible directory exists beside it.
    wrong_root=Path(mapper.base_path+'-wrong_r0');wrong_root.mkdir()
    mn._atomic_write_json(rank_dir/mn.SLAB_META, {'slot_counts':counts,'slot_bytes':sizes})
    mn._atomic_write_text(str(rank_dir/mn.SLAB_EPOCH),'3')
    assert manager.lookup(key(1),b.ReqContext('r')) is b.LookupResult.HIT
    ctx=b.ReqContext('r')
    manager.prepare_load([key(1)],ctx)
    manager.mark_load_failed([key(1)])
    assert manager.lookup(key(1),ctx) is b.LookupResult.MISS
    manager.complete_load([key(1)],ctx)
    pending=manager.prepare_store([key(9)],ctx)
    assert pending is not None
    assert manager.lookup(key(9),ctx) is b.LookupResult.HIT_PENDING
    manager.mark_store_failed([key(9)]);manager.complete_store([key(9)],ctx)
    assert manager.lookup(key(9),ctx) is b.LookupResult.MISS
    epoch=manager.epoch;manager.reset_cache();assert manager.epoch==epoch+1
    manager.shutdown()
    print('CPU: configuration, rank namespaces, completion aggregation, CRC/epoch checks passed',flush=True)


def cuda_checks(root):
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    q=torch.zeros((3,1,32,576),dtype=torch.bfloat16,device='cuda')
    cache=torch.zeros((1,1,64,656),dtype=torch.uint8,device='cuda')
    # Packed FP8 vectors are zero; all four arbitrary-fp32 scales are one.
    cache[...,512:528]=torch.ones((1,1,64,4),dtype=torch.float32,device='cuda').view(torch.uint8)
    indices=torch.full((3,1,2048),-1,dtype=torch.int32,device='cuda')
    for row,count in enumerate((1,3,64)):
        indices[row,0,:count]=torch.arange(count,device='cuda')
    lengths=torch.tensor([1,3,64],dtype=torch.int32,device='cuda')
    workspace=torch.empty(128*1024*1024,dtype=torch.uint8,device='cuda')
    def attention():
        return trtllm_batch_decode_with_kv_cache_mla(
            q,cache,workspace,192,512,64,indices,lengths,2048,
            sparse_mla_top_k=2048,bmm1_scale=1.,kv_scale_format='arbitrary_fp32',return_lse=True)
    out,lse=attention();torch.cuda.synchronize()
    assert torch.isfinite(out).all() and torch.count_nonzero(out)==0
    expected=lengths.float().log2()[:,None].expand(3,32)
    torch.testing.assert_close(lse.reshape(3,32),expected,atol=0.002,rtol=0.002)
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        attention();attention()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gout,glse=attention()
    for _ in range(4):
        graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(glse.reshape(3,32),expected,atol=0.002,rtol=0.002)
    # Execute the shipped backend, including DCP filtering, strided physical
    # pages, candidate compaction and the empty-shard identity.
    from types import SimpleNamespace as NS
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import FlashInferMLASparseSM120Impl
    selections=[[0,2,4],[1,3],list(range(8)),[]]
    selected=torch.full((4,2048),-1,dtype=torch.int32,device='cuda')
    for row,ids in enumerate(selections):
        selected[row,:len(ids)]=torch.tensor(ids,dtype=torch.int32,device='cuda')
    metadata=NS(req_id_per_token=torch.zeros(4,dtype=torch.int32,device='cuda'),
        block_table=torch.tensor([[2]],dtype=torch.int32,device='cuda'),
        block_size=64,topk_tokens=2048,cp_kv_cache_interleave_size=1)
    for page_stride in (128*656, 78*64*(656+132)):
        for rank in (0,1):
            impl=object.__new__(FlashInferMLASparseSM120Impl)
            impl.topk_indices_buffer=selected
            impl.dcp_world_size=2;impl.dcp_rank=rank
            impl.num_heads=16;impl.kv_lora_rank=512;impl.qk_nope_head_dim=192
            impl.qk_rope_head_dim=64;impl.scale=1.;impl.kv_scale_format='arbitrary_fp32'
            impl.need_to_return_lse_for_decode=True;impl._workspace_buffer=workspace
            assert not impl.lse_base_on_e
            backing=torch.full((3,page_stride),255,dtype=torch.uint8,device='cuda')
            # A nonzero storage offset and a block stride not divisible by 656
            # reproduce native BLHNC's mixed MLA/indexer pages.
            paged=backing[:,8448:8448+64*656].view(3,64,656);paged.zero_()
            values=(torch.arange(64,device='cuda')*2+rank).to(torch.float8_e4m3fn)
            paged[2,:,:512]=values.view(torch.uint8)[:,None].expand(64,512)
            paged[...,512:528]=torch.ones((3,64,4),device='cuda').view(torch.uint8)
            actual,local_lse=impl.forward_mqa(torch.zeros((4,32,576),dtype=torch.bfloat16,device='cuda'),paged,metadata,None)
            for row,ids in enumerate(selections):
                owned=[i for i in ids if i%2==rank]
                if not owned:
                    assert torch.count_nonzero(actual[row])==0
                    assert torch.isneginf(local_lse[row]).all()
                else:
                    expected_value=torch.tensor(owned,device='cuda').to(torch.float8_e4m3fn).float().mean()
                    torch.testing.assert_close(actual[row].float(),expected_value.expand(32,512),atol=.03,rtol=.02)
                    torch.testing.assert_close(local_lse[row],torch.full((32,),math.log2(len(owned)),device='cuda'),atol=.002,rtol=.002)
    print('CUDA: shipped DCP backend, padded physical pages and empty shards passed',flush=True)
    # Exercise the new worker API and CUDA copy implementation against a strided page view.
    tensors=[torch.arange(8*256,dtype=torch.int64,device='cuda').to(torch.int8).view(8,256)[:, :128],
             torch.arange(8*64,dtype=torch.int64,device='cuda').to(torch.int8).view(8,64)]
    canonical=b.CanonicalKVCaches([b.CanonicalKVCacheTensor(t, t.shape[1]) for t in tensors],
                                  [[b.CanonicalKVCacheRef(0,128)],[b.CanonicalKVCacheRef(1,64)]])
    spec=mn.MultiNodeSlabOffloadingSpec(config(root/'cuda'))
    worker=spec.get_worker(canonical);manager=spec.get_manager();ctx=b.ReqContext('cuda')
    keys=[key(41),key(42,1)]
    store=manager.prepare_store(keys,ctx);assert store is not None
    expected_pages=[t[1].clone() for t in tensors]
    assert worker.submit_store(1,b.GPULoadStoreSpec([1,1],group_sizes=[1,1],block_indices=[0,0]),store.store_spec)
    worker.wait({1});results=worker.get_finished();assert len(results)==1 and results[0].success
    manager.complete_store(keys,ctx)
    assert manager.lookup(keys[0],ctx) is b.LookupResult.HIT
    for t in tensors:t[2].fill_(-13)
    assert worker.submit_load(2,manager.prepare_load(keys,ctx),b.GPULoadStoreSpec([2,2],group_sizes=[1,1],block_indices=[0,0]))
    worker.wait({2});results=worker.get_finished();assert len(results)==1 and results[0].success
    torch.cuda.synchronize()
    for t,expected_page in zip(tensors,expected_pages):torch.testing.assert_close(t[2],expected_page)
    manager.complete_load(keys,ctx);worker.shutdown();manager.shutdown()
    print('CUDA: sparse base-2 LSE, four graph replays, strided-page slab roundtrip passed',flush=True)


def sparse_semantic_cuda_checks(num_tokens=172, num_heads=32):
    from types import SimpleNamespace as NS
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import FlashInferMLASparseSM120Impl
    selections = [list(range(n+1)) for n in range(num_tokens)]
    rows = len(selections)
    selected = torch.full((rows, 2048), -1, dtype=torch.int32, device='cuda')
    for row, ids in enumerate(selections):
        selected[row, :len(ids)] = torch.tensor(ids, device='cuda')
    metadata = NS(req_id_per_token=torch.zeros(rows, dtype=torch.int32, device='cuda'),
        block_table=torch.tensor([[2, 1]], dtype=torch.int32, device='cuda'),
        block_size=64, topk_tokens=2048, cp_kv_cache_interleave_size=1)
    workspace = torch.empty(128*1024*1024, dtype=torch.uint8, device='cuda')
    q = torch.zeros((rows, num_heads, 576), dtype=torch.bfloat16, device='cuda')
    q[..., 0] = .125
    q[..., 512] = .25
    local_results = []
    for rank in (0, 1):
        impl = object.__new__(FlashInferMLASparseSM120Impl)
        impl.topk_indices_buffer = selected
        impl.dcp_world_size=2; impl.dcp_rank=rank
        impl.num_heads=num_heads//2; impl.kv_lora_rank=512; impl.qk_nope_head_dim=192
        impl.qk_rope_head_dim=64; impl.scale=1.; impl.kv_scale_format='arbitrary_fp32'
        impl.need_to_return_lse_for_decode=True; impl._workspace_buffer=workspace
        backing = torch.zeros((3, 3452160), dtype=torch.uint8, device='cuda')
        cache = backing[:, 8448:8448+64*656].view(3, 64, 656)
        for logical, physical in enumerate((2, 1)):
            values = ((torch.arange(64, device='cuda')+logical*64)*2+rank)%8
            cache[physical, :, :512] = values.to(torch.float8_e4m3fn).view(torch.uint8)[:,None]
            cache[physical,:,528:530].view(torch.bfloat16)[:,0] = values*.5
        cache[...,512:528]=torch.ones((3,64,4),device='cuda').view(torch.uint8)
        output, lse = impl.forward_mqa(q, cache, metadata, None)
        local_results.append((output.float(), lse))
        for row, ids in enumerate(selections):
            owned = [i%8 for i in ids if i%2==rank]
            if not owned:
                assert not output[row].any() and torch.isneginf(lse[row]).all()
                continue
            values = torch.tensor(owned, dtype=torch.float32, device='cuda')
            scores = values*.25
            expected = (scores.softmax(0)*values).sum()
            torch.testing.assert_close(output[row].float(), expected.expand(num_heads,512), atol=.04, rtol=.02, msg=f'rank={rank} row={row} actual={output[row,0,0].item()} lse={lse[row,0].item()} expected={expected.item()}')
            expected_lse = scores.logsumexp(0)/math.log(2)
            torch.testing.assert_close(lse[row], expected_lse.expand(num_heads), atol=.003, rtol=.003)
    print(f'CUDA: {num_tokens} tokens/{num_heads} heads, nonzero multi-page DCP attention and base-2 LSE match dense oracle', flush=True)


def block_table_cuda_checks():
    from vllm.v1.worker.gpu.block_table import BlockTables
    positions = list(range(90108, 90120)) + list(range(170120, 170132))
    mapping = torch.tensor([2, 0], dtype=torch.int32, device='cuda')
    starts = torch.tensor([0, 12, 24], dtype=torch.int32, device='cuda')
    pos = torch.tensor(positions, dtype=torch.int64, device='cuda')
    for cp in (1, 2, 4):
        for rank in range(cp):
            for interleave in (1, 4):
                bt = BlockTables([64, 64], 3, 128,
                    [math.ceil(180224 / (64 * cp)), math.ceil(180224 / 64)],
                    torch.device('cuda'), [64, 64], cp_size=cp, cp_rank=rank,
                    cp_interleave=interleave, cp_sizes=[cp, 1])
                for row in (0, 2):
                    groups = tuple([500 + row * 4000 + 3 * i
                                    for i in range(t.gpu.shape[1])]
                                   for t in bt.block_tables)
                    bt.append_block_ids(row, groups, overwrite=True)
                bt.apply_staged_writes()
                gathered = bt.gather_block_tables(mapping, 3)
                for group, tensor in enumerate(gathered):
                    torch.testing.assert_close(tensor[0], bt.block_tables[group].gpu[2])
                    assert not tensor[2].any()
                actual = bt.compute_slot_mappings(mapping, starts, pos, 32)
                expected = torch.full((2, 32), -1, dtype=torch.int64)
                for group, size in enumerate((cp, 1)):
                    for i, position in enumerate(positions):
                        owner = (position // interleave) % size
                        local = (position // (interleave * size)) * interleave + position % interleave
                        if size == 1 or owner == rank:
                            row = 2 if i < 12 else 0
                            block = 500 + row * 4000 + 3 * (local // 64)
                            expected[group, i] = block * 64 + local % 64
                torch.testing.assert_close(actual.cpu(), expected)
                assert (bt.slot_mappings[:, 32:] == -1).all()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replayed = bt.compute_slot_mappings(mapping, starts, pos, 32)
                graph.replay()
                torch.testing.assert_close(replayed.cpu(), expected)
    print('CUDA: native block-table staging, gather, long slots and graph replay for CP 1/2/4 passed', flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cuda',action='store_true');args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='vllm029-regression-') as root:
        root=Path(root);cpu_checks(root)
        if args.cuda:
            block_table_cuda_checks()
            for count, heads in ((64,32),(65,16),(172,32),(172,64)):
                sparse_semantic_cuda_checks(count,heads)
            cuda_checks(root)
    print(json.dumps({'ok':True,'cuda':args.cuda}),flush=True)

if __name__=='__main__':main()
