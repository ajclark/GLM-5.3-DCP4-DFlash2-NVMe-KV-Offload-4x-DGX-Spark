#!/usr/bin/env python3
"""Exercise release DeepGEMM indexer kernels on SM121 before a model load."""
import torch
from vllm.utils.deep_gemm import fp8_fp4_mqa_logits,fp8_fp4_paged_mqa_logits,get_paged_mqa_logits_metadata
q=torch.ones((4,32,128),device='cuda').to(torch.float8_e4m3fn)
k=torch.ones((64,128),device='cuda').to(torch.float8_e4m3fn)
scales=torch.ones(64,device='cuda')
w=torch.ones((4,32),device='cuda')
starts=torch.zeros(4,dtype=torch.int32,device='cuda')
ends=torch.tensor([1,3,63,64],dtype=torch.int32,device='cuda')
plain=fp8_fp4_mqa_logits((q,None),(k,scales),w,starts,ends,True)
cache=torch.zeros((1,64,1,132),dtype=torch.uint8,device='cuda')
cache.view(-1)[:64*128]=k.view(torch.uint8).reshape(-1)
cache.view(-1)[64*128:]=scales.view(torch.uint8)
lengths=ends[:,None].contiguous()
metadata=get_paged_mqa_logits_metadata(lengths,64,torch.cuda.get_device_properties(0).multi_processor_count)
paged=fp8_fp4_paged_mqa_logits((q[:,None],None),cache,w,lengths,
    torch.zeros((4,8),dtype=torch.int32,device='cuda'),metadata,128,False)
torch.cuda.synchronize()
for row,n in enumerate((1,3,63,64)):
    torch.testing.assert_close(plain[row,:n],torch.full((n,),4096.,device='cuda'))
    torch.testing.assert_close(paged[row,:n],plain[row,:n])
    # Production top-k masks positions using lengths; native clean_logits=True
    # is unsupported for the release's required two-dimensional bounds.
print('Release SM121 prefill and paged indexer logits passed',flush=True)

# The fused GLM path must preserve BLHNC's physical page stride. A wrong
# stride writes through neighbouring MLA layers while staying inside the
# allocation, so memcheck alone cannot catch it.
from vllm.models.deepseek_v32.common.kernels import fused_norm_rope
from vllm import _custom_ops as ops
backing=torch.full((3,3452160),85,dtype=torch.uint8,device='cuda')
reference=backing.clone()
actual_cache=backing[:,8448:8448+64*132].view(3,64,132)
reference_cache=reference[:,8448:8448+64*132].view(3,64,132)
slots=torch.tensor([129,67,-1,128],dtype=torch.int64,device='cuda')
pattern=((torch.arange(128,device='cuda')%2)*2-1).to(torch.bfloat16)
index_k=pattern[None].repeat(4,1)
index_out=torch.zeros_like(index_k)
cos_sin=torch.cat((torch.ones((4,32),device='cuda'),torch.zeros((4,32),device='cuda')),dim=1).to(torch.bfloat16)
fused_norm_rope(torch.arange(4,device='cuda'),torch.ones((4,128),device='cuda',dtype=torch.bfloat16),
    torch.ones(128,device='cuda',dtype=torch.bfloat16),1e-6,
    torch.ones((4,512),device='cuda',dtype=torch.bfloat16),
    torch.ones(512,device='cuda',dtype=torch.bfloat16),1e-6,
    torch.ones((4,64),device='cuda',dtype=torch.bfloat16),cos_sin,index_k,
    torch.ones(128,device='cuda',dtype=torch.bfloat16),torch.zeros(128,device='cuda',dtype=torch.bfloat16),
    1e-6,cos_sin,torch.empty((4,2048),device='cuda',dtype=torch.int32),
    slot_mapping=slots,indexer_k_cache=actual_cache,index_k_out=index_out)
ops.indexer_k_quant_and_cache(index_out,reference_cache,slots,128,'ue8m0')
torch.cuda.synchronize()
torch.testing.assert_close(backing,reference,atol=0,rtol=0)
print('Fused indexer writes match native reference across strided pages; neighbouring bytes and negative slots preserved',flush=True)
