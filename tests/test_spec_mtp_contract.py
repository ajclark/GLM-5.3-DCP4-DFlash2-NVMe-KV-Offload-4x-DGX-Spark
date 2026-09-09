"""Exercise actual captured model/caller methods on CPU, without model weights."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from spec_harness import source_class

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/spec_mtp"


def model_and_worker(patched):
    ns = dict(torch=torch, nn=nn, Module=nn.Module,
              support_torch_compile=lambda cls: cls,
              CUDAGraphMode=NS(NONE="none", PIECEWISE="piecewise"),
              BatchDescriptor=lambda **kw: NS(**kw),
              set_forward_context=lambda *args, **kw: nullcontext())
    layer_type = source_class(FIXTURE / "deepseek_mtp.py",
                              "DeepSeekMultiTokenPredictorLayer", ["forward"],
                              ns, bases=["Module"])
    inner_type = source_class(FIXTURE / "deepseek_mtp.py",
                              "DeepSeekMultiTokenPredictor",
                              ["forward", "compute_logits"], ns, bases=["Module"])
    model_type = source_class(FIXTURE / "deepseek_mtp.py", "DeepSeekMTP",
                              ["forward", "compute_logits"], ns, bases=["Module"])

    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.RMSNorm(4, eps=1e-5)
            self.norm.weight.data.copy_(torch.tensor([1., 2., 3., 4.]))
            self.head = nn.Linear(4, 6, bias=False)
            self.head.weight.data.copy_(torch.arange(24).reshape(6, 4) / 13 - .7)

        def forward(self, x):
            return self.norm(x)

    class Block(nn.Module):
        def forward(self, positions, hidden_states, residual):
            return hidden_states * 1.7, torch.full_like(hidden_states, .3)

    layer = layer_type()
    layer.enorm, layer.hnorm = nn.RMSNorm(4), nn.RMSNorm(4)
    layer.eh_proj = nn.Linear(8, 4, bias=False)
    layer.eh_proj.weight.data.copy_(torch.arange(32).reshape(4, 8) / 19 - .5)
    layer.mtp_block, layer.shared_head = Block(), Shared()
    inner = inner_type()
    inner.mtp_start_layer_idx, inner.num_mtp_layers = 78, 1
    inner.layers = nn.ModuleDict({"78": layer})
    inner.embed_tokens = nn.Embedding(6, 4)
    inner.embed_tokens.weight.data.copy_(torch.arange(24).reshape(6, 4) / 7 + .1)
    inner.logits_processor = lambda head, hidden: head(hidden)
    model = model_type()
    model.model = inner

    source_class(FIXTURE / "autoregressive.py", "AutoRegressiveSpeculator",
                 ["_run_model"], ns, bases=["object"])
    path = ROOT / "experiments/mtp/speculator.py" if patched else FIXTURE / "speculator.py"
    worker_type = source_class(path, "MTPSpeculator", ["model_returns_tuple"], ns)
    worker = worker_type()
    worker.supports_mm_inputs = False
    worker.model, worker.vllm_config = model, NS()
    worker.draft_model_config = NS(hf_config=NS(architectures=["DeepSeekMTPModel"]))
    worker.input_buffers = NS(input_ids=torch.tensor([1, 2]),
                             positions=torch.tensor([0, 8]))
    worker.hidden_states = torch.tensor([[.5, 1., 2., 3.], [1., 4., 2., 3.]])
    return model, worker


def test_captured_runtime_source_identity():
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest


def test_original_v2_contract_reproduces_tuple_as_tensor_failure():
    _, worker = model_and_worker(patched=False)
    logits_hidden, recycled_hidden = worker._run_model(2, None, None, None)
    assert isinstance(logits_hidden, tuple)
    assert logits_hidden is recycled_hidden
    # _prefill indexes this result with the tensor of last-token indices.
    with pytest.raises(TypeError):
        logits_hidden[torch.tensor([0, 1])]


@pytest.mark.parametrize("steps", [1, 2, 3])
@torch.inference_mode()
def test_patched_caller_preserves_logits_and_recycle_normalization(steps):
    model, worker = model_and_worker(patched=True)
    layer = model.model.layers["78"]
    for _ in range(steps):
        logits_hidden, recycled_hidden = worker._run_model(2, None, None, None)
        assert isinstance(logits_hidden, torch.Tensor)
        assert isinstance(recycled_hidden, torch.Tensor)
        assert torch.isfinite(recycled_hidden).all()
        torch.testing.assert_close(recycled_hidden, layer.shared_head(logits_hidden))
        assert not torch.allclose(logits_hidden, recycled_hidden)
        logits = model.compute_logits(logits_hidden)
        torch.testing.assert_close(logits, layer.shared_head.head(recycled_hidden))
        # Passing the recycled tensor to compute_logits would normalize twice.
        assert not torch.allclose(logits, model.compute_logits(recycled_hidden))
        worker.hidden_states = recycled_hidden.clone()
        worker.input_buffers.input_ids = logits.argmax(dim=-1)
        worker.input_buffers.positions += 1


@pytest.mark.parametrize("architecture", ["DeepSeekV4MTPModel", "Glm4MoeMTPModel"])
def test_other_architectures_keep_original_single_tensor_contract(architecture):
    _, worker = model_and_worker(patched=True)
    worker.draft_model_config.hf_config.architectures = [architecture]
    assert worker.model_returns_tuple is False
