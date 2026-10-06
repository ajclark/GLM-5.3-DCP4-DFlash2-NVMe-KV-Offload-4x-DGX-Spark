# SPDX-License-Identifier: Apache-2.0
"""Build-time check that the image's vLLM still has the shapes glm_fast patches.

    PYTHONPATH=/opt/glm-fast python3 -c "from glm_fast.install import verify_targets; ..."

Source-level only (AST of the installed files, located without importing them), so it runs on
a CPU-only build host and never executes vLLM's CUDA platform code.
"""
from __future__ import annotations

import ast
import importlib.util
import os


def vllm_root() -> str:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("vllm is not installed")
    return os.path.dirname(spec.origin)


def _parse(rel: str):
    path = os.path.join(vllm_root(), rel)
    with open(path) as f:
        return ast.parse(f.read()), path


def _cls(tree, name):
    return next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name), None)


def _fn(cls, name):
    if cls is None:
        return None
    return next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name), None)


def _params(fn):
    return [a.arg for a in fn.args.args] if fn is not None else []


def verify_targets() -> list[str]:
    problems: list[str] = []

    def need(cond, what):
        if not cond:
            problems.append(what)

    # vocab-parallel argmax: GPUModelRunner.sample and the pieces it calls.
    tree, _ = _parse("v1/worker/gpu/model_runner.py")
    runner = _cls(tree, "GPUModelRunner")
    need(runner is not None, "GPUModelRunner missing")
    sample = _fn(runner, "sample")
    need(_params(sample)[:4] == ["self", "hidden_states", "input_batch", "grammar_output"],
         f"GPUModelRunner.sample{_params(sample)}")
    tree, _ = _parse("v1/worker/gpu/input_batch.py")
    need(any(isinstance(n, ast.FunctionDef) and n.name == "get_num_sampled_and_rejected" for n in tree.body),
         "input_batch.get_num_sampled_and_rejected missing")
    tree, _ = _parse("model_executor/layers/logits_processor.py")
    lp = _cls(tree, "LogitsProcessor")
    need(_params(_fn(lp, "_apply_head"))[:4] == ["self", "lm_head", "hidden_states", "embedding_bias"],
         "LogitsProcessor._apply_head signature changed")
    tree, _ = _parse("v1/worker/gpu/sample/sampler.py")
    src = ast.unparse(_cls(tree, "Sampler"))
    for attr in ("needs_logits_processing", "trace_replay_state", "compute_nans", "return_sampling_mask",
                 "logprob_token_ids_state", "sampling_states"):
        need(f"self.{attr}" in src, f"Sampler.{attr} missing")

    # L2 prefetch: the decoder layer's fused all-reduce + norm call sites.
    tree, _ = _parse("models/deepseek_v32/nvidia/model.py")
    need(any(isinstance(n, ast.ImportFrom) and any(a.name == "fused_allreduce_rms_norm" for a in n.names)
             for n in tree.body), "model.py no longer imports fused_allreduce_rms_norm by name")
    layer_fwd = ast.unparse(_fn(_cls(tree, "DeepseekV32DecoderLayer"), "forward") or ast.Pass())
    need("fused_allreduce_rms_norm(hidden_states, residual, self.input_layernorm)" in layer_fwd.replace("\n", " ")
         or "self.input_layernorm" in layer_fwd, "decoder layer input_layernorm all-reduce site changed")
    need("self.post_attention_layernorm" in layer_fwd and layer_fwd.count("fused_allreduce_rms_norm") >= 2,
         "decoder layer post-attention all-reduce site changed")
    need(_fn(_cls(tree, "DeepseekV32Model"), "forward") is not None, "DeepseekV32Model.forward missing")
    tree, _ = _parse("v1/attention/ops/dcp.py")
    need("self.query_gather" in ast.unparse(_cls(tree, "MLADCPManager") or ast.Pass()),
         "MLADCPManager.query_gather missing")
    tree, _ = _parse("models/deepseek_v32/attention.py")
    need("self.dcp_manager.query_gather(" in ast.unparse(tree), "attention no longer calls dcp_manager.query_gather")

    # DCP glue (VLLM_DCP_GLUE): dcp_glue copies the stock correction kernel's arithmetic and _cp_lse_common's
    # sequence, so any upstream change to them must fail the build (re-check and re-copy, then update the pins).
    import hashlib
    tree, _ = _parse("v1/attention/ops/dcp.py")
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    pins = {"_correct_attn_cp_out_kernel": "75f800f8407f5098", "_cp_lse_common": "0c64c0b6961253d8",
            "cp_lse_ag_out_rs": "f671449b6ff98352"}
    for name, pin in pins.items():
        fn = fns.get(name)
        got = hashlib.sha256(ast.unparse(fn).encode()).hexdigest()[:16] if fn is not None else None
        need(got == pin, f"dcp.{name} changed upstream (sha {got}, pinned {pin}): re-verify glm_fast/dcp_glue.py")
    need(callable_name_in(tree, "mask_dcp_empty_shards_"), "dcp.mask_dcp_empty_shards_ missing")
    combine = ast.unparse(_fn(_cls(tree, "MLADCPManager"), "_init_combine") or ast.Pass())
    need("cp_lse_ag_out_rs" in combine, "MLADCPManager._init_combine no longer binds cp_lse_ag_out_rs by name")
    return problems


def callable_name_in(tree, name: str) -> bool:
    return any(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body)


if __name__ == "__main__":
    p = verify_targets()
    print("glm_fast verify_targets", p or "ok")
    raise SystemExit(1 if p else 0)
