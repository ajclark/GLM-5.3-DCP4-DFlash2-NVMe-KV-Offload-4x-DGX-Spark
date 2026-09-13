"""Isolated Spark CUDA correctness checks (run only after serving is stopped)."""
import argparse
import json
import hashlib
from pathlib import Path
import tempfile

import torch
from spark_nvme.artifact import publish, restore


def transport_check():
    # Multiple slots, tail chunks, aliasing, and writes on a non-default stream.
    m = torch.nn.Linear(4097, 8192, bias=False, dtype=torch.float16, device="cuda")
    expected = m.weight.detach().cpu()
    with tempfile.TemporaryDirectory(dir="/nvme-artifacts") as root:
        path = Path(root) / "gpu"
        publish(m, path, {"test": "gpu"})
        writer = torch.cuda.Stream()
        with torch.cuda.stream(writer), torch.no_grad():
            m.weight.zero_()
            metrics = restore(m, path, {"test": "gpu"}, depth=8)
            actual = m.weight.cpu()
        assert torch.equal(expected, actual), "CUDA transport changed weight bytes"
    print(json.dumps(metrics), flush=True)


def model_check(model, output):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, load_format="nvme", skip_tokenizer_init=True,
              dtype="float16", max_model_len=128, max_num_seqs=2,
              max_num_batched_tokens=128, kv_cache_memory_bytes=64 << 20,
              enforce_eager=True, enable_prefix_caching=False,
              compilation_config={"mode": 0},
              kernel_config={"enable_flashinfer_autotune": False,
                             "enable_jit_warmup": False,
                             "enable_cutedsl_warmup": False}, seed=739)
    prompts = [{"prompt_token_ids": [1, 7, 13, 19]},
               {"prompt_token_ids": [1, 2, 3, 5, 8, 13, 21]}]
    # Fixed single-request batches keep async admission timing from changing
    # fp16 GEMM shapes and obscuring exact weight-loader parity.
    results = []
    for prompt in prompts:
        results.extend(llm.generate([prompt], SamplingParams(temperature=0, max_tokens=16,
                                      ignore_eos=True, logprobs=1)))
    rows = [{"tokens": r.outputs[0].token_ids,
             "logprobs": [x[token].logprob for x, token in
                          zip(r.outputs[0].logprobs, r.outputs[0].token_ids)]} for r in results]
    Path(output).write_text(json.dumps(rows, indent=2) + "\n")
    def parameter_hashes(model):
        return {name: hashlib.sha256(t.detach().cpu().contiguous().reshape(-1)
                                    .view(torch.uint8).numpy().tobytes()).hexdigest()
                for name, t in model.named_parameters()}
    hashes = llm.apply_model(parameter_hashes)
    Path(output + ".weights.json").write_text(json.dumps(hashes, sort_keys=True) + "\n")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model"); p.add_argument("--output")
    a = p.parse_args()
    if a.model:
        model_check(a.model, a.output)
    else:
        transport_check()
