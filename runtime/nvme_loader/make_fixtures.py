"""Two unrelated tiny architectures for native/prepared output parity tests."""
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM, GPT2Config, GPT2LMHeadModel

torch.manual_seed(739)
root = Path(__file__).resolve().parents[2] / "results/nvme-loader/fixtures"
root.mkdir(parents=True, exist_ok=True)
models = {
    "tiny-llama": LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=256,
        intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=128)),
    "tiny-gpt2": GPT2LMHeadModel(GPT2Config(vocab_size=128, n_embd=128, n_layer=2,
        n_head=4, n_positions=128, bos_token_id=1, eos_token_id=2)),
}
for name, model in models.items():
    model.save_pretrained(root / name, safe_serialization=True)
    print(root / name)
