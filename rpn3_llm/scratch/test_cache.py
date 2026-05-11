import torch
import torch.nn.functional as F
from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

tokenizer = RPNTokenizer("rpn-tokenizer.json")
config = GPTConfig(
    vocab_size=64, n_layer=2, n_head=8, n_embd=256,
    block_size=2048, use_gated_residual=False, use_mohsa=False,
    use_phase_mask=True, universal=True, mlp_ratio=3
)
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = GPT(config)
state_dict = torch.load("models/ut0.7M_2l_8h_256e_mlp3_phaseMask_True_rpn3_3num_104000.pt", map_location=device, weights_only=False)
model.load_state_dict(state_dict['model'])
model.to(device)
model.eval()

prompt_str = "[BOS]85 867+ 9279511819183119213+?"
idx_orig = torch.tensor([tokenizer.encode(prompt_str)], dtype=torch.long, device=device)

# 1. Run without cache autoregressively
idx_nocache = idx_orig.clone()
for _ in range(100):
    with torch.no_grad():
        with torch.autocast(device, dtype=torch.bfloat16):
            logits_nocache, _ = model(idx_nocache, use_cache=False)
    next_token = torch.argmax(logits_nocache[:, -1, :], dim=-1, keepdim=True)
    idx_nocache = torch.cat([idx_nocache, next_token], dim=1)

# 2. Run with cache autoregressively
idx_cache = idx_orig.clone()
past_kv = None
for _ in range(100):
    with torch.no_grad():
        with torch.autocast(device, dtype=torch.bfloat16):
            is_phase_shift = (idx_cache == 10) | (idx_cache == 11) | (idx_cache == 12)
            full_phase_ids = is_phase_shift.cumsum(dim=-1)
            idx_cond = idx_cache[:, -1:] if past_kv is not None else idx_cache
            logits_prefill, _, past_kv = model(idx_cond, use_cache=True, past_key_values=past_kv, full_phase_ids=full_phase_ids)
    next_token = torch.argmax(logits_prefill[:, -1, :], dim=-1, keepdim=True)
    idx_cache = torch.cat([idx_cache, next_token], dim=1)

print(f"Prompt: {prompt_str}")
print(f"No cache gen: {tokenizer.decode(idx_nocache[0].tolist())}")
print(f"Cache gen:    {tokenizer.decode(idx_cache[0].tolist())}")

