import torch
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm"))
from model_rope import GPT
from utils import RPNTokenizer

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
# Prompt with multiple steps to see the ':' token behavior
prompt = "[BOS]12 34+?[REV]21 43+=[MATH]2+4+0=6:1+3+0=4"
idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
tokens = [tokenizer.decode([t]) for t in idx[0].cpu().numpy()]

def get_gate_stats(step):
    model_path = f"rpn_llm/models/ut0.5M_2l_6h_192e_mlp3_phaseMask_True_gated_1-22_phase_lean_{step}.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    x = model.transformer.wte(idx)
    stats = {}
    
    for i in range(model.config.n_layer):
        emb_idx = i % model.config.n_layer
        x = x + model.pass_emb[emb_idx].view(1, 1, -1)
        
        # Attention Gate
        norm_x = model.transformer.h.ln_1(x)
        gate_a = torch.sigmoid(model.transformer.h.attn_gate_proj(norm_x)).squeeze(0).mean(dim=-1).detach().cpu().numpy()
        
        for t_idx, token in enumerate(tokens):
            if token in ['=', ':']:
                key = f"Pass {i} | Token '{token}'"
                stats[key] = gate_a[t_idx]
        
        # Update x for next pass
        attn_out, _, _ = model.transformer.h.attn(norm_x, model.freqs_cis)
        x = x + torch.sigmoid(model.transformer.h.attn_gate_proj(norm_x)) * attn_out
        x = x + torch.sigmoid(model.transformer.h.mlp_gate_proj(model.transformer.h.ln_2(x))) * model.transformer.h.mlp(model.transformer.h.ln_2(x))

    return stats

print("Gate Activations (80k):")
stats_80k = get_gate_stats(80000)
for k, v in stats_80k.items():
    print(f"  {k}: {v:.3f}")

print("\nGate Activations (160k):")
stats_160k = get_gate_stats(160000)
for k, v in stats_160k.items():
    print(f"  {k}: {v:.3f}")
