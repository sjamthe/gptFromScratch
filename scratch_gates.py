import torch
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm"))
sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm/analysis"))

from model_rope import GPT
from utils import RPNTokenizer
from visualize_generation_journey import create_attn_mask

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model_path = "rpn_llm/models/ut0.5M_2l_6h_192e_mlp3_phaseMask_True_gated_1-22_phase_lean_80000.pt"

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = GPT(checkpoint['config']).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
prompt = "[BOS]123 456+?[REV]321 654+=[MATH]3+6+0=9"
idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

attn_mask = create_attn_mask(idx, model.config, device)

attn_gates = []
mlp_gates = []

x = model.transformer.wte(idx)

for i in range(model.config.n_layer):
    emb_idx = i % model.config.n_layer
    x = x + model.pass_emb[emb_idx].view(1, 1, -1)
    
    # Attention
    norm_x_attn = model.transformer.h.ln_1(x)
    attn_out, _, _ = model.transformer.h.attn(norm_x_attn, model.freqs_cis, attn_mask=attn_mask)
    gate_a = torch.sigmoid(model.transformer.h.attn_gate_proj(norm_x_attn))
    attn_gates.append(gate_a.mean(dim=-1).squeeze(0).detach().cpu().numpy()) # [T]
    x = x + gate_a * attn_out
    
    # MLP
    norm_x_mlp = model.transformer.h.ln_2(x)
    mlp_out = model.transformer.h.mlp(norm_x_mlp)
    gate_m = torch.sigmoid(model.transformer.h.mlp_gate_proj(norm_x_mlp))
    mlp_gates.append(gate_m.mean(dim=-1).squeeze(0).detach().cpu().numpy()) # [T]
    x = x + gate_m * mlp_out

tokens = [tokenizer.decode([t]) for t in idx[0].cpu().numpy()]
tokens = [t if t != ' ' else '_' for t in tokens]

attn_gates = np.array(attn_gates) # [Passes, T]
mlp_gates = np.array(mlp_gates) # [Passes, T]

start_idx = tokens.index("[MATH]")

print("Tokens:", tokens[start_idx:])
print("Attn Gates (Pass 0):", np.round(attn_gates[0, start_idx:], 2))
print("MLP Gates (Pass 0):", np.round(mlp_gates[0, start_idx:], 2))
print("Attn Gates (Pass 21):", np.round(attn_gates[-1, start_idx:], 2))
print("MLP Gates (Pass 21):", np.round(mlp_gates[-1, start_idx:], 2))
