import torch
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model_rope import GPT
from utils import RPNTokenizer

def create_attn_mask(idx, config, device):
    B, T = idx.size()
    is_bos = (idx == 2)
    seq_ids = is_bos.cumsum(dim=-1)
    doc_mask = (seq_ids.unsqueeze(1) == seq_ids.unsqueeze(2))
    causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
    
    if config.use_phase_mask:
        is_phase_shift = (idx == 10) | (idx == 11) | (idx == 12)
        global_phase_ids = is_phase_shift.cumsum(dim=-1)
        phase_diff = (global_phase_ids.unsqueeze(-1) - global_phase_ids.unsqueeze(-2))
        phase_mask = (phase_diff == 0) | (phase_diff == 1)
        full_mask = doc_mask & phase_mask & causal_mask
    else:
        full_mask = doc_mask & causal_mask

    return full_mask.unsqueeze(1)

def visualize_gates(step):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_path = f"rpn_llm/models/ut0.5M_2l_6h_192e_mlp3_phaseMask_True_gated_1-22_phase_lean_{step}.pt"
    
    if not os.path.exists(model_path):
        print(f"Skipping {step}, file not found.")
        return
        
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
        if getattr(model.transformer.h, 'use_gated_residual', False):
            gate_a = torch.sigmoid(model.transformer.h.attn_gate_proj(norm_x_attn))
            attn_gates.append(gate_a.mean(dim=-1).squeeze(0).detach().cpu().numpy()) # [T]
            x = x + gate_a * attn_out
        else:
            x = x + attn_out
            attn_gates.append(np.ones(idx.size(1))) # dummy
            
        # MLP
        norm_x_mlp = model.transformer.h.ln_2(x)
        mlp_out = model.transformer.h.mlp(norm_x_mlp)
        if getattr(model.transformer.h, 'use_gated_residual', False):
            gate_m = torch.sigmoid(model.transformer.h.mlp_gate_proj(norm_x_mlp))
            mlp_gates.append(gate_m.mean(dim=-1).squeeze(0).detach().cpu().numpy()) # [T]
            x = x + gate_m * mlp_out
        else:
            x = x + mlp_out
            mlp_gates.append(np.ones(idx.size(1))) # dummy

    tokens = [tokenizer.decode([t]) for t in idx[0].cpu().numpy()]
    tokens = [t if t != ' ' else '_' for t in tokens]
    
    # We only want to plot the generation tokens
    math_idx = tokens.index("[MATH]")
    tokens_to_plot = tokens[math_idx:]
    
    # Convert to arrays and slice: [Passes, Tokens]
    attn_gates = np.array(attn_gates)[:, math_idx:]
    mlp_gates = np.array(mlp_gates)[:, math_idx:]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"Data-Dependent Gate Activations (Step {step})\nGeneration Phase: [MATH]3+6+0=9", fontsize=18)
    
    sns.heatmap(attn_gates, ax=axes[0], cmap="RdBu", vmin=0.0, vmax=1.0, center=0.5, cbar=True, annot=True, fmt=".2f", annot_kws={"size": 8})
    axes[0].set_title("Attention Gate ($0.0$ = Clamped, $1.0$ = Open)", fontsize=14)
    axes[0].set_xlabel("Token", fontsize=12)
    axes[0].set_ylabel("Transformer Pass", fontsize=12)
    axes[0].set_xticks(np.arange(len(tokens_to_plot)) + 0.5)
    axes[0].set_xticklabels([f"T{math_idx+i}: '{t}'" for i, t in enumerate(tokens_to_plot)], rotation=45)
    axes[0].set_yticks(np.arange(model.config.n_layer) + 0.5)
    axes[0].set_yticklabels([f"Pass {i}" for i in range(model.config.n_layer)], rotation=0)
    
    sns.heatmap(mlp_gates, ax=axes[1], cmap="RdBu", vmin=0.0, vmax=1.0, center=0.5, cbar=True, annot=True, fmt=".2f", annot_kws={"size": 8})
    axes[1].set_title("MLP Gate ($0.0$ = Clamped, $1.0$ = Open)", fontsize=14)
    axes[1].set_xlabel("Token", fontsize=12)
    axes[1].set_ylabel("Transformer Pass", fontsize=12)
    axes[1].set_xticks(np.arange(len(tokens_to_plot)) + 0.5)
    axes[1].set_xticklabels([f"T{math_idx+i}: '{t}'" for i, t in enumerate(tokens_to_plot)], rotation=45)
    axes[1].set_yticks(np.arange(model.config.n_layer) + 0.5)
    axes[1].set_yticklabels([f"Pass {i}" for i in range(model.config.n_layer)], rotation=0)
    
    plt.tight_layout()
    out_path = f"rpn_llm/analysis/gate_activations_{step}.png"
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    plt.close()

if __name__ == "__main__":
    visualize_gates(8000)
    visualize_gates(80000)
