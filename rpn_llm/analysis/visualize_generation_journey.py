import torch
import torch.nn.functional as F
import os
import sys
import matplotlib.pyplot as plt
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

def get_residual_journey(model, prompt, tokenizer, device):
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    attn_mask = create_attn_mask(idx, model.config, device)
    
    # Capture states
    states = []
    state_names = []
    
    # Step 0: WTE
    x = model.transformer.wte(idx)
    states.append(x.detach().clone())
    state_names.append("Start(WTE)")
    
    # Step through Universal passes
    for i in range(model.config.n_layer):
        emb_idx = i % model.config.n_layer
        x = x + model.pass_emb[emb_idx].view(1, 1, -1)
        
        # Attention
        attn_out, _, _ = model.transformer.h.attn(
            model.transformer.h.ln_1(x), 
            model.freqs_cis, 
            attn_mask=attn_mask
        )
        x = x + attn_out
        states.append(x.detach().clone())
        state_names.append(f"P{i}_Attn")
        
        # MLP
        mlp_out = model.transformer.h.mlp(model.transformer.h.ln_2(x))
        x = x + mlp_out
        states.append(x.detach().clone())
        state_names.append(f"P{i}_MLP")
        
    return states, state_names, idx

def visualize_generation_journey(step):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_path = f"rpn_llm/models/ut0.4M_2l_6h_192e_mlp3_phaseMask_True_1-22_phase_lean_{step}.pt"
    
    if not os.path.exists(model_path):
        print(f"Skipping {step}, file not found.")
        return
        
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
    # Prompt ending right after the first unit-level calculation
    prompt = "[BOS]123 456+?[REV]321 654+=[MATH]3+6+0=9"
    
    states, state_names, idx = get_residual_journey(model, prompt, tokenizer, device)
    
    tokens = idx[0].cpu().numpy()
    token_strs = [tokenizer.decode([t]) for t in tokens]
    
    # We only want to plot the newly generated tokens starting from [MATH]
    # Find the index of [MATH]
    math_token_id = tokenizer.vocab.get("[MATH]", -1)
    try:
        start_plot_idx = tokens.tolist().index(math_token_id)
    except ValueError:
        print("MATH token not found in prompt.")
        return
        
    T_total = idx.size(1)
    num_states = len(states)
    
    start_state = states[0][0] # [T, C]
    final_state = states[-1][0] # [T, C]
    
    sim_to_start = np.zeros((T_total, num_states))
    sim_to_final = np.zeros((T_total, num_states))
    
    for s_idx in range(num_states):
        curr_state = states[s_idx][0] # [T, C]
        for t_idx in range(start_plot_idx, T_total):
            vec_start = start_state[t_idx].unsqueeze(0)
            vec_curr = curr_state[t_idx].unsqueeze(0)
            vec_final = final_state[t_idx].unsqueeze(0)
            
            sim_to_start[t_idx, s_idx] = F.cosine_similarity(vec_curr, vec_start).item()
            sim_to_final[t_idx, s_idx] = F.cosine_similarity(vec_curr, vec_final).item()
            
    # Plotting
    plt.figure(figsize=(14, 8))
    
    num_plotted_tokens = T_total - start_plot_idx
    colors = plt.cm.tab10(np.linspace(0, 1, num_plotted_tokens))
    
    for i, t_idx in enumerate(range(start_plot_idx, T_total)):
        token_char = token_strs[t_idx]
        if token_char == " ": token_char = "_"
        label_base = f"T{t_idx}: '{token_char}'"
        
        # Solid line: Sim to Start
        plt.plot(range(num_states), sim_to_start[t_idx, :], color=colors[i], linestyle='-', linewidth=3.0, alpha=0.8, label=label_base)
        
        # Dotted line: 1 - Sim to Final
        plt.plot(range(num_states), 1.0 - sim_to_final[t_idx, :], color=colors[i], linestyle=':', linewidth=3.0, alpha=0.8)

    plt.title(f"Residual Stream Journey: Unit Addition (Step {step})\nSolid: Sim to Start | Dotted: 1 - Sim to Final", fontsize=16)
    plt.xlabel("Transformation Stage", fontsize=12)
    plt.ylabel("Similarity Metric (0.0 to 1.0)", fontsize=12)
    plt.xticks(range(num_states), state_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Put legend outside
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    
    out_path = f"rpn_llm/analysis/generation_journey_{step}.png"
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    plt.close()

if __name__ == "__main__":
    visualize_generation_journey(8000)
    visualize_generation_journey(344000)
