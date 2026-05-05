import torch
import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add parent dir to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model_rope import GPT
from utils import RPNTokenizer

def visualize_grokking_milestones(problem_text, device='mps'):
    if not torch.backends.mps.is_available() and device == 'mps':
        device = 'cpu'
        
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
    tokens = torch.tensor(tokenizer.encode(problem_text)).unsqueeze(0).to(device)
    token_labels = [tokenizer.decode([t]) for t in tokens[0]]
    
    milestones = [8000, 16000, 40000, 344000]
    
    for step in milestones:
        path = f"rpn_llm/models/ut0.4M_2l_6h_192e_mlp3_phaseMask_True_1-22_phase_lean_{step}.pt"
        if not os.path.exists(path):
            print(f"Skipping step {step}, checkpoint not found.")
            continue
            
        print(f"Visualizing step {step}...")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = GPT(checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            _, _, weights_list = model(tokens, return_attention=True)
            
        # UT has only one "transformer.h" weight set, but it runs N times.
        # weights_list contains attention from each pass.
        # We'll look at the first and last pass.
        
        passes_to_show = [0, len(weights_list)-1]
        for p_idx in passes_to_show:
            weights = weights_list[p_idx][0].cpu().numpy() # (n_head, T, T)
            n_head = weights.shape[0]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f"Attention Patterns - Step {step} - Pass {p_idx}", fontsize=20)
            
            for h in range(n_head):
                ax = axes[h//3, h%3]
                sns.heatmap(weights[h], ax=ax, cmap='magma', cbar=False)
                ax.set_title(f"Head {h}", fontsize=14)
                
                # Show only few labels to avoid clutter
                if len(token_labels) < 40:
                    ax.set_xticks(np.arange(len(token_labels)) + 0.5)
                    ax.set_xticklabels(token_labels, rotation=90, fontsize=8)
                    ax.set_yticks(np.arange(len(token_labels)) + 0.5)
                    ax.set_yticklabels(token_labels, rotation=0, fontsize=8)
                else:
                    step_tick = len(token_labels) // 10
                    ax.set_xticks(np.arange(0, len(token_labels), step_tick) + 0.5)
                    ax.set_xticklabels(token_labels[::step_tick], rotation=90, fontsize=8)
                    ax.set_yticks(np.arange(0, len(token_labels), step_tick) + 0.5)
                    ax.set_yticklabels(token_labels[::step_tick], rotation=0, fontsize=8)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            out_path = f"rpn_llm/analysis/attn_viz_step_{step}_pass_{p_idx}.png"
            plt.savefig(out_path)
            plt.close()
            print(f"  Saved {out_path}")

if __name__ == "__main__":
    # Use a long problem to see the pointer logic
    long_problem = "[BOS]1234567890 0987654321+? [REV]0987654321 1234567890+="
    visualize_grokking_milestones(long_problem)
