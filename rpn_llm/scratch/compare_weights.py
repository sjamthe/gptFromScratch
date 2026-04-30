import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add rpn_llm to path for unpickling
sys.path.append(os.path.abspath("rpn_llm"))

path1 = "rpn_llm/models/rope3.6M_phaseMask_True_1-22_phase_lean_64000.pt"
path2 = "rpn_llm/models/rope3.6M_phaseMask_False_1-22_phase_lean_64000.pt"

def analyze_diff(p1, p2):
    print(f"Comparing {os.path.basename(p1)} and {os.path.basename(p2)}")
    sd1 = torch.load(p1, map_location='cpu', weights_only=False)['model']
    sd2 = torch.load(p2, map_location='cpu', weights_only=False)['model']
    
    layer_names = []
    l2_norms = []
    cos_sims = []
    
    for name in sd1.keys():
        if name not in sd2: continue
        w1 = sd1[name].float()
        w2 = sd2[name].float()
        
        # Flatten for comparison
        w1_f = w1.view(-1)
        w2_f = w2.view(-1)
        
        # L2 Distance (Relative)
        diff_norm = torch.norm(w1_f - w2_f).item()
        base_norm = torch.norm(w1_f).item()
        rel_diff = diff_norm / base_norm if base_norm > 0 else 0
        
        # Cosine Similarity
        sim = torch.nn.functional.cosine_similarity(w1_f.unsqueeze(0), w2_f.unsqueeze(0)).item()
        
        layer_names.append(name)
        l2_norms.append(rel_diff)
        cos_sims.append(sim)
        
        print(f"Layer: {name:<40} | Rel Diff: {rel_diff:.6f} | Cos Sim: {sim:.6f}")

    # Plotting
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Relative L2 Difference
    plt.subplot(2, 1, 1)
    plt.bar(range(len(layer_names)), l2_norms, color='skyblue')
    plt.xticks(range(len(layer_names)), layer_names, rotation=90, fontsize=8)
    plt.title("Relative L2 Difference (Weights Change)")
    plt.ylabel("Relative Change")
    
    # Subplot 2: Cosine Similarity
    plt.subplot(2, 1, 2)
    plt.bar(range(len(layer_names)), cos_sims, color='salmon')
    plt.xticks(range(len(layer_names)), layer_names, rotation=90, fontsize=8)
    plt.ylim(0.99, 1.0) # Zoom in on the high similarity
    plt.title("Cosine Similarity (Directional Stability)")
    plt.ylabel("Similarity")
    
    plt.tight_layout()
    plt.savefig("rpn_llm/scratch/weight_comparison.png")
    print("\nSaved comparison plot to rpn_llm/scratch/weight_comparison.png")

if __name__ == "__main__":
    analyze_diff(path1, path2)
