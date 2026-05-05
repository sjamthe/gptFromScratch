import torch
import torch.nn.functional as F
import os
import glob
import re
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add parent dir to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pointer_fidelity_test import run_fidelity_test, RPNTokenizer, GPT

def get_weight_metrics(model_path, prev_weights=None, final_weights=None, target_layer="transformer.h.0.attn.c_attn.weight", device='cpu'):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model']
    
    if target_layer not in state_dict:
        # Try to find it or print available keys
        available = [k for k in state_dict.keys() if "weight" in k]
        raise ValueError(f"Layer {target_layer} not found. Available weight layers: {available[:10]}...")

    w = state_dict[target_layer].float().flatten()
    
    norm = torch.norm(w, p=2).item()
    
    consecutive_sim = None
    if prev_weights is not None:
        consecutive_sim = F.cosine_similarity(w.unsqueeze(0), prev_weights.unsqueeze(0)).item()
        
    final_sim = None
    if final_weights is not None:
        final_sim = F.cosine_similarity(w.unsqueeze(0), final_weights.unsqueeze(0)).item()
        
    return w, norm, consecutive_sim, final_sim

def analyze_grokking(checkpoint_pattern, target_layer, run_fidelity=False, device='cpu'):
    paths = sorted(glob.glob(checkpoint_pattern), key=lambda x: int(re.search(r'_(\d+)\.pt', x).group(1)))
    if not paths:
        print(f"No checkpoints found for pattern: {checkpoint_pattern}")
        return
        
    print(f"Found {len(paths)} checkpoints.")
    
    steps = []
    norms = []
    consecutive_sims = []
    final_sims = []
    fidelity_scores = []
    
    # Load final weights first for comparison
    print(f"Loading final weights from {paths[-1]}...")
    final_w, _, _, _ = get_weight_metrics(paths[-1], target_layer=target_layer, device=device)
    
    # Load fidelity benchmark if needed
    test_samples = []
    if run_fidelity:
        benchmark_path = "rpn_llm/analysis/fidelity_benchmark.json"
        if not os.path.exists(benchmark_path):
            print(f"Warning: {benchmark_path} not found. Skipping fidelity.")
            run_fidelity = False
        else:
            with open(benchmark_path, "r") as f:
                benchmark = json.load(f)
            # Use 'short' benchmark for faster processing across many checkpoints
            test_samples = benchmark["short"]
            print(f"Using {len(test_samples)} short trials for fidelity scoring.")
    
    prev_w = None
    for i, p in enumerate(paths):
        step = int(re.search(r'_(\d+)\.pt', p).group(1))
        steps.append(step)
        
        # Weight analysis
        curr_w, norm, c_sim, f_sim = get_weight_metrics(p, prev_w, final_w, target_layer, device)
        norms.append(norm)
        consecutive_sims.append(c_sim if c_sim is not None else 1.0)
        final_sims.append(f_sim)
        prev_w = curr_w
        
        # Fidelity analysis
        if run_fidelity:
            print(f"[{i+1}/{len(paths)}] Analyzing Step {step}...", end="\r")
            score = run_fidelity_test(p, test_samples, device=device, verbose=False)
            fidelity_scores.append(score)
        else:
            print(f"[{i+1}/{len(paths)}] Analyzing Step {step}...", end="\r")

    print("\nAnalysis complete. Plotting...")
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Cosine Similarity')
    
    line1 = ax1.plot(steps, final_sims, label='Similarity to Final', color='blue', linewidth=2)
    line2 = ax1.plot(steps, consecutive_sims, label='Consecutive Similarity', color='cyan', alpha=0.5)
    
    # Secondary axis for Norms
    ax3 = ax1.twinx()
    ax3.set_ylabel('Weight Norm (L2)')
    line3 = ax3.plot(steps, norms, label='Weight Norm', color='green', linestyle='--', alpha=0.6)
    ax3.tick_params(axis='y', labelcolor='green')
    
    if run_fidelity:
        # Another axis for Fidelity? Or just put it on ax2 (right)
        ax2 = ax1.twinx()
        # Offset the second right axis
        ax2.spines['right'].set_position(('outward', 60))
        ax2.set_ylabel('Pointer Fidelity (%)', color='red')
        line4 = ax2.plot(steps, fidelity_scores, label='Pointer Fidelity', color='red', linewidth=3)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 105)
        
        lines = line1 + line2 + line3 + line4
    else:
        lines = line1 + line2 + line3
        
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title(f'Grokking Trajectory: {target_layer}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    layer_slug = target_layer.replace('.', '_')
    out_name = f"rpn_llm/analysis/grokking_plot_{layer_slug}.png"
    plt.savefig(out_name)
    print(f"Plot saved to {out_name}")
    
    # Also save data to JSON for later use
    data = {
        "steps": steps,
        "final_sims": final_sims,
        "consecutive_sims": consecutive_sims,
        "norms": norms,
        "fidelity_scores": fidelity_scores
    }
    data_out = f"rpn_llm/analysis/grokking_data_{layer_slug}.json"
    with open(data_out, "w") as f:
        json.dump(data, f)
    print(f"Data saved to {data_out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default="rpn_llm/models/ut0.4M_2l_6h_192e_mlp3_phaseMask_True_1-22_phase_lean_*.pt")
    parser.add_argument("--layer", type=str, default="transformer.h.0.attn.c_attn.weight")
    parser.add_argument("--fidelity", action="store_true", help="Run behavioral fidelity test (slow)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    # Normalize path pattern if needed
    analyze_grokking(args.pattern, args.layer, args.fidelity, args.device)
