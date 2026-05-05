import torch
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

# Add parent dir to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pointer_fidelity_test import run_fidelity_test, RPNTokenizer, GPT

def analyze_ablation(model_path, benchmark_type="long", device='mps'):
    if not torch.backends.mps.is_available() and device == 'mps':
        device = 'cpu'
        
    print(f"Loading model: {model_path} on {device}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    benchmark_path = "rpn_llm/analysis/fidelity_benchmark.json"
    if not os.path.exists(benchmark_path):
        print(f"Error: {benchmark_path} not found.")
        return
        
    with open(benchmark_path, "r") as f:
        benchmark = json.load(f)
    
    # Use a subset of samples for faster iteration
    samples = benchmark[benchmark_type][:50]
    
    n_layer = config.n_layer
    n_head = config.n_head
    
    print(f"Running baseline fidelity ({len(samples)} samples)...")
    baseline = run_fidelity_test(model, samples, device=device, verbose=False)
    print(f"Baseline Fidelity: {baseline:.1f}%")
    
    head_scores = []
    
    print("\nAblating individual heads (across all passes)...")
    for h in range(n_head):
        # Create mask: (n_layer, n_head)
        mask = torch.ones(n_layer, n_head, device=device)
        mask[:, h] = 0.0
        
        score = run_fidelity_test(model, samples, device=device, verbose=False, head_mask=mask)
        drop = baseline - score
        print(f"Head {h}: {score:.1f}% (Drop: {drop:.1f}%)")
        head_scores.append(score)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    x = np.arange(n_head)
    plt.bar(x, head_scores, color='skyblue', label='Ablated Fidelity')
    plt.axhline(y=baseline, color='red', linestyle='--', label='Baseline')
    plt.xlabel('Ablated Head Index')
    plt.ylabel('Fidelity (%)')
    plt.title(f'Circuit Ablation: Individual Head Impact ({benchmark_type})')
    plt.xticks(x)
    plt.ylim(0, max(baseline + 10, 100))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    out_name = f"rpn_llm/analysis/circuit_ablation_{benchmark_type}.png"
    plt.savefig(out_name)
    print(f"\nPlot saved to {out_name}")
    
    # Save results
    results = {
        "baseline": baseline,
        "head_scores": head_scores,
        "n_head": n_head
    }
    data_out = f"rpn_llm/analysis/circuit_ablation_{benchmark_type}.json"
    with open(data_out, "w") as f:
        json.dump(results, f)
    print(f"Results saved to {data_out}")

if __name__ == "__main__":
    path = "rpn_llm/models/ut0.4M_2l_6h_192e_mlp3_phaseMask_True_1-22_phase_lean_344000.pt"
    analyze_ablation(path, benchmark_type="long")
