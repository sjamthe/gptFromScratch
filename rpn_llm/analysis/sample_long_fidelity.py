import torch
import os
import sys
import json

# Add parent dir to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pointer_fidelity_test import run_fidelity_test

def sample_long_fidelity(device='mps'):
    model_dir = "rpn_llm/models/"
    # Targeted steps to see the curve
    steps = [8000, 16000, 24000, 32000, 40000, 80000, 160000, 240000, 344000]
    
    # Load benchmark
    benchmark_path = "rpn_llm/analysis/fidelity_benchmark.json"
    with open(benchmark_path, "r") as f:
        benchmark = json.load(f)
    long_samples = benchmark["long"]
    
    print(f"{'Step':<10} | {'Long Fidelity':<15}")
    print("-" * 25)
    
    results = []
    for step in steps:
        # Find file matching step
        pattern = f"ut0.4M_2l_6h_192e_mlp3_phaseMask_True_1-22_phase_lean_{step}.pt"
        path = os.path.join(model_dir, pattern)
        if not os.path.exists(path):
            # Try to find it if naming slightly different
            continue
            
        score = run_fidelity_test(path, long_samples[:20], device=device, verbose=False)
        print(f"{step:<10} | {score:>13.1f}%")
        results.append((step, score))
    
    return results

if __name__ == "__main__":
    sample_long_fidelity()
