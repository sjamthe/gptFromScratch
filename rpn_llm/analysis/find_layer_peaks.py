import torch
import glob
import re
import os
import sys

# Add parent dir to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def find_all_layer_peaks(checkpoint_pattern, device='cpu'):
    paths = sorted(glob.glob(checkpoint_pattern), key=lambda x: int(re.search(r'_(\d+)\.pt', x).group(1)))
    if not paths:
        print(f"No checkpoints found.")
        return
    
    # Initialize data structures
    # layer_norms[layer_name] = [(step, norm), ...]
    layer_norms = {}
    
    print(f"Processing {len(paths)} checkpoints...")
    for i, p in enumerate(paths):
        step = int(re.search(r'_(\d+)\.pt', p).group(1))
        checkpoint = torch.load(p, map_location=device, weights_only=False)
        state_dict = checkpoint['model']
        
        for name, param in state_dict.items():
            if "weight" in name or name == "pass_emb":
                norm = torch.linalg.vector_norm(param.float(), ord=2).item()
                if name not in layer_norms:
                    layer_norms[name] = []
                layer_norms[name].append((step, norm))
        
        print(f"[{i+1}/{len(paths)}] Step {step} done.", end="\r")
    
    print("\nLayer Peaks:")
    print(f"{'Layer Name':<40} | {'Peak Step':<10} | {'Peak Norm':<10}")
    print("-" * 65)
    
    results = []
    for name, history in layer_norms.items():
        peak_step, peak_norm = max(history, key=lambda x: x[1])
        print(f"{name:<40} | {peak_step:<10} | {peak_norm:<10.4f}")
        results.append((name, peak_step, peak_norm))
    
    return results

if __name__ == "__main__":
    pattern = "rpn_llm/models/ut0.4M_2l_6h_192e_mlp3_phaseMask_True_1-22_phase_lean_*.pt"
    find_all_layer_peaks(pattern)
