import torch
import torch.nn.functional as F
import os
import sys

# Add parent dir to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import model_rope # This allows torch.load to find the classes

def compare_checkpoints(file1, file2):
    print(f"Comparing {file1} vs {file2}\n")
    
    ckpt1 = torch.load(file1, map_location='cpu', weights_only=False)
    ckpt2 = torch.load(file2, map_location='cpu', weights_only=False)
    
    sd1 = ckpt1['model']
    sd2 = ckpt2['model']
    
    print(f"{'Layer Name':<40} | {'Cosine Sim':<10} | {'L2 Dist':<10}")
    print("-" * 65)
    
    for name in sd1.keys():
        if name not in sd2:
            continue
            
        w1 = sd1[name].view(-1).float()
        w2 = sd2[name].view(-1).float()
        
        # Cosine Similarity
        sim = F.cosine_similarity(w1.unsqueeze(0), w2.unsqueeze(0)).item()
        
        # L2 Distance
        dist = torch.norm(w1 - w2, p=2).item()
        
        print(f"{name:<40} | {sim:10.4f} | {dist:10.2f}")

if __name__ == "__main__":
    # Use the 48k and 80k snapshots
    # Based on the results dir, the pattern is:
    # ut1.8M_phaseMask_True_1-22_phase_lean_48000.pt (not found in results, check models dir)
    
    # I'll check the models directory for the actual .pt files
    model_dir = "rpn_llm/models"
    f48 = os.path.join(model_dir, "ut1.8M_phaseMask_True_1-22_phase_lean_48000.pt")
    f80 = os.path.join(model_dir, "ut1.8M_phaseMask_True_1-22_phase_lean_80000.pt")
    
    if os.path.exists(f48) and os.path.exists(f80):
        compare_checkpoints(f48, f80)
    else:
        # Fallback to check what is available
        print(f"Models not found at {f48} or {f80}")
        print("Available models:")
        for f in os.listdir(model_dir):
            if f.endswith(".pt"):
                print(f" - {f}")
