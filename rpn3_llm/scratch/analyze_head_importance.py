import os
import sys
import torch

# Ensure working directory and rpn3_llm are in python path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "rpn3_llm"))

def analyze_head_importance():
    b0_checkpoint = "rpn3_llm/models/ut1.8M_2l_8h_384e_mlp4_phaseMask_True_sft_1-14_7num_BOS_8000.pt"
    base_checkpoint = "rpn3_llm/models/ut1.8M_2l_8h_384e_mlp4_phaseMask_True_sft_1-14_7num_BOS_352000.pt"
    recal_checkpoint = "rpn3_llm/models/ut1.8M_2l_8h_384e_mlp4_phaseMask_True_theta_sftRecal_sft_1-14_7num_BOS_40000.pt"
    
    checkpoints = {
        "B0 Checkpoint (Theta=10000)": b0_checkpoint,
        "Base Checkpoint (Theta=10000)": base_checkpoint,
        "Recalibrated Checkpoint (Theta=2000)": recal_checkpoint
    }
    
    print("=" * 80)
    print("        ATTENTION HEAD WEIGHT IMPORTANCE ANALYSIS (c_proj PROJECTION)")
    print("=" * 80)
    print("This script calculates the Frobenius norm of each head's slice inside the c_proj")
    print("matrix. A head with a near-zero norm is ignored (dead/redundant) by the model.")
    print("=" * 80)

    for name, path in checkpoints.items():
        if not os.path.exists(path):
            print(f"\n[Skipping {name}: Checkpoint file not found]")
            continue
            
        print(f"\n>>> Analyzing: {name} <<<")
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model_state = checkpoint['model']
        config = checkpoint['config']
        
        n_embd = config.n_embd
        n_head = config.n_head
        head_dim = n_embd // n_head
        
        # Load c_proj weight from model state dict
        # The key should be: 'transformer.h.attn.c_proj.weight'
        c_proj_key = 'transformer.h.attn.c_proj.weight'
        if c_proj_key not in model_state:
            print(f"Error: Could not find '{c_proj_key}' in state dict!")
            continue
            
        weight = model_state[c_proj_key] # shape: [384, 384]
        
        norms = []
        for h in range(n_head):
            # Column slice corresponds to the input dimensions from head h
            start_col = h * head_dim
            end_col = (h + 1) * head_dim
            h_slice = weight[:, start_col:end_col]
            
            # Compute Frobenius norm
            norm = torch.norm(h_slice, p='fro').item()
            norms.append(norm)
            
        total_norm = sum(norms)
        
        print("-" * 65)
        print(f"{'Head Index':<12} | {'Frobenius Norm':<16} | {'Relative Importance (%)':<24}")
        print("-" * 65)
        for h in range(n_head):
            rel = (norms[h] / total_norm) * 100 if total_norm > 0 else 0
            # Draw a tiny ASCII bar chart
            bar_len = int(rel / 2)
            bar = "#" * bar_len
            print(f"Head {h:<7} | {norms[h]:<16.4f} | {rel:<6.2f}% {bar}")
        print("-" * 65)
        
        # Compute head sparsity metrics
        avg_norm = sum(norms) / len(norms)
        max_norm = max(norms)
        min_norm = min(norms)
        variance = sum([(x - avg_norm)**2 for x in norms]) / len(norms)
        print(f"Statistics:")
        print(f"  * Average Head Norm: {avg_norm:.4f}")
        print(f"  * Max/Min Ratio    : {max_norm / min_norm:.2f} (lower means heads are more balanced)")
        print(f"  * Head Variance     : {variance:.6f} (higher means some heads dominate over others)")

if __name__ == "__main__":
    analyze_head_importance()
