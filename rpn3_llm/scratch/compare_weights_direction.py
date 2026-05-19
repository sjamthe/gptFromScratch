import os
import sys
import torch

# Ensure working directory and rpn3_llm are in python path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "rpn3_llm"))

def compare_weights_direction():
    b0_checkpoint = "rpn3_llm/models/ut1.8M_2l_8h_384e_mlp4_phaseMask_True_sft_1-14_7num_BOS_8000.pt"
    base_checkpoint = "rpn3_llm/models/ut1.8M_2l_8h_384e_mlp4_phaseMask_True_sft_1-14_7num_BOS_352000.pt"
    recal_checkpoint = "rpn3_llm/models/ut1.8M_2l_8h_384e_mlp4_phaseMask_True_theta_sftRecal_sft_1-14_7num_BOS_40000.pt"
    
    paths = {
        "B0 (8k)": b0_checkpoint,
        "Base (352k)": base_checkpoint,
        "Recalibrated (40k)": recal_checkpoint
    }
    
    print("=" * 80)
    print("        WEIGHT REPRESENTATION DRIFT: COSINE SIMILARITY COMPARISON")
    print("=" * 80)
    print("This tool computes the cosine similarity (direction overlap) of the c_proj weights")
    print("between checkpoints. A value of 1.0 means identical; 0.0 means completely orthogonal.")
    print("=" * 80)

    # Make sure both B0 and Base exist
    if not os.path.exists(b0_checkpoint) or not os.path.exists(base_checkpoint):
        print("Error: Either B0 or Base checkpoint is missing!")
        return

    # Load state dicts
    print("Loading weights...")
    cp_b0 = torch.load(b0_checkpoint, map_location='cpu', weights_only=False)
    cp_base = torch.load(base_checkpoint, map_location='cpu', weights_only=False)
    
    w_b0 = cp_b0['model']['transformer.h.attn.c_proj.weight']
    w_base = cp_base['model']['transformer.h.attn.c_proj.weight']
    
    config = cp_base['config']
    n_embd = config.n_embd
    n_head = config.n_head
    head_dim = n_embd // n_head
    
    # 1. Global c_proj weight cosine similarity
    glob_sim = torch.cosine_similarity(w_b0.flatten(), w_base.flatten(), dim=0).item()
    print(f"\nGLOBAL c_proj WEIGHT SIMILARITY (8k vs 352k): {glob_sim*100:.2f}%")
    print("This shows how much the overall consensus projection layer has rotated during training.")
    print("-" * 75)
    
    # 2. Head-by-head similarity
    print(f"{'Head Index':<12} | {'Cosine Similarity (8k vs 352k)':<32} | {'Drift/Rotation Description':<25}")
    print("-" * 75)
    
    for h in range(n_head):
        start_col = h * head_dim
        end_col = (h + 1) * head_dim
        
        slice_b0 = w_b0[:, start_col:end_col].flatten()
        slice_base = w_base[:, start_col:end_col].flatten()
        
        sim = torch.cosine_similarity(slice_b0, slice_base, dim=0).item()
        sim_pct = sim * 100
        
        # Describe the change direction
        if sim > 0.95:
            desc = "Almost Unchanged"
        elif sim > 0.8:
            desc = "Slight Rotation"
        elif sim > 0.5:
            desc = "Moderate Drift"
        elif sim > 0.1:
            desc = "Substantial Re-alignment"
        else:
            desc = "Orthogonal (New Role Learned)"
            
        print(f"Head {h:<7} | {sim_pct:<30.2f}% | {desc:<25}")
    print("-" * 75)

    # 3. Compare Base (352k) vs Recalibrated (40k) if it exists
    if os.path.exists(recal_checkpoint):
        print("\nLoading Recalibrated weights...")
        cp_recal = torch.load(recal_checkpoint, map_location='cpu', weights_only=False)
        w_recal = cp_recal['model']['transformer.h.attn.c_proj.weight']
        
        glob_sim_recal = torch.cosine_similarity(w_base.flatten(), w_recal.flatten(), dim=0).item()
        print(f"\nGLOBAL c_proj WEIGHT SIMILARITY (Base 352k vs Recalibrated 40k): {glob_sim_recal*100:.2f}%")
        print("This shows how much the attention projection adjusted during the Theta=2000 recalibration.")
        print("-" * 75)
        
        print(f"{'Head Index':<12} | {'Cosine Similarity (352k vs Recal 40k)':<32} | {'Drift/Rotation Description':<25}")
        print("-" * 75)
        for h in range(n_head):
            start_col = h * head_dim
            end_col = (h + 1) * head_dim
            
            slice_base = w_base[:, start_col:end_col].flatten()
            slice_recal = w_recal[:, start_col:end_col].flatten()
            
            sim = torch.cosine_similarity(slice_base, slice_recal, dim=0).item()
            sim_pct = sim * 100
            
            if sim > 0.98:
                desc = "Near Identical"
            elif sim > 0.95:
                desc = "Highly Retained"
            elif sim > 0.8:
                desc = "Slight Tuning"
            else:
                desc = "Significant Recalibration"
                
            print(f"Head {h:<7} | {sim_pct:<30.2f}% | {desc:<25}")
        print("-" * 75)

if __name__ == "__main__":
    compare_weights_direction()
