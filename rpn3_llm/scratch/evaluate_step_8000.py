import os
import sys
import torch

# Ensure working directory is in python path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "rpn3_llm"))

from utils import RPNTokenizer, DataLoaderLite
from model_rope import GPT, GPTConfig
from train_rpn import run_teacher_forcing_validation

def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    checkpoint_path = "rpn3_llm/models/ut1.8M_2l_8h_384e_mlp4_phaseMask_True_cnt1_sft_1-14_7num_BOS_8000.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
        
    print(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Recreate the model config
    config = checkpoint['config']
    print(f"Loaded config: {config}")
    
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # Initialize DataLoader with the new full-equation padded logic
    B = 64
    T = 384
    val_dataset = "rpn3_llm/data/sft_1-14_7num_BOS_val.txt"
    
    print("Initializing DataLoaderLite (this will regenerate the cache)...")
    val_loader = DataLoaderLite(B, T, val_dataset)
    
    print("Running validation...")
    val_loss_accum, val_perplexity, val_eq_acc, val_rev1_err, val_math1_err, val_rev2_err, val_math2_err, val_ans_err = run_teacher_forcing_validation(
        model, val_loader, device, step=7999
    )
    
    print("\n--- RESULTS ---")
    print(f"Val Loss: {val_loss_accum:.4f}")
    print(f"Val Perplexity: {val_perplexity:.4f}")
    print(f"Val Equation Accuracy: {val_eq_acc:.2f}%")
    print(f"REV1 Error: {val_rev1_err:.2f}%")
    print(f"MATH1 Error: {val_math1_err:.2f}%")
    print(f"REV2 Error: {val_rev2_err:.2f}%")
    print(f"MATH2 Error: {val_math2_err:.2f}%")
    print(f"ANS Error: {val_ans_err:.2f}%")

if __name__ == "__main__":
    main()
