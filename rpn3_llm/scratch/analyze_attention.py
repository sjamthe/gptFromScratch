import os
import sys
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Adjust path to import models
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

def load_model(checkpoint_path, device='cpu'):
    print(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # We must explicitly disable cache for a single dense forward pass
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model, config

def main():
    device = 'cpu'
    model_path = os.path.join(base_dir, "models/ut1.5M_2l_8h_384e_mlp3_phaseMask_True_rpn3_216000.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find checkpoint {model_path}")
        return

    model, config = load_model(model_path, device)
    tokenizer = RPNTokenizer(os.path.join(base_dir, "rpn-tokenizer.json"))

    import random
    import create_dataset
    
    if False:
        # We will generate a 4-operand sequence by patching generate_number to guarantee single digits
        original_gen = create_dataset.generate_number
        create_dataset.generate_number = lambda l: str(random.randint(1, 9))
        
        while True:
            sample_str, _, _ = create_dataset.generate_example(max_numbers=4)
            prompt_part = sample_str.split('?')[0]
            ops_count = prompt_part.count('+') + prompt_part.count('-')
            if ops_count == 3: # 4 numbers
                break
                
        create_dataset.generate_number = original_gen

    sample_str = "[BOS]2 6- 6+ 4+?[REV]2[SEP]6-=[SEP]6+[SEP]4+[MATH]2-6-0=6:[BORROW]1|-:10-6=4[REV]-4[SEP]6+=[SEP]4+[MATH]-4+6+0=2:[BORROW]0|+[REV]2[SEP]4+=[MATH]2+4+0=6:[BORROW]0|+[REV]6[ANS]6[EOS]"
    
    print(f"Analyzing 4-Number Sequence: {sample_str}")
    
    # Tokenize
    tokens = tokenizer.encode(sample_str)
    tokens_text = [tokenizer.decode([t]) for t in tokens]
    
    # Forward Pass
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Pass full_phase_ids if model uses phase masking
        is_phase_shift = (idx == 10) | (idx == 11) | (idx == 12)
        full_phase_ids = is_phase_shift.cumsum(dim=-1)
        
        logits, _, all_weights = model(idx, return_attention=True, full_phase_ids=full_phase_ids)

    print(f"Extracted attention weights for {len(all_weights)} layers.")
    
    # all_weights contains weights for each layer.
    # If use_mohsa is True, each layer has (aw_l, aw_s)
    # Shape of aw is (B, n_head, T, T)
    
    # Create an output directory for the heatmaps
    out_dir = os.path.join(base_dir, "scratch/attention_maps")
    os.makedirs(out_dir, exist_ok=True)
    
    for layer_idx, weights in enumerate(all_weights):
        if isinstance(weights, tuple):
            # MOHSA
            aw_l, aw_s = weights
            # Let's plot the long stream for now as it contains the global context
            aw = aw_l
            stream_name = "LongStream"
        else:
            aw = weights
            stream_name = "Standard"
            
        aw = aw[0].cpu().numpy() # (n_head, T, T)
        n_head = aw.shape[0]
        
        for head_idx in range(n_head):
            head_attn = aw[head_idx] # (T, T)
            
            plt.figure(figsize=(24, 20))
            sns.heatmap(head_attn, xticklabels=tokens_text, yticklabels=tokens_text, cmap="viridis")
            plt.title(f"Layer {layer_idx} | Head {head_idx} ({stream_name})", fontsize=20)
            plt.xlabel("Key Token (Being Attended To)", fontsize=16)
            plt.ylabel("Query Token (Generating)", fontsize=16)
            
            # Rotate tick labels for readability
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            out_file = os.path.join(out_dir, f"L{layer_idx}_H{head_idx}_{stream_name}.png")
            plt.savefig(out_file, dpi=100)
            plt.close()
            
    print(f"\nDone! Saved {len(all_weights) * aw.shape[0]} heatmaps to {out_dir}")
    
    print("\n--- Tracking the Pointer Head for {num_count} Operands ---")
    
    target_indices = []
    for i, t in enumerate(tokens_text):
        if t == '|':
            target_indices.append(i + 1)
            
    print(f"Found {len(target_indices)} math operations. Tracing Layer 1, Head 2:")
    
    for op_idx, target_y_idx in enumerate(target_indices):
        target_token = tokens_text[target_y_idx]
        print(f"\nOperation {op_idx + 1} | Generating '{target_token}' at step {target_y_idx}:")
        
        # We will track Layer 1 (index 1), Head 2
        layer_idx = 1
        head_idx = 2
        
        aw = all_weights[layer_idx]
        aw = aw[0] if isinstance(aw, tuple) else aw
        aw = aw[0].cpu().numpy()
        
        attn_dist = aw[head_idx, target_y_idx, :]
        top_indices = np.argsort(attn_dist)[::-1]
        top_indices = [i for i in top_indices if i < target_y_idx][:5]
        top_tokens = [f"'{tokens_text[i]}' (pos {i}, weight {attn_dist[i]:.2f})" for i in top_indices]
        print(f"  -> Left Operand looks at: {', '.join(top_tokens)}")
        
        target_y_idx2 = target_y_idx + 2
        target_token2 = tokens_text[target_y_idx2]
        print(f"Operation {op_idx + 1} | Generating '{target_token2}' at step {target_y_idx2}:")
        attn_dist2 = aw[head_idx, target_y_idx2, :]
        top_indices2 = np.argsort(attn_dist2)[::-1]
        top_indices2 = [i for i in top_indices2 if i < target_y_idx2][:5]
        top_tokens2 = [f"'{tokens_text[i]}' (pos {i}, weight {attn_dist2[i]:.2f})" for i in top_indices2]
        print(f"  -> Right Operand looks at: {', '.join(top_tokens2)}")

if __name__ == "__main__":
    main()
