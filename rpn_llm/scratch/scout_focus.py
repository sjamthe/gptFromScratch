import torch
import os
import sys

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_rope import GPT
from utils import RPNTokenizer

def scout():
    device = 'cpu'
    ckpt_path = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/models/UT3M_1-22_tens_comp_bracketed_final.pt"
    prompt = "(123)(456)+?<(32"
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    tokenizer = RPNTokenizer("/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/rpn-tokenizer.json")
    
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], device=device)
    
    # Run model return_attention=True
    # GPT.forward returns: logits, loss, present_key_values, all_weights
    _, _, _, all_weights = model(idx, return_attention=True)
    
    # all_weights is a list of weights (one per pass)
    # each element has shape (B, Heads, T, T)
    num_passes = len(all_weights)
    T = all_weights[0].shape[3]
    last_idx = T - 1 # Index 15 for '2'
    
    decoded_tokens = [tokenizer.decode([t]) for t in tokens]
    
    # We compare indices 14 ('3') and 15 ('2')
    indices_to_check = [14, 15]
    
    for target_idx in indices_to_check:
        token_char = decoded_tokens[target_idx]
        print(f"\nREVERSAL LOGIC: Analyzing Token @ Index {target_idx} ('{token_char}')")
        print("="*60)
        
        for p in [2, 4, 7]: # Checking Beginning, Middle, and End of reasoning
            print(f"Pass {p+1}:")
            avg_weights = all_weights[p][0].mean(dim=0)
            attn_for_target = avg_weights[target_idx]
            
            top_vals, top_indices = torch.topk(attn_for_target, k=5)
            for val, idx in zip(top_vals, top_indices):
                token_str = decoded_tokens[idx.item()]
                print(f"  -> {val.item():.4f} focusing on: '{token_str}' (Index {idx.item()})")
            
if __name__ == "__main__":
    scout()
