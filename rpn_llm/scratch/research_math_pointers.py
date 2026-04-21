import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_rope import GPT
from utils import RPNTokenizer

def analyze_math_pointers(ckpt_path, prompt, device='cpu'):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    tokenizer = RPNTokenizer("/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/rpn-tokenizer.json")

    tokens = tokenizer.encode(prompt)
    decoded = [tokenizer.decode([t]) for t in tokens]
    
    idx = torch.tensor([tokens], device=device)
    logits, _, _, all_weights = model(idx, return_attention=True)
    
    last_idx = len(tokens) - 1
    
    model_next_id = torch.argmax(logits[:, -1, :], dim=-1).item()
    model_char = tokenizer.decode([model_next_id])
    
    print(f"Prompt: {prompt}")
    print(f"Logits prediction for next token: {model_char}")
    
    for p in range(len(all_weights)):
        pass_weights = all_weights[p][0] # (Heads, T, T)
        max_attn, _ = torch.max(pass_weights, dim=0) # max pool over heads
        
        attn = max_attn[last_idx]
        
        # Exclude the exit token itself (last_idx) to see what it refers to in the past
        attn_copy = attn.clone()
        attn_copy[last_idx] = 0
        
        # Get top 3 pointers
        top_vals, top_indices = torch.topk(attn_copy, k=3)
        
        ptrs = []
        for val, i in zip(top_vals, top_indices):
            ptrs.append(f"'{decoded[i.item()]}' (idx {i.item()}, {val.item():.3f})")
            
        print(f"Pass {p+1}: Top pointers -> {', '.join(ptrs)}")

if __name__ == "__main__":
    ckpt = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/models/UT3M_1-22_tens_comp_bracketed_final.pt"
    # Testing the adversarial computation step
    print("--- STANDARD PROMPT ---")
    analyze_math_pointers(ckpt, "(3037913)(48)+?<(3197303)(84)+=:3+8+0=")
    
    print("\n--- ADVERSARIAL PROMPT ---")
    analyze_math_pointers(ckpt, "(3037913)(48)+?<(3197303)(84)+=:3+8+0+")
