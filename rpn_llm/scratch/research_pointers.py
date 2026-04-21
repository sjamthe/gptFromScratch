import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_rope import GPT
from utils import RPNTokenizer
import re

def analyze_pointers(ckpt_path, prompt, device='cpu'):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    tokenizer = RPNTokenizer("/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/rpn-tokenizer.json")

    tokens = tokenizer.encode(prompt)
    decoded = [tokenizer.decode([t]) for t in tokens]
    
    matches = list(re.finditer(r'\((\d+)\)', prompt))
    zones = []
    current_zone = []
    in_zone = False
    for i, char in enumerate(decoded):
        if char == '(': in_zone = True; current_zone = []
        elif char == ')': in_zone = False; zones.append(current_zone)
        elif in_zone: current_zone.append(i)

    op1_indices = zones[0]
    op2_indices = zones[1]

    idx = torch.tensor([tokens], device=device)
    _, _, _, all_weights = model(idx, return_attention=True)
    
    last_idx = len(tokens) - 1
    
    print(f"Prompt: {prompt}")
    for p in range(len(all_weights)):
        pass_weights = all_weights[p][0] # (Heads, T, T)
        
        # Max attention in Op1 and Op2 across ANY head
        max_op1_val = -1
        max_op1_idx = -1
        max_op2_val = -1
        max_op2_idx = -1
        
        for h in range(pass_weights.shape[0]):
            attn = pass_weights[h, last_idx]
            
            for i in op1_indices:
                if attn[i] > max_op1_val:
                    max_op1_val = attn[i].item()
                    max_op1_idx = i
                    
            for i in op2_indices:
                if attn[i] > max_op2_val:
                    max_op2_val = attn[i].item()
                    max_op2_idx = i
                    
        d1 = decoded[max_op1_idx]
        d2 = decoded[max_op2_idx]
        print(f"Pass {p+1}: Op1 points -> '{d1}' ({max_op1_val:.3f}) | Op2 points -> '{d2}' ({max_op2_val:.3f})")

if __name__ == "__main__":
    ckpt = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/models/UT3M_1-22_tens_comp_bracketed_final.pt"
    analyze_pointers(ckpt, "(149)(142)+?<(9")
