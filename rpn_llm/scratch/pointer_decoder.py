import torch
import os
import sys

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_rope import GPT
from utils import RPNTokenizer

def decode_by_pointing(ckpt_path, prompt, device='cpu'):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    tokenizer = RPNTokenizer("/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/rpn-tokenizer.json")

    # 1. Identify Operand Zones
    # Prompt example: (123)(456)+?
    tokens = tokenizer.encode(prompt)
    decoded = [tokenizer.decode([t]) for t in tokens]
    
    # Simple regex-like search for parentheses
    import re
    matches = list(re.finditer(r'\((\d+)\)', prompt))
    if len(matches) < 2:
        print("Error: Could not identify two operands in parentheses.")
        return
    
    # We find the token indices for each operand
    # This is a bit tricky with tokenization, so we'll just scan the decoded tokens
    zones = []
    current_zone = []
    in_zone = False
    for i, char in enumerate(decoded):
        if char == '(': 
            in_zone = True
            current_zone = []
        elif char == ')':
            in_zone = False
            zones.append(current_zone)
        elif in_zone:
            current_zone.append(i)
            
    if len(zones) < 2:
        print("Error: Could not map operand zones.")
        return

    op1_indices = zones[0]
    op2_indices = zones[1]
    
    print(f"\nPOINTER DECODER DIAGNOSTIC")
    print("="*60)
    print(f"Prompt: {prompt}")
    print(f"Operand 1 Zone: indices {op1_indices} ({[decoded[i] for i in op1_indices]})")
    print(f"Operand 2 Zone: indices {op2_indices} ({[decoded[i] for i in op2_indices]})")
    print("-" * 60)

    # 2. Run Forward Pass to get attention
    idx = torch.tensor([tokens], device=device)
    _, _, _, all_weights = model(idx, return_attention=True)
    
    # We'll look at the LAST token of the prompt (the current output being refined)
    last_idx = len(tokens) - 1
    target_token = decoded[last_idx]
    
    print(f"Analyzing multi-digit reasoning for token '{target_token}' (Index {last_idx})")
    print("-" * 60)
    print(f"{'HEAD':<8} | {'PEAK TOKEN':<12} | {'INDEX':<6} | {'STRENGTH':<10}")
    print("-" * 60)
    
    # Analyze all 12 heads in Pass 5 (The Maturity Pass)
    logic_pass = 4 
    pass_weights = all_weights[logic_pass][0] # Shape (Heads, T, T)
    
    num_heads = pass_weights.shape[0]
    for h in range(num_heads):
        attn = pass_weights[h, last_idx]
        
        peak_idx = torch.argmax(attn).item()
        peak_token = decoded[peak_idx]
        strength = attn[peak_idx].item()
        
        print(f"Head {h+1:<3} | '{peak_token}':{decoded[peak_idx]} | {peak_idx:<6} | {strength:.4f}")

    # 3. Decision Integration
    print("\n[ROLE DISCOVERY]")
    # Partner: '2' at Index 8
    # Source: '9' at Index 3
    partner_digit_idx = 8
    source_digit_idx = 3
    
    partner_specialists = []
    source_specialists = []
    
    for h in range(num_heads):
        attn = pass_weights[h, last_idx]
        peak = torch.argmax(attn).item()
        if peak == partner_digit_idx:
            partner_specialists.append(h+1)
        if peak == source_digit_idx:
            source_specialists.append(h+1)
            
    print(f"Partner Specialists (looking at '{decoded[partner_digit_idx]}'): {partner_specialists}")
    print(f"Source Specialists  (looking at '{decoded[source_digit_idx]}'): {source_specialists}")

    # 4. Compare with Model Logit
    logits, _ = model(idx)
    model_next_id = torch.argmax(logits[:, -1, :], dim=-1).item()
    model_char = tokenizer.decode([model_next_id])
    
    print(f"\n[FINAL BATTLE]")
    print(f"Model-Predicted Token (Actual Logit): '{model_char.strip()}'")

if __name__ == "__main__":
    ckpt = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/models/UT3M_1-22_tens_comp_bracketed_final.pt"
    # Testing the user's specific prompt
    test_prompt = "(149)(142)+?<(9" 
    decode_by_pointing(ckpt, test_prompt)
