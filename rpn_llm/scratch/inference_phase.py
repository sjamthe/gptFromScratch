import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from utils import RPNTokenizer
from model_rope import GPT, GPTConfig

def generate_with_phases(model, tokenizer, prompt, device, max_new_tokens=256):
    model.eval()
    # Add [BOS] and ? if missing
    if not prompt.startswith("[BOS]"): prompt = "[BOS]" + prompt
    if not prompt.endswith("?"): prompt = prompt + "?"
    
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"Prompt: {prompt}")
    print("Generating...")
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass (model_rope.py handles masking internally for T > 1)
            # For inference T=1 usually, but here T grows.
            # model_rope.py: if T > 1: apply masks. 
            # During generation, T increases. 
            logits, _ = model(idx)
            logits = logits[:, -1, :] # focus on last token
            
            # Sampling (greedy for now)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
            
            # Stop on [EOS] (ID 3)
            if next_token.item() == 3:
                break
                
    full_text = tokenizer.decode(idx[0].tolist())
    print("\nFull Output:")
    print(full_text)
    
    # Split by phase markers for analysis
    for marker in ["[REV]", "[MATH]", "[ANS]", "[EOS]"]:
        full_text = full_text.replace(marker, f"\n{marker} ")
    
    print("\nPhase Analysis:")
    print(full_text)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint_path = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/models/rope3.6M_1-22_phase_lean_8000.pt"
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    
    tokenizer = RPNTokenizer("/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/rpn-tokenizer.json")
    
    # Test cases
    test_prompts = [
        "23 45+?",
        "689 949-?",
        "8 89+?"
    ]
    
    for p in test_prompts:
        generate_with_phases(model, tokenizer, p, device)
        print("-" * 50)

if __name__ == "__main__":
    main()
