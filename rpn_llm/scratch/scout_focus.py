import os
import sys
import torch

# Add the parent directory (rpn_llm) to the path so we can import model_rope
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_rope import GPT
from utils import RPNTokenizer

def scout_attention(checkpoint_path, prompt_str):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device); model.eval()
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
    
    tokens = tokenizer.encode(prompt_str)
    x = torch.tensor([tokens], device=device)
    token_labels = [tokenizer.decode([t]) for t in tokens]
    
    # Target: The '1' in (123) is at index 1
    target_idx = 27
    print(f"Scouting attention for target token '{token_labels[target_idx]}' (Index {target_idx})")
    
    with torch.no_grad():
        _, _, _, all_weights = model(x, return_attention=True)
        
    for layer_idx, layer_weights in enumerate(all_weights):
        # layer_weights is (1, nh, T, T)
        # We look at the LAST generated token (the most recent one)
        last_query_idx = len(tokens) - 1
        head_scores = layer_weights[0, :, last_query_idx, target_idx] # (nh,)
        
        max_head = torch.argmax(head_scores).item()
        max_val = head_scores[max_head].item()
        
        print(f"Layer {layer_idx+1}: Max focus on {target_idx} is Head {max_head+1} with {max_val:.4f} score")
        
if __name__ == "__main__":
    scout_attention("rpn_llm/models/rope25M_1-22_tens_comp_bracketed_final.pt", "(123)(456)+?<(321)(654)+=:3+6+0=")
