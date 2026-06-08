import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

def inspect_model(checkpoint_path, device='cpu'):
    print(f"\n========================================================")
    print(f"Inspecting Checkpoint: {checkpoint_path}")
    print(f"========================================================")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    print(f"n_coord: {config.n_coord}")
    print(f"coord_inject_layers: {config.coord_inject_layers}")
    
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    tokenizer = RPNTokenizer(os.path.join(os.path.dirname(__file__), "rpn-tokenizer.json"))
    
    # 1. Print scales and weight norms
    for idx, head in enumerate(model.coordinate_heads):
        layer_idx = config.coord_inject_layers[idx]
        scale_val = head.scale.item()
        q_norm = head.q_proj.weight.norm().item()
        k_norm = head.k_proj.weight.norm().item()
        out_norm = head.out_proj.weight.norm().item()
        
        print(f"\nCoordinate Head {idx} (Injected at Layer {layer_idx}):")
        print(f"  Scale parameter: {scale_val:.6f}")
        print(f"  q_proj weight norm: {q_norm:.4f}")
        print(f"  k_proj weight norm: {k_norm:.4f}")
        print(f"  out_proj weight norm: {out_norm:.4f}")
        
    # 2. Extract attention maps using hooks
    attn_probs_dict = {}
    
    def get_activation_hook(name):
        def hook(model, input, output):
            # The coordinate head forward returns (out, new_cache)
            # We want to capture the attention probabilities computed inside.
            # Let's temporarily compute it manually since we can't easily grab it from the return.
            x = input[0]
            B, T, C = x.size()
            q = model.q_proj(x)
            k = model.k_proj(x)
            q = q.view(B, T, model.n_heads, model.head_dim).transpose(1, 2)
            k = k.view(B, T, model.n_heads, model.head_dim).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(model.head_dim))
            
            q_idx = torch.arange(T, device=x.device).view(-1, 1)
            k_idx = torch.arange(T, device=x.device).view(1, -1)
            causal_mask = q_idx < k_idx
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), float('-inf'))
            
            attn_probs = F.softmax(scores, dim=-1) # [B, n_heads, T, T]
            attn_probs_dict[name] = attn_probs.cpu()
        return hook

    hooks = []
    for idx, head in enumerate(model.coordinate_heads):
        h = head.register_forward_hook(get_activation_hook(f"head_{idx}"))
        hooks.append(h)
        
    # Test prompt
    prompt = "[REV]<num>1234567890123456789012</num>[ANS]"
    tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # Run forward pass
    phase_tensor = torch.tensor(config.phase_token_ids, device=device)
    is_phase_shift = torch.isin(prompt_tensor, phase_tensor)
    full_phase_ids = is_phase_shift.cumsum(dim=-1)
    
    with torch.no_grad():
        _ = model(prompt_tensor, full_phase_ids=full_phase_ids)
        
    # Remove hooks
    for h in hooks:
        h.remove()
        
    # Analyze the attention maps
    # Let's inspect the attention of query tokens during the reversal part.
    # Specifically, look at query tokens corresponding to digits or delimiters in the prompt.
    print(f"\nAnalyzing Attention Maps for Prompt: {prompt}")
    decoded_tokens = [tokenizer.decode([t]) for t in tokens]
    
    for head_name, probs in attn_probs_dict.items():
        # probs: [1, n_heads, T, T]
        num_heads = probs.size(1)
        T = probs.size(2)
        print(f"\n--- {head_name} Attention Entropy and Affinity ---")
        
        # Calculate mean entropy per query position (excluding first few tokens)
        for h_idx in range(num_heads):
            head_probs = probs[0, h_idx] # [T, T]
            entropies = []
            
            # Let's see what the query token at index T-2 ([ANS]) is attending to
            ans_probs = head_probs[T-2] # attention from [ANS]
            max_val, max_idx = ans_probs.max(dim=-1)
            target_token = decoded_tokens[max_idx.item()]
            
            # Let's print out the top attended tokens for some key queries:
            # 1. The first digit after <num> (index 6: '1')
            # 2. A middle digit (index 15: '0')
            # 3. The [ANS] token (index T-2)
            
            print(f"  Head {h_idx}:")
            # Query: [ANS] (index T-2)
            ans_top_indices = torch.topk(ans_probs, k=min(3, T)).indices.tolist()
            ans_top_tokens = [f"{decoded_tokens[idx]} ({ans_probs[idx].item():.2f})" for idx in ans_top_indices]
            print(f"    Query '[ANS]' attends to: {', '.join(ans_top_tokens)}")
            
            # Query: middle digit (say index 15)
            if T > 15:
                mid_probs = head_probs[15]
                mid_top_indices = torch.topk(mid_probs, k=min(3, T)).indices.tolist()
                mid_top_tokens = [f"{decoded_tokens[idx]} ({mid_probs[idx].item():.2f})" for idx in mid_top_indices]
                print(f"    Query '{decoded_tokens[15]}' (pos 15) attends to: {', '.join(mid_top_tokens)}")

if __name__ == "__main__":
    device = 'cpu'
    checkpoint_path = os.path.join(os.path.dirname(__file__), "models/lesson4_wrappedNum_step40000.pt")
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    inspect_model(checkpoint_path, device)
