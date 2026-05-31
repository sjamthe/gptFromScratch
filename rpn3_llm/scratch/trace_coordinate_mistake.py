import os
import sys
import torch
import torch.nn.functional as F
import math

# Ensure working directory and rpn3_llm are in python path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "rpn3_llm"))

from utils import RPNTokenizer
from model_rope import GPT

def main():
    checkpoint_path = "rpn3_llm/models/ut2.1M_2l_8h_384e_mlp4_phaseMask_True_cnt2_crd2_sft_1-6_4num_BOS_200000.pt"
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tokenizer = RPNTokenizer("rpn3_llm/rpn-tokenizer.json")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # Prompt up to the start of the 3rd reversal block
    # Expected: 95112[SEP]85952+=[SEP]437913968-[SEP]013216-
    # Prefix just before the mistake: 
    # [BOS]21159 25958+ 869319734- 612310-?[REV]95112[SEP]85952+=[SEP]4379
    # (reversing the digits of 869319734: expected is 4, 3, 7, 9, 1, 3, 9, 6, 8. It has generated 4379 so far.)
    prompt_str = "[BOS]21159 25958+ 869319734- 612310-?[REV]95112[SEP]85952+=[SEP]4379"
    tokens = tokenizer.encode(prompt_str)
    
    print("\n--- Token index reference in prompt ---")
    for idx, t_id in enumerate(tokens):
        t_str = tokenizer.decode([t_id])
        if t_str == " ":
            t_str = "' '"
        print(f"  Index {idx:2d}: Token {t_id:4d} -> {t_str}")
        
    inputs = torch.tensor([tokens], dtype=torch.long, device=device)
    is_phase_shift = (inputs == 10) | (inputs == 11) | (inputs == 12)
    full_phase_ids = is_phase_shift.cumsum(dim=-1)
    
    # We want to trace the attention of:
    # 1. Self-attention weights from self.transformer.h.attn (universal block)
    # 2. Coordinate head attention weights from model.coordinate_heads[0]
    
    # Hooks to capture CoordinateHead Module 0 activations
    c_q_val, c_k_val = None, None
    def c_q_hook(module, input, output):
        nonlocal c_q_val
        c_q_val = output.detach()
        
    def c_k_hook(module, input, output):
        nonlocal c_k_val
        c_k_val = output.detach()
        
    h_q = model.coordinate_heads[0].q_proj.register_forward_hook(c_q_hook)
    h_k = model.coordinate_heads[0].k_proj.register_forward_hook(c_k_hook)
    
    with torch.no_grad():
        logits, _, all_weights = model(inputs, return_attention=True, full_phase_ids=full_phase_ids)
        
    # Remove hooks
    h_q.remove()
    h_k.remove()
    
    # Get last query token attention weights
    # Last query index is len(tokens) - 1
    q_idx = len(tokens) - 1
    q_str = tokenizer.decode([tokens[q_idx]])
    
    print(f"\n=======================================================")
    print(f"ANALYZING CURRENT QUERY TOKEN AT INDEX {q_idx}: '{q_str}'")
    print(f"=======================================================")
    
    # 1. Predictions for the next token (after '9')
    next_logits = logits[0, -1, :]
    probs = torch.softmax(next_logits, dim=-1)
    top_k = 5
    top_probs, top_indices = torch.topk(probs, k=top_k)
    print(f"\nTop {top_k} Predictions for the next token:")
    for rank in range(top_k):
        idx = top_indices[rank].item()
        p = top_probs[rank].item() * 100
        print(f"  Rank {rank+1}: '{tokenizer.decode([idx])}' (ID: {idx:4d}) -> {p:5.2f}%")
        
    # 2. Coordinate Head attention weights reconstruction
    # c_q_val shape: [1, T, n_heads * head_dim]
    # c_k_val shape: [1, T, n_heads * head_dim]
    coord_head = model.coordinate_heads[0]
    head_dim = coord_head.head_dim
    n_heads = coord_head.n_heads
    T = c_q_val.shape[1]
    
    Q = c_q_val[0].view(T, n_heads, head_dim).transpose(0, 1) # [n_heads, T, head_dim]
    K = c_k_val[0].view(T, n_heads, head_dim).transpose(0, 1) # [n_heads, T, head_dim]
    
    scores = torch.matmul(Q, K.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim)) # [n_heads, T, T]
    causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
    
    # Apply phase masking if applicable
    if model.config.use_phase_mask:
        is_bos = (inputs == model.config.bos_token_id)
        seq_ids = is_bos.cumsum(dim=-1)
        doc_mask = (seq_ids.unsqueeze(1) == seq_ids.unsqueeze(2))[0]
        
        is_phase_shift_cpu = (inputs == 10) | (inputs == 11) | (inputs == 12)
        global_phase_ids = is_phase_shift_cpu.cumsum(dim=-1)
        phase_diff = (global_phase_ids.unsqueeze(-1) - global_phase_ids.unsqueeze(-2))[0]
        phase_mask = (phase_diff == 0) | (phase_diff == 1)
        
        full_mask = ~(doc_mask & phase_mask & (~causal_mask))
        scores = scores.masked_fill(full_mask.unsqueeze(0), float('-inf'))
    else:
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
        
    coord_attn = F.softmax(scores, dim=-1).cpu().numpy() # [n_heads, T, T]
    
    print("\n--- Coordinate Head Module 0 Attention weights for last query ---")
    for h in range(n_heads):
        h_weights = coord_attn[h, q_idx, :]
        top_keys = np_topk(h_weights, tokens, tokenizer, k=5)
        print(f"  Head {h}: {top_keys}")
        
    # 3. Standard Self-Attention weights
    # all_weights is a list of tensors of shape [B, n_head, T, T]
    # One tensor per pass in universal loop
    print("\n--- Standard Self-Attention Weights for last query (pass-by-pass) ---")
    for pass_idx, aw in enumerate(all_weights):
        # aw shape: [B, n_head, T, T]
        aw_numpy = aw[0].cpu().numpy()
        print(f"  Pass {pass_idx}:")
        for h in range(aw.shape[1]):
            h_weights = aw_numpy[h, q_idx, :]
            top_keys = np_topk(h_weights, tokens, tokenizer, k=5)
            print(f"    Head {h}: {top_keys}")

def np_topk(weights, tokens, tokenizer, k=5):
    # Sort indices by weight descending
    sorted_idxs = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
    res = []
    for idx in sorted_idxs[:k]:
        w = weights[idx] * 100
        t_id = tokens[idx]
        t_str = tokenizer.decode([t_id])
        if t_str == " ":
            t_str = "' '"
        res.append(f"'{t_str}' (idx {idx}): {w:.1f}%")
    return ", ".join(res)

if __name__ == "__main__":
    main()
