import os
import sys
import torch
import numpy as np

# Ensure working directory and rpn3_llm are in python path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "rpn3_llm"))

from utils import RPNTokenizer
from model_rope import GPT

def get_ascii_char(val):
    chars = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i"]
    idx = int(clip(val * 10, 0, 9))
    return chars[idx]

def clip(val, l, h):
    return max(l, min(val, h))

def format_token_char(token_str):
    if len(token_str) > 1:
        if token_str.startswith("[") and token_str.endswith("]"):
            return token_str[1]  # 1st letter after [ (e.g. 'R' for [REV])
        else:
            return token_str[0]
    return token_str

def interpretability_lens():
    # Hardcoded test checkpoints (can be chosen by user)
    base_checkpoint = "rpn3_llm/models/ut1.8M_2l_8h_384e_mlp4_phaseMask_True_sft_1-14_7num_BOS_352000.pt"
    recal_checkpoint = "rpn3_llm/models/ut1.8M_2l_8h_384e_mlp4_phaseMask_True_theta_sftRecal_sft_1-14_7num_BOS_40000.pt"
    
    checkpoint_path = recal_checkpoint if os.path.exists(recal_checkpoint) else base_checkpoint
    
    device = 'cpu'
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    config.universal = True
    
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    tokenizer = RPNTokenizer("rpn3_llm/rpn-tokenizer.json")
    
    # Prompt representing the harvested duplicate '3' failure boundary:
    # prompt: [BOS]57 313733-?[REV]75[SEP]3 -> expected next token '3'
    prompt_str = "[BOS]57 313733-?[REV]75[SEP]3"
    tokens = tokenizer.encode(prompt_str)
    
    inputs = torch.tensor([tokens], dtype=torch.long, device=device)
    is_phase_shift = (inputs == 10) | (inputs == 11) | (inputs == 12)
    full_phase_ids = is_phase_shift.cumsum(dim=-1)
    
    # Run forward pass extracting attention weights
    with torch.no_grad():
        logits, _, all_weights = model(inputs, targets=inputs, return_attention=True, full_phase_ids=full_phase_ids)
        
    print("\n" + "="*80)
    print("                UNIVERSAL TRANSFORMER MECHANISTIC INTERPRETABILITY LENS")
    print("="*80)
    print(f"Active Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Prompt: {prompt_str}")
    print("-" * 80)
    
    # 1. LOGIT LENS (Universal pass-by-pass token probabilities for Query Index 16)
    q_idx = 16
    q_str = tokenizer.decode([tokens[q_idx]])
    print(f"\n[LOGIT LENS] Query Token at Index {q_idx} ('{q_str}'):")
    print("-" * 55)
    print(f"{'Pass Index':<12} | {'Top Prediction':<20} | {'Probability':<12}")
    print("-" * 55)
    
    # In Universal Transformer, all_weights contains attention weights for each step/pass.
    # To trace logit progression, we can hook layers, but here we show prediction after final pass:
    probs = torch.softmax(logits[0, q_idx, :], dim=-1)
    top_prob, top_id = torch.topk(probs, k=3)
    for rank in range(3):
        pred_tok = tokenizer.decode([top_id[rank].item()])
        pred_prob = top_prob[rank].item() * 100
        print(f"Rank {rank+1:<9} | '{pred_tok}':{top_id[rank].item():<16} | {pred_prob:.2f}%")
        
    # 2. ASCII ATTENTION HEATMAPS (For Pass 0 Head 0 and Head 5)
    print("\n" + "="*80)
    print("                     ASCII ATTENTION MATRIX MAPS (PASS 0)")
    print("="*80)
    print("Intensity representation: [  . : - = + * # % @ ] (Empty = <3%, @ = 100%)")
    
    # Get attention weights for Pass 0
    # Shape of all_weights[0]: [B, n_head, T_q, T_k]
    pass0_weights = all_weights[0]
    if pass0_weights is not None:
        B, n_head, T_q, T_k = pass0_weights.shape
        # We'll print the attention grids for Head 0 (Copy Head) and Head 5 (Lookahead Head)
        for head in [0,1,2,3,4,5,6,7]:
            print(f"\n>>> Head {head} Attention Map (Causal Query x Key) <<<")
            print("      " + "".join([f"{format_token_char(tokenizer.decode([tokens[k]])):^2s}" for k in range(T_k)]))
            print("     " + "".join([f"{k:2d}" for k in range(T_k)]))
            print("   " + "-" * (T_k * 2 + 5))
            
            for q in range(T_q):
                q_token_str = tokenizer.decode([tokens[q]])
                line_chars = []
                for k in range(T_k):
                    if k <= q:
                        val = pass0_weights[0, head, q, k].item()
                        line_chars.append(f"{get_ascii_char(val)} ")
                    else:
                        line_chars.append("  ")  # Causal mask
                print(f"{format_token_char(q_token_str):^2s}{q:2d}| " + "".join(line_chars))
                
if __name__ == "__main__":
    interpretability_lens()
