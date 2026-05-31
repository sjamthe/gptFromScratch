import os
import sys
import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import math

# Add parent directory to path to import model_rope and utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

def analyze_coordinate_attention_by_phase(checkpoint_path, num_examples=100, tokenizer_path="rpn3_llm/rpn-tokenizer.json", dataset_path="rpn3_llm/data/sft_1-6_4num_BOS_val.txt"):
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = RPNTokenizer(tokenizer_path)
    vocab_size = len(tokenizer.vocab)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    config = checkpoint['config']
    print(f"Model config: universal={getattr(config, 'universal', False)}, n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}, n_coord={getattr(config, 'n_coord', 0)}, n_coord_heads={getattr(config, 'n_coord_heads', 4)}")
    
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    n_coord = getattr(config, 'n_coord', 0)
    n_coord_heads = getattr(config, 'n_coord_heads', 4)
    if n_coord == 0:
        print("Error: No coordinate heads in this model checkpoint.")
        return

    # Set up forward hooks to capture CoordinateHead Q and K projections
    q_projections = {}
    k_projections = {}
    
    def make_q_hook(name):
        def hook(module, input, output):
            q_projections[name] = output.detach()
        return hook

    def make_k_hook(name):
        def hook(module, input, output):
            k_projections[name] = output.detach()
        return hook

    hooks = []
    for c_idx in range(n_coord):
        head = model.coordinate_heads[c_idx]
        h_q = head.q_proj.register_forward_hook(make_q_hook(f"coord_{c_idx}"))
        h_k = head.k_proj.register_forward_hook(make_k_hook(f"coord_{c_idx}"))
        hooks.extend([h_q, h_k])

    # Read validation examples
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(parent_dir, "data/sft_1-6_4num_BOS_val.txt")
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset not found at {dataset_path}")
            return
            
    print(f"Reading first {num_examples} examples from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = [f.readline().strip() for _ in range(num_examples)]
    
    # Filter empty lines
    lines = [l for l in lines if l]
    
    # We want to accumulate attention probabilities:
    # key: (c_idx, head_idx, q_phase) -> 2D matrix of shape [vocab_size, vocab_size] (query_tok, key_tok)
    accum_attn = defaultdict(lambda: np.zeros((vocab_size, vocab_size), dtype=float))
    accum_count = defaultdict(lambda: np.zeros(vocab_size, dtype=float))
    
    interesting_tokens = [
        "[BOS]", "[EOS]", "[SEP]", "[REV]", "[BORROW]", "[MATH]", "[ANS]", " ", "+", "-",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    ]
    interesting_ids = [tokenizer.vocab.get(t) for t in interesting_tokens if t in tokenizer.vocab]

    for line_idx, line in enumerate(lines):
        tokens = tokenizer.encode(line)
        if not tokens:
            continue
            
        idx = torch.tensor([tokens], dtype=torch.long)
        
        # Clear hooked projections cache
        q_projections.clear()
        k_projections.clear()
        
        # Run forward pass to trigger hooks
        with torch.no_grad():
            _ = model(idx)
            
        # Get phase classifications for each position in this sequence
        seq_len = len(tokens)
        phases = []
        curr_phase = "BASE"
        
        for i in range(seq_len):
            t_id = tokens[i]
            t_str = tokenizer.inverse_vocab.get(t_id, "")
            
            if t_str == "[REV]":
                curr_phase = "REV"
            elif t_str == "[MATH]":
                curr_phase = "MATH"
            elif t_str == "[ANS]":
                curr_phase = "ANS"
                
            phases.append(curr_phase)
            
        # Process attention for each coordinate module
        for c_idx in range(n_coord):
            proj_name = f"coord_{c_idx}"
            if proj_name not in q_projections or proj_name not in k_projections:
                continue
                
            # output is [1, T, n_heads * head_dim]
            q_val = q_projections[proj_name][0] # [T, n_heads * head_dim]
            k_val = k_projections[proj_name][0] # [T, n_heads * head_dim]
            
            T_len = q_val.shape[0]
            head_dim = model.coordinate_heads[c_idx].head_dim
            
            # Reshape Q, K to [n_heads, T, head_dim]
            Q = q_val.view(T_len, n_coord_heads, head_dim).transpose(0, 1)
            K = k_val.view(T_len, n_coord_heads, head_dim).transpose(0, 1)
            
            # scores: [n_heads, T, T]
            scores = torch.matmul(Q, K.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
            
            # Apply causal mask (j > i is masked)
            causal_mask = torch.triu(torch.ones(T_len, T_len), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
            
            # attn_probs: [n_heads, T, T]
            attn_probs = F.softmax(scores, dim=-1).numpy()
            
            for i in range(seq_len):
                q_tok = tokens[i]
                q_phase = phases[i]
                
                for h in range(n_coord_heads):
                    for j in range(i + 1):
                        k_tok = tokens[j]
                        p = attn_probs[h, i, j]
                        
                        accum_attn[(c_idx, h, q_phase)][q_tok, k_tok] += p
                        
                    accum_count[(c_idx, h, q_phase)][q_tok] += 1.0

    # Remove hooks
    for h in hooks:
        h.remove()
        
    # Generate markdown output report
    print("\n# CoordinateHead Dynamic Phase-Shift Attention Analysis Report")
    
    for c_idx in range(n_coord):
        scale = model.coordinate_heads[c_idx].scale.item()
        print(f"\n## Analysis of CoordinateHead Module {c_idx} (Scale: {scale:.4f})")
        
        if abs(scale) < 0.01:
            print(f"> [!NOTE]")
            print(f"> CoordinateHead Module {c_idx} has been classified as **INACTIVE** (scale magnitude {abs(scale):.4f} < 0.01).")
            print("> Detailed tables are omitted to reduce clutter.")
            continue
            
        phase_tags = ["BASE", "REV", "MATH", "ANS"]
        
        for h in range(n_coord_heads):
            print(f"\n### Head {h} Analysis")
            
            for tag in phase_tags:
                print(f"\n#### Phase Context: `{tag}`")
                print("| Query Token | Top 5 Key Tokens (average attention %, raw count) |")
                print("|---|---|")
                
                for q_tok in interesting_ids:
                    denom = accum_count[(c_idx, h, tag)][q_tok]
                    q_str = tokenizer.inverse_vocab.get(q_tok, f"[ID:{q_tok}]")
                    if q_str == ' ':
                        q_str = "' '"
                        
                    if denom == 0:
                        print(f"| {q_str} | (no occurrences) |")
                        continue
                        
                    # Find key tokens
                    k_sums = accum_attn[(c_idx, h, tag)][q_tok]
                    active_keys = np.where(k_sums > 0.0)[0]
                    
                    # Sort keys by avg prob
                    avg_probs = k_sums / denom
                    active_keys = active_keys[np.argsort(avg_probs[active_keys])[::-1]]
                    
                    top_strs = []
                    for k_tok in active_keys[:5]:
                        pct = avg_probs[k_tok] * 100
                        k_str = tokenizer.inverse_vocab.get(k_tok, f"[ID:{k_tok}]")
                        if k_str == ' ':
                            k_str = "' '"
                        elif k_str == '\n':
                            k_str = "'\\n'"
                        top_strs.append(f"**{k_str}** ({pct:.1f}%)")
                        
                    print(f"| {q_str} | {', '.join(top_strs)} |")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of validation examples to run")
    parser.add_argument("--tokenizer", type=str, default="rpn3_llm/rpn-tokenizer.json", help="Path to tokenizer json")
    parser.add_argument("--dataset", type=str, default="rpn3_llm/data/sft_1-6_4num_BOS_val.txt", help="Path to validation text dataset")
    
    args = parser.parse_args()
    
    analyze_coordinate_attention_by_phase(args.checkpoint, args.num_examples, args.tokenizer, args.dataset)
