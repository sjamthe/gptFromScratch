import os
import sys
import torch
import torch.nn.functional as F

# Add the parent directory to the path so we can import model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import RPNTokenizer

def analyze_checkpoint(checkpoint_path, tokenizer_path):
    print(f"Analyzing checkpoint: {checkpoint_path}")
    
    # Load tokenizer
    tokenizer = RPNTokenizer(tokenizer_path)
    vocab_size = len(tokenizer.vocab)
    
    # Build complete vocabulary map
    vocab = {}
    for i in range(vocab_size):
        try:
            vocab[i] = tokenizer.decode([i])
        except Exception:
            vocab[i] = f"[ID:{i}]"
            
    # Load checkpoint state dict
    device = 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    # Check if we have CoordinateHeads
    coord_keys = [k for k in state_dict.keys() if 'coordinate_heads' in k]
    if not coord_keys:
        print("Error: No coordinate_heads found in this checkpoint.")
        return
        
    # Find n_coord
    n_coords = max([int(k.split('.')[1]) for k in coord_keys]) + 1
    print(f"Found {n_coords} CoordinateHead module(s) in checkpoint.")
    
    # Extract WTE weight
    wte_weight = state_dict['transformer.wte.weight'] # [vocab_size, n_embd]
    
    # Analyze each CoordinateHead module
    for idx in range(n_coords):
        print(f"\n==================================================")
        print(f"CoordinateHead Module {idx} Analysis")
        print(f"==================================================")
        
        q_weight = state_dict[f'coordinate_heads.{idx}.q_proj.weight'] # [n_heads * head_dim, n_embd]
        q_bias = state_dict[f'coordinate_heads.{idx}.q_proj.bias']     # [n_heads * head_dim]
        k_weight = state_dict[f'coordinate_heads.{idx}.k_proj.weight'] # [n_heads * head_dim, n_embd]
        k_bias = state_dict[f'coordinate_heads.{idx}.k_proj.bias']     # [n_heads * head_dim]
        scale = state_dict.get(f'coordinate_heads.{idx}.scale', torch.tensor(1.0)).item()
        
        total_dim, n_embd = q_weight.shape
        # Get config/n_heads from out_proj weight if possible
        out_weight = state_dict[f'coordinate_heads.{idx}.out_proj.weight'] # [n_embd, n_heads]
        n_heads = out_weight.shape[1]
        head_dim = total_dim // n_heads
        
        print(f"Number of heads: {n_heads}, head dimension: {head_dim}, embedding dimension: {n_embd}, Scale: {scale:.4f}")
        
        # Get model vocab size from embedding shape
        vocab_size_model = wte_weight.shape[0]
        
        # Compute projected Q and K for all tokens:
        # Q: [vocab_size_model, n_heads * head_dim]
        # K: [vocab_size_model, n_heads * head_dim]
        Q = torch.matmul(wte_weight, q_weight.t()) + q_bias
        K = torch.matmul(wte_weight, k_weight.t()) + k_bias
        
        # Reshape to [vocab_size_model, n_heads, head_dim]
        Q = Q.view(vocab_size_model, n_heads, head_dim)
        K = K.view(vocab_size_model, n_heads, head_dim)
        
        for h in range(n_heads):
            q_h = Q[:, h, :] # [vocab_size_model, head_dim]
            k_h = K[:, h, :] # [vocab_size_model, head_dim]
            scores = torch.matmul(q_h, k_h.t()) / (head_dim ** 0.5) # [vocab_size_model, vocab_size_model]
            probs = F.softmax(scores, dim=-1) # [vocab_size_model, vocab_size_model]
            
            score_std = scores.std().item()
            print(f"\n--- Head {h} Query-Key Affinity (Score Spread Std Dev: {score_std:.4f}) ---")
            
            # We want to show what each query token looks for.
            # Specifically, list important query tokens (like space, operators, digits, BOS, phase tokens)
            # and show their top 5 preferred key tokens.
            interesting_tokens = [
                "[UNK]","[BOS]", "[EOS]","[SEP]", "[REV]", "[BORROW]","[PASS]", "[MATH]", "[ANS]", " ", "+", "-",
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
            ]

            # Find the token IDs for interesting tokens
            token_ids_to_analyze = []
            for t_str in interesting_tokens:
                # encode returns a list of token ids
                ids = tokenizer.encode(t_str)
                if ids:
                    token_ids_to_analyze.append((t_str, ids[0]))
                    
            # Restricted vocab softmax (only active RPN tokens)
            interesting_ids = [t_id for _, t_id in token_ids_to_analyze]
            interesting_ids_tensor = torch.tensor(interesting_ids, dtype=torch.long, device=scores.device)
            scores_restricted = scores[:, interesting_ids_tensor]
            probs_restricted = F.softmax(scores_restricted, dim=-1)
            
            print("\n| Query Token | Top 5 Key Tokens (restricted softmax %, raw score) |")
            print("|---|---|")
            for t_str, t_id in token_ids_to_analyze:
                # Find top keys within the interesting tokens set
                prob_rest = probs_restricted[t_id]  # [len(interesting_ids)]
                top_rest_vals, top_rest_indices = torch.topk(prob_rest, k=min(5, len(interesting_ids)))
                
                top_strs = []
                for val_rest, idx_rest in zip(top_rest_vals.tolist(), top_rest_indices.tolist()):
                    kid = interesting_ids[idx_rest]
                    raw_score = scores[t_id, kid].item()
                    k_str = vocab.get(kid, f"[PAD:{kid}]")
                    if k_str == ' ':
                        k_label = "' '"
                    elif k_str == '\n':
                        k_label = "'\\n'"
                    else:
                        k_label = k_str
                    top_strs.append(f"{k_label} ({val_rest:.1%}, score:{raw_score:.3f})")
                
                q_label = t_str
                if q_label == ' ':
                    q_label = "' '"
                elif q_label == '\n':
                    q_label = "'\\n'"
                    
                print(f"| {q_label} | {', '.join(top_strs)} |")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--tokenizer", type=str, default="rpn3_llm/rpn-tokenizer.json", help="Path to tokenizer json")
    args = parser.parse_args()
    
    analyze_checkpoint(args.checkpoint, args.tokenizer)
