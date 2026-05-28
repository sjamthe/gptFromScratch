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
        
    # Check if we have CounterHeads
    counter_keys = [k for k in state_dict.keys() if 'counter_heads' in k]
    if not counter_keys:
        print("Error: No counter_heads found in this checkpoint.")
        return
        
    # Find n_counter
    n_counters = max([int(k.split('.')[1]) for k in counter_keys]) + 1
    print(f"Found {n_counters} CounterHead(s) in checkpoint.")
    
    # Extract WTE weight
    wte_weight = state_dict['transformer.wte.weight'] # [vocab_size, n_embd]
    
    # Analyze each CounterHead
    for i in range(n_counters):
        print(f"\n--- CounterHead {i} Analysis ---")
        query_weight = state_dict[f'counter_heads.{i}.query.weight'] # [n_buckets, n_embd]
        query_bias = state_dict[f'counter_heads.{i}.query.bias']     # [n_buckets]
        scale = state_dict.get(f'counter_heads.{i}.scale', torch.tensor(1.0)).item()
        
        n_buckets, n_embd = query_weight.shape
        print(f"Number of buckets: {n_buckets}, embedding dimension: {n_embd}, Scale: {scale:.4f}")
        
        # Compute affinities: logits = wte_weight @ query_weight.T + query_bias
        logits = torch.matmul(wte_weight, query_weight.t()) + query_bias # [vocab_size, n_buckets]
        
        # Softmax over buckets
        probs = F.softmax(logits, dim=-1) # [vocab_size, n_buckets]
        
        # Build assignments: which tokens go to which buckets
        bucket_assignments = {b: [] for b in range(n_buckets)}
        for token_id in range(vocab_size):
            token_str = vocab[token_id]
            # Replace spaces and newlines with readable labels
            if token_str == ' ':
                token_label = "' '"
            elif token_str == '\n':
                token_label = "'\\n'"
            else:
                token_label = token_str
                
            prob_dist = probs[token_id].tolist()
            max_prob = max(prob_dist)
            assigned_bucket = prob_dist.index(max_prob)
            
            # Record assignments if probability is reasonably high (e.g. > 0.1)
            # or if it's the max probability
            bucket_assignments[assigned_bucket].append((token_label, max_prob))
            
        # Sort and print assignments for each bucket
        print("\n| Bucket | Dominant Tokens (with softmax assignment probabilities) |")
        print("|---|---|")
        for b in range(n_buckets):
            sorted_tokens = sorted(bucket_assignments[b], key=lambda x: x[1], reverse=True)
            token_strs = [f"{t} ({p:.1%})" for t, p in sorted_tokens[:15]]
            tokens_joined = ", ".join(token_strs)
            if len(sorted_tokens) > 15:
                tokens_joined += f", ... (+{len(sorted_tokens) - 15} more)"
            elif not sorted_tokens:
                tokens_joined = "(empty)"
            print(f"| Bucket {b} | {tokens_joined} |")
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--tokenizer", type=str, default="rpn3_llm/rpn-tokenizer.json", help="Path to tokenizer json")
    args = parser.parse_args()
    
    analyze_checkpoint(args.checkpoint, args.tokenizer)
