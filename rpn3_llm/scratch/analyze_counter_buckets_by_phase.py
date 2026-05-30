import os
import sys
import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np

# Add parent directory to path to import model_rope and utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

def analyze_counter_buckets_by_phase(checkpoint_path, num_examples=100, tokenizer_path="rpn3_llm/rpn-tokenizer.json", dataset_path="rpn3_llm/data/sft_1-6_4num_BOS_val.txt"):
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = RPNTokenizer(tokenizer_path)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    config = checkpoint['config']
    print(f"Model config: universal={getattr(config, 'universal', False)}, n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}, n_counter={getattr(config, 'n_counter', 0)}, n_buckets={getattr(config, 'n_buckets', 4)}")
    
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    n_counter = getattr(config, 'n_counter', 0)
    n_buckets = getattr(config, 'n_buckets', 4)
    if n_counter == 0:
        print("Error: No counter heads in this model checkpoint.")
        return

    # Set up forward hooks to capture counter head query logits
    current_logits = {}
    def make_hook(name):
        def hook(module, input, output):
            current_logits[name] = output.detach()
        return hook

    hooks = []
    for h_idx in range(n_counter):
        head = model.counter_heads[h_idx]
        h = head.query.register_forward_hook(make_hook(f"head_{h_idx}"))
        hooks.append(h)

    # Read validation examples
    if not os.path.exists(dataset_path):
        # Try relative to repo root
        dataset_path = os.path.join(parent_dir, "data/sft_1-6_4num_BOS_val.txt")
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset not found at {dataset_path}")
            return
            
    print(f"Reading first {num_examples} examples from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = [f.readline().strip() for _ in range(num_examples)]
    
    # Filter empty lines
    lines = [l for l in lines if l]
    
    # We want to count occurrences of (token_id, bucket) for each head and phase tag
    # phase tags: "BASE", "REV", "MATH", "ANS"
    vocab_size = len(tokenizer.vocab)
    counts = defaultdict(lambda: np.zeros((2 * vocab_size, n_buckets), dtype=int))
    
    # Track phase summaries
    phase_counts = defaultdict(int)
    
    eq_id = tokenizer.vocab.get("=")
    digits = [str(d) for d in range(10)]
    digit_vocab_ids = {tokenizer.vocab.get(d) for d in digits if d in tokenizer.vocab}
    
    for line_idx, line in enumerate(lines):
        tokens = tokenizer.encode(line)
        if not tokens:
            continue
            
        idx = torch.tensor([tokens], dtype=torch.long)
        
        # Clear current logits cache
        current_logits.clear()
        
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
            
        # Accumulate assignments
        for h_idx in range(n_counter):
            logits_name = f"head_{h_idx}"
            if logits_name not in current_logits:
                continue
                
            # shape: [1, seq_len, n_buckets]
            logits = current_logits[logits_name][0] # [seq_len, n_buckets]
            buckets = logits.argmax(dim=-1).numpy() # [seq_len]
            
            for i in range(seq_len):
                t_id = tokens[i]
                bucket = buckets[i]
                tag = phases[i]
                
                # Check if it's a carry digit in MATH phase
                if tag == "MATH" and t_id in digit_vocab_ids:
                    if i + 1 < seq_len and tokens[i+1] == eq_id:
                        # Use virtual token ID to represent carry
                        recorded_id = t_id + vocab_size
                    else:
                        recorded_id = t_id
                else:
                    recorded_id = t_id
                    
                counts[(h_idx, tag)][recorded_id, bucket] += 1
                phase_counts[tag] += 1

    # Remove hooks
    for h in hooks:
        h.remove()
        
    print("\n--- Phase Distribution Statistics ---")
    for tag, c in phase_counts.items():
        print(f"  {tag}: {c} tokens analyzed")
        
    # Generate markdown output report
    print("\n# CounterHead Dynamic Phase-Shift Analysis Report")
    
    for h_idx in range(n_counter):
        scale = model.counter_heads[h_idx].scale.item()
        print(f"\n## Analysis of CounterHead {h_idx} (Scale: {scale:.4f})")
        
        if abs(scale) < 0.01:
            print(f"> [!NOTE]")
            print(f"> CounterHead {h_idx} has been classified as **INACTIVE** (scale magnitude {abs(scale):.4f} < 0.01).")
            print("> Its assignments are static/uniform across all phases. Detailed tables are omitted to reduce clutter.")
            continue
            
        phase_tags = ["BASE", "REV", "MATH", "ANS"]
        
        for tag in phase_tags:
            matrix = counts[(h_idx, tag)]
            total_tokens = matrix.sum()
            print(f"\n### Phase context: `{tag}` (Total tokens: {total_tokens})")
            
            if total_tokens == 0:
                print("No tokens observed in this phase.")
                continue
                
            print("| Bucket | Dominant Tokens (Count, bucket assignment %) |")
            print("|---|---|")
            
            for b in range(4):
                # Find all tokens assigned to this bucket
                bucket_counts = matrix[:, b]
                active_token_ids = np.where(bucket_counts > 0)[0]
                
                # Sort by count descending
                active_token_ids = active_token_ids[np.argsort(bucket_counts[active_token_ids])[::-1]]
                
                token_strs = []
                for t_id in active_token_ids:
                    cnt = bucket_counts[t_id]
                    # Compute total count for this token across all buckets in this phase
                    tok_sum = matrix[t_id].sum()
                    pct = (cnt / tok_sum) * 100 if tok_sum > 0 else 0
                    
                    # Get name
                    if t_id >= vocab_size:
                        base_id = t_id - vocab_size
                        base_str = tokenizer.inverse_vocab.get(base_id, f"[ID:{base_id}]")
                        k_str = f"{base_str} (carry)"
                    else:
                        k_str = tokenizer.inverse_vocab.get(t_id, f"[ID:{t_id}]")
                        
                    if k_str == ' ':
                        k_str = "' '"
                    elif k_str == '\n':
                        k_str = "'\\n'"
                        
                    token_strs.append(f"{k_str} ({cnt}, {pct:.1f}%)")
                    
                tokens_str = ", ".join(token_strs) if token_strs else "(empty)"
                print(f"| Bucket {b} | {tokens_str} |")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of validation examples to run")
    parser.add_argument("--tokenizer", type=str, default="rpn3_llm/rpn-tokenizer.json", help="Path to tokenizer json")
    parser.add_argument("--dataset", type=str, default="rpn3_llm/data/sft_1-6_4num_BOS_val.txt", help="Path to validation text dataset")
    
    args = parser.parse_args()
    
    analyze_counter_buckets_by_phase(args.checkpoint, args.num_examples, args.tokenizer, args.dataset)
