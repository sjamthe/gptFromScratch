import torch
import torch.nn.functional as F
from model_rope import GPT
from utils import RPNTokenizer
import sys

def analyze_attributions(checkpoint_path, prompt, expected_full):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

    tokens = tokenizer.encode(prompt)
    expected_tokens = tokenizer.encode(expected_full)
    
    reversal_deltas = [0.0, 0.0] # [Layer1, Layer2]
    math_deltas = [0.0, 0.0]
    rev_count = 0
    math_count = 0
    
    print(f"\nAnalyzing Prompt: {prompt}")
    print(f"{'Step':<5} | {'Target':<6} | {'Phase':<8} | {'Init Score':<10} | {'L1 Work %':<9} | {'L2 Work %':<9} | {'Final Score':<10}")
    print("-" * 75)

    for step in range(len(expected_tokens)):
        target_token = expected_tokens[step]
        target_char = tokenizer.decode([target_token])
        
        # Teacher forcing: feed the correct tokens up to this point
        current_tokens = tokens + expected_tokens[:step]
        
        x_idx = torch.tensor([current_tokens], dtype=torch.long, device=device)
        B, T = x_idx.size()
        
        with torch.no_grad():
            x = model.transformer.wte(x_idx)
            freqs_cis = model.freqs_cis
            
            # Masks
            causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
            is_bos = (x_idx == 2)
            seq_ids = is_bos.cumsum(dim=-1)
            doc_mask = (seq_ids.unsqueeze(1) == seq_ids.unsqueeze(2))
            attn_mask = (doc_mask & causal_mask).unsqueeze(1)

            def get_logit(hidden_state):
                # Apply final LayerNorm and Unembed for the LAST token
                normed = model.transformer.ln_f(hidden_state[:, -1, :])
                return (normed @ model.lm_head.weight[target_token]).item()

            # Score 0: Raw embedding
            score_0 = get_logit(x)
            
            # Score 1: After Layer 1
            x, _, _ = model.transformer.h[0](x, freqs_cis, attn_mask=attn_mask)
            score_1 = get_logit(x)
            
            # Score 2: After Layer 2
            x, _, _ = model.transformer.h[1](x, freqs_cis, attn_mask=attn_mask)
            score_2 = get_logit(x)
            
            delta_1 = score_1 - score_0
            delta_2 = score_2 - score_1
            total_delta = score_2 - score_0
            
            # Calculate percentages
            if abs(total_delta) > 1e-6:
                pct_1 = (delta_1 / total_delta) * 100
                pct_2 = (delta_2 / total_delta) * 100
            else:
                pct_1, pct_2 = 0.0, 0.0
            
            # Identify phase
            generated_so_far = tokenizer.decode(expected_tokens[1:step+1])
            phase = "Math" if '+' in generated_so_far else "Reversal"
            
            if phase == "Reversal":
                reversal_deltas[0] += pct_1
                reversal_deltas[1] += pct_2
                rev_count += 1
            else:
                math_deltas[0] += pct_1
                math_deltas[1] += pct_2
                math_count += 1
                
            print(f"{step+1:<5} | '{target_char}'    | {phase:<8} | {score_0:<10.2f} | {pct_1:>8.1f}% | {pct_2:>8.1f}% | {score_2:<10.2f}")

    print("\n--- Summary: Average Work Percentage per Phase ---")
    if rev_count > 0:
        print(f"Reversal Phase : Layer 1 does {reversal_deltas[0]/rev_count:>6.1f}%, Layer 2 does {reversal_deltas[1]/rev_count:>6.1f}%")
    if math_count > 0:
        print(f"Math Phase     : Layer 1 does {math_deltas[0]/math_count:>6.1f}%, Layer 2 does {math_deltas[1]/math_count:>6.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Using default parameters")
        print("Usage: python analyze_layer_attributions.py <checkpoint_path> <optional_prompt> <optional_expected>")
    
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "rpn_llm/models/rope3.6M_1-22_uniform_BOS_64000.pt"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "[BOS]123 456+?"
    expected = sys.argv[3] if len(sys.argv) > 3 else "<321 654 +=3+6+0=9:2+5+0=7:1+4+0=5:975>579"
    
    analyze_attributions(ckpt, prompt, expected)
