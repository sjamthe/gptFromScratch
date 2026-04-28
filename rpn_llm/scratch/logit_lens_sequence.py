import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn.functional as F
from model_rope import GPT
from utils import RPNTokenizer

def run_logit_lens(checkpoint_path, prompt):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

    tokens = tokenizer.encode(prompt)
    x_idx = torch.tensor([tokens], dtype=torch.long, device=device)
    B, T = x_idx.size()
    
    with torch.no_grad():
        x = model.transformer.wte(x_idx)
        freqs_cis = model.freqs_cis
        
        causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        is_bos = (x_idx == 2)
        seq_ids = is_bos.cumsum(dim=-1)
        doc_mask = (seq_ids.unsqueeze(1) == seq_ids.unsqueeze(2))
        attn_mask = (doc_mask & causal_mask).unsqueeze(1)

        def get_top_preds(hidden_state):
            # Apply LayerNorm
            normed = model.transformer.ln_f(hidden_state) # Shape: [1, T, n_embd]
            logits = normed @ model.lm_head.weight.T # Shape: [1, T, vocab_size]
            probs = F.softmax(logits, dim=-1)
            
            top_probs, top_indices = torch.max(probs, dim=-1)
            return top_indices[0], top_probs[0] # Return shape: [T], [T]

        # Get predictions at each layer
        l0_idx, l0_prob = get_top_preds(x)
        
        x, _, _ = model.transformer.h[0](x, freqs_cis, attn_mask=attn_mask)
        l1_idx, l1_prob = get_top_preds(x)
        
        x, _, _ = model.transformer.h[1](x, freqs_cis, attn_mask=attn_mask)
        l2_idx, l2_prob = get_top_preds(x)
        
        print(f"\nLogit Lens Sequence Analysis for Prompt: {repr(prompt)}\n")
        print(f"{'Pos':<4} | {'Input Token':<12} | {'Layer 0 (Raw)':<20} | {'Layer 1':<20} | {'Layer 2 (Final)':<20}")
        print("-" * 85)
        
        for i in range(T):
            in_tok = repr(tokenizer.decode([tokens[i]]))
            
            l0_tok = repr(tokenizer.decode([l0_idx[i].item()]))
            l0_str = f"{l0_tok} ({l0_prob[i]:.1%})"
            
            l1_tok = repr(tokenizer.decode([l1_idx[i].item()]))
            l1_str = f"{l1_tok} ({l1_prob[i]:.1%})"
            
            l2_tok = repr(tokenizer.decode([l2_idx[i].item()]))
            l2_str = f"{l2_tok} ({l2_prob[i]:.1%})"
            
            # The last token is special (it predicts the NEXT token)
            is_last = " <--- (Predicts next token)" if i == T-1 else ""
            
            print(f"{i:<4} | {in_tok:<12} | {l0_str:<20} | {l1_str:<20} | {l2_str:<20}{is_last}")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "rpn_llm/models/rope3.6M_1-22_uniform_BOS_64000.pt"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "[BOS]123 456+?<321 654 +=3+6+0=9:2+5+0="
    run_logit_lens(ckpt, prompt)
