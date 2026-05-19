import os
import sys
import torch

# Ensure working directory and rpn3_llm are in python path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "rpn3_llm"))

from utils import RPNTokenizer
from model_rope import GPT

def trace_attention():
    checkpoint_path = "rpn3_llm/models/ut1.8M_2l_8h_384e_mlp4_phaseMask_True_sft_1-14_7num_BOS_352000.pt"
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load Model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    config.universal = True # UT architecture
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    tokenizer = RPNTokenizer("rpn3_llm/rpn-tokenizer.json")

    # Failure #2 prompt up to the mismatch token
    prompt_str = "[BOS]57 313733-?[REV]75[SEP]3"
    tokens = tokenizer.encode(prompt_str)
    
    print("\n--- Token Breakdown ---")
    for idx, tok in enumerate(tokens):
        print(f"Index {idx:2d}: Token {tok:4d} -> '{tokenizer.decode([tok])}'")

    # Let's run a forward pass and extract attention weights!
    # Shape of inputs: [B, T]
    inputs = torch.tensor([tokens], dtype=torch.long, device=device)
    
    is_phase_shift = (inputs == 10) | (inputs == 11) | (inputs == 12)
    full_phase_ids = is_phase_shift.cumsum(dim=-1)

    with torch.no_grad():
        logits, _, all_weights = model(inputs, return_attention=True, full_phase_ids=full_phase_ids)

    print("\n--- Forward Pass Done ---")
    print(f"Logits shape: {logits.shape}")
    print(f"Number of layers/passes weights: {len(all_weights)}")

    # We want to analyze attention at Index 15 ([SEP]) and Index 16 ('3')
    target_indices = [15, 16]

    for q_idx in target_indices:
        q_tok = tokens[q_idx]
        q_str = tokenizer.decode([q_tok])
        print(f"\n=======================================================")
        print(f"ANALYZING QUERY TOKEN AT INDEX {q_idx}: '{q_str}'")
        print(f"=======================================================")

        # Predictions if we stopped at this token
        # Wait, logits from model is size [1, 1, 64] because it only returns the last token logits during target-free forward.
        # To get logits at ALL tokens, we must pass targets=inputs to the forward pass!
        # Let's do that!
        with torch.no_grad():
            full_logits, _, _ = model(inputs, targets=inputs, return_attention=True, full_phase_ids=full_phase_ids)
        
        q_logits = full_logits[0, q_idx, :]
        probs = torch.softmax(q_logits, dim=-1)
        top_k = 5
        top_probs, top_indices = torch.topk(probs, k=top_k)
        print(f"\nTop {top_k} Predictions after '{q_str}':")
        for rank in range(top_k):
            idx = top_indices[rank].item()
            p = top_probs[rank].item() * 100
            print(f"  Rank {rank+1}: '{tokenizer.decode([idx])}' (ID: {idx:4d}) -> {p:5.2f}%")

        # Now let's analyze the attention weights of this query token
        for pass_idx, aw in enumerate(all_weights):
            print(f"\n--- Pass {pass_idx} ---")
            if aw is not None:
                B, n_head, T_q, T_k = aw.shape
                # Get the attention weights from the query token at q_idx
                query_attn = aw[0, :, q_idx, :] # shape: [n_head, T_k]
                
                for h in range(n_head):
                    h_attn = query_attn[h].cpu().numpy()
                    # Find the indices in prompt that receive significant attention (>5%)
                    sig_indices = [i for i, val in enumerate(h_attn) if val > 0.03]
                    sig_str = ", ".join([f"'{tokenizer.decode([tokens[idx]])}' (idx {idx}): {h_attn[idx]*100:.1f}%" for idx in sig_indices])
                    print(f"  Head {h}: {sig_str}")

if __name__ == "__main__":
    trace_attention()
