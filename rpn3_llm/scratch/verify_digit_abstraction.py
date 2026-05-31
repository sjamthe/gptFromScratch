import os
import sys
import torch
import numpy as np

# Add rpn3_llm to sys.path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "rpn3_llm"))

from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

def main():
    print("=== STARTING DIGIT-IDENTITY ABSTRACTION VERIFICATION ===")
    
    # 1. Initialize GPTConfig and Model
    tokenizer = RPNTokenizer("rpn3_llm/rpn-tokenizer.json")
    
    config = GPTConfig(
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_embd=64,
        block_size=128,
        universal=True,
        use_phase_mask=True,
        n_counter=1,
        n_coord=1,
        use_digit_abstraction=True,
        freeze_coord_scale=True,
        phase_token_ids=[10, 11, 12]
    )
    
    model = GPT(config)
    model.eval()
    
    # Define two sequences of identical structure but different digit values
    # Structure: [BOS] <digits> [REV] <reversed_digits>
    seq_A_str = "[BOS]123[REV]32"
    seq_B_str = "[BOS]456[REV]65"
    
    tokens_A = tokenizer.encode(seq_A_str)
    tokens_B = tokenizer.encode(seq_B_str)
    
    print(f"Sequence A: {tokens_A} -> {seq_A_str}")
    print(f"Sequence B: {tokens_B} -> {seq_B_str}")
    
    inputs_A = torch.tensor([tokens_A], dtype=torch.long)
    inputs_B = torch.tensor([tokens_B], dtype=torch.long)
    
    # Run forward pass capturing attention weights
    with torch.no_grad():
        logits_A, _, weights_A = model(inputs_A, return_attention=True)
        logits_B, _, weights_B = model(inputs_B, return_attention=True)
        
    print("\n--- Testing Attention Score Identity ---")
    # For both layer/pass attention weights:
    # weights is a list of length n_layer. Each element is [B, n_head, T, T].
    for pass_idx in range(len(weights_A)):
        w_A = weights_A[pass_idx][0] # [n_head, T, T]
        w_B = weights_B[pass_idx][0] # [n_head, T, T]
        
        # We check the query tokens in the REV phase.
        # Tokens are: [BOS], digit, digit, digit, [REV], digit, digit
        # Indices:
        # 0: [BOS]
        # 1, 2, 3: digits
        # 4: [REV] (phase token: changes phase to 1)
        # 5, 6: digits in phase 1 (reversal phase)
        # So queries at index 5 and 6 are in the REV phase.
        rev_query_indices = [5, 6]
        
        for idx in rev_query_indices:
            diff = torch.abs(w_A[:, idx, :] - w_B[:, idx, :]).max().item()
            print(f"Pass {pass_idx}, Query index {idx} ('{tokenizer.decode([tokens_A[idx]])}' vs '{tokenizer.decode([tokens_B[idx]])}'):")
            print(f"  Max absolute attention weight difference: {diff:.6e}")
            if diff < 1e-6:
                print("  => SUCCESS: Attention weights are identical!")
            else:
                print("  => FAILURE: Attention weights differ!")
                sys.exit(1)
                
    # 2. Verify KV cache parity
    print("\n--- Testing KV Cache Parity ---")
    model.eval()
    
    # Generate autoregressively step-by-step
    generated_no_cache = []
    generated_with_cache = []
    
    # Prompt: [BOS]123[REV]
    prompt_str = "[BOS]123[REV]"
    prompt_tokens = tokenizer.encode(prompt_str)
    
    # No Cache run
    idx_no_cache = torch.tensor([prompt_tokens], dtype=torch.long)
    for _ in range(5):
        # We need to compute phase shift
        is_phase_shift = (idx_no_cache == 10) | (idx_no_cache == 11) | (idx_no_cache == 12)
        full_phase_ids = is_phase_shift.cumsum(dim=-1)
        with torch.no_grad():
            logits, _ = model(idx_no_cache, full_phase_ids=full_phase_ids)
        next_tok = torch.argmax(logits[0, -1, :]).item()
        generated_no_cache.append(next_tok)
        idx_no_cache = torch.cat([idx_no_cache, torch.tensor([[next_tok]])], dim=1)
        
    # With Cache run
    idx_cache = torch.tensor([prompt_tokens], dtype=torch.long)
    past_kv = None
    for i in range(5):
        # We need full_phase_ids representing the full generated sequence
        is_phase_shift = (idx_cache == 10) | (idx_cache == 11) | (idx_cache == 12)
        full_phase_ids = is_phase_shift.cumsum(dim=-1)
        # For cache, we pass only the last token if cache is present
        cond_idx = idx_cache[:, -1:] if past_kv is not None else idx_cache
        with torch.no_grad():
            logits, _, past_kv = model(cond_idx, use_cache=True, past_key_values=past_kv, full_phase_ids=full_phase_ids)
        next_tok = torch.argmax(logits[0, -1, :]).item()
        generated_with_cache.append(next_tok)
        idx_cache = torch.cat([idx_cache, torch.tensor([[next_tok]])], dim=1)
        
    print(f"Generated (No Cache):   {generated_no_cache} -> '{tokenizer.decode(generated_no_cache)}'")
    print(f"Generated (With Cache): {generated_with_cache} -> '{tokenizer.decode(generated_with_cache)}'")
    
    if generated_no_cache == generated_with_cache:
        print("=> SUCCESS: KV Cache matches No-Cache generation exactly!")
    else:
        print("=> FAILURE: KV Cache output differs from No-Cache!")
        sys.exit(1)
        
    # 3. Verify Gradient Flow
    print("\n--- Testing Gradient Flow ---")
    model.train()
    
    inputs = torch.tensor([tokens_A], dtype=torch.long)
    targets = torch.tensor([[ -100 if t == 2 else t for t in tokens_A ]], dtype=torch.long)
    
    logits, loss = model(inputs, targets=targets)
    loss.backward()
    
    # Check that gradients flow to embedding, c_attn, c_proj, and coordinate heads
    grad_ok = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"  WARNING: Parameter {name} has NO gradient!")
                grad_ok = False
            else:
                grad_norm = param.grad.norm().item()
                print(f"  Parameter {name}: grad norm = {grad_norm:.6f}")
                if grad_norm == 0.0 and "scale" not in name:
                    print(f"  WARNING: Parameter {name} gradient is ZERO!")
                    grad_ok = False
                    
    # Check that scale does not have gradient if freeze_coord_scale=True
    if model.coordinate_heads[0].scale.grad is not None:
        print("  WARNING: CoordinateHead scale has a gradient but it was supposed to be frozen!")
        grad_ok = False
    else:
        print("  SUCCESS: CoordinateHead scale gradient is None (frozen).")
        
    if grad_ok:
        print("=> SUCCESS: Gradient flow is healthy!")
    else:
        print("=> FAILURE: Gradient health checks failed!")
        sys.exit(1)

    print("\n=== ALL VERIFICATION TESTS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
