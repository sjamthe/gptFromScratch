import os
import sys
import torch

# Add parent directory to path so we can import model_rope and utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from model_rope import GPT, GPTConfig

def verify_coord_head():
    print("--- Running Coordinate Distance Head Verification ---")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Configure a small model with n_counter=1, n_coord=2
    config = GPTConfig(
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_embd=128,
        block_size=64,
        universal=True,
        n_counter=1,
        n_coord=2,
        n_coord_heads=4
    )
    
    model = GPT(config)
    model.to(device)
    
    # Set to eval for cache parity test
    model.eval()
    
    # 1. Generate a random input sequence
    B = 2
    T_total = 10
    idx = torch.randint(0, config.vocab_size, (B, T_total), dtype=torch.long, device=device)
    
    # Run full sequence without cache
    with torch.no_grad():
        logits_nocache, _ = model(idx, use_cache=False)
    
    # Run without cache step-by-step for each sequence prefix
    prefill_len = 5
    logits_nocache_list = []
    with torch.no_grad():
        logits_pre, _ = model(idx[:, :prefill_len], use_cache=False)
        logits_nocache_list.append(logits_pre)
        
    for t in range(prefill_len, T_total):
        with torch.no_grad():
            logits_step_nocache, _ = model(idx[:, :t+1], use_cache=False)
            logits_nocache_list.append(logits_step_nocache)
            
    # Run with cache step-by-step
    idx_prefill = idx[:, :prefill_len]
    with torch.no_grad():
        logits_prefill, _, past_kv = model(idx_prefill, use_cache=True)
    
    current_kv = past_kv
    logits_cache_list = [logits_prefill]
    
    for t in range(prefill_len, T_total):
        idx_t = idx[:, t:t+1]
        with torch.no_grad():
            logits_step, _, current_kv = model(idx_t, use_cache=True, past_key_values=current_kv)
        logits_cache_list.append(logits_step)
    
    # Compare predictions and check discrepancy
    pred_nocache_list = [l[:, -1, :].argmax(dim=-1, keepdim=True) for l in logits_nocache_list]
    pred_nocache = torch.cat(pred_nocache_list, dim=1)
    
    pred_cache_list = [l[:, -1, :].argmax(dim=-1, keepdim=True) for l in logits_cache_list]
    pred_cache = torch.cat(pred_cache_list, dim=1)
    
    print(f"pred_nocache shape: {pred_nocache.shape}")
    print(f"pred_cache shape: {pred_cache.shape}")
    
    # Check max absolute difference between logits at the final step
    max_logit_diff = torch.max(torch.abs(logits_nocache_list[-1] - logits_cache_list[-1])).item()
    print(f"  Max logit discrepancy at final token: {max_logit_diff:.6f}")
    
    diff = (pred_nocache != pred_cache).sum().item()
    print(f"Logits check at final token:")
    val_nocache = logits_nocache[:, -1, :]
    val_cache = logits_cache_list[-1][:, -1, :]
    max_logit_diff = torch.max(torch.abs(val_nocache - val_cache)).item()
    print(f"  Max logit discrepancy: {max_logit_diff:.6f}")
    
    if diff == 0 and max_logit_diff < 1e-4:
        print("✅ Cache Parity Test Passed! Output logits match exactly.")
    else:
        print(f"❌ Cache Parity Test Failed! Prediction mismatch count: {diff}")
        print(f"  No Cache Preds: {pred_nocache}")
        print(f"  Cache Preds:    {pred_cache}")
        
    # 2. Gradient flow check
    print("\n--- Running Gradient Flow Verification ---")
    model.train()
    
    # Create fake targets
    targets = torch.randint(0, config.vocab_size, (B, T_total), dtype=torch.long, device=device)
    logits, loss = model(idx, targets=targets)
    
    loss.backward()
    
    # Verify that coordinate heads parameters have non-zero gradients
    grad_ok = True
    for name, param in model.named_parameters():
        if "coordinate_heads" in name:
            if param.grad is None:
                print(f"❌ Parameter {name} has no gradient!")
                grad_ok = False
            elif torch.norm(param.grad) == 0:
                print(f"❌ Parameter {name} has zero gradient!")
                grad_ok = False
            else:
                print(f"✅ Parameter {name} grad norm: {torch.norm(param.grad).item():.8f} | max: {param.grad.abs().max().item():.8e} | mean: {param.grad.abs().mean().item():.8e}")
                
    if grad_ok:
        print("✅ Gradient Flow Test Passed! All coordinate head parameters received healthy gradients.")
    else:
        print("❌ Gradient Flow Test Failed!")

if __name__ == "__main__":
    verify_coord_head()
