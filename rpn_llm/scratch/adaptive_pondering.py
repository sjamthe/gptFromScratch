import os
import sys
import torch

# Add the parent directory (rpn_llm) to the path so we can import model_rope
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

def generate_with_halt(model, tokenizer, prompt, threshold, max_new_tokens=50, device='cpu'):
    model.eval()
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    
    generated = []
    past_kv = None
    pass_counts = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -1:] if past_kv is not None else idx
            with torch.autocast(device, dtype=torch.bfloat16):
                # Use halt_threshold to enable early exit
                logits, _, past_kv = model(idx_cond, use_cache=True, past_key_values=past_kv, halt_threshold=threshold)
            
            # Since our model returns all passes' KV caches, but pads them if stopped early,
            # we can count how many 'active' (non-padded) passes were run.
            # However, for simplicity, we'll just track the number of distinct attention weights if needed.
            # Actually, I'll just return the number of passes used for the LAST token.
            
            logits = logits[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            next_id = idx_next.item()
            if next_id == tokenizer.encode("\n")[0]:
                break
            
            generated.append(next_id)
            idx = torch.cat((idx, idx_next), dim=1)
            
    return tokenizer.decode(generated)

def measure_convergence(model, tokenizer, prompt, device='cpu'):
    model.eval()
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    
    print(f"\nAnalyzing Convergence for: '{prompt}'")
    print("-" * 40)
    print("Pass | L2 Delta (Distance from previous)")
    print("-" * 40)
    
    with torch.no_grad():
        # We'll run one forward pass and manually inspect the distances
        # We need a hook or a manual loop to see internal states.
        # I'll just use a small for loop here to simulate the forward passes.
        
        x = model.transformer.wte(idx)
        prev_x = None
        for i in range(model.config.n_layer):
            curr_x = x
            x = x + model.pass_emb[i].view(1, 1, -1)
            x, _, _ = model.transformer.h(x, model.freqs_cis, return_attention=False)
            
            if prev_x is not None:
                delta = torch.norm(x - prev_x, p=2, dim=-1).mean().item()
                print(f" {i+1:2d}  | {delta:.6f}")
            prev_x = x

def run_adaptive_pondering_tests(ckpt_path):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

    # 1. Easy Problem
    measure_convergence(model, tokenizer, "(1)(1)+?", device=device)
    
    # 2. Hard Problem
    measure_convergence(model, tokenizer, "(999)(999)+?", device=device)

    # 3. Demonstration of Early Exit
    threshold = 0.05 # Chosen based on previous observation
    print(f"\nDEMONSTRATION: Early Exit (Threshold={threshold})")
    print("-" * 40)
    
    # Test if it still gets the math right while pondering less
    print(f"Result (1+1): {generate_with_halt(model, tokenizer, '(1)(1)+?', threshold, device=device)}")
    print(f"Result (999+999): {generate_with_halt(model, tokenizer, '(999)(999)+?', threshold, device=device)}")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "rpn_llm/models/UT3M_1-22_tens_comp_bracketed_final.pt"
    run_adaptive_pondering_tests(ckpt)
