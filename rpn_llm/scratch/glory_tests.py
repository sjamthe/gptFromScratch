import os
import sys
import torch
import torch.nn.functional as F

# Add the parent directory (rpn_llm) to the path so we can import model_rope
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

def generate_with_passes(model, tokenizer, prompt, num_passes, max_new_tokens=100, device='cpu'):
    model.eval()
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    
    generated = []
    past_kv = None
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -1:] if past_kv is not None else idx
            with torch.autocast(device, dtype=torch.bfloat16):
                # Using our NEW num_passes override!
                logits, _, past_kv = model(idx_cond, use_cache=True, past_key_values=past_kv, num_passes=num_passes)
            
            logits = logits[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            next_id = idx_next.item()
            if next_id == tokenizer.encode("\n")[0]:
                break
            
            generated.append(next_id)
            idx = torch.cat((idx, idx_next), dim=1)
            
    return tokenizer.decode(generated)

def run_glory_tests(ckpt_path):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

    print("\n" + "="*60)
    print("QUEST FOR GLORY: UNIVERSAL TRANSFORMER STRESS TESTS")
    print("="*60)

    # TEST 1: Depth Extrapolation (Recursive Quality)
    # We take a complex prompt and see if it improves with more passes.
    prompt_complex = "(999)(1)+?"
    print(f"\nTEST 1: Depth Sweep on '{prompt_complex}'")
    for p in [4, 8, 10, 12, 16]:
        output = generate_with_passes(model, tokenizer, prompt_complex, p, device=device)
        print(f"Passes {p:2d}: {output}")

    # TEST 2: Positional Jitter (Structural Invariance)
    # Adding random spaces inside the brackets to break 'fixed' index lookups.
    jitters = [
        "(123)(456)+?",
        "(  123)(456  )+?",
        "(1 2 3)(4 5 6)+?",
    ]
    print(f"\nTEST 2: Positional Jitter (Robustness to index-shifting)")
    for prompt in jitters:
        output = generate_with_passes(model, tokenizer, prompt, 8, device=device)
        print(f"Prompt '{prompt}': {output}")

    # TEST 3: Infinite Carry Check
    # This failed before at 8 passes. Does it solve at 16?
    prompt_carry = "(9)(9)+?<9+9+2="
    print(f"\nTEST 3: Carry-2 Depth Check on '{prompt_carry}'")
    for p in [8, 16, 24]:
        output = generate_with_passes(model, tokenizer, prompt_carry, p, device=device)
        print(f"Passes {p:2d}: {output}")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "rpn_llm/models/UT3M_1-22_tens_comp_bracketed_final.pt"
    run_glory_tests(ckpt)
