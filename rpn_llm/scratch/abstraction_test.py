import os
import sys
import torch
import torch.nn.functional as F

# Add the parent directory (rpn_llm) to the path so we can import model_rope
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

def generate_output(model, tokenizer, prompt, max_new_tokens=100, device='cpu'):
    model.eval()
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    
    generated = []
    past_kv = None
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -1:] if past_kv is not None else idx
            with torch.autocast(device, dtype=torch.bfloat16):
                logits, _, past_kv = model(idx_cond, use_cache=True, past_key_values=past_kv)
            
            logits = logits[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            next_id = idx_next.item()
            if next_id == tokenizer.encode("\n")[0]:
                break
            
            generated.append(next_id)
            idx = torch.cat((idx, idx_next), dim=1)
            
    return tokenizer.decode(generated)

def run_stress_test(checkpoint_path):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Loading Universal model: {checkpoint_path} on {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
    
    tests = [
        # 1. Carry Generalization
        # Standard training only sees carry 0 or 1. Can it handle 2?
        ("(9)(9)+?<9+9+2=", "Should predict 20"),
        ("(4)(8)+?<4+8+2=", "Should predict 14"),
        
        # 2. Operand Order Stability
        # Subtraction uses [BORROW]. Does it stick to the script if we mix it?
        ("(123)(456)-?<(321)(654)+=", "Operator Flip Test (Math vs Scratchpad conflict)"),
        
        # 3. Intermediate Logic
        # It's used to ":". What if we give it two ":" steps?
        ("(1)(1)+?<1+1+0=2:1+1+0=", "Recursion Stability Test"),
        
        # 4. Long Numbers
        # 50 Digits (Training max was 22)
        ("(" + "9"*50 + ")(" + "1"*50 + ")+?", "50-Digit Marathon"),
    ]
    
    print("\n" + "="*50)
    print("STRESS TEST RESULTS")
    print("="*50)
    
    for prompt, desc in tests:
        print(f"\nTEST: {desc}")
        print(f"PROMPT: {prompt}")
        output = generate_output(model, tokenizer, prompt, device=device)
        print(f"OUTPUT: {output}")

if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "rpn_llm/models/UT3M_1-22_tens_comp_bracketed_final.pt"
    run_stress_test(ckpt)
