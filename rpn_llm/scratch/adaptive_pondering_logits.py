import os
import sys
import torch

# Add the parent directory (rpn_llm) to the path so we can import model_rope
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

def analyze_logit_lockin(model, tokenizer, prompt, device='cpu'):
    model.eval()
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    
    print(f"\nAnalyzing Logit Lock-in for: '{prompt}'")
    print("-" * 50)
    print("Pass | Predicted Token | Stable?")
    print("-" * 50)
    
    prev_token = None
    stable_since = None
    
    with torch.no_grad():
        # Manual loop to see intermediate logits after each pass
        x = model.transformer.wte(idx)
        for i in range(model.config.n_layer):
            x = x + model.pass_emb[i].view(1, 1, -1)
            x, _, _ = model.transformer.h(x, model.freqs_cis, return_attention=False)
            
            # Project to logits
            last_x = model.transformer.ln_f(x[:, -1, :])
            logits = model.lm_head(last_x)
            token_id = torch.argmax(logits, dim=-1).item()
            token_str = tokenizer.decode([token_id]).strip()
            
            is_stable = "YES" if token_id == prev_token else "no"
            print(f" {i+1:2d}  | '{token_str}' ({token_id:4d}) | {is_stable}")
            
            if token_id == prev_token and stable_since is None:
                stable_since = i
            elif token_id != prev_token:
                stable_since = None
                
            prev_token = token_id

def generate_with_stability(model, tokenizer, prompt, stability_n, device='cpu'):
    model.eval()
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    
    generated = []
    past_kv = None
    
    with torch.no_grad():
        for _ in range(max_new_tokens:=50):
            idx_cond = idx[:, -1:] if past_kv is not None else idx
            with torch.autocast(device, dtype=torch.bfloat16):
                # Using our NEW halt_on_logit_stability!
                logits, _, past_kv = model(idx_cond, use_cache=True, past_key_values=past_kv, halt_on_logit_stability=stability_n)
            
            logits = logits[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            next_id = idx_next.item()
            if next_id == tokenizer.encode("\n")[0]:
                break
            
            generated.append(next_id)
            idx = torch.cat((idx, idx_next), dim=1)
            
    return tokenizer.decode(generated)

def run_global_sweep(model, tokenizer, device='cpu'):
    print("\n" + "="*60)
    print("GLOBAL PASS SWEEP: FINDING THE MATURITY POINT")
    print("="*60)
    
    prompts = [
        {"desc": "Easy (1+1)", "text": "(1)(1)+?"},
        {"desc": "Hard (999+999)", "text": "(999)(999)+?"},
        {"desc": "Subtraction (100-1)", "text": "(100)(1)-?"},
    ]
    
    for p_info in prompts:
        print(f"\nTesting {p_info['desc']}: '{p_info['text']}'")
        print("-" * 40)
        for p in [2, 3, 4, 5, 6, 7, 8]:
            output = generate_with_passes(model, tokenizer, p_info['text'], p, device=device)
            # Check if correct (Simple check for this specific test)
            status = "PASS" if (">2" in output or ">1998" in output) else "fail"
            print(f"Passes {p}: [{status}] -> {output}")

def generate_with_passes(model, tokenizer, prompt, num_passes, max_new_tokens=50, device='cpu'):
    model.eval()
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    
    generated = []
    past_kv = None
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -1:] if past_kv is not None else idx
            with torch.autocast(device, dtype=torch.bfloat16):
                # Pass num_passes to the model
                logits, _ = model(idx_cond, use_cache=False, num_passes=num_passes)
            
            logits = logits[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            next_id = idx_next.item()
            if next_id == tokenizer.encode("\n")[0]:
                break
            
            generated.append(next_id)
            idx = torch.cat((idx, idx_next), dim=1)
            
    return tokenizer.decode(generated)

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "rpn_llm/models/UT3M_1-22_tens_comp_bracketed_final.pt"
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

    run_global_sweep(model, tokenizer, device=device)
