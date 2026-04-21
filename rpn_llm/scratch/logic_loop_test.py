import os
import sys
import torch
import torch.nn.functional as F

# Add the parent directory (rpn_llm) to the path so we can import model_rope
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

def generate_step(model, tokenizer, prompt, num_passes, device='cpu'):
    model.eval()
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        with torch.autocast(device, dtype=torch.bfloat16):
            # use_cache=False means model returns (logits, loss)
            logits, _ = model(idx, use_cache=False, num_passes=num_passes)
        
        logits = logits[:, -1, :] # Final token logits
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        return tokenizer.decode([idx_next.item()]).strip()

def run_logic_loop_comparison():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
    
    std_path = "rpn_llm/models/rope25M_1-22_tens_comp_bracketed_final.pt"
    uni_path = "rpn_llm/models/UT3M_1-22_tens_comp_bracketed_final.pt"
    
    print("\n" + "="*60)
    print("LOGIC LOOP CHALLENGE: MENTAL CARRY PROPAGATION")
    print("="*60)

    results = []
    
    # We will sweep from 1 digit (9+1) to 15 digits (999...9 + 1)
    # The carry has to ripple to the far left to produce a '1'.
    for length in range(1, 16):
        n1 = "9" * length
        n2 = "1"
        prompt = f"({n1})({n2})+? >"
        expected = "1" # The result of 99...9 + 1 always starts with 1
        
        # 1. Test Standard 25M (8 Layers)
        checkpoint_std = torch.load(std_path, map_location=device, weights_only=False)
        model_std = GPT(checkpoint_std['config'])
        model_std.load_state_dict(checkpoint_std['model'])
        model_std.to(device)
        std_out = generate_step(model_std, tokenizer, prompt, None, device=device)
        del model_std
        
        # 2. Test Universal 3M (8, 16, 32 Passes)
        checkpoint_uni = torch.load(uni_path, map_location=device, weights_only=False)
        model_uni = GPT(checkpoint_uni['config'])
        model_uni.load_state_dict(checkpoint_uni['model'])
        model_uni.to(device)
        
        uni_8 = generate_step(model_uni, tokenizer, prompt, 8, device=device)
        uni_16 = generate_step(model_uni, tokenizer, prompt, 16, device=device)
        uni_32 = generate_step(model_uni, tokenizer, prompt, 32, device=device)
        del model_uni
        
        print(f"Digits {length:2d} | Std: {std_out} | Uni8: {uni_8} | Uni16: {uni_16} | Uni32: {uni_32}")
        results.append((length, std_out, uni_8, uni_16, uni_32))

    # Summary table
    print("\n" + "="*60)
    print("FINAL COMPARISON TABLE")
    print("="*60)
    print("| Digits | Standard | Uni (8) | Uni (16) | Uni (32) |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        print(f"| {r[0]:2d} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")

if __name__ == "__main__":
    run_logic_loop_comparison()
