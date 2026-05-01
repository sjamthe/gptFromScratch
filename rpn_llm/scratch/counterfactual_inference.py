import torch
import torch.nn.functional as F
import os
import sys

# Add parent dir to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model_rope import GPT, GPTConfig

def hack_inference(model_path, device='cpu'):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    from utils import RPNTokenizer
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
    
    test_cases = [
        ("Original (2,8,7,8)", "2:6+1+1=8:7+0+0=7:8+0+0=8:"),
        ("Sequential (1,2,3,4)", "2:6+1+1=1:7+0+0=2:8+0+0=3:"),
        ("Descending (9,8,7,6)", "2:6+1+1=9:7+0+0=8:8+0+0=7:"),
        ("Repeated (5,5,5,5)", "5:6+1+1=5:7+0+0=5:8+0+0=5:"),
        ("Swapped Orig (2,7,8,8)", "2:6+1+1=7:7+0+0=8:8+0+0=8:"),
        ("Large Gap (0,9,0,9)", "0:6+1+1=9:7+0+0=0:8+0+0=9:"),
    ]
    
    def generate(model, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = model(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
            if next_token.item() == tokenizer.vocab.get("[EOS]", -1):
                break
        return idx[0]

    prefix = "[BOS]8766 16+? [REV]6678 61+=[MATH]6+6+0="
    
    print(f"{'Test Case':<25} | {'Scratchpad Digits':<20} | {'Model Completion'}")
    print("-" * 80)
    
    with torch.no_grad():
        for label, scratch_suffix in test_cases:
            full_prompt = prefix + scratch_suffix
            idx = torch.tensor(tokenizer.encode(full_prompt), dtype=torch.long).unsqueeze(0).to(device)
            
            # Extract the actual digits being injected for display
            # suffix looks like "2:6+1+1=8:7+0+0=7:8+0+0=8:"
            # We want the digits after the '=' signs
            import re
            injected_digits = re.findall(r"=(\d)", full_prompt)
            # Add the first one which is hardcoded in the suffix start
            if ":" in scratch_suffix:
                injected_digits = [scratch_suffix[0]] + injected_digits
            
            out = generate(model, idx, max_new_tokens=15)
            completion = tokenizer.decode(out.tolist())[len(full_prompt):]
            
            print(f"{label:<25} | {str(injected_digits):<20} | {completion}")

if __name__ == "__main__":
    model_path = "rpn_llm/models/ut1.8M_phaseMask_True_1-22_phase_lean_80000.pt"
    if os.path.exists(model_path):
        hack_inference(model_path)
    else:
        print(f"Model not found: {model_path}")
