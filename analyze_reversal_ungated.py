import torch
import sys
import os
import re

sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm"))
from model_rope import GPT
from utils import RPNTokenizer

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

def test_reversal(step, use_gated=False):
    if use_gated:
        model_path = f"rpn_llm/models/ut0.5M_2l_6h_192e_mlp3_phaseMask_True_gated_1-22_phase_lean_{step}.pt"
    else:
        model_path = f"rpn_llm/models/ut0.4M_2l_6h_192e_mlp3_phaseMask_True_1-22_phase_lean_{step}.pt"
        
    if not os.path.exists(model_path):
        return None
        
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Test cases: (prompt, expected_rev_digits)
    # 22-digit case (In-distribution boundary)
    n1 = "1234567890123456789012" # 22 digits
    n2 = "9876543210987654321098" # 22 digits
    n1_rev = n1[::-1]
    n2_rev = n2[::-1]
    
    test_cases = [
        ("[BOS]1234 5678+?", "4321 8765+="),
        (f"[BOS]{n1} {n2}+?", f"{n1_rev} {n2_rev}+=")
    ]
    
    results = []
    for prompt, expected_digits in test_cases:
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(70):
                logits, _ = model(idx)
                next_tok = torch.argmax(logits[0, -1, :])
                idx = torch.cat([idx, next_tok.view(1, 1)], dim=1)
                if next_tok.item() == tokenizer.vocab.get("[MATH]", -1):
                    break
        
        generated = tokenizer.decode(idx[0].tolist())
        if "[REV]" in generated:
            rev_content = generated.split("[REV]")[1].split("[MATH]")[0].strip()
            results.append(rev_content == expected_digits)
        else:
            results.append(False)
            
    return results

steps_ungated = [8000, 32000, 80000, 160000, 344000]
steps_gated = [8000, 80000, 160000]

print("Analyzing Reversal Mastery (Short=4, Long=22 digits)")
print("-" * 50)
print("UNGATED MODELS:")
print(f"{'Step':<8} | {'Short':<10} | {'Long (22)':<10}")
for s in steps_ungated:
    res = test_reversal(s, use_gated=False)
    if res:
        print(f"{s:<8} | {'✅' if res[0] else '❌':<10} | {'✅' if res[1] else '❌':<10}")

print("\nGATED MODELS:")
print(f"{'Step':<8} | {'Short':<10} | {'Long (22)':<10}")
for s in steps_gated:
    res = test_reversal(s, use_gated=True)
    if res:
        print(f"{s:<8} | {'✅' if res[0] else '❌':<10} | {'✅' if res[1] else '❌':<10}")
