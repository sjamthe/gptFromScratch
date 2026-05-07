import torch
import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm"))
from model_rope import GPT
from utils import RPNTokenizer

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

# Load the GATED 200k model
model_path = "rpn_llm/models/ut0.5M_2l_6h_192e_mlp3_phaseMask_True_gated_1-22_phase_lean_200000.pt"
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = GPT(checkpoint['config']).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

def test_reversal(digits, num_trials=20):
    correct = 0
    for _ in range(num_trials):
        n1 = "".join([str(random.randint(1,9)) for _ in range(digits)])
        n2 = "9876"
        expected = f"{n1[::-1]} 6789+="
        
        prompt = f"[BOS]{n1} {n2}+?"
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(digits + 15):
                logits, _ = model(idx)
                next_tok = torch.argmax(logits[0, -1, :])
                idx = torch.cat([idx, next_tok.view(1, 1)], dim=1)
                if next_tok.item() == tokenizer.vocab.get("[MATH]", -1):
                    break
        
        generated = tokenizer.decode(idx[0].tolist())
        if "[REV]" in generated:
            rev_content = generated.split("[REV]")[1].split("[MATH]")[0].strip()
            if rev_content == expected:
                correct += 1
    
    return (correct / num_trials) * 100

print(f"Testing GATED 200k Model on the 'Unbreakable' 22-digit limit...")
acc_22 = test_reversal(22, num_trials=20)
print(f"Accuracy (22 digits): {acc_22}%")

print(f"\nTesting GATED 200k Model on 23-digit reversal (OOD Challenge)...")
acc_23 = test_reversal(23, num_trials=20)
print(f"Accuracy (23 digits): {acc_23}%")
