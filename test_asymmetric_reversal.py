import torch
import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm"))
from model_rope import GPT
from utils import RPNTokenizer

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

# Load the 344k model
model_path = "rpn_llm/models/ut0.4M_2l_6h_192e_mlp3_phaseMask_True_1-22_phase_lean_344000.pt"
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = GPT(checkpoint['config']).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

def test_asymmetric(n1_len, n2_len, num_trials=10):
    correct = 0
    for _ in range(num_trials):
        n1 = "".join([str(random.randint(1,9)) for _ in range(n1_len)])
        n2 = "".join([str(random.randint(1,9)) for _ in range(n2_len)])
        n1_rev = n1[::-1]
        n2_rev = n2[::-1]
        expected = f"{n1_rev} {n2_rev}+="
        
        prompt = f"[BOS]{n1} {n2}+?"
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(n1_len + n2_len + 15):
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
            else:
                if n1_len == 22 and n2_len == 4:
                    print(f"FAILED: EXP={expected} | GEN={rev_content}")
    
    return (correct / num_trials) * 100

print(f"Testing 22+4 Reversal (20 trials)...")
acc = test_asymmetric(22, 4, num_trials=20)
print(f"Accuracy: {acc}%")
