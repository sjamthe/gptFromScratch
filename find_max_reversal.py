import torch
import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm"))
from model_rope import GPT
from utils import RPNTokenizer

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

def generate_reversal_prompt(digits):
    a = "".join([str(random.randint(1, 9)) if i == 0 else str(random.randint(0, 9)) for i in range(digits)])
    b = "".join([str(random.randint(1, 9)) if i == 0 else str(random.randint(0, 9)) for i in range(digits)])
    prompt = f"[BOS]{a} {b}+?"
    expected = f"{a[::-1]} {b[::-1]}+="
    return prompt, expected

def test_length(model, digits, num_trials=20):
    correct = 0
    for _ in range(num_trials):
        prompt, expected = generate_reversal_prompt(digits)
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        
        with torch.no_grad():
            # Max tokens = digits*2 (for numbers) + 2 (spaces/delimiters) + safety
            for _ in range(digits * 2 + 10):
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

# Load the 344k model
model_path = "rpn_llm/models/ut0.4M_2l_6h_192e_mlp3_phaseMask_True_1-22_phase_lean_344000.pt"
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = GPT(checkpoint['config']).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

print(f"{'Digits':<10} | {'Accuracy':<10}")
print("-" * 25)

for d in range(22, 0, -1):
    acc = test_length(model, d, num_trials=20)
    print(f"{d:<10} | {acc:>9.1f}%")
    if acc >= 95.0:
        print(f"\nTarget achieved at {d} digits!")
        # Optional: confirm with more trials
        final_acc = test_length(model, d, num_trials=50)
        print(f"Confirmed Accuracy ({d} digits, 50 trials): {final_acc:.1f}%")
        break

