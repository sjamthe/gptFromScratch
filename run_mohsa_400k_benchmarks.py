import torch
import sys
import os
import random
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm"))
sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm/analysis"))
from model_rope import GPT, GPTConfig
from utils import RPNTokenizer
from pointer_fidelity_test import run_fidelity_test

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
model_path = "rpn_llm/models/ut0.5M_2l_6h_192e_mlp3_phaseMask_True_gated_1-22_phase_lean_80000.pt"

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = GPT(checkpoint['config']).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

def test_reversal(n1_len, n2_len, num_trials=20):
    correct = 0
    for _ in range(num_trials):
        n1 = "".join([str(random.randint(1,9)) for _ in range(n1_len)])
        n2 = "".join([str(random.randint(1,9)) for _ in range(n2_len)])
        expected = f"{n1[::-1]} {n2[::-1]}+="
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
    return (correct / num_trials) * 100

print("--- BOUNDARY TEST A: 22-Digit Reversal ---")
acc_22 = test_reversal(22, 4)
print(f"Result: {acc_22}%")

print("\n--- BOUNDARY TEST B: 25-Digit Strict Fidelity ---")
with open("rpn_llm/analysis/fidelity_benchmark.json", "r") as f:
    benchmark = json.load(f)
run_fidelity_test(model, benchmark["long"][:50], device=device, verbose=True)

print("\n--- BOUNDARY TEST C: Positional Shift-Invariance (10-token padding) ---")
padding = " " * 10
prompt = f"[BOS]{padding}1234 5678+?"
idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
with torch.no_grad():
    for _ in range(30):
        logits, _ = model(idx)
        next_tok = torch.argmax(logits[0, -1, :])
        idx = torch.cat([idx, next_tok.view(1, 1)], dim=1)
        if next_tok.item() == tokenizer.vocab.get("[MATH]", -1):
            break
generated = tokenizer.decode(idx[0].tolist())
if "[REV]" in generated:
    rev_content = generated.split("[REV]")[1].split("[MATH]")[0].strip()
    status = "✅" if rev_content == "4321 8765+=" else "❌"
    print(f"Status: {status} (Generated: {rev_content})")
else:
    print("Status: ❌ (No [REV] token)")

print("\n--- BOUNDARY TEST D: Asymmetric Capacity (N1=21, N2=4) ---")
acc_21_4 = test_reversal(21, 4)
print(f"Result: {acc_21_4}%")

print("\n--- EXTRA: Short Fidelity (4-digit math) ---")
run_fidelity_test(model, benchmark["short"][:50], device=device, verbose=True)

