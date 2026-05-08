import torch
import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model_rope import GPT
from utils import RPNTokenizer

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
tokenizer = RPNTokenizer("rpn-tokenizer.json")
model_path = "models/ut0.5M_2l_6h_192e_mlp4_phaseMask_True_1-22_rev_only_40000.pt"

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = GPT(checkpoint['config']).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

correct = 0
num_trials = 100
for i in range(num_trials):
    n1 = "".join([str(random.randint(1,9)) for _ in range(18)])
    n2 = "".join([str(random.randint(1,9)) for _ in range(14)])
    expected = f"{n1[::-1]} {n2[::-1]}+="
    prompt = f"[BOS]{n1} {n2}+?"
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(22 + 4 + 15):
            logits, _ = model(idx)
            next_tok = torch.argmax(logits[0, -1, :])
            idx = torch.cat([idx, next_tok.view(1, 1)], dim=1)
            if next_tok.item() == tokenizer.vocab.get("[MATH]", -1) or next_tok.item() == tokenizer.vocab.get("[EOS]", -1):
                break
    generated = tokenizer.decode(idx[0].tolist())
    if "[REV]" in generated:
        rev_content = generated.split("[REV]")[1].split("[MATH]")[0].split("[EOS]")[0].strip()
        if rev_content == expected:
            correct += 1
        else:
            print(f"Trial {i+1} FAILED!")
            print(f"  Prompt : {n1} {n2}+?")
            print(f"  Target : {expected}")
            print(f"  Output : {rev_content}")
            
            # Find exact point of failure
            exp_parts = expected.replace("+=","").split()
            out_parts = rev_content.replace("+=","").split()
            if len(out_parts) > 0 and exp_parts[0] != out_parts[0]:
                for j in range(min(len(exp_parts[0]), len(out_parts[0]))):
                    if exp_parts[0][j] != out_parts[0][j]:
                        print(f"    Mismatch in N1 at pos {j}: expected {exp_parts[0][j]}, got {out_parts[0][j]}")
                        break
    else:
        print("Status: ❌ (No [REV] token)")

print(f"Result: {(correct / num_trials) * 100}%")
