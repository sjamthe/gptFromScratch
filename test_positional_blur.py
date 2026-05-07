import torch
import sys
import os

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

def test_padded_prompt(padding_len):
    # Short 4-digit problem that the model usually NAILED (100%)
    a, b = "1234", "5678"
    expected = "4321 8765+="
    
    # Add dummy padding at the start
    padding = " " * padding_len
    prompt = f"[BOS]{padding}{a} {b}+?"
    
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(30):
            logits, _ = model(idx)
            next_tok = torch.argmax(logits[0, -1, :])
            idx = torch.cat([idx, next_tok.view(1, 1)], dim=1)
            if next_tok.item() == tokenizer.vocab.get("[MATH]", -1):
                break
    
    generated = tokenizer.decode(idx[0].tolist())
    rev_content = ""
    if "[REV]" in generated:
        rev_content = generated.split("[REV]")[1].split("[MATH]")[0].strip()
    
    return rev_content == expected, rev_content

print(f"{'Padding':<10} | {'Result':<10} | {'Generated'}")
print("-" * 50)
for p in [0, 10, 20, 30, 40, 50, 60]:
    success, gen = test_padded_prompt(p)
    status = "✅" if success else "❌"
    print(f"{p:<10} | {status:<10} | {gen}")

