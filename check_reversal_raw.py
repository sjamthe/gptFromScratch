import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm"))
from model_rope import GPT
from utils import RPNTokenizer

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

step = 80000
model_path = f"rpn_llm/models/ut0.4M_2l_6h_192e_mlp3_phaseMask_True_1-22_phase_lean_{step}.pt"
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = GPT(checkpoint['config']).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

prompt = "[BOS]1234 5678+?"
idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
with torch.no_grad():
    for _ in range(40):
        logits, _ = model(idx)
        next_tok = torch.argmax(logits[0, -1, :])
        idx = torch.cat([idx, next_tok.view(1, 1)], dim=1)
        if next_tok.item() == tokenizer.vocab.get("[EOS]", -1):
            break

generated = tokenizer.decode(idx[0].tolist())
print(f"PROMPT   : {prompt}")
print(f"GENERATED: {generated}")
