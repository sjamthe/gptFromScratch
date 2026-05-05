import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm"))
from model_rope import GPT
from utils import RPNTokenizer

device = 'cpu'
model_path = "rpn_llm/models/ut0.4M_2l_6h_192e_mlp3_phaseMask_True_1-22_phase_lean_8000.pt"
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = GPT(checkpoint['config']).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
prompt = "[BOS]123 456+? [REV]321 654+="
idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

with torch.no_grad():
    logits, _ = model(idx, targets=idx)
    preds = torch.argmax(logits[0], dim=-1)

in_toks = [tokenizer.decode([t]) for t in idx[0].tolist()]
out_toks = [tokenizer.decode([t]) for t in preds.tolist()]

print("IN: ", [t if t != ' ' else '_' for t in in_toks])
print("OUT:", [t if t != ' ' else '_' for t in out_toks])
