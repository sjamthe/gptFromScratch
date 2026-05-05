import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm"))
from model_rope import GPT
from utils import RPNTokenizer

device = 'cpu'
tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
prompt = "[BOS]123 456+?[REV]321 654+=[MATH]3+6+0=9"
idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

def get_preds(step):
    model_path = f"rpn_llm/models/ut0.4M_2l_6h_192e_mlp3_phaseMask_True_1-22_phase_lean_{step}.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    with torch.no_grad():
        logits, _ = model(idx, targets=idx)
        preds = torch.argmax(logits[0], dim=-1)
    
    return [tokenizer.decode([t]) for t in preds.tolist()]

preds_8k = get_preds(8000)
preds_344k = get_preds(344000)

math_idx = tokenizer.encode("[MATH]")[0]
try:
    start_idx = idx[0].tolist().index(math_idx)
except ValueError:
    start_idx = 0

in_toks = [tokenizer.decode([t]) for t in idx[0].tolist()][start_idx:]
out_8k = preds_8k[start_idx:]
out_344k = preds_344k[start_idx:]

print("IN:   ", [t if t != ' ' else '_' for t in in_toks])
print("344k: ", [t if t != ' ' else '_' for t in out_344k])
print("8k:   ", [t if t != ' ' else '_' for t in out_8k])

