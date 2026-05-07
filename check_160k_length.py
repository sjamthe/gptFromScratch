import torch
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm"))
sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm/analysis"))
from pointer_fidelity_test import generate_until_math_done, RPNTokenizer, GPT

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model_path = "rpn_llm/models/ut0.5M_2l_6h_192e_mlp3_phaseMask_True_gated_1-22_phase_lean_160000.pt"

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = GPT(checkpoint['config']).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

with open("rpn_llm/analysis/fidelity_benchmark.json", "r") as f:
    benchmark = json.load(f)

prompt = benchmark["long"][0].split("[MATH]")[0] + "[MATH]"
idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

print(f"Generating for prompt: {prompt[-20:]}")
out_idx = generate_until_math_done(model, idx, tokenizer, max_new_tokens=500)
print(f"Total tokens generated: {out_idx.size(1) - idx.size(1)}")
print(f"Sample end of generation: {tokenizer.decode(out_idx[0, -20:].tolist())}")
