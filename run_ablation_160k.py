import torch
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm"))
sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm/analysis"))
from pointer_fidelity_test import run_fidelity_test, RPNTokenizer, GPT

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model_path = "rpn_llm/models/ut0.5M_2l_6h_192e_mlp3_phaseMask_True_gated_1-22_phase_lean_160000.pt"

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = GPT(checkpoint['config']).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

with open("rpn_llm/analysis/fidelity_benchmark.json", "r") as f:
    benchmark = json.load(f)

# Run ablation on 6 heads
results = {}
baseline = run_fidelity_test(model, benchmark["long"][:50], device=device, verbose=False)
print(f"Baseline Fidelity (160k): {baseline:.1f}%")

for h in range(6):
    mask = torch.ones(6).to(device)
    mask[h] = 0.0
    score = run_fidelity_test(model, benchmark["long"][:50], device=device, verbose=False, head_mask=mask)
    print(f"Ablating Head {h}: {score:.1f}%")

