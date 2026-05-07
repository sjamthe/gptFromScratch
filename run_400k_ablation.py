import sys
import os
import torch
import json
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm"))
sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm/analysis"))
from model_rope import GPT
from utils import RPNTokenizer
from pointer_fidelity_test import run_fidelity_test

device = 'cpu' # Use CPU for faster small batches
model_path = "rpn_llm/models/ut0.5M_2l_6h_192e_mlp3_phaseMask_True_gated_1-22_phase_lean_400000.pt"

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = GPT(checkpoint['config']).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

with open("rpn_llm/analysis/fidelity_benchmark.json", "r") as f:
    benchmark = json.load(f)

samples = benchmark["short"][:25] # 25 samples is enough for ablation
baseline = run_fidelity_test(model, samples, device=device, verbose=False)
print(f"Baseline: {baseline:.1f}%")

head_drops = []
for h in range(6):
    mask = torch.ones(2, 6, device=device)
    mask[:, h] = 0.0
    score = run_fidelity_test(model, samples, device=device, verbose=False, head_mask=mask)
    drop = baseline - score
    print(f"Head {h}: {score:.1f}% (Drop: {drop:.1f}%)")
    head_drops.append(drop)

