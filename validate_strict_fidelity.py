import torch
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm"))
sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm/analysis"))
from pointer_fidelity_test import run_fidelity_test, RPNTokenizer, GPT

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

with open("rpn_llm/analysis/fidelity_benchmark.json", "r") as f:
    benchmark = json.load(f)

checkpoints = [
    "rpn_llm/models/ut0.4M_2l_6h_192e_mlp3_phaseMask_True_1-22_phase_lean_344000.pt",
    "rpn_llm/models/ut0.4M_2l_6h_192e_mlp3_phaseMask_True_1-22_phase_lean_160000.pt"
]

for cp_path in checkpoints:
    print(f"\n" + "="*60)
    print(f"CHECKPOINT: {cp_path}")
    print("="*60)
    
    checkpoint = torch.load(cp_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print("\n[SHORT BENCHMARK - 4 Digits]")
    run_fidelity_test(model, benchmark["short"][:50], device=device, verbose=True)
    
    print("\n[LONG BENCHMARK - 10 Digits]")
    run_fidelity_test(model, benchmark["long"][:50], device=device, verbose=True)

