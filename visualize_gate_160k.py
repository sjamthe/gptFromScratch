import torch
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm"))
from model_rope import GPT, RPNTokenizer

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model_path = "rpn_llm/models/ut0.5M_2l_6h_192e_mlp3_phaseMask_True_gated_1-22_phase_lean_160000.pt"

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = GPT(checkpoint['config']).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

# A typical long problem
prompt = "[BOS]2318 2232+? [REV]8132 2322+=[MATH]"
idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

with torch.no_grad():
    # We'll just run one forward pass and let the model generate a bit
    # To capture gates, we'll need to hook them. 
    # But wait, visualize_gate_activations.py already does this!
    pass

# Instead of writing a new script, let's just modify the existing one to point to 160k
