import torch
import sys
import os
import json
from collections import Counter

sys.path.append("rpn_llm")
from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

# Load model
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

# Use Gated 80k model as seen in user's snippet
model_path = "rpn_llm/models/ut0.4M_2l_6h_192e_mlp3_phaseMask_True_1-22_phase_lean_80000.pt"
print(f"Loading model from {model_path}")
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = GPT(checkpoint['config']).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

# Load benchmark data
with open("rpn_llm/analysis/fidelity_benchmark.json", "r") as f:
    benchmark = json.load(f)

examples = benchmark["short"] + benchmark["long"]
print(f"Loaded {len(examples)} examples.")

rev_id = tokenizer.encode("[REV]")[0]
math_id = tokenizer.encode("[MATH]")[0]

failure_positions = []

for full_str in examples:
    tokens = tokenizer.encode(full_str)
    
    # We want to predict y based on x
    x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits, _ = model(x, targets=y) # Pass targets to get all logits
        preds = torch.argmax(logits, dim=-1)
        
    y_cpu = y.cpu()
    preds_cpu = preds.cpu()
    
    # Find masks
    math_pos = (y_cpu == math_id).cumsum(dim=1)
    rev_start_pos = (y_cpu == rev_id).cumsum(dim=1)
    rev_mask = (rev_start_pos > 0) & (math_pos == 0)
    
    # Get indices where mask is True
    rev_indices = torch.where(rev_mask[0])[0]
    
    for idx in rev_indices:
        pos_in_rev = idx.item() - rev_indices[0].item() # Position relative to [REV]
        
        if preds_cpu[0, idx] != y_cpu[0, idx]:
            failure_positions.append(pos_in_rev)

# Count frequencies
dist = Counter(failure_positions)

print(f"--- FAILURE DISTRIBUTION BY POSITION IN REVERSAL ---")
print(f"Total failures recorded: {len(failure_positions)}")
for pos in sorted(dist.keys()):
    print(f"Position {pos}: {dist[pos]} failures")
