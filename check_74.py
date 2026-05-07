import torch
import sys
import os
import json
import re

sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm"))
sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm/analysis"))
from pointer_fidelity_test import generate_until_math_done, RPNTokenizer, GPT

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model_path = "rpn_llm/models/ut0.5M_2l_6h_192e_mlp3_phaseMask_True_gated_1-22_phase_lean_80000.pt"

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = GPT(checkpoint['config']).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

with open("rpn_llm/analysis/fidelity_benchmark.json", "r") as f:
    benchmark = json.load(f)

for sample in benchmark["long"][:5]:
    prompt = sample.split("[MATH]")[0] + "[MATH]"
    
    # Generate scratchpad
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    out_idx = generate_until_math_done(model, idx, tokenizer, max_new_tokens=200)
    generated_str = tokenizer.decode(out_idx[0].tolist())
    
    # Hack it
    try:
        math_part = generated_str.split("[MATH]")[1].split("[ANS]")[0]
    except IndexError:
        continue
        
    hacked_chunks = []
    hacked_digits = []
    for chunk in math_part.split(':'):
        if '=' in chunk:
            left, right = chunk.split('=')
            new_val = '9' if '0' not in right else '1'
            hacked_chunks.append(f"{left}={new_val}")
            if right.strip():
                hacked_digits.append(new_val)
        else:
            hacked_chunks.append(chunk)
            
    hacked_scratchpad = prompt + ":".join(hacked_chunks) + ":"
    
    # Resume
    idx_hack = torch.tensor(tokenizer.encode(hacked_scratchpad), dtype=torch.long).unsqueeze(0).to(device)
    ans_token = tokenizer.vocab.get("[ANS]", -1)
    
    final_out = idx_hack
    for _ in range(50):
        logits, _ = model(final_out)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        final_out = torch.cat((final_out, next_token), dim=1)
        if next_token.item() == ans_token:
            break
            
    completion = tokenizer.decode(final_out[0].tolist())
    ans_indices = (final_out[0] == ans_token).nonzero()
    
    if len(ans_indices) > 0:
        ans_pos = ans_indices[0].item()
        completion_up_to_ans = tokenizer.decode(final_out[0, :ans_pos].tolist())
        harvested_part = completion_up_to_ans.rsplit(":", 1)[1].strip() if ":" in completion_up_to_ans else ""
    else:
        harvested_part = ""
        
    print("\n---")
    print("Hacked Prompt ending in... ", hacked_scratchpad.split("[MATH]")[1][-20:])
    print("Hacked digits inserted  : ", "".join(hacked_digits))
    print("Model Harvested string  : ", harvested_part)
