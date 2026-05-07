import torch
import sys
import os
import json
import re

sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm"))
sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm/analysis"))
from pointer_fidelity_test import generate_until_math_done, RPNTokenizer, GPT

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model_path = "rpn_llm/models/ut0.5M_2l_6h_192e_mlp3_phaseMask_True_gated_1-22_phase_lean_32000.pt"

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = GPT(checkpoint['config']).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")

with open("rpn_llm/analysis/fidelity_benchmark.json", "r") as f:
    benchmark = json.load(f)

print(f"Showing raw 32k outputs for 5 samples...")

for sample in benchmark["long"][:5]:
    prompt = sample.split("[MATH]")[0] + "[MATH]"
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    # 1. Generate normal scratchpad
    out_idx = generate_until_math_done(model, idx, tokenizer, max_new_tokens=200)
    generated_str = tokenizer.decode(out_idx[0].tolist())
    
    print("\n" + "="*50)
    print(f"PROMPT: {prompt[-30:]}")
    
    # Show the model's "natural" scratchpad (before we hack it)
    try:
        math_part = generated_str.split("[MATH]")[1].split("[ANS]")[0]
        print(f"NATURAL SCRATCHPAD (32k): {math_part[:100]}...")
    except:
        print(f"NATURAL SCRATCHPAD (32k): [FAILED TO GENERATE MATH]")
        continue

    # 2. Hack it and see what it does when asked for the answer
    original_digits = re.findall(r"=(\d):", generated_str)
    if not original_digits: continue
    
    hacked_chunks = []
    for chunk in math_part.split(':'):
        if '=' in chunk:
            left, _ = chunk.split('=')
            hacked_chunks.append(f"{left}=9")
        else:
            hacked_chunks.append(chunk)
            
    hacked_scratchpad = prompt + ":".join(hacked_chunks) + ":"
    idx_hack = torch.tensor(tokenizer.encode(hacked_scratchpad), dtype=torch.long).unsqueeze(0).to(device)
    
    final_out = idx_hack
    ans_token = tokenizer.vocab.get("[ANS]", -1)
    for _ in range(30): # generate a short bit after hacked scratchpad
        logits, _ = model(final_out)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        final_out = torch.cat((final_out, next_token), dim=1)
        if next_token.item() == ans_token: break
            
    raw_completion = tokenizer.decode(final_out[0].tolist())
    after_hack = raw_completion[len(hacked_scratchpad):]
    print(f"OUTPUT AFTER HACK: {after_hack}")
