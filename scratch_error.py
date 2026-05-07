import torch
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "rpn_llm"))
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

samples = benchmark["long"][:5]

for prompt, expected in samples:
    # Need to extract the prompt correctly.
    # Actually `expected` in the benchmark is the FULL string: "[BOS]...[MATH]...[ANS]".
    # We should just pass the prompt part to generate_until_math_done.
    
    # Expected format: "[BOS]a b+? [REV]a_rev b_rev+="
    prompt_str = expected.split("[MATH]")[0] + "[MATH]"
    
    # generate_until_math_done takes the prompt and generates until [ANS]
    idx = torch.tensor([tokenizer.encode(prompt_str)], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(150):
            logits, _ = model(idx)
            next_tok = torch.argmax(logits[0, -1, :])
            idx = torch.cat([idx, next_tok.view(1, 1)], dim=1)
            if next_tok.item() == tokenizer.vocab.get("[ANS]", -1) or next_tok.item() == tokenizer.vocab.get("[END]", -1):
                break
                
    generated = tokenizer.decode(idx[0].tolist())
    if generated != expected:
        print("FAILED")
        print("Expected :", expected[len(prompt_str):])
        print("Generated:", generated[len(prompt_str):])
        print("-" * 40)
