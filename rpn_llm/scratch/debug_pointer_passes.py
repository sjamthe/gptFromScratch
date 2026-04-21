import torch
import os, sys, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_rope import GPT
from utils import RPNTokenizer

ckpt_path = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/models/UT3M_1-22_tens_comp_bracketed_final.pt"
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model = GPT(checkpoint['config'])
model.load_state_dict(checkpoint['model'])
model.eval()
tokenizer = RPNTokenizer("/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/rpn-tokenizer.json")

base = tokenizer.encode("(3037913)(48)+?")
current_seq = base + tokenizer.encode("<(")
idx = torch.tensor([current_seq])
logits, _, _, all_weights = model(idx, return_attention=True)

zones = []
in_zone = False
current_zone = []
for i, char in enumerate([tokenizer.decode([t]) for t in base]):
    if char == '(': in_zone = True; current_zone = []
    elif char == ')': in_zone = False; zones.append(current_zone)
    elif in_zone: current_zone.append(i)
op1_indices = zones[0]
decoded = [tokenizer.decode([t]) for t in base]

for p in range(len(all_weights)):
    max_attn, _ = torch.max(all_weights[p][0], dim=0)
    attn = max_attn[-1, op1_indices]
    peak = torch.argmax(attn).item()
    print(f"Pass {p+1}: Points to '{decoded[op1_indices[peak]]}' (val: {attn[peak]:.4f})")
