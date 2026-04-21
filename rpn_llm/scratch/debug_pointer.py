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

line = "(3037913)(48)+?<(3197303)(84)+=:"
match = re.search(r'\((.*?)\)\((.*?)\)(.)\?<\((.*?)\)\((.*?)\)', line)
o1_str, o2_str, op, r1_str, r2_str = match.groups()

base_tokens = tokenizer.encode(f"({o1_str})({o2_str}){op}?")
decoded = [tokenizer.decode([t]) for t in base_tokens]

print(f"Decoded Prompt: {decoded}")
zones = []
in_zone = False
current_zone = []
for i, char in enumerate(decoded):
    if char == '(': in_zone = True; current_zone = []
    elif char == ')': in_zone = False; zones.append(current_zone)
    elif in_zone: current_zone.append(i)

op1_indices = zones[0]
print(f"Op1 indices: {op1_indices}")

current_seq = list(base_tokens)
current_seq.extend(tokenizer.encode("<("))

print(f"\nReversal 1 String: {r1_str}")
for step_idx, true_char in enumerate(r1_str[:3]):
    idx = torch.tensor([current_seq])
    logits, _, _, all_weights = model(idx, return_attention=True)
    model_next_id = torch.argmax(logits[:, -1, :], dim=-1).item()
    logit_char = tokenizer.decode([model_next_id])
    
    pass_weights = all_weights[4][0] 
    max_pooled_attn, _ = torch.max(pass_weights, dim=0)
    last_idx = len(current_seq) - 1
    
    attn_for_exit = max_pooled_attn[last_idx]
    op1_attn = attn_for_exit[op1_indices]
    
    peak_rel = torch.argmax(op1_attn).item()
    peak_abs = op1_indices[peak_rel]
    pointer_char = decoded[peak_abs]
    
    print(f"--- Step {step_idx} | Current Seq suffix: {tokenizer.decode(current_seq[-5:])}")
    print(f"Exit Token evaluated: {tokenizer.decode([current_seq[-1]])}")
    print(f"Pointer points to [{peak_abs}] '{pointer_char}' (val: {op1_attn[peak_rel]:.4f})")
    print(f"Logit predicts: '{logit_char}'")
    print(f"Ground Truth: '{true_char}'")
    current_seq.append(tokenizer.encode(true_char)[0])
