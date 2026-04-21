import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_rope import GPT
from utils import RPNTokenizer

ckpt_path = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/models/UT3M_1-22_tens_comp_bracketed_final.pt"
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model = GPT(checkpoint['config'])
model.load_state_dict(checkpoint['model'])
model.eval()
tokenizer = RPNTokenizer("/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/rpn-tokenizer.json")

prompt = "(149)(142)+?"
tokens = tokenizer.encode(prompt)
idx = torch.tensor([tokens])

out = model.generate(idx, max_new_tokens=10, temperature=0.0)
print(f"Generated: {tokenizer.decode(out[0].tolist())}")
