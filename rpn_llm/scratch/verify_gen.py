import sys
import os
import random
# Add the project root to path
sys.path.append("/Users/sjamthe/Documents/GithubRepos/gptFromScratch")

from rpn_llm.RPNDataset import RPNDataset
from rpn_llm.utils import RPNTokenizer

tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
dataset = RPNDataset(
    num_samples=100,
    max_operands=2,
    operations=('+', '-'),
    max_number=10**22-1,
    tokenizer=tokenizer,
    max_seq_len=256
)

print("Generated 100 samples.")
for i in range(5):
    prompt, full, rev = dataset.examples[i]
    print(f"Sample {i+1}:")
    print(f"  Prompt: {prompt}")
    print(f"  Full:   {full}")
    print(f"  Reverse: {rev}")
    print("-" * 20)
