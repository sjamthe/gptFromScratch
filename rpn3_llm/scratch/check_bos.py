import torch
import numpy as np
from utils import DataLoaderLite, RPNTokenizer
tokenizer = RPNTokenizer("rpn3_llm/rpn-tokenizer.json")
loader = DataLoaderLite(B=1, T=768, split="val")
x, y = loader.next_batch()
bos_id = tokenizer.encode("[BOS]")[0]
nl_id = tokenizer.encode("\n")[0]
print(f"bos_id: {bos_id}, nl_id: {nl_id}")
bos_indices = (y == bos_id).nonzero()
nl_indices = (y == nl_id).nonzero()
print(f"BOS in y: {bos_indices}")
print(f"NL in y: {nl_indices}")
