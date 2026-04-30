import torch
import os
import sys

# Add the directory containing model_rope to path
sys.path.append(os.path.abspath("rpn_llm"))

path = "rpn_llm/models/rope3.6M_1-22_uniform_BOS_64000.pt"
out_path = "rpn_llm/models/rope3.6M_1-22_uniform_BOS_64000_stripped.pt"

if os.path.exists(path):
    print(f"Loading {path}...")
    # Using cpu to avoid taking up GPU memory for this large file
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    stripped_checkpoint = {
        'model': checkpoint['model'],
        'config': checkpoint['config'],
        'step': checkpoint.get('step', 64000),
    }
    
    print(f"Saving stripped version to {out_path}...")
    torch.save(stripped_checkpoint, out_path)
    
    old_size = os.path.getsize(path) / (1024*1024)
    new_size = os.path.getsize(out_path) / (1024*1024)
    print(f"Done! Reduced from {old_size:.1f} MB to {new_size:.1f} MB")
else:
    print(f"File {path} not found.")
