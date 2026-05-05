import torch
import os
import sys

# Add current and rpn_llm directory to sys.path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'rpn_llm'))

import model_rope
from model_rope import GPTConfig

ckpt_path = 'rpn_llm/models/ut1.8M_phaseMask_True_1-22_phase_lean_48000.pt'
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(ckpt['config'])
else:
    print(f"File not found: {ckpt_path}")
