import os
import sys
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "rpn3_llm"))

from utils import RPNTokenizer
from model_rope import GPT

def trace():
    checkpoint_path = "rpn3_llm/models/ut1.8M_2l_8h_384e_mlp4_phaseMask_True_sft_1-14_7num_BOS_352000.pt"
    device = 'cpu' # Cpu is fine for structure inspection
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    config.universal = True
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    tokenizer = RPNTokenizer("rpn3_llm/rpn-tokenizer.json")
    prompt_str = "[BOS]57 313733-?[REV]75[SEP]3"
    tokens = tokenizer.encode(prompt_str)
    inputs = torch.tensor([tokens], dtype=torch.long)
    
    is_phase_shift = (inputs == 10) | (inputs == 11) | (inputs == 12)
    full_phase_ids = is_phase_shift.cumsum(dim=-1)

    with torch.no_grad():
        logits, _, all_weights = model(inputs, return_attention=True, full_phase_ids=full_phase_ids)
        
    print(f"all_weights type: {type(all_weights)}")
    print(f"all_weights length: {len(all_weights)}")
    for i, w in enumerate(all_weights):
        print(f"Item {i} type: {type(w)}")
        if isinstance(w, tuple) or isinstance(w, list):
            print(f"  Length: {len(w)}")
            for j, val in enumerate(w):
                if val is not None:
                    print(f"    Sub-item {j} shape: {val.shape}")
                else:
                    print(f"    Sub-item {j} is None")
        else:
            print(f"  Value: {w}")

if __name__ == "__main__":
    trace()
