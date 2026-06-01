import os
import sys
import torch
import random

# Add paths to sys.path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "rpn3_llm"))

from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

def reverse_string(s):
    return s[::-1]

def run_reversal_test(model, tokenizer, device, length, num_trials=50):
    model.eval()
    correct = 0
    eos_id = tokenizer.encode("[EOS]")[0]
    unk_id = tokenizer.encode("[UNK]")[0]
    
    for _ in range(num_trials):
        digits = "".join([str(random.randint(0, 9)) for _ in range(length)])
        prompt_str = f"[REV]{digits}[ANS]"
        expected_str = reverse_string(digits) + "[EOS]"
        
        prompt_tokens = tokenizer.encode(prompt_str)
        idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        
        generated = []
        # Allow extra tokens in case of negative or signs, but reversal is just length+1 (with EOS)
        for _ in range(length + 5):
            is_phase_shift = (idx == 10) | (idx == 11) | (idx == 12)
            full_phase_ids = is_phase_shift.cumsum(dim=-1)
            
            with torch.no_grad():
                logits, _ = model(idx, full_phase_ids=full_phase_ids)
            logits = logits[0, -1, :]
            next_id = torch.argmax(logits).item()
            generated.append(next_id)
            idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
            
            if next_id == eos_id or next_id == unk_id:
                break
                
        pred_str = tokenizer.decode(generated).strip()
        if pred_str == expected_str:
            correct += 1
            
    return (correct / num_trials) * 100.0

def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tokenizer = RPNTokenizer("rpn3_llm/rpn-tokenizer.json")
    
    models_to_test = {
        "Old Model (cnt2_crd2)": "rpn3_llm/models/ut2.1M_2l_8h_384e_mlp4_phaseMask_True_cnt2_crd2_sft_1-6_4num_BOS_200000.pt",
        "New Model (96k snapshot)": "rpn3_llm/models/ut2.1M_2l_8h_384e_mlp4_phaseMask_True_cnt2_crd2_digitAbs_freezeCoordScale_sft_1-6_4num_BOS_96000.pt",
        "New Model (300k checkpoint)": "rpn3_llm/models/ut2.1M_2l_8h_384e_mlp4_phaseMask_True_cnt2_crd2_digitAbs_freezeCoordScale_sft_1-6_4num_BOS_300000.pt"
    }
    
    loaded_models = {}
    for name, path in models_to_test.items():
        if os.path.exists(path):
            print(f"Loading {name} from {path}...")
            ckpt = torch.load(path, map_location=device, weights_only=False)
            config = ckpt['config']
            model = GPT(config)
            model.load_state_dict(ckpt['model'])
            model.to(device)
            loaded_models[name] = model
        else:
            print(f"WARNING: Could not find checkpoint file for {name} at {path}")
            
    print("\nStarting evaluation of [REV]{digits}[ANS] reversal capability across lengths 2 to 16...")
    header = f"{'Length':<8} | "
    for name in loaded_models.keys():
        header += f"{name:<25} | "
    print(header)
    print("-" * (8 + 3 + 28 * len(loaded_models)))
    
    for length in range(2, 17):
        accs = {}
        row_str = f"{length:<8d} | "
        for name, model in loaded_models.items():
            acc = run_reversal_test(model, tokenizer, device, length, num_trials=50)
            row_str += f"{acc:>23.1f}% | "
        print(row_str)

if __name__ == "__main__":
    main()
