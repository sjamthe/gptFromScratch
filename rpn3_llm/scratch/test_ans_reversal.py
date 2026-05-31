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

def run_reversal_test(model, tokenizer, device, length, num_trials=20):
    model.eval()
    correct = 0
    eos_id = tokenizer.encode("[EOS]")[0]
    unk_id = tokenizer.encode("[UNK]")[0]
    
    # We generate digits of given length
    for _ in range(num_trials):
        digits = "".join([str(random.randint(0, 9)) for _ in range(length)])
        # prompt: [BOS][REV][MATH][REV]{digits}[ANS]
        # We must use this exact prefix to trigger the Phase 3 -> Phase 4 transition (reversal),
        # otherwise a Phase 1 -> Phase 2 transition is triggered, which the model interprets
        # as the MATH calculation phase (doing column math on the units digit and terminating).
        prompt_str = f"[BOS][REV][MATH][REV]{digits}[ANS]"
        expected_str = reverse_string(digits) + "[EOS]"
        
        prompt_tokens = tokenizer.encode(prompt_str)
        idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        
        generated = []
        for _ in range(length + 5): # generate up to length + 2 tokens
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
        else:
            print(f"Failed trial: {prompt_str} {digits} -> {expected_str} vs {pred_str}")
            
    return (correct / num_trials) * 100.0

def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tokenizer = RPNTokenizer("rpn3_llm/rpn-tokenizer.json")
    
    # Check paths
    model_paths = {
        "Base Model (ut1.8M)": [
            "rpn3_llm/models/ut1.8M_2l_8h_384e_mlp4_phaseMask_True_sft_1-6_4num_BOS_200000.pt",
        ],
        "New Model (ut2.1M)": [
            "rpn3_llm/models/ut2.1M_2l_8h_384e_mlp4_phaseMask_True_cnt2_crd2_sft_1-6_4num_BOS_200000.pt",
        ]
    }
    
    loaded_models = {}
    for name, paths in model_paths.items():
        found = False
        for p in paths:
            if os.path.exists(p):
                print(f"Loading {name} from {p}...")
                ckpt = torch.load(p, map_location=device, weights_only=False)
                config = ckpt['config']
                # Create model
                model = GPT(config)
                model.load_state_dict(ckpt['model'])
                model.to(device)
                loaded_models[name] = model
                found = True
                break
        if not found:
            print(f"ERROR: Could not find checkpoint file for {name} in {paths}")
            sys.exit(1)
            
    print("\nStarting evaluation of ANS-phase reversal capability across lengths 1 to 22...")
    print(f"{'Length':<8} | {'Base Model Accuracy':<22} | {'New Model Accuracy':<22}")
    print("-" * 60)
    
    for length in range(2,5):
        accs = {}
        for name, model in loaded_models.items():
            print(f"Testing {name} for length {length}...")
            acc = run_reversal_test(model, tokenizer, device, length, num_trials=20)
            accs[name] = acc
            
        print(f"{length:<8d} | {accs['Base Model (ut1.8M)']:>20.1f}% | {accs['New Model (ut2.1M)']:>20.1f}%")

if __name__ == "__main__":
    main()
