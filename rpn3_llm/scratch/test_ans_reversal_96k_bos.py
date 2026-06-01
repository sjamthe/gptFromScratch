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

def run_reversal_test(model, tokenizer, device, length, with_bos=False, num_trials=20):
    model.eval()
    correct = 0
    eos_id = tokenizer.encode("[EOS]")[0]
    unk_id = tokenizer.encode("[UNK]")[0]
    
    for _ in range(num_trials):
        digits = "".join([str(random.randint(0, 9)) for _ in range(length)])
        if with_bos:
            prompt_str = f"[BOS][REV]{digits}[ANS]"
        else:
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
    
    checkpoint_path = "rpn3_llm/models/ut2.1M_2l_8h_384e_mlp4_phaseMask_True_cnt2_crd2_digitAbs_freezeCoordScale_sft_1-6_4num_BOS_96000.pt"
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Could not find checkpoint file at {checkpoint_path}")
        sys.exit(1)
        
    print(f"Loading model from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = GPT(config)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    
    print("\nEvaluating [REV]{digits}[ANS] reversal (Without BOS vs With BOS)...")
    print(f"{'Length':<8} | {'Without BOS':<15} | {'With BOS':<15}")
    print("-" * 45)
    
    for length in range(2, 11):
        acc_no_bos = run_reversal_test(model, tokenizer, device, length, with_bos=False, num_trials=20)
        acc_bos = run_reversal_test(model, tokenizer, device, length, with_bos=True, num_trials=20)
        print(f"{length:<8d} | {acc_no_bos:>12.1f}% | {acc_bos:>12.1f}%")

if __name__ == "__main__":
    main()
