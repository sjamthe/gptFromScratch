import torch
import torch.nn.functional as F
import os
import sys
import random
import re

# Add parent dir to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

def generate_random_problem(digits):
    a_int = random.getrandbits(digits * 4) # roughly large enough
    b_int = random.getrandbits(digits * 4)
    # Ensure they are the right number of digits
    a = str(random.randint(10**(digits-1), 10**digits - 1))
    b = str(random.randint(10**(digits-1), 10**digits - 1))
    
    # Calculate ground truth for math verification
    expected_sum = str(int(a) + int(b))
    
    # Return in reversed prompt format
    return f"[BOS]{a} {b}+? [REV]{a[::-1]} {b[::-1]}+=", expected_sum

def run_length_test(model_path, digits, trials=20, device='cpu'):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
    correct_count = 0
    
    for _ in range(trials):
        prompt, ground_truth = generate_random_problem(digits)
        idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        
        # Generate full answer
        # We need a large enough max_new_tokens for long problems
        max_gen = digits * 4 + 50 
        
        full_out = idx
        for _ in range(max_gen):
            logits, _ = model(full_out)
            logits = logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            full_out = torch.cat((full_out, next_token), dim=1)
            if next_token.item() == tokenizer.vocab.get("[EOS]", -1):
                break
        
        completion = tokenizer.decode(full_out[0].tolist())
        
        # Extract the final answer after [ANS]
        if "[ANS]" in completion:
            ans_part = completion.split("[ANS]")[-1].split("[EOS]")[0].strip()
            # The model outputs reversed digits usually, but let's check
            # Ground truth is e.g. "123", model might output "321" (reversed result)
            if ans_part == ground_truth[::-1]:
                correct_count += 1
                
    accuracy = (correct_count / trials) * 100
    return accuracy

if __name__ == "__main__":
    random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Running OOD Length Stress Test on {device}...")
    
    models = [
        ("UT 1.8M (MLP4)", "rpn_llm/models/ut1.8M_phaseMask_True_1-22_phase_lean_48000.pt"),
        ("UT 1.5M (MLP3)", "rpn_llm/models/ut1.5M_mlp3_phaseMask_True_1-22_phase_lean_56000.pt"),
        ("UT 1.2M (MLP2)", "rpn_llm/models/ut1.2M_mlp2_phaseMask_True_1-22_phase_lean_64000.pt"),
        ("RoPE 2.4M", "rpn_llm/models/rope2.4M_phaseMask_True_1-22_phase_lean_80000.pt"),
        ("RoPE 3.6M", "rpn_llm/models/rope3.6M_phaseMask_True_1-22_phase_lean_64000.pt"),
    ]
    
    lengths = [25, 30, 35]
    trials_per_length = 20
    
    print(f"{'Model':<20} | {'25-dig':<8} | {'30-dig':<8} | {'35-dig':<8}")
    print("-" * 55)
    
    for name, path in models:
        if not os.path.exists(path):
            continue
        
        row = [name]
        for length in lengths:
            acc = run_length_test(path, length, trials=trials_per_length, device=device)
            row.append(f"{acc:>6.1f}%")
            
        print(f"{row[0]:<20} | {row[1]:<8} | {row[2]:<8} | {row[3]:<8}")
