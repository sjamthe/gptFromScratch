"""
POINTER FIDELITY TEST (The "Gaslighting" Diagnostic)
=====================================================

This test measures the degree to which a model relies on its Attention Pointers (Logic) 
versus its MLP Weights (Memory) to generate the final answer digits (i.e., whether it actually reads its own scratchpad or ignores it).

METHODOLOGY:
1.  GENERATE: We provide a math prompt and let the model generate its natural scratchpad 
    (e.g., [REV]...[MATH]8+7=5:2+1+1=4:[ANS]).
    
2.  GASLIGHT (THE HACK): We identify the result digits of each intermediate addition step 
    in the scratchpad. We replace these with random "junk" digits.
    Example: 
        Original: 8+7=5: 2+1+1=4:
        Hacked:   8+7=9: 2+1+1=0:  <-- We changed the results to 9 and 0.

3.  STRIP SUMMARY: Models often generate a "summary" string of the answer right before 
     the [ANS] token. We strip this summary to force the model to re-read the 
     individual logic steps we just hacked.

4.  RESUME & HARVEST: We feed the hacked scratchpad back to the model and ask it for 
    the final answer.
    
5.  FIDELITY SCORING (Soft Fidelity):
    - Logic-Driven Model: Will "trust its eyes" and output the hacked digits (9 and 0).
    - Memory-Driven Model: Will "trust its memory" and output the true math (5 and 4).
    - FIDELITY SCORE: The % of generated digits that match our Hacked Junk.

HIGHER SCORE = More logical (Model follows pointers).
LOWER SCORE  = More memorized (Model ignores pointers and hallucinations its training data).
"""

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

def generate_until_math_done(model, idx, tokenizer, max_new_tokens=1024):
    """Generates until the scratchpad is finished (reaches [ANS] or [EOS])"""
    ans_token = tokenizer.vocab.get("[ANS]", -1)
    eos_token = tokenizer.vocab.get("[EOS]", -1)
    
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=1)
        
        if next_token.item() == ans_token or next_token.item() == eos_token:
            break
    return idx

def run_fidelity_test(model_path, test_samples, device='cpu', verbose=True):
    if verbose:
        print(f"\nTesting Model: {os.path.basename(model_path)}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
    
    followed_count = 0
    total_trials = 0
    
    for prompt in test_samples:
        # 1. Generate normal scratchpad first to find the stop point
        idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        full_gen = generate_until_math_done(model, idx, tokenizer)
        scratchpad = tokenizer.decode(full_gen[0].tolist())
        
        # 2. Find all digits after '=' in the scratchpad
        original_digits = re.findall(r"=(\d):", scratchpad)
        if not original_digits:
            continue
            
        # 3. Create "Gaslit" scratchpad
        hacked_digits = [str(random.randint(0, 9)) for _ in original_digits]
        if len(hacked_digits) > 0:
            hacked_digits[-1] = original_digits[-1] # Anchor
        
        digit_iter = iter(hacked_digits)
        hacked_scratchpad = re.sub(r"=(\d):", lambda m: f"={next(digit_iter)}:", scratchpad)
        
        if ":" in hacked_scratchpad:
            hacked_scratchpad = hacked_scratchpad.rsplit(":", 1)[0] + ":"
            
        # 4. Resume generation
        idx_hack = torch.tensor(tokenizer.encode(hacked_scratchpad), dtype=torch.long).unsqueeze(0).to(device)
        ans_token = tokenizer.vocab.get("[ANS]", -1)
        final_out = idx_hack
        for _ in range(50):
            logits, _ = model(final_out)
            logits = logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            final_out = torch.cat((final_out, next_token), dim=1)
            if next_token.item() == ans_token:
                break
        
        completion = tokenizer.decode(final_out[0].tolist())
        ans_indices = (final_out[0] == ans_token).nonzero()
        
        if len(ans_indices) > 0:
            ans_pos = ans_indices[0].item()
            completion_up_to_ans = tokenizer.decode(final_out[0, :ans_pos].tolist())
            harvested_part = completion_up_to_ans.rsplit(":", 1)[1].strip() if ":" in completion_up_to_ans else ""
        else:
            harvested_part = ""
        
        # 5. Score it
        expected_hacked = "".join(hacked_digits)
        matches = 0
        min_len = min(len(expected_hacked), len(harvested_part))
        for i in range(min_len):
            if expected_hacked[i] == harvested_part[i]:
                matches += 1
        
        fidelity_per_trial = (matches / len(expected_hacked)) if len(expected_hacked) > 0 else 0
        followed_count += fidelity_per_trial
        total_trials += 1
        
        if verbose and total_trials % 20 == 0:
            print(f"  Progress: {total_trials}/100, Current Avg: {(followed_count/total_trials)*100:.1f}%")
        
        if total_trials >= 100:
            break

    avg_fidelity = (followed_count / total_trials) * 100 if total_trials > 0 else 0
    if verbose:
        print(f"Fidelity Score: {avg_fidelity:.1f}% ({followed_count:.1f}/{total_trials})")
    return avg_fidelity

def generate_random_problem(digits):
    a = "".join([str(random.randint(1, 9)) if i == 0 else str(random.randint(0, 9)) for i in range(digits)])
    b = "".join([str(random.randint(1, 9)) if i == 0 else str(random.randint(0, 9)) for i in range(digits)])
    # Return in reversed prompt format
    return f"[BOS]{a} {b}+? [REV]{a[::-1]} {b[::-1]}+="

if __name__ == "__main__":
    import json
    
    # Set seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Load benchmark
    benchmark_path = "rpn_llm/analysis/fidelity_benchmark.json"
    if not os.path.exists(benchmark_path):
        print(f"Error: {benchmark_path} not found. Run create_fidelity_benchmark.py first.")
        sys.exit(1)
        
    with open(benchmark_path, "r") as f:
        benchmark = json.load(f)
    
    short_prompts = benchmark["short"]
    long_prompts = benchmark["long"]
    
    print(f"Loaded {len(short_prompts)} short and {len(long_prompts)} long trials from benchmark.")
    
    models_to_test = [
        "rpn_llm/models/ut0.4M_2l_6h_192e_mlp3_phaseMask_True_1-22_phase_lean_344000.pt",
        #"rpn_llm/models/ut0.2M_mlp1_phaseMask_True_1-22_phase_lean_64000.pt",
        #"rpn_llm/models/ut1.8M_phaseMask_True_1-22_phase_lean_48000.pt", # MLP4 (1.8M)
        #"rpn_llm/models/ut1.5M_mlp3_phaseMask_True_1-22_phase_lean_56000.pt", # MLP3 (1.5M)
        #"rpn_llm/models/ut1.2M_mlp2_phaseMask_True_1-22_phase_lean_64000.pt", # MLP2 (1.2M)
        #"rpn_llm/models/ut0.9M_mlp1_phaseMask_True_1-22_phase_lean_64000.pt", # MLP1 (0.9M)
        #"rpn_llm/models/rope3.6M_phaseMask_True_1-22_phase_lean_64000.pt", # RoPE 3.6M
    ]
    
    final_results = []
    
    for m_path in models_to_test:
        if not os.path.exists(m_path):
            continue
            
        # We need to run it twice for each model
        # One for short, one for long
        m_name = os.path.basename(m_path).split("_")[0] + "_" + os.path.basename(m_path).split("_")[1]
        
        print(f"\nEvaluating {m_name}...")
        short_score = run_fidelity_test(m_path, short_prompts)
        long_score = run_fidelity_test(m_path, long_prompts)
        
        final_results.append({
            "name": m_name,
            "short": short_score,
            "long": long_score
        })
            
    print("\n" + "="*60)
    print(f"{'Model Name':<20} | {'Short (4-dig) Fidelity':<20} | {'Long (25-dig) Fidelity'}")
    print("-" * 60)
    for res in final_results:
        print(f"{res['name']:<20} | {res['short']:18.1f}% | {res['long']:18.1f}%")
    print("="*60)
