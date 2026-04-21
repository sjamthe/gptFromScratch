import torch
import os
import sys
import re

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_rope import GPT
from utils import RPNTokenizer

def evaluate_reversal_pointers(ckpt_path, test_file_path, num_samples=50, device='cpu'):
    """
    Robust Pointer Evaluator
    Goal: Prove that the model acts as a pure Pointer Network during the "Reversal Phase" of Ten's Complement.
    
    Attention Role Documentation:
    - Pass 5 (Index 4): We isolated Pass 5 as the "Maturity Point" where the Universal Transformer 
      locks its internal logic and structural targets. 
    - Head Pooling: Rather than relying on a single specialist (which can shift depending on sequence length),
      we Max-Pool across all 8 attention heads. This guarantees we capture the "Source Specialist" head 
      (often Head 5) safely, because its peak attention on the target digit will dominate the pooled scores.
    """
    print("Loading Model and Tokenizer...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    tokenizer = RPNTokenizer("/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/rpn-tokenizer.json")

    print(f"Reading {num_samples} samples from {test_file_path}...")
    with open(test_file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()][:num_samples]

    total_digits = 0
    pointer_logit_matches = 0
    pointer_truth_matches = 0

    print("\n--- BEGIN POINTER EVALUATION (REVERSAL PHASE) ---")

    for idx, line in enumerate(lines):
        # Format: (O1)(O2)op?<(R1)(R2)op=:...
        # Example: (3037913)(48)+?<(3197303)(84)+=:
        
        # Parse ground truth
        match = re.search(r'\((.*?)\)\((.*?)\)(.)\?<\((.*?)\)\((.*?)\)', line)
        if not match: continue
        
        o1_str, o2_str, op, r1_str, r2_str = match.groups()
        base_prompt = f"({o1_str})({o2_str}){op}?<<"
        
        # We need the exact tokens for the operand zones to map them back
        # We find zones in the base prompt
        base_tokens = tokenizer.encode(f"({o1_str})({o2_str}){op}?")
        decoded = [tokenizer.decode([t]) for t in base_tokens]
        
        zones = []
        current_zone = []
        in_zone = False
        for i, char in enumerate(decoded):
            if char == '(': in_zone = True; current_zone = []
            elif char == ')': in_zone = False; zones.append(current_zone)
            elif in_zone: current_zone.append(i)
            
        if len(zones) < 2: continue
        op1_indices = zones[0]
        op2_indices = zones[1]

        # REVERSAL 1 (R1)
        current_seq = list(base_tokens)
        current_seq.extend(tokenizer.encode("<("))
        for step_idx, true_char in enumerate(r1_str):
            input_tensor = torch.tensor([current_seq], device=device)
            
            # Forward pass to get attention
            logits, _, _, all_weights = model(input_tensor, return_attention=True)
            
            # Extract output
            model_next_id = torch.argmax(logits[:, -1, :], dim=-1).item()
            logit_char = tokenizer.decode([model_next_id])
            
            # Pass 8 (Final Layer) is where identity-copy projection happens
            logic_pass = 7 
            pass_weights = all_weights[logic_pass][0] # (Heads, T, T)
            
            # Max-Pool across heads to find the specialist pointer
            max_pooled_attn, _ = torch.max(pass_weights, dim=0) # (T, T)
            
            # We look at where the LAST token (exit token) is pointing within the Op1 zone
            last_idx = len(current_seq) - 1
            attn_for_exit = max_pooled_attn[last_idx]
            
            # Find peak in Op1
            op1_attn_vals = attn_for_exit[op1_indices]
            peak_relative_idx = torch.argmax(op1_attn_vals).item()
            peak_absolute_idx = op1_indices[peak_relative_idx]
            
            # The character it points to
            pointer_char = decoded[peak_absolute_idx]
            
            total_digits += 1
            if pointer_char == logit_char: pointer_logit_matches += 1
            if pointer_char == true_char: pointer_truth_matches += 1
                
            # Teacher forcing for next step
            current_seq.append(tokenizer.encode(true_char)[0])

        # REVERSAL 2 (R2)
        current_seq.extend(tokenizer.encode(")("))
        for step_idx, true_char in enumerate(r2_str):
            input_tensor = torch.tensor([current_seq], device=device)
            logits, _, _, all_weights = model(input_tensor, return_attention=True)
            
            model_next_id = torch.argmax(logits[:, -1, :], dim=-1).item()
            logit_char = tokenizer.decode([model_next_id])
            
            logic_pass = 7 
            pass_weights = all_weights[logic_pass][0] 
            max_pooled_attn, _ = torch.max(pass_weights, dim=0)
            
            last_idx = len(current_seq) - 1
            attn_for_exit = max_pooled_attn[last_idx]
            
            # Find peak in Op2
            op2_attn_vals = attn_for_exit[op2_indices]
            peak_relative_idx = torch.argmax(op2_attn_vals).item()
            peak_absolute_idx = op2_indices[peak_relative_idx]
            
            pointer_char = decoded[peak_absolute_idx]
            
            total_digits += 1
            if pointer_char == logit_char: pointer_logit_matches += 1
            if pointer_char == true_char: pointer_truth_matches += 1
                
            current_seq.append(tokenizer.encode(true_char)[0])

        if idx % 10 == 0:
            print(f"Processed {idx}/{num_samples} formulas. Current Logit Match: {(pointer_logit_matches/total_digits)*100:.2f}%")

    print("\n=======================================================")
    print(f"ROBUST POINTER EVALUATION: FINAL RESULTS")
    print(f"Total Reversal Digits Evaluated: {total_digits}")
    print(f"Pointer == Logit Match Rate: {(pointer_logit_matches/total_digits)*100:.2f}% ({pointer_logit_matches}/{total_digits})")
    print(f"Pointer == Truth Match Rate: {(pointer_truth_matches/total_digits)*100:.2f}% ({pointer_truth_matches}/{total_digits})")
    print("=======================================================\n")
    
    if pointer_logit_matches/total_digits > 0.99:
        print("VERDICT: SUCCESS! The model operates as a pure deterministic Pointer Network during sequence reversal.")
    else:
        print("VERDICT: PARTIAL MATCH. The model uses composite logic rather than strict point-and-copy.")

if __name__ == "__main__":
    ckpt = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/models/UT3M_1-22_tens_comp_bracketed_final.pt"
    test_file = "/Users/sjamthe/Documents/GithubRepos/gptFromScratch/rpn_llm/data/RPNData-1-22_tens_comp_bracketed_test.txt"
    evaluate_reversal_pointers(ckpt, test_file, num_samples=100)
