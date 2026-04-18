import os
import re
import torch
import json
import time
from collections import defaultdict
from model_rope import GPT, GPTConfig
from utils import RPNTokenizer, DataLoaderLite

VALIDATION_SET_RATIO = 0.025

def calculate_carries(a_str, b_str, op):
    a, b = int(a_str), int(b_str)
    if op == '-' and a < b:
        a_str, b_str = b_str, str(a).zfill(len(b_str))
        
    a_digits = [int(x) for x in reversed(a_str)]
    b_digits = [int(x) for x in reversed(b_str)]
    
    max_len = max(len(a_digits), len(b_digits))
    a_digits += [0] * (max_len - len(a_digits))
    b_digits += [0] * (max_len - len(b_digits))
    
    carries = 0
    carry_val = 0
    if op == '+':
        for i in range(max_len):
            if a_digits[i] + b_digits[i] + carry_val > 9:
                carries += 1
                carry_val = 1
            else:
                carry_val = 0
    elif op == '-':
        for i in range(max_len):
            if a_digits[i] - b_digits[i] - carry_val < 0:
                carries += 1
                carry_val = 1
            else:
                carry_val = 0
    return carries

def validate_model(checkpoint_path, test_file_path, output_fail_path):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
    
    # 2. Parse Test File
    print(f"Parsing test file {test_file_path}...")
    with open(test_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # 3. Group by prompt token length to avoid padding issues
    # A prompt is everything up to and including the '=' sign and the trailing space. 
    length_groups = defaultdict(list)
    eq_id = tokenizer.encode("?")[0]
    nl_id = tokenizer.encode("\n")[0]
    
    print("Grouping by prompt length...")
    for line in lines:
        if "?" not in line:
            continue
        
        # Find index of '?'
        try:
            sep_idx = line.index("?")
        except ValueError:
            continue
            
        length_groups[sep_idx + 1].append(line.rstrip())
        
    # free lines memory
    del lines
    
    print(f"Grouped into {len(length_groups)} different prompt lengths.")
    
    # 4. Batched Generation
    max_batch_size = 256 # Dropped surgically to prevent Apple MPS fragmenting the KV Cache memory allocation bounds!
    failures = []
    total_processed = 0
    total_correct = 0
    
    total_by_carry = defaultdict(int)
    correct_by_carry = defaultdict(int)
    
    edge_stats = {
        'zero_operand': {'total': 0, 'correct': 0},
        'negative_result': {'total': 0, 'correct': 0},
        'normal': {'total': 0, 'correct': 0}
    }
    
    # Max generation steps for an answer
    # A padded 5-digit math evaluates to ~112 tokens dynamically, pushing past the bounds of 96 natively!
    # For Ten's Complement subtraction, sequences can reach nearly 200 tokens.
    max_new_tokens = 256

    total_items = sum(len(items) for items in length_groups.values())
    start_time = time.time()
    overall_tokens_gen = 0

    # Ensure output file is freshly clean before append looping!
    with open(output_fail_path, "w", encoding="utf-8") as f:
        f.write("--- Real-time Validation Failures ---\n\n")

    max_rows = int(VALIDATION_SET_RATIO*total_items/len(length_groups))
    print(f"Beginning batched evaluation on {max_rows} rows per group...")
    accuracy_by_length = {}
    group_idx = 1
    for length, items in length_groups.items():
        print(f"\n--- Evaluating group {group_idx}/{len(length_groups)} (prompt length {length}) - {len(items)} items ---")
        group_total_processed = 0
        group_total_correct = 0
        group_idx += 1
        
        # Process in chunks of max_batch_size. only validate a fraction of the data
        for i in range(0, min(max_rows, len(items)), max_batch_size):
            batch_items = items[i:i+max_batch_size]
            B = len(batch_items)
            
            batch_prompt_tokens = []
            expected_strs = []
            prompt_strs = []
            
            for line_str in batch_items:
                line_tokens = tokenizer.encode(line_str)
                sep_idx = line_tokens.index(eq_id)
                prompt_tokens = line_tokens[:sep_idx + 1]
                
                batch_prompt_tokens.append(prompt_tokens)
                prompt_strs.append(tokenizer.decode(prompt_tokens).strip())
                expected_strs.append(line_str.split('?', 1)[1].strip())
            
            # Construct (B, L) tensor natively (no padding required because all logic lengths identically match)
            prompts = torch.tensor(batch_prompt_tokens, dtype=torch.long, device=device)
            
            # Generate max_new_tokens sequentially using Argmax
            idx = prompts
            
            t0 = time.time()
            
            past_kv = None
            for step in range(max_new_tokens):
                # If KV Cache is fully loaded, ONLY pass the raw isolated new single token forward!
                idx_cond = idx[:, -1:] if past_kv is not None else idx
                
                with torch.no_grad():
                    with torch.autocast(device, dtype=torch.bfloat16):
                        logits, _, past_kv = model(idx_cond, use_cache=True, past_key_values=past_kv)
                
                logits = logits[:, -1, :] # Pluck final step logits
                idx_next = torch.argmax(logits, dim=-1, keepdim=True) # Deterministic greedy decision
                idx = torch.cat((idx, idx_next), dim=1) # Append
                
                # Apple MPS has a structural bug where tight internal `torch.cat` matrices 
                # heavily defer Garbage Collection! Natively sync the caches securely!
                if device == "mps":
                    torch.mps.empty_cache()
            
            t1 = time.time()
            batch_time = t1 - t0
            overall_tokens_gen += B * max_new_tokens
            tokens_per_sec = (B * max_new_tokens) / batch_time if batch_time > 0 else 0
            
            total_processed += B
            group_total_processed += B
            pct_complete = (total_processed / total_items) * 100
                            
            # Extract and verify the generated characters
            for b in range(B):
                full_generated_tokens = idx[b].tolist()
                gen_answer_tokens = full_generated_tokens[length:]
                
                # Truncate string sequence heavily at newline (to match exactly)
                if nl_id in gen_answer_tokens:
                    nl_idx = gen_answer_tokens.index(nl_id)
                    gen_answer_tokens = gen_answer_tokens[:nl_idx+1]
                
                # Compare
                expected_str = expected_strs[b]
                predicted_str = tokenizer.decode(gen_answer_tokens).strip() 
                prompt_str = prompt_strs[b]
                
                # New regex-based operand extraction for (n1)(n2)op? format
                m = re.search(r"\((\d+)\)\((\d+)\)([+\-])\?", prompt_str)
                
                expected_ans_str = expected_str.split('>')[-1].split('[UNK]')[0].split('\n')[0].strip() if '>' in expected_str else ""
                predicted_ans_str = predicted_str.split('>')[-1].split('[UNK]')[0].split('\n')[0].strip() if '>' in predicted_str else ""
                
                carries = 0
                is_zero = False
                if m:
                    n1_str, n2_str, op = m.groups()
                    try:
                        p0_val = int(n1_str); p1_val = int(n2_str)
                        is_zero = (p0_val == 0 or p1_val == 0)
                        carries = calculate_carries(n1_str, n2_str, op)
                    except ValueError: pass
                
                is_neg = expected_ans_str.startswith('-')
                is_normal = not is_zero and not is_neg
                
                total_by_carry[carries] += 1
                if is_zero: edge_stats['zero_operand']['total'] += 1
                if is_neg: edge_stats['negative_result']['total'] += 1
                if is_normal: edge_stats['normal']['total'] += 1
                
                # Check absolute truth of final numerical output!
                if expected_ans_str != predicted_ans_str:
                    # Document failure
                    fail_str = f"Q: {prompt_str} | Expected: {expected_ans_str} | Predicted: {predicted_ans_str} | Full Expected: {expected_str} | Full Pred: {predicted_str}"
                    failures.append(fail_str)
                    # Stream directly to disk live!
                    with open(output_fail_path, "a", encoding="utf-8") as f:
                        f.write(fail_str + "\n")
                else:
                    total_correct += 1
                    group_total_correct += 1
                    correct_by_carry[carries] += 1
                    if is_zero: edge_stats['zero_operand']['correct'] += 1
                    if is_neg: edge_stats['negative_result']['correct'] += 1
                    if is_normal: edge_stats['normal']['correct'] += 1

            # print accuracy
            accuracy = (total_correct / total_processed) * 100
            group_accuracy = (group_total_correct / group_total_processed) * 100
            accuracy_by_length[length] = group_accuracy
            print(f"Progress: {total_processed}/{total_items} ({pct_complete:.2f}%) | Accuracy: {accuracy:.2f}% | Group {group_idx-1} Accuracy: {group_accuracy:.2f}% | Batch inference time: {batch_time:.2f}s | Tokens/sec: {tokens_per_sec:.1f}")

    # 5. Output Results
    accuracy = (total_correct / total_processed) * 100
    print(f"\n=====================")
    print(f"Validation Complete!")
    print(f"Total Evaluated: {total_processed}")
    print(f"Total Correct: {total_correct}")
    print(f"Total Failures: {len(failures)}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    with open(output_fail_path, "a", encoding="utf-8") as f:
        f.write(f"\nValidation Accuracy: {accuracy:.2f}% ({total_correct}/{total_processed})\n")
        f.write("=========================================\n\n")

        f.write("\n--- Breakdown by Prompt Length ---\n")
        f.write("Token Length | Total Items | Accuracy\n")
        for g in sorted(accuracy_by_length.keys()):
            stats = f"{g:2d} | {len(length_groups[g]):<10} | {accuracy_by_length[g]:.2f}%"
            f.write(stats + "\n")
            
        f.write("\n--- Breakdown by Carry Operations ---\n")
        f.write("Carries | Total   | Correct | Failures | Accuracy\n")
        
    print("\n--- Breakdown by Carry Operations ---")
    print("Carries | Total   | Correct | Failures | Accuracy")
    for c in sorted(total_by_carry.keys()):
        tot = total_by_carry[c]
        cor = correct_by_carry[c]   
        fls = tot - cor
        acc = (cor / tot) * 100 if tot > 0 else 0
        stats = f"{c:<8}| {tot:<8}| {cor:<8}| {fls:<8}| {acc:.2f}%"
        print(stats)
        with open(output_fail_path, "a", encoding="utf-8") as f:
            f.write(stats + "\n")
            
    with open(output_fail_path, "a", encoding="utf-8") as f:
        f.write("\n--- Edge Case Analysis ---\n")
        f.write(f"{'Category':<16} | {'Total':<8} | {'Correct':<8} | Accuracy\n")
        for cat, stats in edge_stats.items():
            tot = stats['total']
            cor = stats['correct']
            acc = (cor / tot) * 100 if tot > 0 else 0
            f.write(f"{cat:<16} | {tot:<8} | {cor:<8} | {acc:.2f}%\n")
            
    print("\n--- Edge Case Analysis ---")
    print(f"{'Category':<16} | {'Total':<8} | {'Correct':<8} | Accuracy")
    for cat, stats in edge_stats.items():
        tot = stats['total']
        cor = stats['correct']
        acc = (cor / tot) * 100 if tot > 0 else 0
        print(f"{cat:<16} | {tot:<8} | {cor:<8} | {acc:.2f}%")
        
    print(f"\nFailures dumped to {output_fail_path}")

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "rpn_llm/models/rope25M_1-22_tens_comp_bracketed_final.pt"
    validate_model(model_path, "rpn_llm/data/RPNData-1-22_tens_comp_bracketed_test.txt",    
                            "rpn_llm/results/rope25M_1-22_tens_comp_bracketed_final_failures.txt")
